"""
ein notation:
b - batch
t - time
h - attention heads
i - contact bodies
j - terrain points
d - dimension
"""

from __future__ import annotations

from collections import namedtuple
from functools import partial

import einx
import torch
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from perceiver_pytorch import PerceiverBlock
from perceiver_pytorch.perceiver_pytorch import Attention as PerceiverAttention
from perceiver_pytorch.perceiver_pytorch import FeedForward as PerceiverFeedForward
from perceiver_pytorch.perceiver_pytorch import PreNorm
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch_einops_utils import masked_mean
from torch_einops_utils.stable_softmax import binary_cross_entropy_with_logits
from x_mlps_pytorch import MLP

# named tuples

CrossAttentionReturn = namedtuple('CrossAttentionReturn', [
    'body_features',
    'attn_logits',
    'contact_probs'
])

TerrainEncoderReturn = namedtuple('TerrainEncoderReturn', [
    'tokens',
    'context'
])

AttentionContactPlanningReturn = namedtuple('AttentionContactPlanningReturn', [
    'body_features',
    'attn_logits',
    'contact_probs',
    'occupancy_logits',
    'occupancy_probs',
    'occupancy_loss'
])

TerrainContactPlannerReturn = namedtuple('TerrainContactPlannerReturn', [
    'body_features',
    'attn_logits',
    'contact_probs',
    'occupancy_logits',
    'occupancy_probs',
    'terrain_tokens',
    'terrain_context',
    'occupancy_loss',
    'contact_loss'
])

ContactPlannerWithOutcomeModelReturn = namedtuple('ContactPlannerWithOutcomeModelReturn', [
    'plan',
    'outcome_model',
    'aux_loss'
])

# helpers

def exists(v):
    return v is not None

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

def to_3d_kernel(kernel_size):
    if isinstance(kernel_size, int):
        return (1, kernel_size, kernel_size)

    kernel_size = tuple(kernel_size)

    if len(kernel_size) == 2:
        return (1, *kernel_size)

    if len(kernel_size) == 3:
        return kernel_size

    raise ValueError('kernel_size must be int, a 2-tuple, or a 3-tuple')

LinearNoBias = partial(nn.Linear, bias = False)

# loss

def no_contact_target_from_soft_targets(
    soft_target_probs: Tensor,       # b i j
    point_mask: Tensor | None = None # b j
) -> Tensor:
    if exists(point_mask):
        soft_target_probs = einx.where(
            'b j, b i j, -> b i j',
            point_mask.bool(),
            soft_target_probs,
            0.
        )

    return (soft_target_probs.sum(dim = -1) <= 0).float()

def contact_probs_loss(
    attn_logits: Tensor,        # b h i (1 + j), no-contact at column 0
    soft_target_probs: Tensor,  # b i j
    point_mask: Tensor | None = None,
    contact_mask: Tensor | None = None
) -> Tensor:
    point_mask = point_mask.bool() if exists(point_mask) else None

    if exists(point_mask):
        soft_target_probs = einx.where(
            'b j, b i j, -> b i j',
            point_mask,
            soft_target_probs,
            0.
        )

    soft_target_probs = safe_div(soft_target_probs, soft_target_probs.sum(dim = -1, keepdim = True))
    no_contact_target = no_contact_target_from_soft_targets(soft_target_probs)

    target = torch.cat((
        no_contact_target[..., None],
        soft_target_probs * (1. - no_contact_target[..., None])
    ), dim = -1)

    log_probs = attn_logits.log_softmax(dim = -1)
    point_loss = -einsum(log_probs, target, 'b h i j, b i j -> b h i')

    valid = contact_mask.bool() if exists(contact_mask) else torch.ones_like(no_contact_target, dtype = torch.bool)
    return masked_mean(point_loss, rearrange(valid, 'b i -> b 1 i'))

def occupancy_loss(
    occupancy_logits: Tensor,   # b i
    occupancy_targets: Tensor,  # b i
    body_mask: Tensor | None = None
) -> Tensor:
    loss = binary_cross_entropy_with_logits(
        occupancy_logits,
        occupancy_targets,
        reduction = 'none'
    )

    return masked_mean(loss, body_mask.bool()) if exists(body_mask) else loss.mean()

# modules

class CausalConv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        kt, kh, kw = kernel_size

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pad = (
            (kw - 1) // 2, kw // 2,
            (kh - 1) // 2, kh // 2,
            kt - 1, 0
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad))

class TerrainEncoder(Module):
    def __init__(
        self,
        dim = 96,
        point_dim = 3,
        point_embed_dim = 16,
        channels = (16, 48),
        kernel_size = (1, 5, 5),
        context_dim = 64
    ):
        super().__init__()
        channels = tuple(channels)
        assert len(channels) > 0

        # int / 2-tuple -> spatial kernel with time kernel 1; 3-tuple -> (kT, kH, kW)
        self.kernel_size = to_3d_kernel(kernel_size)

        self.to_point = nn.Sequential(
            nn.Linear(point_dim, point_embed_dim),
            nn.ELU()
        )

        convs = []
        in_channels = 1

        for out_channels in channels:
            convs.extend((
                CausalConv3d(in_channels, out_channels, self.kernel_size),
                nn.ELU()
            ))
            in_channels = out_channels

        self.convs = nn.Sequential(*convs)
        self.to_tokens = nn.Linear(point_embed_dim + channels[-1], dim)
        self.to_context = MLP(dim, 64, context_dim)

        self.dim = dim
        self.context_dim = context_dim

    def forward(
        self,
        height: Tensor,                  # b t h w | b h w
        points: Tensor,                  # b t n 3 | b n 3
        point_mask: Tensor | None = None # b t n | b n
    ) -> TerrainEncoderReturn:
        if height.ndim == 3:
            height = rearrange(height, 'b h w -> b 1 h w')

        height = rearrange(height, 'b t h w -> b 1 t h w')

        if points.ndim == 3:
            points = rearrange(points, 'b n d -> b 1 n d')

        b, _, t, h, w = height.shape
        n = h * w

        assert points.shape[:2] == (b, t)
        assert points.shape[2] == n

        if exists(point_mask) and point_mask.ndim == 2:
            point_mask = rearrange(point_mask, 'b n -> b 1 n')

        height_feat = self.convs(height)
        height_feat = rearrange(height_feat, 'b c t h w -> b t (h w) c')

        tokens = self.to_tokens(torch.cat((self.to_point(points), height_feat), dim = -1))
        context = self.to_context(tokens)

        if exists(point_mask):
            context = context.masked_fill(~point_mask[..., None], -torch.inf)

        context = context.amax(dim = -2)

        if exists(point_mask):
            context = torch.where(torch.isfinite(context), context, torch.zeros_like(context))

        return TerrainEncoderReturn(tokens, context)

class FiLM(Module):
    def __init__(
        self,
        dim,
        hidden_dim = 64,
        cond_dim = None
    ):
        super().__init__()
        cond_dim = cond_dim if exists(cond_dim) else dim
        self.to_gamma_beta = MLP(cond_dim, hidden_dim, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        scaled = einx.multiply('b ... d, b d -> b ... d', x, gamma + 1.)
        return einx.add('b ... d, b d -> b ... d', scaled, beta)

class SelfAttention(Module):
    """pre-norm self attention / feedforward stack borrowed from perceiver block"""

    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        depth = 1,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, PerceiverAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, PerceiverFeedForward(dim, dropout = dropout))
            ])
            for _ in range(depth)
        ])

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        for self_attn, self_ff in self.layers:
            x = self_attn(x, mask = mask) + x
            x = self_ff(x) + x

        return x

class TalkingHeads(Module):
    """small residual mlp over the head axis, from https://arxiv.org/abs/2003.02436"""

    def __init__(
        self,
        heads,
        hidden_dim = 8
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(heads, hidden_dim, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, heads, 1, bias = False)
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, logits):
        return self.net(logits) + logits

class CrossAttention(Module):
    def __init__(
        self,
        dim = 96,
        heads = 32,
        dim_head = 3,
        point_dim = 3,
        dropout = 0.,
        talking_heads = True,
        talking_heads_hidden_dim = 8
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads

        self.scale = dim_head ** -0.5

        self.talking_heads = TalkingHeads(heads, talking_heads_hidden_dim) if talking_heads else None

        self.no_contact_key_value = nn.Parameter(torch.zeros(dim))

        # af2 attention gating, projected from the query input
        self.to_gate = nn.Linear(dim, inner_dim)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.constant_(self.to_gate.bias, 1.)

        self.to_point = nn.Linear(point_dim, dim)
        self.to_q = LinearNoBias(dim, inner_dim)
        self.to_kv = LinearNoBias(dim, inner_dim * 2)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        query_input: Tensor,              # b i d, becomes queries via to_q
        points: Tensor,                   # b j point_dim | b j d
        point_mask: Tensor | None = None, # b j
        point_features: bool = False      # points are already d-dimensional tokens
    ) -> CrossAttentionReturn:
        point_tokens = points if point_features else self.to_point(points)

        b = point_tokens.shape[0]
        no_contact = repeat(self.no_contact_key_value, 'd -> b 1 d', b = b)
        point_tokens = torch.cat((no_contact, point_tokens), dim = 1)

        if exists(point_mask):
            point_mask = torch.cat((point_mask.new_ones(b, 1), point_mask.bool()), dim = -1)

        q = self.to_q(query_input)
        k, v = self.to_kv(point_tokens).chunk(2, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        attn_logits = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        if exists(self.talking_heads):
            attn_logits = self.talking_heads(attn_logits)

        if exists(point_mask):
            attn_logits = einx.where(
                'b j, b h i j, -> b h i j',
                point_mask,
                attn_logits,
                torch.finfo(attn_logits.dtype).min
            )

        contact_probs = attn_logits.softmax(dim = -1)

        body_features = einsum(contact_probs, v, 'b h i j, b h j d -> b h i d')

        gate = self.to_gate(query_input)
        gate = rearrange(gate, 'b i (h d) -> b h i d', h = self.heads)
        body_features = body_features * gate.sigmoid()

        body_features = self.merge_heads(body_features)
        body_features = self.to_out(body_features)

        return CrossAttentionReturn(body_features, attn_logits, contact_probs)

class OccupancyPredictor(Module):
    def __init__(
        self,
        dim,
        context_dim,
        hidden_dims = (512, 512, 512)
    ):
        super().__init__()
        self.net = MLP(dim + context_dim, *hidden_dims, 1, squeeze_out = True)

    def forward(
        self,
        body_features: Tensor,  # b i d
        context: Tensor          # b c
    ) -> Tensor:                # b i
        context = repeat(context, 'b c -> b i c', i = body_features.shape[-2])
        return self.net(torch.cat((body_features, context), dim = -1))

class AttentionContactPlanning(Module):
    def __init__(
        self,
        dim = 96,
        heads = 32,
        dim_head = 3,
        point_dim = 3,
        num_bodies = 9,
        dropout = 0.,
        perceiver_depth = 2,
        perceiver_queries_contribute_kv = True,
        terrain_self_attend = False,
        terrain_self_attend_depth = 1,
        occupancy_from_attention = False,
        occupancy_context_dim = None,
        occupancy_hidden_dims = (512, 512, 512),
        talking_heads = True,
        talking_heads_hidden_dim = 8,
        film_timestep_cond = False
    ):
        super().__init__()
        self.body_embed = nn.Parameter(torch.randn(num_bodies, dim) * 0.02)
        self.film_timestep_cond = film_timestep_cond
        self.film = FiLM(dim, cond_dim = dim + 1 if film_timestep_cond else dim) # +1 for past / present flag
        self.perceiver = nn.ModuleList([
            PerceiverBlock(
                dim = dim,
                context_dim = dim,
                cross_heads = 1,
                latent_heads = heads,
                cross_dim_head = 64,
                latent_dim_head = dim_head,
                attn_dropout = dropout,
                ff_dropout = dropout,
                queries_contribute_kv = perceiver_queries_contribute_kv
            )
            for _ in range(perceiver_depth)
        ])
        self.terrain_self_attn = SelfAttention(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            depth = terrain_self_attend_depth,
            dropout = dropout
        ) if terrain_self_attend else None
        self.cross_attn = CrossAttention(
            dim,
            heads,
            dim_head,
            point_dim,
            dropout,
            talking_heads = talking_heads,
            talking_heads_hidden_dim = talking_heads_hidden_dim
        )

        occupancy_context_dim = occupancy_context_dim if exists(occupancy_context_dim) else dim

        # occupancy - faithful: mlp on cross attention output + pooled terrain context
        # or single source - fold context into body query upfront, occupancy is just 1 - no-contact prob
        self.occupancy_from_attention = occupancy_from_attention
        self.to_body_with_context = nn.Linear(dim + occupancy_context_dim, dim) if occupancy_from_attention else None
        self.occupancy_predictor = None if occupancy_from_attention else OccupancyPredictor(dim, occupancy_context_dim, occupancy_hidden_dims)

    def forward(
        self,
        context: Tensor,                    # b d
        points: Tensor,                     # b j point_dim | b j d
        point_mask: Tensor | None = None,   # b j
        point_features: bool = False,
        occupancy_context: Tensor | None = None, # b c
        occupancy_targets: Tensor | None = None, # b i
        body_mask: Tensor | None = None, # b i
        is_current_timestep: Tensor | bool = False # b | scalar, 1 being present
    ) -> AttentionContactPlanningReturn:
        body_embed = repeat(self.body_embed, 'i d -> b i d', b = context.shape[0])
        occupancy_context = occupancy_context if exists(occupancy_context) else context

        if exists(self.to_body_with_context):
            body_context = repeat(occupancy_context, 'b c -> b i c', i = self.body_embed.shape[0])
            body_embed = self.to_body_with_context(torch.cat((body_embed, body_context), dim = -1))

        film_cond = context

        if self.film_timestep_cond:
            is_current = torch.as_tensor(is_current_timestep, device = context.device, dtype = context.dtype)
            is_current = repeat(is_current, '-> b', b = context.shape[0]) if is_current.ndim == 0 else is_current
            film_cond = torch.cat((context, rearrange(is_current, 'b -> b 1')), dim = -1)

        queries = self.film(body_embed, film_cond)

        point_mask = point_mask.bool() if exists(point_mask) else None
        terrain_tokens = points if point_features else self.cross_attn.to_point(points)

        if exists(self.terrain_self_attn):
            terrain_tokens = self.terrain_self_attn(terrain_tokens, mask = point_mask)

        for block in self.perceiver:
            queries = block(queries, context = terrain_tokens, mask = point_mask)

        out = self.cross_attn(
            queries,
            terrain_tokens,
            point_mask = point_mask,
            point_features = True
        )

        if self.occupancy_from_attention:
            no_contact_probs = out.contact_probs[..., 0].mean(dim = 1) # b h i -> b i
            occupancy_probs = 1. - no_contact_probs
            occupancy_logits = occupancy_probs.clamp(1e-6, 1. - 1e-6).logit()
        else:
            occupancy_logits = self.occupancy_predictor(out.body_features, occupancy_context)
            occupancy_probs = occupancy_logits.sigmoid()

        loss = occupancy_loss(occupancy_logits, occupancy_targets, body_mask) if exists(occupancy_targets) else None

        return AttentionContactPlanningReturn(
            out.body_features,
            out.attn_logits,
            out.contact_probs,
            occupancy_logits,
            occupancy_probs,
            loss
        )

class TerrainContactPlanner(Module):
    def __init__(
        self,
        dim = 96,
        heads = 32,
        dim_head = 3,
        num_bodies = 9,
        proprio_dim = 375,
        goal_dim = 3,
        action_dim = 0,
        terrain_encoder: TerrainEncoder | None = None,
        context_hidden_dim = 128,
        dropout = 0.,
        perceiver_depth = 2,
        perceiver_queries_contribute_kv = True,
        terrain_self_attend = False,
        terrain_self_attend_depth = 1,
        occupancy_from_attention = False,
        occupancy_hidden_dims = (512, 512, 512),
        talking_heads = True,
        talking_heads_hidden_dim = 8,
        film_timestep_cond = False
    ):
        super().__init__()
        self.terrain_encoder = terrain_encoder if exists(terrain_encoder) else TerrainEncoder(dim = dim)
        assert self.terrain_encoder.dim == dim

        self.action_dim = action_dim

        self.to_context = MLP(
            proprio_dim + goal_dim + action_dim + self.terrain_encoder.context_dim,
            context_hidden_dim,
            context_hidden_dim,
            dim
        )

        self.attention = AttentionContactPlanning(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            point_dim = dim,
            num_bodies = num_bodies,
            dropout = dropout,
            perceiver_depth = perceiver_depth,
            perceiver_queries_contribute_kv = perceiver_queries_contribute_kv,
            terrain_self_attend = terrain_self_attend,
            terrain_self_attend_depth = terrain_self_attend_depth,
            occupancy_from_attention = occupancy_from_attention,
            occupancy_context_dim = self.terrain_encoder.context_dim,
            occupancy_hidden_dims = occupancy_hidden_dims,
            talking_heads = talking_heads,
            talking_heads_hidden_dim = talking_heads_hidden_dim,
            film_timestep_cond = film_timestep_cond
        )

    def forward(
        self,
        height: Tensor,                  # b t h w | b h w
        points: Tensor,                  # b t n 3 | b n 3
        proprio: Tensor,                 # b t d | b d
        goal: Tensor,                    # b t d | b d
        action: Tensor | None = None,    # b t action_dim | b action_dim
        point_mask: Tensor | None = None, # b t n | b n
        occupancy_targets: Tensor | None = None, # b t i | b i
        body_mask: Tensor | None = None, # b t i | b i
        contact_targets: Tensor | None = None, # b t i j | b i j
        contact_mask: Tensor | None = None, # b t i | b i
        is_current_timestep: Tensor | bool = False # b | scalar, 1 being present
    ) -> TerrainContactPlannerReturn:
        if proprio.ndim == 2:
            proprio = rearrange(proprio, 'b d -> b 1 d')

        if goal.ndim == 2:
            goal = rearrange(goal, 'b d -> b 1 d')

        if exists(action) and action.ndim == 2:
            action = rearrange(action, 'b d -> b 1 d')

        terrain = self.terrain_encoder(height, points, point_mask)
        tokens = terrain.tokens

        b, t = tokens.shape[:2]

        assert proprio.shape[:2] == (b, t)
        assert goal.shape[:2] == (b, t)

        if self.action_dim > 0:
            assert exists(action), 'action must be passed when action_dim > 0'

            assert action.shape[:2] == (b, t)
            assert action.shape[-1] == self.action_dim

            context_input = torch.cat((proprio, goal, action, terrain.context), dim = -1)
        else:
            assert not exists(action), 'set action_dim > 0 to condition on action'
            context_input = torch.cat((proprio, goal, terrain.context), dim = -1)

        context = self.to_context(context_input)

        if exists(point_mask):
            if point_mask.ndim == 2:
                point_mask = rearrange(point_mask, 'b n -> b 1 n')

            if point_mask.shape[1] == 1 and t > 1:
                point_mask = repeat(point_mask, 'b 1 n -> b t n', t = t)

            point_mask = rearrange(point_mask, 'b t n -> (b t) n')

        if exists(occupancy_targets):
            if occupancy_targets.ndim == 2:
                occupancy_targets = repeat(occupancy_targets, 'b i -> b t i', t = t)

            assert occupancy_targets.shape[:2] == (b, t)
            occupancy_targets = rearrange(occupancy_targets, 'b t i -> (b t) i')

        if exists(body_mask):
            if body_mask.ndim == 2:
                body_mask = repeat(body_mask, 'b i -> b t i', t = t)

            assert body_mask.shape[:2] == (b, t)
            body_mask = rearrange(body_mask, 'b t i -> (b t) i')

        if exists(contact_targets):
            if contact_targets.ndim == 3:
                contact_targets = repeat(contact_targets, 'b i j -> b t i j', t = t)

            assert contact_targets.ndim == 4 and contact_targets.shape[:2] == (b, t)
            contact_targets = rearrange(contact_targets, 'b t i j -> (b t) i j')

        if exists(contact_mask):
            if contact_mask.ndim == 2:
                contact_mask = repeat(contact_mask, 'b i -> b t i', t = t)

            assert contact_mask.shape[:2] == (b, t)
            contact_mask = rearrange(contact_mask, 'b t i -> (b t) i')

        out = self.attention(
            rearrange(context, 'b t d -> (b t) d'),
            rearrange(tokens, 'b t n d -> (b t) n d'),
            point_mask = point_mask,
            point_features = True,
            occupancy_context = rearrange(terrain.context, 'b t d -> (b t) d'),
            occupancy_targets = occupancy_targets,
            body_mask = body_mask,
            is_current_timestep = is_current_timestep
        )

        contact_loss = contact_probs_loss(out.attn_logits, contact_targets, point_mask, contact_mask) if exists(contact_targets) else None

        return TerrainContactPlannerReturn(
            rearrange(out.body_features, '(b t) i d -> b t i d', b = b, t = t),
            rearrange(out.attn_logits, '(b t) h i j -> b t h i j', b = b, t = t),
            rearrange(out.contact_probs, '(b t) h i j -> b t h i j', b = b, t = t),
            rearrange(out.occupancy_logits, '(b t) i -> b t i', b = b, t = t),
            rearrange(out.occupancy_probs, '(b t) i -> b t i', b = b, t = t),
            tokens,
            terrain.context,
            out.occupancy_loss,
            contact_loss
        )

class ContactPlannerWithOutcomeModel(Module):
    """contact planner on past action, with an optional action-conditioned contact outcome model"""

    def __init__(
        self,
        *,
        action_dim,
        outcome_model = False,
        tie_outcome_model = False,
        **planner_kwargs
    ):
        super().__init__()
        assert not tie_outcome_model or outcome_model, 'tie_outcome_model requires outcome_model = True'

        terrain_encoder = planner_kwargs.pop('terrain_encoder', None)

        planner_kwargs.setdefault('film_timestep_cond', outcome_model and tie_outcome_model)

        self.planner = TerrainContactPlanner(
            action_dim = action_dim,
            terrain_encoder = terrain_encoder,
            **planner_kwargs
        )

        self.outcome_model = None if not outcome_model else (
            self.planner if tie_outcome_model else TerrainContactPlanner(
                action_dim = action_dim,
                terrain_encoder = self.planner.terrain_encoder,
                **planner_kwargs
            )
        )

    def forward(
        self,
        height: Tensor,
        points: Tensor,
        proprio: Tensor,
        goal: Tensor,
        prev_action: Tensor,
        action: Tensor | None = None,
        point_mask: Tensor | None = None,
        occupancy_targets: Tensor | None = None,
        body_mask: Tensor | None = None,
        contact_targets: Tensor | None = None,
        contact_mask: Tensor | None = None
    ) -> ContactPlannerWithOutcomeModelReturn:
        if exists(action):
            assert exists(self.outcome_model), 'enable outcome_model to pass action'
        else:
            assert not exists(self.outcome_model), 'action must be passed when outcome_model is enabled'

        plan = self.planner(
            height,
            points,
            proprio,
            goal,
            action = prev_action,
            point_mask = point_mask,
            occupancy_targets = occupancy_targets,
            body_mask = body_mask,
            contact_targets = contact_targets,
            contact_mask = contact_mask,
            is_current_timestep = False
        )

        if not exists(self.outcome_model):
            return ContactPlannerWithOutcomeModelReturn(plan, None, None)

        outcome_out = self.outcome_model(
            height,
            points,
            proprio,
            goal,
            action = action,
            point_mask = point_mask,
            occupancy_targets = occupancy_targets,
            body_mask = body_mask,
            contact_targets = contact_targets,
            contact_mask = contact_mask,
            is_current_timestep = True
        )

        losses = [
            loss for loss in (outcome_out.occupancy_loss, outcome_out.contact_loss)
            if exists(loss)
        ]

        aux_loss = sum(losses) if losses else None

        return ContactPlannerWithOutcomeModelReturn(plan, outcome_out, aux_loss)
