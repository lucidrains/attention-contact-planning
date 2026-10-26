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
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch_einops_utils import masked_mean
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

TerrainContactPlannerReturn = namedtuple('TerrainContactPlannerReturn', [
    'body_features',
    'attn_logits',
    'contact_probs',
    'terrain_tokens',
    'terrain_context'
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
        hidden_dim = 64
    ):
        super().__init__()
        self.to_gamma_beta = MLP(dim, hidden_dim, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        scaled = einx.multiply('b ... d, b d -> b ... d', x, gamma + 1.)
        return einx.add('b ... d, b d -> b ... d', scaled, beta)

class CrossAttention(Module):
    def __init__(
        self,
        dim = 96,
        heads = 32,
        dim_head = 3,
        point_dim = 3,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads

        self.scale = dim_head ** -0.5

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

class AttentionContactPlanning(Module):
    def __init__(
        self,
        dim = 96,
        heads = 32,
        dim_head = 3,
        point_dim = 3,
        num_bodies = 9,
        dropout = 0.
    ):
        super().__init__()
        self.body_embed = nn.Parameter(torch.randn(num_bodies, dim) * 0.02)
        self.film = FiLM(dim)
        self.cross_attn = CrossAttention(dim, heads, dim_head, point_dim, dropout)

    def forward(
        self,
        context: Tensor,                  # b d
        points: Tensor,                   # b j point_dim | b j d
        point_mask: Tensor | None = None, # b j
        point_features: bool = False
    ) -> CrossAttentionReturn:
        body_embed = repeat(self.body_embed, 'i d -> b i d', b = context.shape[0])
        queries = self.film(body_embed, context)

        return self.cross_attn(
            queries,
            points,
            point_mask = point_mask,
            point_features = point_features
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
        terrain_encoder: TerrainEncoder | None = None,
        context_hidden_dim = 128,
        dropout = 0.
    ):
        super().__init__()
        self.terrain_encoder = terrain_encoder if exists(terrain_encoder) else TerrainEncoder(dim = dim)
        assert self.terrain_encoder.dim == dim

        self.to_context = MLP(
            proprio_dim + goal_dim + self.terrain_encoder.context_dim,
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
            dropout = dropout
        )

    def forward(
        self,
        height: Tensor,                  # b t h w | b h w
        points: Tensor,                  # b t n 3 | b n 3
        proprio: Tensor,                 # b t d | b d
        goal: Tensor,                    # b t d | b d
        point_mask: Tensor | None = None # b t n | b n
    ) -> TerrainContactPlannerReturn:
        if proprio.ndim == 2:
            proprio = rearrange(proprio, 'b d -> b 1 d')

        if goal.ndim == 2:
            goal = rearrange(goal, 'b d -> b 1 d')

        terrain = self.terrain_encoder(height, points, point_mask)
        tokens = terrain.tokens

        b, t = tokens.shape[:2]

        assert proprio.shape[:2] == (b, t)
        assert goal.shape[:2] == (b, t)

        context = self.to_context(torch.cat((proprio, goal, terrain.context), dim = -1))

        if exists(point_mask):
            if point_mask.ndim == 2:
                point_mask = rearrange(point_mask, 'b n -> b 1 n')

            if point_mask.shape[1] == 1 and t > 1:
                point_mask = repeat(point_mask, 'b 1 n -> b t n', t = t)

            point_mask = rearrange(point_mask, 'b t n -> (b t) n')

        out = self.attention(
            rearrange(context, 'b t d -> (b t) d'),
            rearrange(tokens, 'b t n d -> (b t) n d'),
            point_mask = point_mask,
            point_features = True
        )

        return TerrainContactPlannerReturn(
            rearrange(out.body_features, '(b t) i d -> b t i d', b = b, t = t),
            rearrange(out.attn_logits, '(b t) h i j -> b t h i j', b = b, t = t),
            rearrange(out.contact_probs, '(b t) h i j -> b t h i j', b = b, t = t),
            tokens,
            terrain.context
        )
