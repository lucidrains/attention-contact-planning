"""
ein notation:
b - batch
h - attention heads
i - contact bodies
j - terrain points
d - dimension
"""

from __future__ import annotations

from functools import partial
from collections import namedtuple

import torch
from torch import nn, Tensor
from torch.nn import Module

import einx
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from torch_einops_utils import masked_mean, masked_sum

from x_mlps_pytorch import MLP

# named tuples

CrossAttentionReturn = namedtuple('CrossAttentionReturn', [
    'body_features',
    'attn_logits',
    'contact_probs'
])

# helpers

def exists(v):
    return v is not None

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

LinearNoBias = partial(nn.Linear, bias = False)

# loss

def contact_probs_loss(
    attn_logits: Tensor,        # b h i j
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

    valid = contact_mask.bool() if exists(contact_mask) else soft_target_probs.sum(dim = -1) > 0

    log_probs = attn_logits.log_softmax(dim = -1)
    point_loss = -einsum(log_probs, soft_target_probs, 'b h i j, b i j -> b h i j')

    point_loss_mask = rearrange(point_mask, 'b j -> b 1 1 j') if exists(point_mask) else None
    point_loss = masked_sum(point_loss, point_loss_mask, dim = -1)

    return masked_mean(point_loss, rearrange(valid, 'b i -> b 1 i'))

# modules

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

        self.scale = dim_head ** -0.5

        self.to_point = nn.Linear(point_dim, dim)
        self.to_q = LinearNoBias(dim, inner_dim)
        self.to_kv = LinearNoBias(dim, inner_dim * 2)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,                  # b i d
        points: Tensor,                   # b j point_dim
        point_mask: Tensor | None = None  # b j
    ) -> CrossAttentionReturn:
        point_tokens = self.to_point(points)

        q = self.to_q(queries)
        k, v = self.to_kv(point_tokens).chunk(2, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        attn_logits = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        if exists(point_mask):
            attn_logits = einx.where(
                'b j, b h i j, -> b h i j',
                point_mask.bool(),
                attn_logits,
                torch.finfo(attn_logits.dtype).min
            )

        attn = attn_logits.softmax(dim = -1)
        contact_probs = attn  # b h i j

        attn = self.attn_dropout(attn)
        body_features = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        body_features = self.merge_heads(body_features)
        body_features = self.to_out(body_features)

        return CrossAttentionReturn(body_features, attn_logits, contact_probs)

# main class

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
        points: Tensor,                   # b j point_dim
        point_mask: Tensor | None = None  # b j
    ) -> CrossAttentionReturn:
        body_embed = repeat(self.body_embed, 'i d -> b i d', b = context.shape[0])
        queries = self.film(body_embed, context)

        return self.cross_attn(queries, points, point_mask = point_mask)
