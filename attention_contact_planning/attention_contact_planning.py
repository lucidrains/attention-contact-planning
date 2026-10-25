from __future__ import annotations

from torch import Tensor, nn


class AttnContact(nn.Module):
    """Attention-based contact planner."""

    def __init__(
        self,
        dim = 256,
        depth = 4,
        heads = 8,
        dim_head = 32,
        point_dim = 3,
        dropout = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.point_dim = point_dim
        self.dropout = dropout

    def forward(
        self,
        proprio: Tensor,  # (b, obs_dim)
        points: Tensor,  # (b, n_points, point_dim)
        point_mask: Tensor | None = None,  # (b, n_points)
    ) -> tuple[Tensor, Tensor]:
        # -> contact logits (b, n_points), occupancy logits (b, n_points)
        raise NotImplementedError("AttnContact is not implemented yet")
