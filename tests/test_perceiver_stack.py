import torch

from attention_contact_planning import AttentionContactPlanning


def test_attention_planner_perceiver_body_to_terrain():
    planner = AttentionContactPlanning(dim = 16, heads = 4, dim_head = 4, point_dim = 16)

    out = planner(
        torch.randn(2, 16),
        torch.randn(2, 30, 16),
        point_features = True
    )

    assert len(planner.perceiver) == 2
    assert out.body_features.shape == (2, 9, 16)
