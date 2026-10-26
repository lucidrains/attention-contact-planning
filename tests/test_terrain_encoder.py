import pytest
import torch

from attention_contact_planning import (
    TerrainContactPlanner,
    TerrainEncoder,
    contact_probs_loss,
)


@pytest.mark.parametrize('t', [1, 3])
def test_terrain_encoder_to_cross_attention(t):
    # height scans -> terrain tokens + context -> query context -> cross attention

    torch.manual_seed(0)

    b, h, w = 2, 17, 11
    n = h * w

    terrain_encoder = TerrainEncoder() if t == 1 else TerrainEncoder(kernel_size = (3, 5, 5))

    planner = TerrainContactPlanner(
        terrain_encoder = terrain_encoder,
        proprio_dim = 10,
        goal_dim = 3
    )

    height = torch.randn(b, t, h, w)
    points = torch.randn(b, t, n, 3)
    proprio = torch.randn(b, t, 10)
    goal = torch.randn(b, t, 3)

    out = planner(height, points, proprio, goal)

    assert out.terrain_tokens.shape == (b, t, n, 96)
    assert out.terrain_context.shape == (b, t, 64)
    assert out.body_features.shape == (b, t, 9, 96)
    assert out.attn_logits.shape == (b, t, 32, 9, n + 1)
    assert out.contact_probs.shape == (b, t, 32, 9, n + 1)
    assert torch.allclose(out.contact_probs, out.attn_logits.softmax(dim = -1))

    # changing the future must not change past terrain features

    if t > 1:
        future = height.clone()
        future[:, -1] += 1.

        tokens = planner.terrain_encoder(height, points).tokens
        future_tokens = planner.terrain_encoder(future, points).tokens

        assert torch.allclose(tokens[:, :-1], future_tokens[:, :-1])

    target = torch.randn(b, t, 9, n).softmax(dim = -1)
    target[:, :, 6:] = 0.  # blank maps become no-contact targets

    loss = contact_probs_loss(
        out.attn_logits.reshape(b * t, 32, 9, n + 1),
        target.reshape(b * t, 9, n)
    )

    loss.backward()

    assert planner.terrain_encoder.convs[0].conv.weight.grad is not None
    assert planner.attention.body_embed.grad is not None
    assert planner.attention.cross_attn.no_contact_key_value.grad is not None

    assert len(planner.attention.perceiver) == 2
    assert any(param.grad is not None for param in planner.attention.perceiver.parameters())
