import torch

from attention_contact_planning import AttentionContactPlanning, contact_probs_loss


def test_attention_contact_planning():
    planner = AttentionContactPlanning(dim = 96, heads = 32, dim_head = 3, point_dim = 3, num_bodies = 9)

    context = torch.randn(2, 96)
    points = torch.randn(2, 187, 3)
    point_mask = torch.ones(2, 187, dtype = torch.bool)
    point_mask[1, 100:] = False

    out = planner(context, points, point_mask = point_mask)

    assert out.body_features.shape == (2, 9, 96)
    assert out.attn_logits.shape == (2, 32, 9, 188)
    assert out.contact_probs.shape == (2, 32, 9, 188)
    assert torch.allclose(out.contact_probs, out.attn_logits.softmax(dim = -1))
    assert (out.contact_probs[1, :, :, 101:] == 0).all()

    soft_target_probs = torch.randn(2, 9, 187).softmax(dim = -1)
    contact_mask = torch.tensor([[True] * 9, [True] * 4 + [False] * 5])

    loss = contact_probs_loss(
        out.attn_logits,
        soft_target_probs,
        point_mask = point_mask,
        contact_mask = contact_mask
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
