import torch

from attention_contact_planning import (
    contact_probs_loss,
    no_contact_target_from_soft_targets,
)


def test_blank_soft_target_sets_no_contact_to_one():
    soft_target_probs = torch.zeros(2, 3, 5)
    soft_target_probs[1, 0, 2] = 1.

    no_contact_target = no_contact_target_from_soft_targets(soft_target_probs)

    assert no_contact_target.tolist() == [
        [1., 1., 1.],
        [0., 1., 1.],
    ]

    # mass only on a masked point also counts as blank

    point_mask = torch.ones(2, 5, dtype = torch.bool)
    point_mask[1, 2] = False

    masked_no_contact_target = no_contact_target_from_soft_targets(
        soft_target_probs,
        point_mask = point_mask
    )

    assert masked_no_contact_target[1, 0] == 1.

    # no-contact logit is column 0

    attn_logits = torch.zeros(2, 1, 3, 6, requires_grad = True)

    loss = contact_probs_loss(attn_logits, soft_target_probs)
    loss.backward()

    assert attn_logits.grad is not None
    assert attn_logits.grad[1, 0, 0, 0] > 0.  # valid contact
    assert attn_logits.grad[1, 0, 1, 0] < 0.  # blank target
