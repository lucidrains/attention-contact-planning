import pytest
import torch

from attention_contact_planning import (
    AttentionContactPlanning,
    ContactPlannerWithOutcomeModel,
    CrossAttention,
    TerrainContactPlanner,
    TerrainEncoder,
    contact_probs_loss,
    occupancy_loss,
)
from attention_contact_planning.attention_contact_planning import (
    SelfAttention,
    TalkingHeads,
    exists,
    no_contact_target_from_soft_targets,
)


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


def test_attention_planner_perceiver_body_to_terrain():
    planner = AttentionContactPlanning(dim = 16, heads = 4, dim_head = 4, point_dim = 16)

    out = planner(
        torch.randn(2, 16),
        torch.randn(2, 30, 16),
        point_features = True
    )

    assert len(planner.perceiver) == 2
    assert out.body_features.shape == (2, 9, 16)


def test_self_attention_is_prenormed_with_residual():
    torch.manual_seed(0)

    attn = SelfAttention(dim = 16, heads = 4, dim_head = 4, depth = 2)
    x = torch.randn(2, 10, 16)

    assert attn(x).shape == x.shape

    # zero out both residual branches -> identity

    with torch.no_grad():
        for self_attn, self_ff in attn.layers:
            self_attn.fn.to_out.weight.zero_()
            self_attn.fn.to_out.bias.zero_()
            self_ff.fn.net[-2].weight.zero_()
            self_ff.fn.net[-2].bias.zero_()

    assert torch.allclose(attn(x), x)


def test_talking_heads_residual_conv_mlp():
    torch.manual_seed(0)

    heads = 4
    talking_heads = TalkingHeads(heads = heads, hidden_dim = 2)
    attn_logits = torch.randn(2, heads, 3, 5)

    assert isinstance(talking_heads.net[0], torch.nn.Conv2d)
    assert isinstance(talking_heads.net[1], torch.nn.ReLU)
    assert isinstance(talking_heads.net[2], torch.nn.Conv2d)

    # zero initialized output conv makes the branch an identity at init

    assert torch.allclose(talking_heads(attn_logits), attn_logits)

    with torch.no_grad():
        talking_heads.net[-1].weight.normal_(std = 0.1)

    mixed = talking_heads(attn_logits)

    assert mixed.shape == attn_logits.shape
    assert not torch.allclose(mixed, attn_logits)


@pytest.mark.parametrize('talking_heads', [False, True])
def test_cross_attention_talking_heads(talking_heads):
    torch.manual_seed(0)

    cross_attn = CrossAttention(
        dim = 16,
        heads = 4,
        dim_head = 4,
        point_dim = 16,
        talking_heads = talking_heads,
        talking_heads_hidden_dim = 2
    )

    assert exists(cross_attn.talking_heads) == talking_heads

    out = cross_attn(
        torch.randn(2, 9, 16),
        torch.randn(2, 20, 16),
        point_features = True
    )

    assert out.attn_logits.shape == (2, 4, 9, 21)
    assert torch.allclose(out.contact_probs, out.attn_logits.softmax(dim = -1))

    out.attn_logits.mean().backward()

    if talking_heads:
        assert any(exists(param.grad) for param in cross_attn.talking_heads.parameters())


def test_self_attention_mask_ignores_masked_tokens():
    torch.manual_seed(0)

    attn = SelfAttention(dim = 16, heads = 4, dim_head = 4)
    x = torch.randn(2, 10, 16)
    mask = torch.ones(2, 10, dtype = torch.bool)
    mask[0, 5:] = False

    out = attn(x, mask = mask)

    other_x = x.clone()
    other_x[0, 5:] = torch.randn_like(other_x[0, 5:])

    assert torch.allclose(out[0, :5], attn(other_x, mask = mask)[0, :5])


@pytest.mark.parametrize('terrain_self_attend', [False, True])
def test_optional_terrain_self_attention(terrain_self_attend):
    torch.manual_seed(0)

    planner = AttentionContactPlanning(
        dim = 16,
        heads = 4,
        dim_head = 4,
        point_dim = 16,
        perceiver_depth = 1,
        terrain_self_attend = terrain_self_attend,
        terrain_self_attend_depth = 2
    )

    if terrain_self_attend:
        assert len(planner.terrain_self_attn.layers) == 2
    else:
        assert planner.terrain_self_attn is None

    context = torch.randn(2, 16)
    points = torch.randn(2, 20, 16)
    point_mask = torch.ones(2, 20, dtype = torch.bool)
    point_mask[1, 10:] = False

    out = planner(
        context,
        points,
        point_mask = point_mask,
        point_features = True
    )

    assert out.body_features.shape == (2, 9, 16)
    assert (out.contact_probs[1, :, :, 11:] == 0).all()

    out.attn_logits.mean().backward()

    if terrain_self_attend:
        assert any(param.grad is not None for param in planner.terrain_self_attn.parameters())


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


def make_planner(occupancy_from_attention):
    return AttentionContactPlanning(
        dim = 16,
        heads = 4,
        dim_head = 4,
        point_dim = 16,
        num_bodies = 9,
        occupancy_from_attention = occupancy_from_attention,
        occupancy_context_dim = 8,
        occupancy_hidden_dims = (32, 32)
    )


def test_occupancy_head_faithful_to_paper():
    torch.manual_seed(0)

    b, n, dim, context_dim, num_bodies = 2, 20, 16, 8, 9

    planner = make_planner(occupancy_from_attention = False)

    assert planner.occupancy_predictor is not None
    assert planner.to_body_with_context is None

    context = torch.randn(b, dim)
    occupancy_context = torch.randn(b, context_dim)
    points = torch.randn(b, n, dim)

    out = planner(
        context,
        points,
        point_features = True,
        occupancy_context = occupancy_context
    )

    assert out.occupancy_logits.shape == (b, num_bodies)
    assert out.occupancy_probs.shape == (b, num_bodies)
    assert torch.allclose(out.occupancy_probs, out.occupancy_logits.sigmoid())

    # head must see the pooled terrain context

    other = planner(
        context,
        points,
        point_features = True,
        occupancy_context = torch.randn(b, context_dim)
    )

    assert not torch.allclose(out.occupancy_logits, other.occupancy_logits)

    targets = torch.randint(0, 2, (b, num_bodies)).float()
    loss = occupancy_loss(out.occupancy_logits, targets)

    loss.backward()

    assert any(param.grad is not None for param in planner.occupancy_predictor.parameters())
    assert planner.body_embed.grad is not None


def test_optional_occupancy_loss_from_labels():
    torch.manual_seed(0)

    b, n, dim, context_dim, num_bodies = 2, 20, 16, 8, 9

    planner = make_planner(occupancy_from_attention = False)

    context = torch.randn(b, dim)
    occupancy_context = torch.randn(b, context_dim)
    points = torch.randn(b, n, dim)

    out = planner(
        context,
        points,
        point_features = True,
        occupancy_context = occupancy_context
    )
    assert not exists(out.occupancy_loss)

    targets = torch.randint(0, 2, (b, num_bodies))
    body_mask = torch.tensor([
        [True] * 9,
        [True] * 4 + [False] * 5
    ])

    out = planner(
        context,
        points,
        point_features = True,
        occupancy_context = occupancy_context,
        occupancy_targets = targets,
        body_mask = body_mask
    )

    expected = occupancy_loss(out.occupancy_logits, targets, body_mask)

    assert out.occupancy_loss.ndim == 0
    assert torch.allclose(out.occupancy_loss, expected)

    out.occupancy_loss.backward()
    assert any(exists(param.grad) for param in planner.occupancy_predictor.parameters())


def test_occupancy_derived_from_cross_attention():
    torch.manual_seed(0)

    b, n, dim, context_dim, num_bodies = 2, 20, 16, 8, 9

    planner = make_planner(occupancy_from_attention = True)

    assert planner.occupancy_predictor is None
    assert planner.to_body_with_context is not None

    context = torch.randn(b, dim)
    occupancy_context = torch.randn(b, context_dim)
    points = torch.randn(b, n, dim)

    out = planner(
        context,
        points,
        point_features = True,
        occupancy_context = occupancy_context
    )

    # occupancy is just the no-contact column, averaged over heads

    expected_occupancy = 1. - out.contact_probs[..., 0].mean(dim = 1)

    assert out.occupancy_probs.shape == (b, num_bodies)
    assert torch.allclose(out.occupancy_probs, expected_occupancy)
    assert torch.allclose(
        out.occupancy_logits,
        expected_occupancy.clamp(1e-6, 1. - 1e-6).logit()
    )

    # context is folded in early, so body features and occupancy both move

    other = planner(
        context,
        points,
        point_features = True,
        occupancy_context = torch.randn(b, context_dim)
    )

    assert not torch.allclose(out.body_features, other.body_features)
    assert not torch.allclose(out.occupancy_probs, other.occupancy_probs)

    targets = torch.randint(0, 2, (b, num_bodies)).float()
    loss = occupancy_loss(out.occupancy_logits, targets)

    loss.backward()

    assert planner.to_body_with_context.weight.grad is not None
    assert planner.cross_attn.no_contact_key_value.grad is not None


def test_terrain_planner_terrain_self_attend():
    torch.manual_seed(0)

    b, t, h, w = 2, 2, 3, 4
    n = h * w

    terrain_encoder = TerrainEncoder(dim = 16, context_dim = 8, channels = (4,))

    planner = TerrainContactPlanner(
        dim = 16,
        heads = 4,
        dim_head = 4,
        terrain_encoder = terrain_encoder,
        proprio_dim = 10,
        goal_dim = 3,
        terrain_self_attend = True,
        terrain_self_attend_depth = 2
    )

    assert len(planner.attention.terrain_self_attn.layers) == 2

    point_mask = torch.ones(b, t, n, dtype = torch.bool)
    point_mask[:, :, 6:] = False

    out = planner(
        torch.randn(b, t, h, w),
        torch.randn(b, t, n, 3),
        torch.randn(b, t, 10),
        torch.randn(b, t, 3),
        point_mask = point_mask
    )

    assert out.body_features.shape == (b, t, 9, 16)

    out.attn_logits.mean().backward()
    assert any(param.grad is not None for param in planner.attention.terrain_self_attn.parameters())


@pytest.mark.parametrize('occupancy_from_attention', [False, True])
def test_terrain_planner_occupancy(occupancy_from_attention):
    torch.manual_seed(0)

    b, t, h, w = 2, 2, 3, 4
    n = h * w

    terrain_encoder = TerrainEncoder(dim = 16, context_dim = 8, channels = (4,))

    planner = TerrainContactPlanner(
        dim = 16,
        heads = 4,
        dim_head = 4,
        terrain_encoder = terrain_encoder,
        proprio_dim = 10,
        goal_dim = 3,
        occupancy_from_attention = occupancy_from_attention,
        occupancy_hidden_dims = (16,)
    )

    out = planner(
        torch.randn(b, t, h, w),
        torch.randn(b, t, n, 3),
        torch.randn(b, t, 10),
        torch.randn(b, t, 3)
    )

    assert out.occupancy_logits.shape == (b, t, 9)
    assert out.occupancy_probs.shape == (b, t, 9)

    if occupancy_from_attention:
        assert planner.attention.occupancy_predictor is None
        expected = 1. - out.contact_probs[..., 0].mean(dim = 2)
        assert torch.allclose(out.occupancy_probs, expected)
    else:
        assert planner.attention.occupancy_predictor is not None

    out.occupancy_logits.mean().backward()
    assert planner.terrain_encoder.convs[0].conv.weight.grad is not None


def test_terrain_planner_optional_occupancy_loss_from_labels():
    torch.manual_seed(0)

    b, t, h, w = 2, 2, 3, 4
    n, num_bodies = h * w, 9

    terrain_encoder = TerrainEncoder(dim = 16, context_dim = 8, channels = (4,))

    planner = TerrainContactPlanner(
        dim = 16,
        heads = 4,
        dim_head = 4,
        terrain_encoder = terrain_encoder,
        proprio_dim = 10,
        goal_dim = 3,
        occupancy_hidden_dims = (16,)
    )

    height = torch.randn(b, t, h, w)
    points = torch.randn(b, t, n, 3)
    proprio = torch.randn(b, t, 10)
    goal = torch.randn(b, t, 3)

    out = planner(height, points, proprio, goal)
    assert not exists(out.occupancy_loss)

    targets = torch.randint(0, 2, (b, t, num_bodies))
    body_mask = torch.ones(b, t, num_bodies, dtype = torch.bool)
    body_mask[:, :, 5:] = False

    out = planner(
        height,
        points,
        proprio,
        goal,
        occupancy_targets = targets,
        body_mask = body_mask
    )

    expected = occupancy_loss(
        out.occupancy_logits.reshape(b * t, num_bodies),
        targets.reshape(b * t, num_bodies),
        body_mask.reshape(b * t, num_bodies)
    )

    assert out.occupancy_loss.ndim == 0
    assert torch.allclose(out.occupancy_loss, expected)

    out.occupancy_loss.backward()
    assert exists(planner.terrain_encoder.convs[0].conv.weight.grad)


@pytest.mark.parametrize('outcome_model, tie_outcome_model', [
    (False, False),
    (True, False),
    (True, True)
])
def test_contact_planner_with_outcome_model(outcome_model, tie_outcome_model):
    torch.manual_seed(0)

    b, t, h, w = 2, 2, 3, 4
    n, num_bodies = h * w, 9

    planner = ContactPlannerWithOutcomeModel(
        action_dim = 6,
        outcome_model = outcome_model,
        tie_outcome_model = tie_outcome_model,
        dim = 16,
        heads = 4,
        dim_head = 4,
        terrain_encoder = TerrainEncoder(dim = 16, context_dim = 8, channels = (4,)),
        proprio_dim = 10,
        goal_dim = 3,
        context_hidden_dim = 16,
        occupancy_hidden_dims = (16,)
    )

    height = torch.randn(b, t, h, w)
    points = torch.randn(b, t, n, 3)
    proprio = torch.randn(b, t, 10)
    goal = torch.randn(b, t, 3)
    prev_action = torch.randn(b, t, 6)
    action = torch.randn(b, t, 6) if outcome_model else None

    out = planner(
        height, points, proprio, goal,
        prev_action = prev_action,
        action = action,
        occupancy_targets = torch.randint(0, 2, (b, t, num_bodies)).float(),
        contact_targets = torch.randn(b, num_bodies, n).softmax(dim = -1)
    )

    assert out.plan.contact_probs.shape == (b, t, 4, num_bodies, n + 1)
    assert exists(out.plan.contact_loss)

    if not outcome_model:
        assert not exists(out.outcome_model)
        assert not exists(out.aux_loss)
        assert not planner.planner.attention.film_timestep_cond

        with pytest.raises(AssertionError):
            planner(height, points, proprio, goal, prev_action = prev_action, action = torch.randn(b, t, 6))

        return

    assert out.outcome_model.contact_probs.shape == (b, t, 4, num_bodies, n + 1)
    assert exists(out.aux_loss)

    if tie_outcome_model:
        assert planner.outcome_model is planner.planner
        assert planner.planner.attention.film_timestep_cond
    else:
        assert planner.outcome_model is not planner.planner
        assert not planner.planner.attention.film_timestep_cond

    with pytest.raises(AssertionError):
        planner(height, points, proprio, goal, prev_action = prev_action)

    out.aux_loss.backward()
