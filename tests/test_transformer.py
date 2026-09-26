import torch

from transformer import SwiGLU, Transformer


def test_default_architecture_feeds_embeddings_directly_and_ties_head():
    model = Transformer(
        dim=32,
        depth=2,
        heads=4,
        ff_mult=2.5,
        vocab_size=64,
        max_seq_len=16,
    )

    assert isinstance(model.input_projection, torch.nn.Identity)
    assert isinstance(model.input_norm, torch.nn.Identity)
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.blocks[0].ff.hidden_dim == 128  # aligned to 64


def test_explicit_feedforward_width_is_respected():
    feedforward = SwiGLU(32, hidden_dim=48)
    assert feedforward.hidden_dim == 48
    assert feedforward.proj_in.out_features == 96


def test_causal_forward_loss_and_backward_are_finite():
    torch.manual_seed(7)
    model = Transformer(
        dim=32,
        depth=2,
        heads=4,
        ff_mult=2.0,
        vocab_size=64,
        max_seq_len=12,
    )
    inputs = torch.randint(0, 64, (2, 12))
    targets = torch.randint(0, 64, (2, 12))
    loss, logits = model(inputs, targets, return_logits=True)
    loss.backward()

    assert logits.shape == (2, 12, 64)
    assert loss.isfinite()
    assert all(
        parameter.grad is None or parameter.grad.isfinite().all()
        for parameter in model.parameters()
    )

    changed = inputs.clone()
    changed[:, 6:] = torch.randint(0, 64, changed[:, 6:].shape)
    with torch.no_grad():
        original_logits = model(inputs)
        changed_logits = model(changed)
    assert torch.allclose(original_logits[:, :6], changed_logits[:, :6], atol=1e-6)


def test_resize_preserves_weight_tying_and_existing_rows():
    model = Transformer(32, 1, 4, 2.0, 32, 8)
    original = model.token_embedding.weight.detach().clone()

    model.resize_token_embeddings(40)

    assert model.lm_head.weight is model.token_embedding.weight
    assert torch.equal(model.token_embedding.weight[:32], original)
    assert model(torch.randint(0, 40, (1, 8))).shape == (1, 8, 40)
