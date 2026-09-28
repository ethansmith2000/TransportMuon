import torch

from muon import adam_update, get_muon_param_groups
from muon_warm import MuonWarm


def test_adam_update_matches_torch_adamw_direction():
    initial = torch.tensor([1.0, -2.0])
    gradient = torch.tensor([0.3, -0.7])
    reference = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW(
        [reference], lr=1.0, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0
    )
    reference.grad = gradient.clone()
    optimizer.step()

    exp_avg = torch.zeros_like(initial)
    exp_avg_sq = torch.zeros_like(initial)
    direction = adam_update(
        gradient, exp_avg, exp_avg_sq, 1, (0.9, 0.99), 1e-8
    )

    assert torch.allclose(initial - direction, reference, atol=1e-6)


def test_param_groups_route_embeddings_and_vectors_to_adam():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.linear = torch.nn.Linear(8, 4)

    model = Model()
    groups = get_muon_param_groups(model)
    muon_ids = {
        id(parameter)
        for group in groups
        if group["use_muon"]
        for parameter in group["params"]
    }
    adam_ids = {
        id(parameter)
        for group in groups
        if not group["use_muon"]
        for parameter in group["params"]
    }

    assert id(model.linear.weight) in muon_ids
    assert id(model.embedding.weight) in adam_ids
    assert id(model.linear.bias) in adam_ids


def test_param_groups_can_mark_fused_matrices_for_logical_row_splitting():
    model = torch.nn.Sequential(torch.nn.Linear(8, 24, bias=False))
    groups = get_muon_param_groups(
        model,
        muon_split_predicate=lambda name, _parameter: 3 if name == "0.weight" else 1,
    )

    muon_group = next(group for group in groups if group["use_muon"])
    assert muon_group["muon_split_count"] == 3
    assert muon_group["params"] == [model[0].weight]


def test_logically_split_muon_keeps_independent_block_state(monkeypatch):
    parameter = torch.nn.Parameter(torch.randn(12, 4))
    optimizer = MuonWarm(
        [{"params": [parameter], "use_muon": True, "muon_split_count": 3}],
        muon_warm_full_ns_steps=0,
        muon_warm_anchor_every=1,
    )
    import muon_warm

    monkeypatch.setattr(
        muon_warm,
        "_muon_ns5_prepared",
        getattr(muon_warm._muon_ns5_prepared, "_torchdynamo_orig_callable"),
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    block_states = optimizer.state[parameter]["muon_block_states"]
    assert len(block_states) == 3
    assert all(state["muon_warm_q"].shape == (4, 4) for state in block_states)
    assert all(state["momentum_fast"].shape == (4, 4) for state in block_states)


def test_mixed_optimizer_runs_without_an_import_stub(monkeypatch):
    torch.manual_seed(61)
    model = torch.nn.Linear(8, 4)
    groups = get_muon_param_groups(model)
    optimizer = MuonWarm(
        groups,
        muon_ns_steps=5,
        muon_warm_full_ns_steps=0,
        muon_warm_anchor_every=1,
    )
    # Avoid compile startup in this integration test; the same function body is
    # exercised eagerly while the optimizer/import/state wiring remains real.
    import muon_warm

    monkeypatch.setattr(
        muon_warm,
        "_muon_ns5_prepared",
        getattr(muon_warm._muon_ns5_prepared, "_torchdynamo_orig_callable"),
    )

    output = model(torch.randn(3, 8)).square().mean()
    output.backward()
    optimizer.step()

    assert model.weight.isfinite().all()
    assert model.bias.isfinite().all()
    assert "muon_warm_q" in optimizer.state[model.weight]
    assert "exp_avg" in optimizer.state[model.bias]
