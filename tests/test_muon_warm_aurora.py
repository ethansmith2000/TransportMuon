import importlib
import sys
import types

import torch


muon_stub = types.ModuleType("muon")
muon_stub.adam_update = lambda *args, **kwargs: args[0]
sys.modules.setdefault("muon", muon_stub)
muon_warm = importlib.import_module("muon_warm")
muon_warm_aurora = importlib.import_module("muon_warm_aurora")


def _eager(function):
    return getattr(function, "_torchdynamo_orig_callable", function)


def _eager_polar(matrix, steps, method):
    assert method == "muon"
    return _eager(muon_warm._muon_ns5_prepared)(matrix, steps)


def _leverage_cv(prepared_polar: torch.Tensor) -> torch.Tensor:
    leverage = prepared_polar.float().square().sum(dim=0)
    return leverage.std(unbiased=False) / leverage.mean()


def test_aurora_anchor_balances_tall_matrix_leverage(monkeypatch):
    torch.manual_seed(61)
    gradient = torch.randn(32, 8)
    gradient[:8].mul_(0.03)
    work_matrix, transposed = muon_warm._prepare_muon_matrix(gradient)
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    optimizer = muon_warm_aurora.MuonWarmAurora(
        [{"params": [parameter], "use_muon": True}],
        muon_ns_steps=5,
        muon_warm_full_ns_steps=0,
        aurora_anchor_iterations=3,
    )
    state = {"aurora_balance_enabled": True}
    monkeypatch.setattr(
        muon_warm_aurora,
        "_polar_solve_prepared",
        _eager_polar,
    )

    exact_polar = torch.linalg.svd(work_matrix.float(), full_matrices=False)
    ordinary = exact_polar.U @ exact_polar.Vh
    balanced = optimizer._compute_anchor_direction(
        work_matrix, state, transposed
    )

    assert transposed
    assert _leverage_cv(balanced) < 0.15 * _leverage_cv(ordinary)
    assert torch.allclose(
        balanced.float() @ balanced.float().T,
        torch.eye(8),
        atol=1.5e-2,
        rtol=1.5e-2,
    )
    assert state["aurora_column_scale"].shape == (32,)


def test_warm_balance_updates_column_scale(monkeypatch):
    torch.manual_seed(62)
    gradient = torch.randn(16, 4)
    work_matrix, transposed = muon_warm._prepare_muon_matrix(gradient)
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    optimizer = muon_warm_aurora.MuonWarmAurora(
        [{"params": [parameter], "use_muon": True}],
        muon_warm_full_ns_steps=0,
        aurora_anchor_iterations=2,
    )
    state = {"aurora_balance_enabled": True, "step": 2}
    monkeypatch.setattr(
        muon_warm_aurora,
        "_polar_solve_prepared",
        _eager_polar,
    )
    q_previous = optimizer._compute_anchor_direction(
        work_matrix, state, transposed
    )
    old_scale = state["aurora_column_scale"].clone()

    torch.rand(100)
    tracking_matrix = optimizer._prepare_tracking_matrix(
        work_matrix, q_previous, state, transposed
    )

    assert tracking_matrix.shape == work_matrix.shape
    assert tracking_matrix.isfinite().all()
    assert not torch.equal(state["aurora_column_scale"], old_scale)


def test_wide_matrix_keeps_standard_muon_target():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm_aurora.MuonWarmAurora(
        [{"params": [parameter], "use_muon": True}],
        muon_warm_full_ns_steps=0,
    )
    state = {"aurora_balance_enabled": False}
    work_matrix, transposed = muon_warm._prepare_muon_matrix(
        torch.randn_like(parameter)
    )

    tracking = optimizer._prepare_tracking_matrix(
        work_matrix, None, state, transposed
    )

    assert not transposed
    assert tracking is work_matrix
    assert "aurora_column_scale" not in state
