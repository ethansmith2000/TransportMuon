import importlib
import sys
import types

import torch


muon_stub = types.ModuleType("muon")
muon_stub.adam_update = lambda *args, **kwargs: args[0]
sys.modules.setdefault("muon", muon_stub)
muon_warm = importlib.import_module("muon_warm")
procrustes = importlib.import_module("muon_warm_procrustes")


def _eager(function):
    return getattr(function, "_torchdynamo_orig_callable", function)


def test_procrustes_improves_dense_stretch_rotation(monkeypatch):
    torch.manual_seed(83)
    rows, columns = 8, 16
    q_previous = torch.cat(
        (torch.eye(rows), torch.zeros(rows, columns - rows)), dim=1
    )
    skew = torch.randn(rows, rows)
    rotation = torch.matrix_exp(0.03 * (skew - skew.T))
    eigenvectors = torch.linalg.qr(torch.randn(rows, rows)).Q
    stretch = eigenvectors @ torch.diag(torch.linspace(0.2, 2.0, rows))
    stretch = stretch @ eigenvectors.T
    target = rotation @ q_previous
    work_matrix = (stretch @ target).to(torch.bfloat16)
    q_previous = q_previous.to(torch.bfloat16)
    monkeypatch.setattr(
        procrustes,
        "_muon_ns5_prepared",
        _eager(muon_warm._muon_ns5_prepared),
    )

    jacobi = _eager(muon_warm._warm_polar_jacobi_step)(
        work_matrix,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        2,
        True,
        None,
        "tikhonov",
    )
    solved, _ = _eager(procrustes._procrustes_transport_step)(
        work_matrix,
        q_previous,
        None,
        1.0,
        1e-3,
        "tikhonov",
        "higham_cubic",
        2,
        "higham_cubic",
        5,
        2,
        0.0,
        -1.0,
    )

    jacobi_error = (jacobi.float() - target).norm()
    procrustes_error = (solved.float() - target).norm()
    assert procrustes_error < 0.2 * jacobi_error


def test_normal_trust_region_caps_applied_motion(monkeypatch):
    rows = 4
    q_previous = torch.cat((torch.eye(rows), torch.zeros(rows, rows)), dim=1)
    angle = 0.2
    target = torch.cat(
        (
            torch.cos(torch.tensor(angle)) * torch.eye(rows),
            torch.sin(torch.tensor(angle)) * torch.eye(rows),
        ),
        dim=1,
    )
    monkeypatch.setattr(
        procrustes,
        "_muon_ns5_prepared",
        _eager(muon_warm._muon_ns5_prepared),
    )

    _, stats = _eager(procrustes._procrustes_transport_step)(
        target.to(torch.bfloat16),
        q_previous.to(torch.bfloat16),
        None,
        1.0,
        1e-3,
        "tikhonov",
        "higham_cubic",
        2,
        "higham_cubic",
        5,
        2,
        0.01,
        -1.0,
    )

    effective_eta, normal_rms, applied_normal_rms, accepted = stats
    assert effective_eta < 1.0
    assert normal_rms > 0.01
    assert applied_normal_rms <= 0.010001
    assert accepted == 1.0


def test_optimizer_records_procrustes_step_statistics(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = procrustes.MuonWarmProcrustes(
        [{"params": [parameter], "use_muon": True}],
        muon_warm_full_ns_steps=0,
        muon_warm_anchor_every=0,
        muon_warm_max_normal_rms=0.02,
    )
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": torch.cat(
            (torch.eye(4), torch.zeros(4, 4)), dim=1
        ).to(torch.bfloat16),
    }
    monkeypatch.setattr(
        procrustes,
        "_muon_ns5_prepared",
        _eager(muon_warm._muon_ns5_prepared),
    )
    monkeypatch.setattr(
        procrustes,
        "_procrustes_transport_step",
        _eager(procrustes._procrustes_transport_step),
    )

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert state["muon_warm_effective_eta_tensor"].ndim == 0
    assert state["muon_warm_normal_rms_tensor"].isfinite()
    assert state["muon_warm_applied_normal_rms_tensor"] <= 0.020001
    assert state["muon_warm_step_accepted_tensor"] in (0.0, 1.0)
