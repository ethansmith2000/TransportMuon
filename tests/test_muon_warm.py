import importlib
import math
import sys
import types

import torch


muon_stub = types.ModuleType("muon")
muon_stub.adam_update = lambda *args, **kwargs: args[0]
sys.modules.setdefault("muon", muon_stub)
muon_warm = importlib.import_module("muon_warm")


def _eager(function):
    return getattr(function, "_torchdynamo_orig_callable", function)


def test_rectangular_transport_tracks_row_space_motion():
    size = 4
    angle = 0.1
    q_previous = torch.cat((torch.eye(size), torch.zeros(size, size)), dim=1)
    target = torch.cat(
        (
            math.cos(angle) * torch.eye(size),
            math.sin(angle) * torch.eye(size),
        ),
        dim=1,
    )
    warm_step = _eager(muon_warm._warm_polar_jacobi_step)

    rotation_only = warm_step(
        target, q_previous, 1.0, 1e-3, "higham_cubic", 1, False
    )
    full_tangent = warm_step(
        target, q_previous, 1.0, 1e-3, "higham_cubic", 1, True
    )

    rotation_error = (rotation_only - target).norm()
    tangent_error = (full_tangent - target).norm()
    assert tangent_error < 1e-4 * rotation_error


def test_alignment_metrics_detect_row_space_motion_with_zero_skew():
    size = 4
    angle = 0.1
    q_previous = torch.cat((torch.eye(size), torch.zeros(size, size)), dim=1)
    target = torch.cat(
        (math.cos(angle) * torch.eye(size), math.sin(angle) * torch.eye(size)),
        dim=1,
    )
    metrics_function = _eager(muon_warm._warm_alignment_metrics)

    _, metrics = metrics_function(target, q_previous)

    rotation_error, subspace_error, relative_alignment = metrics.tolist()
    assert rotation_error == 0.0
    assert abs(subspace_error - math.sin(angle)) < 1e-5
    assert relative_alignment > 0.99


def test_tikhonov_jacobi_scale_is_bounded_near_zero_denominator():
    alignment = torch.tensor([[1.0, 0.0], [0.0, -1.0]])

    scale = muon_warm._jacobi_scale(alignment, 1e-2, "tikhonov")

    assert torch.isfinite(scale).all()
    assert scale[0, 1] == 0.0


def test_residual_form_retractions_match_original_polynomials_in_float64():
    torch.manual_seed(19)
    matrix = torch.randn(5, 9, dtype=torch.float64)
    matrix /= matrix.norm()
    gram = matrix @ matrix.T

    original_quadratic = 1.5 * matrix - 0.5 * gram @ matrix
    residual_quadratic = muon_warm._row_ns_quadratic(matrix, 1)
    gram_square = gram @ gram
    original_cubic = 0.125 * (
        15.0 * matrix + (-10.0 * gram + 3.0 * gram_square) @ matrix
    )
    residual_cubic = muon_warm._row_ns_higham_cubic(matrix, 1)

    assert torch.allclose(residual_quadratic, original_quadratic, atol=1e-12)
    assert torch.allclose(residual_cubic, original_cubic, atol=1e-12)


def test_residual_form_avoids_bfloat16_cancellation_near_identity():
    torch.manual_seed(20)
    q = torch.linalg.qr(torch.randn(16, 4)).Q.T
    tangent_seed = torch.randn_like(q)
    symmetric = 0.5 * (
        tangent_seed @ q.T + q @ tangent_seed.T
    )
    tangent = tangent_seed - symmetric @ q
    tangent *= 0.1 * math.sqrt(q.shape[0]) / tangent.norm()
    candidate = (q + tangent).to(torch.bfloat16)
    gram = candidate @ candidate.T
    original = 1.5 * candidate - 0.5 * gram @ candidate
    residual = muon_warm._row_ns_quadratic(candidate, 1)
    identity = torch.eye(q.shape[0])

    original_error = (original.float() @ original.float().T - identity).norm()
    residual_error = (residual.float() @ residual.float().T - identity).norm()
    assert residual_error < 0.8 * original_error


def test_uncapped_anchor_polish_avoids_loose_gershgorin_scale():
    torch.manual_seed(21)
    rows, columns = 32, 64
    left = torch.linalg.qr(torch.randn(rows, rows)).Q
    right = torch.linalg.qr(torch.randn(columns, rows)).Q.T
    singular_values = torch.linspace(0.7, 1.2, rows)
    candidate = (
        left @ torch.diag(singular_values) @ right
    ).to(torch.bfloat16)
    capped = muon_warm._row_ns_retract(
        candidate, 2, "higham_cubic", True
    )
    uncapped = muon_warm._row_ns_retract(
        candidate, 2, "higham_cubic", False
    )
    identity = torch.eye(rows)

    capped_error = (capped.float() @ capped.float().T - identity).norm()
    uncapped_error = (uncapped.float() @ uncapped.float().T - identity).norm()
    assert uncapped_error < 0.5 * capped_error


def test_polar_express_six_steps_need_no_extra_polish():
    torch.manual_seed(22)
    matrix = torch.randn(32, 64)
    prepared, _ = muon_warm._prepare_muon_matrix(matrix)

    polar = _eager(muon_warm._polar_express_prepared)(prepared, 6)
    orthogonality_error = (
        polar.float() @ polar.float().T - torch.eye(32)
    ).norm() / math.sqrt(32)

    assert orthogonality_error < 0.015


def test_tangent_trust_region_reduces_effective_step():
    size = 4
    q_previous = torch.cat((torch.eye(size), torch.zeros(size, size)), dim=1)
    target = torch.cat((0.8 * torch.eye(size), 0.6 * torch.eye(size)), dim=1)
    warm_step = _eager(muon_warm._warm_polar_jacobi_step_with_stats)

    _, stats = warm_step(
        target,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        1,
        True,
        None,
        "floor",
        0.05,
        -1.0,
    )

    effective_eta, tangent_rms, accepted = stats
    assert effective_eta < 1.0
    assert tangent_rms > 0.05
    assert effective_eta * tangent_rms <= 0.050001
    assert accepted == 1.0


def test_alignment_gate_rejects_an_overshooting_step():
    angle = 0.3
    q_previous = torch.eye(2)
    target = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    warm_step = _eager(muon_warm._warm_polar_jacobi_step_with_stats)

    q_next, stats = warm_step(
        target.to(torch.bfloat16),
        q_previous.to(torch.bfloat16),
        4.0,
        1e-3,
        "higham_cubic",
        1,
        True,
        None,
        "floor",
        0.0,
        0.0,
    )

    assert stats[2] == 0.0
    assert torch.equal(q_next, q_previous.to(torch.bfloat16))


def test_optimizer_records_device_side_controller_statistics(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_max_tangent_rms=0.05,
        muon_warm_alignment_tolerance=1e-4,
    )
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": torch.cat(
            (torch.eye(4), torch.zeros(4, 4)), dim=1
        ).to(torch.bfloat16),
    }
    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_with_stats",
        _eager(muon_warm._warm_polar_jacobi_step_with_stats),
    )

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert state["muon_warm_effective_eta_tensor"].ndim == 0
    assert state["muon_warm_tangent_rms_tensor"].isfinite()
    assert state["muon_warm_step_accepted_tensor"] in (0.0, 1.0)
    assert not state["muon_warm_did_anchor"]
    assert state["muon_warm_age"] == 1


def test_stats_mode_measures_tangent_without_changing_eta(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_record_stats=True,
    )
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": torch.cat(
            (torch.eye(4), torch.zeros(4, 4)), dim=1
        ).to(torch.bfloat16),
    }
    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_with_stats",
        _eager(muon_warm._warm_polar_jacobi_step_with_stats),
    )

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert state["muon_warm_effective_eta_tensor"] == 1.0
    assert state["muon_warm_tangent_rms_tensor"] > 0.0
    assert state["muon_warm_step_accepted_tensor"] == 1.0


def test_tensor_scalar_history_caps_tangent_spikes(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_max_tangent_ratio=2.0,
        muon_warm_tangent_ema_beta=0.9,
    )
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": torch.cat(
            (torch.eye(4), torch.zeros(4, 4)), dim=1
        ).to(torch.bfloat16),
        "muon_warm_tangent_ema_sq_tensor": torch.tensor(0.01**2),
    }
    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_with_stats",
        _eager(muon_warm._warm_polar_jacobi_step_with_stats),
    )

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    applied_rms = (
        state["muon_warm_effective_eta_tensor"]
        * state["muon_warm_tangent_rms_tensor"]
    )
    assert state["muon_warm_dynamic_cap_tensor"] == 0.02
    assert applied_rms <= 0.020001
    assert state["muon_warm_tangent_ema_sq_tensor"] > 0.01**2


def test_adaptive_tracking_error_triggers_anchor(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_max_tracking_error=0.05,
    )
    angle = 0.1
    gradient = torch.cat(
        (math.cos(angle) * torch.eye(4), math.sin(angle) * torch.eye(4)),
        dim=1,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous,
    }
    calls = []

    def fake_anchor(work_matrix, _steps):
        calls.append(True)
        return work_matrix

    monkeypatch.setattr(muon_warm, "_warm_alignment_metrics", _eager(muon_warm._warm_alignment_metrics))
    monkeypatch.setattr(muon_warm, "_muon_ns5_prepared", fake_anchor)
    monkeypatch.setattr(muon_warm, "_row_ns_retract", lambda value, *_: value)

    optimizer._muon_update_warm(gradient, state, optimizer.param_groups[0])

    assert calls == [True]
    assert state["muon_warm_subspace_error"] > 0.05
    assert state["muon_warm_did_anchor"]
    assert state["muon_warm_age"] == 0


def test_minimum_alignment_alone_can_trigger_anchor(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_min_alignment=0.5,
    )
    gradient = -torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1)
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous,
    }
    calls = []

    def fake_anchor(work_matrix, _steps):
        calls.append(True)
        return work_matrix

    monkeypatch.setattr(
        muon_warm,
        "_warm_alignment_metrics",
        _eager(muon_warm._warm_alignment_metrics),
    )
    monkeypatch.setattr(muon_warm, "_muon_ns5_prepared", fake_anchor)
    monkeypatch.setattr(muon_warm, "_row_ns_retract", lambda value, *_: value)

    optimizer._muon_update_warm(gradient, state, optimizer.param_groups[0])

    assert calls == [True]
    assert state["muon_warm_relative_alignment"] < 0.5


def test_aspect_ratio_scaling_does_not_modify_cached_polar(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(8, 4))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_warm_full_ns_steps=0,
    )
    group = optimizer.param_groups[0]
    state = {
        "step": 1,
        "momentum_fast": torch.zeros_like(parameter),
    }
    cached = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(torch.bfloat16)
    monkeypatch.setattr(muon_warm, "_muon_ns5_prepared", lambda *_: cached.contiguous())
    monkeypatch.setattr(muon_warm, "_row_ns_retract", lambda value, *_: value.contiguous())

    update = optimizer._muon_update_warm(torch.zeros_like(parameter), state, group)

    assert torch.allclose(
        torch.linalg.svdvals(state["muon_warm_q"].float()), torch.ones(4)
    )
    assert torch.allclose(
        torch.linalg.svdvals(update.float()), torch.full((4,), math.sqrt(2.0)), atol=1e-2
    )
