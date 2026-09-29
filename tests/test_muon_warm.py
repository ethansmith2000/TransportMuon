import importlib
import math

import pytest
import torch


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


def test_separate_normal_scale_controls_row_space_motion():
    size = 4
    q_previous = torch.cat((torch.eye(size), torch.zeros(size, size)), dim=1)
    target = torch.cat((0.8 * torch.eye(size), 0.6 * torch.eye(size)), dim=1)
    warm_step = _eager(muon_warm._warm_polar_jacobi_step)

    without_normal = warm_step(
        target,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        True,
        None,
        "floor",
        0.0,
        -1.0,
        1.0,
        0.0,
    )
    with_normal = warm_step(
        target,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        True,
        None,
        "floor",
        0.0,
        -1.0,
        1.0,
        1.0,
    )

    assert torch.equal(without_normal, q_previous)
    assert (with_normal - target).norm() < (without_normal - target).norm()


def test_separate_angular_scale_controls_in_space_rotation():
    angle = 0.1
    q_previous = torch.eye(2)
    target = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    warm_step = _eager(muon_warm._warm_polar_jacobi_step)

    disabled = warm_step(
        target, q_previous, 1.0, 1e-3, "higham_cubic", 0,
        True, None, "floor", 0.0, -1.0, 0.0, 1.0,
    )
    enabled = warm_step(
        target, q_previous, 1.0, 1e-3, "higham_cubic", 0,
        True, None, "floor", 0.0, -1.0, 1.0, 1.0,
    )

    assert torch.equal(disabled, q_previous)
    assert (enabled - target).norm() < (disabled - target).norm()


def test_lazy_retraction_follows_configured_warm_cadence(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_warm_retract_steps=1,
        muon_warm_retract_every=2,
    )
    calls = []

    def fake_step(*arguments, **kwargs):
        calls.append(arguments[5])
        return arguments[1]

    monkeypatch.setattr(muon_warm, "_warm_polar_jacobi_step", fake_step)
    q = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1)
    state = {"muon_warm_age": 0}
    optimizer._compute_warm_direction(q, q, None, state, False)
    assert state["muon_warm_did_retract"] is False
    state["muon_warm_age"] = 1
    optimizer._compute_warm_direction(q, q, None, state, False)
    assert state["muon_warm_did_retract"] is True
    assert calls == [0, 1]


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

    (
        rotation_error,
        subspace_error,
        relative_alignment,
        relative_min_stretch,
    ) = metrics.tolist()
    assert rotation_error == 0.0
    assert abs(subspace_error - math.sin(angle)) < 1e-5
    assert relative_alignment > 0.99
    assert relative_min_stretch > 0.99


def test_alignment_metrics_detect_symmetric_indefinite_wrong_branch():
    q_previous = torch.cat((torch.eye(2), torch.zeros(2, 2)), dim=1)
    stretch = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    work_matrix = stretch @ q_previous
    metrics_function = _eager(muon_warm._warm_alignment_metrics)

    _, metrics = metrics_function(work_matrix, q_previous)

    rotation_error, subspace_error, _, relative_min_stretch = metrics.tolist()
    assert rotation_error == 0.0
    assert subspace_error == 0.0
    assert relative_min_stretch < 0.0


def test_zero_retraction_steps_are_an_exact_noop():
    torch.manual_seed(23)
    matrix = torch.randn(4, 8)

    retracted = muon_warm._row_ns_retract(
        matrix, 0, "higham_cubic", cap_spectral_norm=True
    )

    assert retracted is matrix


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


def test_power_cap_is_less_conservative_than_dense_gershgorin_bound():
    torch.manual_seed(21)
    rows, columns = 32, 64
    left = torch.linalg.qr(torch.randn(rows, rows)).Q
    right = torch.linalg.qr(torch.randn(columns, rows)).Q.T
    singular_values = torch.linspace(0.7, 1.2, rows)
    candidate = (
        left @ torch.diag(singular_values) @ right
    ).to(torch.bfloat16)
    gershgorin = muon_warm._row_ns_retract(
        candidate, 1, "higham_cubic", True
    )
    power, vector, estimate, scale = muon_warm._row_ns_retract_power_cap(
        candidate,
        1,
        "higham_cubic",
        None,
        2,
        1.25,
    )
    identity = torch.eye(rows)
    gershgorin_error = (
        gershgorin.float() @ gershgorin.float().T - identity
    ).norm()
    power_error = (
        power.float() @ power.float().T - identity
    ).norm()

    assert vector.shape == (rows,)
    assert estimate > 1.0
    assert 0.0 < scale < 1.0
    assert power_error < 0.6 * gershgorin_error


def test_power_cap_fresh_seed_detects_a_direction_missed_by_cache():
    singular_values = torch.tensor([1.0, 1.0, 1.0, 2.0])
    candidate = torch.diag(singular_values).to(torch.bfloat16)
    stale_vector = torch.tensor([1.0, 0.0, 0.0, 0.0])

    _, next_vector, estimate, scale = muon_warm._row_ns_retract_power_cap(
        candidate,
        1,
        "higham_cubic",
        stale_vector,
        2,
        1.25,
    )

    assert estimate == pytest.approx(2.0)
    assert scale == pytest.approx(0.5)
    assert next_vector.argmax() == 3


def test_optimizer_caches_power_cap_vector_and_diagnostics(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_spectral_cap_mode="power",
        muon_warm_power_steps=2,
        muon_warm_power_safety_factor=1.25,
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
        "_warm_polar_jacobi_step_power_cap",
        _eager(muon_warm._warm_polar_jacobi_step_power_cap),
    )

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert state["muon_warm_power_vector"].shape == (4,)
    assert state["muon_warm_power_vector"].isfinite().all()
    assert state["muon_warm_power_sigma_tensor"].isfinite()
    assert 0.0 < state["muon_warm_spectral_cap_scale_tensor"] <= 1.0


def test_power_step_cap_preserves_unaffected_singular_directions():
    rows = 4
    q_previous = torch.cat(
        (torch.eye(rows), torch.zeros(rows, rows)), dim=1
    ).to(torch.bfloat16)
    correction = torch.cat(
        (
            torch.zeros(rows, rows),
            torch.diag(torch.tensor([0.1, 1.0, 10.0, 100.0])),
        ),
        dim=1,
    ).to(torch.bfloat16)
    whole_candidate = q_previous + correction
    whole_scaled, _, _, _ = muon_warm._row_ns_retract_power_cap(
        whole_candidate,
        1,
        "higham_cubic",
        None,
        2,
        1.05,
    )
    (
        step_capped,
        _,
        _,
        final_scale,
        probe_sigma,
        step_scale,
        _,
        _,
        _,
    ) = muon_warm._row_ns_retract_power_step_cap(
        q_previous,
        correction,
        torch.tensor(1.0),
        1,
        "higham_cubic",
        None,
        2,
        1.05,
    )
    identity = torch.eye(rows)
    whole_error = (
        whole_scaled.float() @ whole_scaled.float().T - identity
    ).norm()
    step_error = (
        step_capped.float() @ step_capped.float().T - identity
    ).norm()

    assert probe_sigma > 90.0
    assert step_scale < 0.01
    assert final_scale == 1.0
    assert step_error < 0.1 * whole_error


def test_adaptive_power_refines_only_when_first_direction_is_uncertain():
    gram = torch.tensor(
        [[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 0.25]],
        dtype=torch.bfloat16,
    )

    refined = muon_warm._power_spectral_cap_adaptive(
        gram, None, 2, 1.25, 0.01
    )
    kept = muon_warm._power_spectral_cap_adaptive(
        gram, None, 2, 1.25, 0.5
    )
    fixed_one = muon_warm._power_spectral_cap(gram, None, 1, 1.25)
    fixed_two = muon_warm._power_spectral_cap(gram, None, 2, 1.25)

    assert refined[4] == 1.0
    assert kept[4] == 0.0
    assert refined[5] > 0.01
    assert torch.equal(refined[1], fixed_two[1])
    assert torch.equal(kept[1], fixed_one[1])


def test_optimizer_records_power_step_scale(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_spectral_cap_mode="power_step",
        muon_warm_power_steps=2,
        muon_warm_power_safety_factor=1.05,
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
        "_warm_polar_jacobi_step_power_step_cap",
        _eager(muon_warm._warm_polar_jacobi_step_power_step_cap),
    )

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert state["muon_warm_power_step_samples"] == 1
    assert state["muon_warm_power_probe_sigma_tensor"].isfinite()
    assert 0.0 < state["muon_warm_spectral_step_scale_tensor"] <= 1.0


def test_legacy_retraction_oracle_matches_full_candidate_filter():
    q_previous = torch.cat((torch.eye(2), torch.zeros(2, 2)), dim=1)
    work_matrix = torch.tensor(
        [[1.0, 0.0, 10.0, 0.0], [0.0, 1.0, 0.0, 0.5]]
    )
    core = _eager(muon_warm._warm_polar_jacobi_core)

    retained = core(
        work_matrix,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        1,
        True,
        jacobi_damping="tikhonov",
        normal_inv_cap=4.0,
    )
    oracle = core(
        work_matrix,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        1,
        True,
        jacobi_damping="tikhonov",
        spectral_cap_mode="power_step",
        power_steps=2,
        power_safety_factor=1.05,
        normal_inv_cap=4.0,
        legacy_retraction_output=True,
    )

    assert torch.equal(oracle[7], retained[0])
    assert not torch.allclose(oracle[0], oracle[7])


def test_legacy_retraction_oracle_filters_output_when_cache_retraction_is_lazy():
    q_previous = torch.cat((torch.eye(2), torch.zeros(2, 2)), dim=1)
    work_matrix = torch.tensor(
        [[1.0, 0.0, 10.0, 0.0], [0.0, 1.0, 0.0, 0.5]]
    )
    core = _eager(muon_warm._warm_polar_jacobi_core)

    retained = core(
        work_matrix,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        1,
        True,
        jacobi_damping="tikhonov",
        normal_inv_cap=4.0,
    )
    lazy_oracle = core(
        work_matrix,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        True,
        jacobi_damping="tikhonov",
        spectral_cap_mode="power_step",
        power_steps=2,
        power_safety_factor=1.05,
        normal_inv_cap=4.0,
        legacy_retraction_output=True,
    )

    assert torch.equal(lazy_oracle[7], retained[0])
    cache_error = (
        lazy_oracle[0] @ lazy_oracle[0].mT - torch.eye(2)
    ).norm()
    assert cache_error > 0.0


def test_legacy_retraction_oracle_separates_cache_and_output(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_spectral_cap_mode="power_step",
        muon_warm_legacy_retraction_output=True,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    q_cache = 0.75 * q_previous
    q_output = 0.25 * q_previous
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous.clone(),
    }

    def fake_oracle(*args):
        scalar = q_previous.new_tensor(1.0, dtype=torch.float32)
        return (
            q_cache,
            q_output,
            torch.ones(4, dtype=q_previous.dtype),
            scalar,
            scalar,
            scalar,
            scalar,
        )

    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_power_step_cap_with_legacy_output",
        fake_oracle,
    )

    update = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert torch.equal(state["muon_warm_q"], q_cache)
    assert torch.equal(update, q_output)
    assert "muon_warm_legacy_output_direction" not in state


def test_legacy_retraction_oracle_uses_separate_lazy_compiled_path(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_retract_every=2,
        muon_warm_spectral_cap_mode="power_step",
        muon_warm_legacy_retraction_output=True,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    q_cache = 0.75 * q_previous
    q_output = 0.25 * q_previous
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous.clone(),
        "muon_warm_age": 0,
    }
    called = []

    def fake_lazy(*args):
        called.append(True)
        scalar = q_previous.new_tensor(1.0, dtype=torch.float32)
        return (
            q_cache,
            q_output,
            torch.ones(4, dtype=q_previous.dtype),
            scalar,
            scalar,
            scalar,
            scalar,
        )

    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_power_step_cap_with_legacy_output_no_cache_retract",
        fake_lazy,
    )

    update = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert called == [True]
    assert torch.equal(state["muon_warm_q"], q_cache)
    assert torch.equal(update, q_output)
    assert not state["muon_warm_did_retract"]
    assert state["muon_warm_did_output_retract"]
    assert state["muon_warm_output_retractions"] == 1
    assert "muon_warm_power_cap_samples" not in state
    assert state["muon_warm_power_step_samples"] == 1


def test_power_step_angular_gate_uses_lightweight_lazy_path(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_retract_steps=0,
        muon_warm_spectral_cap_mode="power_step",
        muon_warm_legacy_retraction_output=True,
        muon_warm_max_angular_rms=8.0,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    q_cache = 0.75 * q_previous
    q_output = 0.25 * q_previous
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous.clone(),
        "muon_warm_age": 0,
    }
    called = []

    def fake_angular_step(*args):
        called.append(True)
        scalar = q_previous.new_tensor(1.0, dtype=torch.float32)
        angular = q_previous.new_tensor(12.0, dtype=torch.float32)
        return (
            q_cache,
            q_output,
            angular,
            torch.ones(4, dtype=q_previous.dtype),
            scalar,
            scalar,
            scalar,
            scalar,
        )

    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_power_step_cap_with_angular_signal_no_cache_retract",
        fake_angular_step,
    )

    update = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert called == [True]
    assert torch.equal(state["muon_warm_q"], q_cache)
    assert torch.equal(update, q_output)
    assert state["muon_warm_angular_signal_tensor"] == 12.0


def test_skew_ratio_is_scale_free_polar_stationarity_residual():
    angle = 0.3
    q_previous = torch.eye(2)
    target = 7.0 * torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )

    result = _eager(muon_warm._warm_polar_jacobi_core)(
        target,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        measure_skew_ratio=True,
    )

    assert result[1][9] == pytest.approx(abs(math.sin(angle)), rel=1e-6)


def test_power_step_skew_ratio_gate_uses_lightweight_lazy_path(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_retract_steps=0,
        muon_warm_spectral_cap_mode="power_step",
        muon_warm_legacy_retraction_output=True,
        muon_warm_max_skew_ratio=0.1,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    q_cache = 0.75 * q_previous
    q_output = 0.25 * q_previous
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous.clone(),
        "muon_warm_age": 0,
    }
    called = []

    def fake_skew_ratio_step(*args):
        called.append(True)
        scalar = q_previous.new_tensor(1.0, dtype=torch.float32)
        signal = q_previous.new_tensor(0.25, dtype=torch.float32)
        return (
            q_cache,
            q_output,
            signal,
            torch.ones(4, dtype=q_previous.dtype),
            scalar,
            scalar,
            scalar,
            scalar,
        )

    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_power_step_cap_with_skew_ratio_signal_no_cache_retract",
        fake_skew_ratio_step,
    )

    update = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert called == [True]
    assert torch.equal(state["muon_warm_q"], q_cache)
    assert torch.equal(update, q_output)
    assert state["muon_warm_skew_ratio_signal_tensor"] == 0.25


def test_skew_signal_is_only_evaluated_when_next_check_consumes_it(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=8,
        muon_warm_full_ns_steps=0,
        muon_warm_retract_steps=0,
        muon_warm_spectral_cap_mode="power_step",
        muon_warm_legacy_retraction_output=True,
        muon_warm_max_skew_ratio=0.1,
        muon_warm_check_every=4,
        muon_warm_signal_check_only=True,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 1,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous.clone(),
        "muon_warm_age": 0,
    }
    calls = []

    def fake_legacy(*args):
        calls.append("legacy")
        scalar = q_previous.new_tensor(1.0, dtype=torch.float32)
        return (
            q_previous,
            q_previous,
            torch.ones(4, dtype=q_previous.dtype),
            scalar,
            scalar,
            scalar,
            scalar,
        )

    def fake_skew(*args):
        calls.append("skew")
        scalar = q_previous.new_tensor(1.0, dtype=torch.float32)
        return (
            q_previous,
            q_previous,
            scalar.new_tensor(0.25),
            torch.ones(4, dtype=q_previous.dtype),
            scalar,
            scalar,
            scalar,
            scalar,
        )

    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_power_step_cap_with_legacy_output_no_cache_retract",
        fake_legacy,
    )
    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_power_step_cap_with_skew_ratio_signal_no_cache_retract",
        fake_skew,
    )

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )
    assert calls == ["legacy"]
    assert "muon_warm_skew_ratio_signal_tensor" not in state
    assert "muon_warm_skew_ratio_signal_evaluations" not in state

    state["step"] = 3
    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )
    assert calls == ["legacy", "skew"]
    assert state["muon_warm_skew_ratio_signal_tensor"] == 0.25
    assert state["muon_warm_skew_ratio_signal_evaluations"] == 1

    # Step eight is already a fixed anchor, so measuring a step-seven signal
    # cannot affect the next update and is skipped as well.
    state["step"] = 7
    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )
    assert calls == ["legacy", "skew", "legacy"]
    assert state["muon_warm_skew_ratio_signal_evaluations"] == 1


def test_skew_signal_cadence_is_opt_in(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=8,
        muon_warm_full_ns_steps=0,
        muon_warm_retract_steps=0,
        muon_warm_spectral_cap_mode="power_step",
        muon_warm_legacy_retraction_output=True,
        muon_warm_max_skew_ratio=0.1,
        muon_warm_check_every=4,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 1,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous.clone(),
        "muon_warm_age": 0,
    }
    calls = []

    def fake_skew(*args):
        calls.append("skew")
        scalar = q_previous.new_tensor(1.0, dtype=torch.float32)
        return (
            q_previous,
            q_previous,
            scalar.new_tensor(0.25),
            torch.ones(4, dtype=q_previous.dtype),
            scalar,
            scalar,
            scalar,
            scalar,
        )

    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_power_step_cap_with_skew_ratio_signal_no_cache_retract",
        fake_skew,
    )
    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert calls == ["skew"]
    assert state["muon_warm_skew_ratio_signal_evaluations"] == 1


def test_separate_skew_signal_uses_one_direction_path(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=8,
        muon_warm_full_ns_steps=0,
        muon_warm_retract_steps=0,
        muon_warm_spectral_cap_mode="power_step",
        muon_warm_legacy_retraction_output=True,
        muon_warm_max_skew_ratio=0.1,
        muon_warm_check_every=4,
        muon_warm_signal_check_only=True,
        muon_warm_separate_skew_signal=True,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 1,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous.clone(),
        "muon_warm_age": 0,
    }
    calls = []

    def fake_terms(*args):
        calls.append("direction")
        scalar = q_previous.new_tensor(1.0, dtype=torch.float32)
        square = torch.eye(4, dtype=q_previous.dtype)
        return (
            q_previous,
            q_previous,
            square,
            square,
            torch.ones(4, dtype=q_previous.dtype),
            scalar,
            scalar,
            scalar,
            scalar,
        )

    def fake_reduction(*args):
        calls.append("reduction")
        return q_previous.new_tensor(0.25, dtype=torch.float32)

    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_power_step_cap_with_skew_terms_no_cache_retract",
        fake_terms,
    )
    monkeypatch.setattr(muon_warm, "_skew_ratio_from_terms", fake_reduction)

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )
    assert calls == ["direction"]
    assert "muon_warm_skew_ratio_signal_tensor" not in state

    state["step"] = 3
    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )
    assert calls == ["direction", "direction", "reduction"]
    assert state["muon_warm_skew_ratio_signal_tensor"] == 0.25
    assert state["muon_warm_skew_ratio_signal_evaluations"] == 1

    state["step"] = 7
    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )
    assert calls == ["direction", "direction", "reduction", "direction"]
    assert state["muon_warm_skew_ratio_signal_evaluations"] == 1


def test_signal_free_and_skew_measured_paths_emit_identical_directions():
    torch.manual_seed(31)
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    work_matrix = torch.randn(4, 8).to(torch.bfloat16)
    common = (
        work_matrix,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        True,
        None,
        "tikhonov",
    )
    legacy = _eager(
        muon_warm._warm_polar_jacobi_step_power_step_cap_with_legacy_output_no_cache_retract
    )(*common, None, 1, 1.05, 1.0, 1.0, 4.0)
    measured = _eager(
        muon_warm._warm_polar_jacobi_step_power_step_cap_with_skew_ratio_signal_no_cache_retract
    )(*common, None, 1, 1.05, True, 1.0, 1.0, 4.0)

    assert torch.equal(legacy[0], measured[0])
    assert torch.equal(legacy[1], measured[1])
    for legacy_value, measured_value in zip(legacy[2:], measured[3:]):
        assert torch.equal(legacy_value, measured_value)


def test_raw_skew_terms_reproduce_combined_signal_and_direction():
    torch.manual_seed(37)
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    work_matrix = torch.randn(4, 8).to(torch.bfloat16)
    common = (
        work_matrix,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        True,
        None,
        "tikhonov",
    )
    measured = _eager(
        muon_warm._warm_polar_jacobi_step_power_step_cap_with_skew_ratio_signal_no_cache_retract
    )(*common, None, 1, 1.05, True, 1.0, 1.0, 4.0)
    terms = _eager(
        muon_warm._warm_polar_jacobi_step_power_step_cap_with_skew_terms_no_cache_retract
    )(*common, None, 1, 1.05, True, 1.0, 1.0, 4.0)
    signal = _eager(muon_warm._skew_ratio_from_terms)(terms[2], terms[3])

    assert torch.equal(terms[0], measured[0])
    assert torch.equal(terms[1], measured[1])
    assert signal == measured[2]
    for terms_value, measured_value in zip(terms[4:], measured[3:]):
        assert torch.equal(terms_value, measured_value)

def test_adaptive_power_skew_path_records_device_side_refinement(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_retract_steps=0,
        muon_warm_spectral_cap_mode="power_step",
        muon_warm_power_steps=2,
        muon_warm_power_refine_threshold=0.02,
        muon_warm_legacy_retraction_output=True,
        muon_warm_max_skew_ratio=0.52,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous.clone(),
        "muon_warm_age": 0,
    }
    called = []

    def fake_adaptive(*args):
        called.append(args)
        scalar = q_previous.new_tensor(1.0, dtype=torch.float32)
        return (
            q_previous,
            q_previous,
            scalar.new_tensor(0.1),
            torch.ones(4, dtype=q_previous.dtype),
            scalar,
            scalar,
            scalar,
            scalar,
            scalar,
            scalar.new_tensor(0.125),
        )

    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_power_step_cap_with_adaptive_power_skew_ratio_signal_no_cache_retract",
        fake_adaptive,
    )

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert len(called) == 1
    assert called[0][12] == 0.02
    assert state["muon_warm_power_refinement_samples"] == 1
    assert state["muon_warm_power_refinement_sum_tensor"] == 1.0
    assert state["muon_warm_power_uncertainty_sum_tensor"] == 0.125


def test_adaptive_power_requires_supported_lightweight_skew_configuration():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    common = {
        "params": [{"params": [parameter], "use_muon": True}],
        "muon_warm_spectral_cap_mode": "power_step",
        "muon_warm_power_steps": 2,
        "muon_warm_power_refine_threshold": 0.02,
    }
    with pytest.raises(ValueError, match="max_skew_ratio"):
        muon_warm.MuonWarm(**common)
    with pytest.raises(ValueError, match="lightweight"):
        muon_warm.MuonWarm(
            **common,
            muon_warm_max_skew_ratio=0.52,
            muon_warm_record_stats=True,
        )


def test_separate_skew_signal_requires_check_only_lightweight_controller():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    common = {
        "params": [{"params": [parameter], "use_muon": True}],
        "muon_warm_spectral_cap_mode": "power_step",
        "muon_warm_max_skew_ratio": 0.52,
        "muon_warm_separate_skew_signal": True,
    }
    with pytest.raises(ValueError, match="signal_check_only"):
        muon_warm.MuonWarm(**common)
    with pytest.raises(ValueError, match="lightweight"):
        muon_warm.MuonWarm(
            **common,
            muon_warm_signal_check_only=True,
            muon_warm_record_stats=True,
        )


def test_legacy_retraction_oracle_requires_power_step_mode():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    with pytest.raises(ValueError, match="power_step"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_legacy_retraction_output=True,
        )


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

    effective_eta, tangent_rms, accepted, angular_rms, normal_rms = stats[:5]
    assert effective_eta < 1.0
    assert tangent_rms > 0.05
    assert effective_eta * tangent_rms <= 0.050001
    assert accepted == 1.0
    assert angular_rms >= 0.0
    assert normal_rms > 0.0


def test_normal_inverse_cap_limits_ill_conditioned_rectangular_correction():
    q_previous = torch.cat((torch.eye(2), torch.zeros(2, 2)), dim=1)
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1e-3, 0.0, 1.0],
        ]
    )
    warm_step = _eager(muon_warm._warm_polar_jacobi_step_with_stats)

    uncapped, uncapped_stats = warm_step(
        target,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        True,
        None,
        "tikhonov",
        0.0,
        -1.0,
        None,
        1.0,
        1.0,
        0.0,
    )
    capped, capped_stats = warm_step(
        target,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        True,
        None,
        "tikhonov",
        0.0,
        -1.0,
        None,
        1.0,
        1.0,
        2.0,
    )

    assert uncapped_stats[6] > 100.0
    assert torch.equal(capped_stats[6], uncapped_stats[6])
    assert capped_stats[7] > 0.0
    assert capped_stats[4] < 0.01 * uncapped_stats[4]
    assert (capped - q_previous).norm() < 0.01 * (uncapped - q_previous).norm()


def test_angular_signal_path_matches_instrumented_warm_step():
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1)
    target = torch.randn(4, 8)
    signal_step = _eager(
        muon_warm._warm_polar_jacobi_step_with_angular_signal
    )
    stats_step = _eager(muon_warm._warm_polar_jacobi_step_with_stats)

    signaled, angular_signal = signal_step(
        target, q_previous, 1.0, 1e-3, "higham_cubic", 1,
        True, None, "tikhonov", 1.0, 1.0, 4.0,
    )
    instrumented, stats = stats_step(
        target, q_previous, 1.0, 1e-3, "higham_cubic", 1,
        True, None, "tikhonov", 0.0, -1.0, None, 1.0, 1.0, 4.0,
    )

    assert torch.equal(signaled, instrumented)
    assert torch.equal(angular_signal, stats[3])


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
    assert state["muon_warm_angular_rms_tensor"].isfinite()
    assert state["muon_warm_normal_rms_tensor"].isfinite()
    assert state["muon_warm_max_tangent_rms_tensor"].isfinite()
    assert state["muon_warm_max_applied_tangent_rms_tensor"] <= 0.050001
    assert state["muon_warm_min_effective_eta_tensor"] <= 1.0
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


def test_update_stats_compare_warm_direction_with_fresh_anchor(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True, "lr": 0.01}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_record_stats=True,
        muon_warm_update_stats_every=1,
        muon_warm_reference_lr_ratio=0.25,
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
    monkeypatch.setattr(
        muon_warm,
        "_muon_ns5_prepared",
        _eager(muon_warm._muon_ns5_prepared),
    )

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert state["muon_update_samples"] == 1
    assert state["muon_reference_samples"] == 1
    assert state["muon_update_elements"] == parameter.numel()
    assert state["muon_reference_elements"] == parameter.numel()
    for key in (
        "muon_update_direction_sq_sum_tensor",
        "muon_update_applied_sq_sum_tensor",
        "muon_reference_direction_sq_sum_tensor",
        "muon_reference_direction_dot_sum_tensor",
        "muon_reference_difference_sq_sum_tensor",
        "muon_reference_candidate_applied_sq_sum_tensor",
        "muon_reference_applied_sq_sum_tensor",
    ):
        assert state[key].isfinite()
    assert state["muon_reference_applied_sq_sum_tensor"] < state[
        "muon_reference_candidate_applied_sq_sum_tensor"
    ]
    assert state["muon_update_stats_by_age"][1]["samples"] == 1
    assert state["muon_update_stats_by_age"][1]["reference_samples"] == 1
    age_stats = state["muon_update_stats_by_age"][1]
    for predictor in (
        "momentum_innovation_ratio",
        "tangent_rms",
        "angular_rms",
        "normal_rms",
        "relative_max_abs_inverse",
        "normal_inverse_clipped_fraction",
    ):
        for suffix in (
            "sample",
            "value",
            "value_sq",
            "error",
            "error_sq",
            "value_error",
        ):
            assert age_stats[
                f"refresh_predictor_{predictor}_{suffix}_sum_tensor"
            ].isfinite()


def test_output_normalization_restores_muon_direction_norm(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_normalize_output=True,
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
        "_warm_polar_jacobi_step",
        _eager(muon_warm._warm_polar_jacobi_step),
    )

    update = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert update.float().norm() == pytest.approx(math.sqrt(4), rel=2e-2)


def test_output_normalization_uses_flattened_convolution_rows(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(6, 4, 3, 3))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=1,
        muon_warm_full_ns_steps=0,
        muon_warm_normalize_output=True,
    )
    state = {"step": 1, "momentum_fast": torch.zeros_like(parameter)}
    monkeypatch.setattr(
        muon_warm,
        "_muon_ns5_prepared",
        _eager(muon_warm._muon_ns5_prepared),
    )

    update = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert update.shape == parameter.shape
    assert update.float().norm() == pytest.approx(math.sqrt(6), rel=2e-2)


def test_warm_output_scale_filters_update_without_scaling_cached_basis(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_output_scale=0.25,
        muon_warm_update_stats_every=1,
    )
    cached = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": cached.clone(),
    }
    monkeypatch.setattr(
        optimizer,
        "_compute_warm_direction",
        lambda *_: cached.clone(),
    )
    monkeypatch.setattr(
        optimizer,
        "_compute_anchor_direction",
        lambda *_: cached.clone(),
    )

    update = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    expected = cached.float() * 0.25
    assert torch.allclose(update.float(), expected)
    assert torch.equal(state["muon_warm_q"], cached)
    assert state["muon_update_direction_sq_sum_tensor"] == expected.square().sum()
    assert (
        state["muon_reference_candidate_applied_sq_sum_tensor"]
        == expected.square().sum() * optimizer.param_groups[0]["lr"] ** 2
    )


def test_warm_output_scale_leaves_anchor_update_unscaled(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=1,
        muon_warm_full_ns_steps=0,
        muon_warm_output_scale=0.25,
    )
    anchor = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {"step": 1, "momentum_fast": torch.zeros_like(parameter)}
    monkeypatch.setattr(
        optimizer,
        "_compute_anchor_direction",
        lambda *_: anchor.clone(),
    )

    update = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert torch.equal(update, anchor)


def test_warm_output_scale_can_decay_with_cache_age(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_output_scale=0.2,
        muon_warm_output_scale_start=0.4,
        muon_warm_output_scale_decay=math.log(2.0),
    )
    cached = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": cached.clone(),
    }
    monkeypatch.setattr(
        optimizer,
        "_compute_warm_direction",
        lambda *_: cached.clone(),
    )

    age_one = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )
    state["step"] = 3
    age_two = optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert torch.allclose(age_one.float(), cached.float() * 0.4, atol=2e-3)
    assert torch.allclose(age_two.float(), cached.float() * 0.3, atol=2e-3)


def test_warm_output_scale_must_be_non_negative():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    with pytest.raises(ValueError, match="output_scale"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_output_scale=-0.1,
        )
    with pytest.raises(ValueError, match="output_scale_start"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_output_scale_start=-0.1,
        )
    with pytest.raises(ValueError, match="output_scale_decay"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_output_scale_decay=-0.1,
        )


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


def test_minimum_stretch_rejects_symmetric_indefinite_branch(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(2, 4))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_min_stretch=0.01,
    )
    q_previous = torch.cat((torch.eye(2), torch.zeros(2, 2)), dim=1)
    gradient = torch.tensor([[1.0, 2.0], [2.0, 1.0]]) @ q_previous
    state = {
        "step": 2,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous.to(torch.bfloat16),
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
    assert state["muon_warm_relative_min_stretch"] < 0.0
    assert state["muon_warm_did_anchor"]


def test_maximum_warm_age_triggers_anchor_with_reason(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_max_age=3,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 4,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous,
        "muon_warm_age": 3,
    }
    monkeypatch.setattr(muon_warm, "_muon_ns5_prepared", lambda value, _: value)
    monkeypatch.setattr(muon_warm, "_row_ns_retract", lambda value, *_: value)

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert state["muon_warm_did_anchor"]
    assert state["muon_warm_anchor_reason"] == "max_age"
    assert state["muon_warm_age"] == 0
    assert state["muon_warm_anchor_counts"] == {"max_age": 1}


def test_angular_signal_batches_checks_and_triggers_anchor(monkeypatch):
    parameters = [
        torch.nn.Parameter(torch.zeros(4, 8)),
        torch.nn.Parameter(torch.zeros(4, 8)),
    ]
    optimizer = muon_warm.MuonWarm(
        [{"params": parameters, "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_check_every=4,
        muon_warm_max_angular_rms=10.0,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    for parameter, signal in zip(parameters, (20.0, 5.0)):
        optimizer.state[parameter].update(
            {
                "step": 3,
                "momentum_fast": torch.zeros_like(parameter),
                "muon_warm_q": q_previous.clone(),
                "muon_warm_angular_signal_tensor": torch.tensor(signal),
            }
        )

    optimizer._prepare_angular_anchor_flags()

    assert optimizer.state[parameters[0]]["muon_warm_force_angular_anchor"]
    assert not optimizer.state[parameters[1]]["muon_warm_force_angular_anchor"]
    assert optimizer.state[parameters[0]]["muon_warm_angular_checks"] == 1
    assert optimizer.state[parameters[1]]["muon_warm_angular_checks"] == 1

    monkeypatch.setattr(muon_warm, "_muon_ns5_prepared", lambda value, _: value)
    monkeypatch.setattr(muon_warm, "_row_ns_retract", lambda value, *_: value)
    optimizer.state[parameters[0]]["step"] = 4
    optimizer._muon_update_warm(
        torch.randn_like(parameters[0]),
        optimizer.state[parameters[0]],
        optimizer.param_groups[0],
    )

    assert optimizer.state[parameters[0]]["muon_warm_did_anchor"]
    assert optimizer.state[parameters[0]]["muon_warm_anchor_reason"] == "angular_rms"
    assert optimizer.state[parameters[0]]["muon_warm_anchor_counts"] == {
        "angular_rms": 1
    }


def test_skew_ratio_signal_batches_checks_and_triggers_anchor(monkeypatch):
    parameters = [
        torch.nn.Parameter(torch.zeros(4, 8)),
        torch.nn.Parameter(torch.zeros(4, 8)),
    ]
    optimizer = muon_warm.MuonWarm(
        [{"params": parameters, "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_check_every=4,
        muon_warm_max_skew_ratio=0.1,
        muon_warm_spectral_cap_mode="power_step",
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    for parameter, signal in zip(parameters, (0.2, 0.05)):
        optimizer.state[parameter].update(
            {
                "step": 3,
                "momentum_fast": torch.zeros_like(parameter),
                "muon_warm_q": q_previous.clone(),
                "muon_warm_skew_ratio_signal_tensor": torch.tensor(signal),
            }
        )

    optimizer._prepare_angular_anchor_flags()

    assert optimizer.state[parameters[0]]["muon_warm_force_skew_ratio_anchor"]
    assert not optimizer.state[parameters[1]]["muon_warm_force_skew_ratio_anchor"]
    assert optimizer.state[parameters[0]]["muon_warm_skew_ratio_checks"] == 1
    assert optimizer.state[parameters[1]]["muon_warm_skew_ratio_checks"] == 1

    monkeypatch.setattr(muon_warm, "_muon_ns5_prepared", lambda value, _: value)
    monkeypatch.setattr(muon_warm, "_row_ns_retract", lambda value, *_: value)
    optimizer.state[parameters[0]]["step"] = 4
    optimizer._muon_update_warm(
        torch.randn_like(parameters[0]),
        optimizer.state[parameters[0]],
        optimizer.param_groups[0],
    )

    assert optimizer.state[parameters[0]]["muon_warm_did_anchor"]
    assert optimizer.state[parameters[0]]["muon_warm_anchor_reason"] == "skew_ratio"
    assert optimizer.state[parameters[0]]["muon_warm_anchor_counts"] == {
        "skew_ratio": 1
    }


def test_async_drift_check_poll_is_nonblocking_and_discards_stale_results():
    parameters = [
        torch.nn.Parameter(torch.zeros(4, 8)),
        torch.nn.Parameter(torch.zeros(4, 8)),
    ]
    optimizer = muon_warm.MuonWarm(
        [{"params": parameters, "use_muon": True}],
        muon_warm_check_every=4,
        muon_warm_max_skew_ratio=0.1,
        muon_warm_async_checks=True,
        muon_warm_spectral_cap_mode="power_step",
    )
    current_state = optimizer.state[parameters[0]]
    stale_state = optimizer.state[parameters[1]]
    current_state.update({"step": 0, "muon_warm_anchor_serial": 2})
    stale_state.update({"step": 0, "muon_warm_anchor_serial": 3})

    class FakeEvent:
        def __init__(self, ready):
            self.ready = ready

        def query(self):
            return self.ready

    pending = {
        "entries": [(current_state, 2), (stale_state, 2)],
        "host_signals": torch.tensor([0.2, 0.2]),
        "event": FakeEvent(False),
        "threshold": 0.1,
        "flag_key": "muon_warm_force_skew_ratio_anchor",
        "checks_key": "muon_warm_skew_ratio_checks",
    }
    optimizer._muon_warm_pending_drift_checks = {"fake-device": pending}

    optimizer._prepare_angular_anchor_flags()

    assert "muon_warm_force_skew_ratio_anchor" not in current_state
    assert "fake-device" in optimizer._muon_warm_pending_drift_checks

    pending["event"].ready = True
    optimizer._prepare_angular_anchor_flags()

    assert current_state["muon_warm_force_skew_ratio_anchor"]
    assert current_state["muon_warm_skew_ratio_checks"] == 1
    assert current_state["muon_warm_async_checks_completed"] == 1
    assert "muon_warm_force_skew_ratio_anchor" not in stale_state
    assert stale_state["muon_warm_skew_ratio_checks"] == 1
    assert stale_state["muon_warm_async_stale_checks"] == 1
    assert not optimizer._muon_warm_pending_drift_checks


def test_rejection_streak_is_checked_without_per_step_sync(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_momentum=0.0,
        muon_nesterov=False,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
        muon_warm_alignment_tolerance=0.0,
        muon_warm_max_rejection_streak=2,
        muon_warm_check_every=1,
    )
    q_previous = torch.cat((torch.eye(4), torch.zeros(4, 4)), dim=1).to(
        torch.bfloat16
    )
    state = {
        "step": 4,
        "momentum_fast": torch.zeros_like(parameter),
        "muon_warm_q": q_previous,
        "muon_warm_rejection_streak_tensor": torch.tensor(2.0),
    }
    monkeypatch.setattr(
        muon_warm,
        "_warm_alignment_metrics",
        _eager(muon_warm._warm_alignment_metrics),
    )
    monkeypatch.setattr(muon_warm, "_muon_ns5_prepared", lambda value, _: value)
    monkeypatch.setattr(muon_warm, "_row_ns_retract", lambda value, *_: value)

    optimizer._muon_update_warm(
        torch.randn_like(parameter), state, optimizer.param_groups[0]
    )

    assert state["muon_warm_did_anchor"]
    assert state["muon_warm_anchor_reason"] == "rejection_streak"
    assert state["muon_warm_rejection_streak_tensor"] == 0.0


def test_rejection_streak_requires_alignment_gate():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    try:
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_max_rejection_streak=2,
        )
    except ValueError as error:
        assert "alignment_tolerance" in str(error)
    else:
        raise AssertionError("rejection streak without a gate should be rejected")


def test_normal_inverse_cap_must_be_non_negative():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    with pytest.raises(ValueError, match="normal_inv_cap"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_normal_inv_cap=-1.0,
        )


def test_normal_inverse_ema_requires_absolute_safety_cap():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    with pytest.raises(ValueError, match="safety ceiling"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_normal_inv_ema_ratio=2.0,
        )
    with pytest.raises(ValueError, match="ema_beta"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_normal_inv_cap=4.0,
            muon_warm_normal_inv_ema_ratio=2.0,
            muon_warm_normal_inv_ema_beta=1.0,
        )


def test_dynamic_normal_inverse_cap_uses_fixed_cap_observation():
    q_previous = torch.cat((torch.eye(2), torch.zeros(2, 2)), dim=1)
    work_matrix = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 1.0]]
    )
    core = _eager(muon_warm._warm_polar_jacobi_core)

    fixed_q, fixed_stats, *_ = core(
        work_matrix,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        True,
        jacobi_damping="tikhonov",
        measure_tangent=True,
        normal_inv_cap=4.0,
    )
    adaptive_q, adaptive_stats, *_ = core(
        work_matrix,
        q_previous,
        1.0,
        1e-3,
        "higham_cubic",
        0,
        True,
        jacobi_damping="tikhonov",
        measure_tangent=True,
        normal_inv_cap=4.0,
        dynamic_normal_inv_cap=torch.tensor(2.0),
    )

    assert torch.equal(fixed_stats[8], adaptive_stats[8])
    assert adaptive_stats[4] < fixed_stats[4]
    assert (adaptive_q - q_previous).norm() < (fixed_q - q_previous).norm()


def test_normal_inverse_ema_state_stays_on_device(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(2, 4))
    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_warm_normal_inv_cap=4.0,
        muon_warm_normal_inv_ema_ratio=2.0,
        muon_warm_anchor_every=0,
        muon_warm_full_ns_steps=0,
    )
    monkeypatch.setattr(
        muon_warm,
        "_warm_polar_jacobi_step_with_normal_inv_observation",
        _eager(muon_warm._warm_polar_jacobi_step_with_normal_inv_observation),
    )
    q_previous = torch.cat((torch.eye(2), torch.zeros(2, 2)), dim=1)
    work_matrix = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 1.0]]
    )
    state = {"muon_warm_age": 0}

    q_next = optimizer._compute_warm_direction(
        work_matrix, q_previous, None, state, False
    )
    state["muon_warm_age"] = 1
    optimizer._compute_warm_direction(
        work_matrix, q_next, None, state, False
    )

    assert state["muon_warm_normal_inv_ema_sq_tensor"].ndim == 0
    assert state["muon_warm_normal_inv_ema_sq_tensor"].device == parameter.device
    assert state["muon_warm_normal_inv_ema_samples"] == 2
    assert state["muon_warm_normal_inv_dynamic_cap_samples"] == 1
    assert state["muon_warm_normal_inv_dynamic_cap_tensor"] > 0.0
    assert state["muon_warm_normal_inv_effective_cap_sum_tensor"] > 0.0
    assert state["muon_warm_normal_inv_ema_active_sum_tensor"] in (0.0, 1.0)


def test_angular_anchor_threshold_validates_configuration():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    with pytest.raises(ValueError, match="max_angular_rms"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_max_angular_rms=-1.0,
        )
    with pytest.raises(ValueError, match="gershgorin"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_max_angular_rms=10.0,
            muon_warm_spectral_cap_mode="power",
        )


def test_angular_anchor_threshold_supports_power_step_mode():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))

    optimizer = muon_warm.MuonWarm(
        [{"params": [parameter], "use_muon": True}],
        muon_warm_max_angular_rms=10.0,
        muon_warm_spectral_cap_mode="power_step",
    )

    assert optimizer.muon_warm_max_angular_rms == 10.0


def test_skew_ratio_anchor_threshold_validates_configuration():
    parameter = torch.nn.Parameter(torch.zeros(4, 8))
    with pytest.raises(ValueError, match="max_skew_ratio"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_max_skew_ratio=-0.1,
        )
    with pytest.raises(ValueError, match="power_step"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_max_skew_ratio=0.1,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        muon_warm.MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_max_angular_rms=1.0,
            muon_warm_max_skew_ratio=0.1,
            muon_warm_spectral_cap_mode="power_step",
        )


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
