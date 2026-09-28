import torch
import pytest

from train_llm import (
    SyntheticSequenceDataset,
    _compact_token_ids,
    _tokenize_iterable_split,
    build_datasets,
    build_parser,
    optimizer_diagnostics,
)


def test_synthetic_dataset_returns_shifted_learnable_sequences():
    dataset = SyntheticSequenceDataset(4, 16, 64, seed=9)
    inputs, targets = dataset[0]

    assert inputs.shape == targets.shape == (16,)
    assert torch.equal(inputs[1:], targets[:-1])
    stride = (targets[0] - inputs[0]).remainder(64)
    assert torch.equal((targets - inputs).remainder(64), stride.expand_as(inputs))


def test_local_optimizer_is_the_default():
    args = build_parser().parse_args([])
    assert args.optimizer in {"transport_muon", "soap"}
    assert not args.input_projection
    assert args.tie_embeddings
    assert args.soap_fallback_lr == 3e-4
    assert args.muon_retract_steps == 1
    assert args.muon_spectral_cap_mode == "gershgorin"
    assert args.muon_power_steps == 2
    assert args.muon_power_safety_factor == 1.25
    assert not args.muon_async_checks
    assert not args.hf_streaming


class _TinyTokenizer:
    eos_token_id = 99
    pad_token_id = None

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [[index + 1, index + 2] for index, _ in enumerate(texts)]}


def test_stream_tokenization_obeys_a_hard_token_budget():
    documents = ({"text": f"document {index}"} for index in range(20))

    tokens = _tokenize_iterable_split(
        documents,
        _TinyTokenizer(),
        "text",
        maximum_documents=0,
        maximum_tokens=17,
    )

    assert tokens.shape == (17,)
    assert tokens[2].item() == 99


def test_token_cache_uses_int32_without_changing_ids():
    tokens = torch.tensor([0, 1, 50_256], dtype=torch.int64)

    compact = _compact_token_ids(tokens)

    assert compact.dtype == torch.int32
    assert torch.equal(compact.long(), tokens)


def test_openwebtext_requires_streaming_or_an_explicit_large_download_override():
    args = build_parser().parse_args(
        ["--dataset", "hf", "--dataset-name", "Skylion007/openwebtext"]
    )

    with pytest.raises(ValueError, match="use --hf-streaming"):
        build_datasets(args)


def test_optimizer_diagnostics_summarizes_muon_reference_statistics():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    optimizer.state[parameter].update(
        {
            "muon_update_direction_sq_sum_tensor": torch.tensor(4.0),
            "muon_update_momentum_sq_sum_tensor": torch.tensor(1.0),
            "muon_update_direction_momentum_dot_sum_tensor": torch.tensor(2.0),
            "muon_update_applied_sq_sum_tensor": torch.tensor(0.04),
            "muon_update_elements": 4,
            "muon_update_samples": 1,
            "muon_reference_direction_sq_sum_tensor": torch.tensor(4.0),
            "muon_reference_direction_dot_sum_tensor": torch.tensor(4.0),
            "muon_reference_difference_sq_sum_tensor": torch.tensor(0.0),
            "muon_reference_candidate_applied_sq_sum_tensor": torch.tensor(0.04),
            "muon_reference_applied_sq_sum_tensor": torch.tensor(0.0025),
            "muon_reference_elements": 4,
            "muon_reference_samples": 1,
            "muon_warm_q": torch.eye(2),
            "muon_warm_output_retractions": 7,
            "muon_warm_power_cap_samples": 2,
            "muon_warm_power_sigma_sum_tensor": torch.tensor(2.2),
            "muon_warm_spectral_cap_scale_sum_tensor": torch.tensor(1.8),
            "muon_warm_spectral_cap_min_scale_tensor": torch.tensor(0.8),
            "muon_warm_power_step_samples": 2,
            "muon_warm_power_probe_sigma_sum_tensor": torch.tensor(4.0),
            "muon_warm_spectral_step_scale_sum_tensor": torch.tensor(0.6),
            "muon_warm_spectral_step_min_scale_tensor": torch.tensor(0.2),
            "muon_update_stats_by_age": {
                3: {
                    "direction_sq_sum_tensor": torch.tensor(4.0),
                    "momentum_sq_sum_tensor": torch.tensor(1.0),
                    "direction_momentum_dot_sum_tensor": torch.tensor(2.0),
                    "applied_sq_sum_tensor": torch.tensor(0.04),
                    "orthogonality_error_sum_tensor": torch.tensor(0.2),
                    "max_orthogonality_error_tensor": torch.tensor(0.2),
                    "reference_direction_sq_sum_tensor": torch.tensor(4.0),
                    "reference_direction_dot_sum_tensor": torch.tensor(4.0),
                    "reference_difference_sq_sum_tensor": torch.tensor(0.0),
                    "candidate_applied_sq_sum_tensor": torch.tensor(0.04),
                    "reference_applied_sq_sum_tensor": torch.tensor(0.0025),
                    "refresh_predictor_momentum_innovation_ratio_sample_sum_tensor": torch.tensor(2.0),
                    "refresh_predictor_momentum_innovation_ratio_value_sum_tensor": torch.tensor(3.0),
                    "refresh_predictor_momentum_innovation_ratio_value_sq_sum_tensor": torch.tensor(5.0),
                    "refresh_predictor_momentum_innovation_ratio_error_sum_tensor": torch.tensor(6.0),
                    "refresh_predictor_momentum_innovation_ratio_error_sq_sum_tensor": torch.tensor(20.0),
                    "refresh_predictor_momentum_innovation_ratio_value_error_sum_tensor": torch.tensor(10.0),
                    "elements": 4,
                    "samples": 1,
                    "reference_elements": 4,
                    "reference_samples": 1,
                }
            },
        }
    )

    diagnostics = optimizer_diagnostics(optimizer)

    assert diagnostics["update_stats"]["direction_rms"] == 1.0
    assert diagnostics["update_stats"]["direction_momentum_cosine"] == 1.0
    assert diagnostics["fresh_reference_stats"]["direction_cosine"] == 1.0
    assert diagnostics["fresh_reference_stats"]["direction_rms_ratio"] == 1.0
    assert diagnostics["fresh_reference_stats"]["relative_direction_error"] == 0.0
    assert diagnostics["fresh_reference_stats"][
        "candidate_applied_direction_rms"
    ] == pytest.approx(0.1)
    assert diagnostics["fresh_reference_stats"][
        "reference_applied_direction_rms"
    ] == pytest.approx(0.025)
    age_three = diagnostics["update_stats_by_age"]["3"]
    assert age_three["kind"] == "warm"
    assert age_three["direction_rms"] == 1.0
    assert age_three["mean_orthogonality_error"] == pytest.approx(0.2)
    assert age_three["fresh_reference"]["direction_cosine"] == 1.0
    predictor = diagnostics["refresh_predictor_stats"][
        "momentum_innovation_ratio"
    ]
    assert predictor["samples"] == 2
    assert predictor["mean_value"] == pytest.approx(1.5)
    assert predictor["mean_relative_reference_error"] == pytest.approx(3.0)
    assert predictor["pearson_error_correlation"] == pytest.approx(1.0)
    assert age_three["refresh_predictors"]["momentum_innovation_ratio"] == predictor
    assert diagnostics["refresh_predictor_stats_by_shape"]["2x2"][
        "momentum_innovation_ratio"
    ] == predictor
    assert diagnostics["power_cap_stats"]["samples"] == 2
    assert diagnostics["power_cap_stats"][
        "mean_estimated_top_singular"
    ] == pytest.approx(1.1)
    assert diagnostics["power_cap_stats"]["mean_input_scale"] == pytest.approx(0.9)
    assert diagnostics["power_cap_stats"]["minimum_input_scale"] == pytest.approx(0.8)
    assert diagnostics["power_cap_stats"]["mean_probe_top_singular"] == 2.0
    assert diagnostics["power_cap_stats"]["mean_transport_step_scale"] == pytest.approx(0.3)
    assert diagnostics["power_cap_stats"]["minimum_transport_step_scale"] == pytest.approx(0.2)
    assert diagnostics["output_retractions"] == 7


def test_optimizer_diagnostics_reports_step_only_power_statistics():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    optimizer.state[parameter].update(
        {
            "muon_warm_power_step_samples": 2,
            "muon_warm_power_probe_sigma_sum_tensor": torch.tensor(4.0),
            "muon_warm_spectral_step_scale_sum_tensor": torch.tensor(0.6),
            "muon_warm_spectral_step_min_scale_tensor": torch.tensor(0.2),
        }
    )

    diagnostics = optimizer_diagnostics(optimizer)

    assert diagnostics["power_cap_stats"] == {
        "step_samples": 2,
        "mean_probe_top_singular": 2.0,
        "mean_transport_step_scale": pytest.approx(0.3),
        "minimum_transport_step_scale": pytest.approx(0.2),
    }
