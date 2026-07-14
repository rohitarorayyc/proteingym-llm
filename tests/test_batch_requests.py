from config.models import MODELS, PRIMARY_MODELS, benchmark_spec
from src import batch as batch_module
from src.batch import _build_requests, _normalize_openai_body, _provider_request


def test_primary_registry_is_exactly_ten_models_without_sonnet():
    assert len(PRIMARY_MODELS) == 10
    assert set(PRIMARY_MODELS) == set(MODELS)
    assert not any("sonnet" in model for model in PRIMARY_MODELS)


def test_effort_protocol_is_max_at_n50_and_uniform_high_for_set_size_controls():
    assert benchmark_spec("gpt-5.6-sol", 50)["reasoning"] == "max"
    assert benchmark_spec("gpt-5.5", 50)["reasoning"] == "xhigh"
    assert benchmark_spec("claude-opus-4.8", 50)["reasoning"] == "max"
    for model in PRIMARY_MODELS:
        assert benchmark_spec(model, 10)["reasoning"] == "high"
        assert benchmark_spec(model, 100)["reasoning"] == "high"


def test_openai_batch_request_carries_reasoning_effort():
    spec = MODELS["gpt-5.5"]
    request = _provider_request("openai", spec, "assay", "prompt")
    assert request["body"]["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert request["body"]["max_output_tokens"] == 128000


def test_anthropic_batch_request_carries_max_effort():
    spec = MODELS["claude-opus-4.8"]
    request = _provider_request("anthropic", spec, "assay", "prompt")
    assert request["params"]["output_config"] == {"effort": "max"}
    assert request["params"]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert request["params"]["max_tokens"] == 128000


def test_google_batch_request_includes_thoughts():
    spec = MODELS["gemini-3.5-flash"]
    request = _provider_request("google", spec, "assay", "prompt")
    thinking = request["request"]["generationConfig"]["thinkingConfig"]
    assert thinking == {"thinkingLevel": "high", "includeThoughts": True}
    key = request["request"]["labels"]["assay"]
    assert len(key) == 32
    assert set(key) <= set("0123456789abcdef")


def test_batch_normalizer_preserves_provider_returned_model_revision():
    normalized = _normalize_openai_body(
        {
            "id": "resp_1",
            "model": "gpt-backend-2026-07-14",
            "created_at": 1784020800,
            "status": "completed",
            "output": [],
            "usage": {},
        }
    )
    assert normalized["response_model_id"] == "gpt-backend-2026-07-14"
    assert normalized["provider_created_at"] == 1784020800


def test_batch_preflight_overflow_stays_outside_final_tree(tmp_path, monkeypatch):
    spec = {
        "provider": "openai",
        "model_id": "test-model",
        "reasoning": "high",
        "max_tokens": 2,
        "ctx": 10,
        "supports_batch": True,
    }
    meta = {
        "assay": {
            "reference_sequence": "AAAA",
            "target_name": "TEST",
            "organism": "test",
            "fitness_description": "activity",
            "fitness_description_source": "Activity; assay",
            "fitness_description_repair": None,
        }
    }
    data_bundle = {
        "bundle_version": "test-v1",
        "manifest_sha256": "1" * 64,
        "selection": {"sizes": [50], "seeds": [1]},
    }
    monkeypatch.setattr(batch_module, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(batch_module, "FAILURES", tmp_path / "failures")
    monkeypatch.setattr(
        batch_module,
        "shared_subset",
        lambda *_args: [("v1", "AAAT", 1.0), ("v2", "AATA", 0.0)],
    )

    requests, included, _episodes = _build_requests(
        "test",
        spec,
        50,
        1,
        ["assay"],
        meta,
        data_bundle=data_bundle,
        write_overflows=True,
    )

    assert requests == []
    assert included == []
    assert not (tmp_path / "results" / "test" / "n50" / "b1" / "assay.json").exists()
    assert (tmp_path / "failures" / "test__n50__b1" / "assay.json").exists()
