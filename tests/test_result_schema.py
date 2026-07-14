import json

import pytest

from src import client
from src.run import (
    RESULT_SCHEMA_VERSION,
    base_result_record,
    record_response,
    run_assay,
    should_run,
    write_result,
)

SPEC = {
    "provider": "openai-compatible",
    "api_style": "responses",
    "model_id": "test-model",
    "api_key_env": "LAB_API_KEY",
    "base_url_env": "LAB_BASE_URL",
    "reasoning": "max",
    "send_reasoning": True,
    "max_tokens": 128000,
    "ctx": 1000000,
}


@pytest.fixture(autouse=True)
def _endpoint_environment(monkeypatch):
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )


META = {
    "reference_sequence": "WT",
    "fitness_description": "repaired description",
    "fitness_description_source": "generic description",
    "fitness_description_repair": "hard-prompt-repairs-v1",
}
DATA_BUNDLE = {
    "bundle_version": "test-v1",
    "manifest_sha256": "1" * 64,
    "selection": {"sizes": [50], "seeds": [1, 2, 3]},
}


def test_prompt_token_planning_uses_frozen_tokenizer_or_safe_utf8_bound():
    text = "ACDEFGHIKLMNPQRSTVWY" * 100 + " β"
    assert client.estimate_tokens(text, SPEC) == len(text.encode("utf-8"))
    tokenized = dict(SPEC, tokenizer_encoding="cl100k_base")
    assert client.token_estimator(tokenized) == "tiktoken:cl100k_base"
    assert 0 < client.estimate_tokens(text, tokenized) < len(text.encode("utf-8"))


def test_result_schema_preserves_full_provider_metadata(tmp_path):
    raw = "reasoning and answer " * 1000
    ids = ["M01", "M02"]
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    record = base_result_record(
        "test",
        SPEC,
        50,
        2,
        "assay",
        META,
        "prompt",
        2,
        data_bundle=DATA_BUNDLE,
        subset=subset,
    )
    response = {
        "text": raw,
        "reasoning_text": "summary",
        "response_content": [{"type": "reasoning"}, {"type": "message"}],
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "output_tokens": 20,
        "reasoning_tokens": 12,
        "response_id": "resp_1",
        "response_model_id": "test-model-2026-07-14",
        "provider_response_version": "2026-07-14",
        "provider_created_at": 1784020800,
        "status": "completed",
        "incomplete_reason": None,
        "stop_reason": None,
        "service_tier": "default",
        "elapsed_s": 1.2,
        "error": None,
    }
    response["text"] += '\n{"ranking":["M01","M02"]}'
    record_response(record, response, ids, subset)

    output = tmp_path / "result.json"
    write_result(output, record)
    saved = json.loads(output.read_text())
    assert saved["schema_version"] == RESULT_SCHEMA_VERSION
    assert saved["raw_output"] == response["text"]
    assert len(saved["raw_output"]) > 4000
    assert saved["reasoning_summary"] == "summary"
    assert saved["usage"]["output_tokens"] == 20
    assert saved["assay_prompt_repair"] == "hard-prompt-repairs-v1"
    assert saved["prompt_sha256"]
    assert saved["split_sha256"]
    assert saved["run_label"] == "canonical"
    assert saved["eval_bundle_manifest_sha256"] == "1" * 64
    assert saved["request_fingerprint"]
    assert saved["response_model_id"] == "test-model-2026-07-14"
    assert saved["provider_response_version"] == "2026-07-14"
    assert saved["provider_created_at"] == 1784020800
    assert saved["attempt_started_at_utc"].endswith("Z")
    assert saved["attempt_completed_at_utc"].endswith("Z")
    assert len(saved["runtime"]["code_sha256"]) == 64
    assert saved["runtime"]["python_version"]


def test_resume_policy_retries_only_requested_failures(tmp_path):
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"error": "rate limit", "stop_reason": None}))
    assert not should_run(output)
    assert should_run(output, retry_errors=True)
    output.write_text(json.dumps({"error": None, "stop_reason": "max_tokens"}))
    assert should_run(output, retry_truncated=True)


def test_resume_rejects_mixed_provenance(tmp_path):
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"schema_version": 1}))
    expected = {"schema_version": RESULT_SCHEMA_VERSION}
    with pytest.raises(RuntimeError, match="incompatible provenance"):
        should_run(output, expected=expected)


def test_resume_rejects_tampered_cell_identity(tmp_path):
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    expected = base_result_record(
        "test",
        SPEC,
        50,
        1,
        "assay",
        META,
        "prompt",
        2,
        data_bundle=DATA_BUNDLE,
        subset=subset,
    )
    tampered = dict(expected, model="other-model", batch=2)
    output = tmp_path / "result.json"
    output.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(RuntimeError, match="model, batch"):
        should_run(output, expected=expected)


def test_truncated_response_is_preserved_but_not_scored():
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    record = base_result_record(
        "test",
        SPEC,
        50,
        1,
        "assay",
        META,
        "prompt",
        2,
        data_bundle=DATA_BUNDLE,
        subset=subset,
    )
    response = {
        "text": '{"ranking":["M01","M02"]}',
        "reasoning_text": "partial reasoning",
        "status": "incomplete",
        "incomplete_reason": "max_output_tokens",
        "error": None,
    }
    record_response(record, response, ["M01", "M02"], subset)
    assert record["raw_output"] == response["text"]
    assert record["truncated"] is True
    assert record["parsed"] is False
    assert record["spearman"] is None


def test_complete_response_without_provider_model_identity_is_not_scorable():
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    record = base_result_record(
        "test",
        SPEC,
        50,
        1,
        "assay",
        META,
        "prompt",
        2,
        data_bundle=DATA_BUNDLE,
        subset=subset,
    )
    record_response(
        record,
        {"text": '{"ranking":["M01","M02"]}', "status": "completed"},
        ["M01", "M02"],
        subset,
    )
    assert record["error"] == "provider response missing model identity"
    assert record["parsed"] is False
    assert record["spearman"] is None


@pytest.mark.parametrize(
    ("response_fields", "message"),
    [
        ({"status": "failed"}, "status is not successful: failed"),
        ({"status": "cancelled"}, "status is not successful: cancelled"),
        ({"status": "completed", "stop_reason": "SAFETY"}, "stop reason"),
        ({"status": "completed", "stop_reason": "content_filter"}, "stop reason"),
    ],
)
def test_terminal_provider_failures_are_preserved_but_not_scored(response_fields, message):
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    record = base_result_record(
        "test",
        SPEC,
        50,
        1,
        "assay",
        META,
        "prompt",
        2,
        data_bundle=DATA_BUNDLE,
        subset=subset,
    )
    response = {
        "text": '{"ranking":["M01","M02"]}',
        "response_model_id": "test-model-2026-07-14",
        **response_fields,
    }
    record_response(record, response, ["M01", "M02"], subset)
    assert message in record["error"]
    assert record["raw_output"] == response["text"]
    assert record["truncated"] is False
    assert record["parsed"] is False
    assert record["spearman"] is None


@pytest.mark.parametrize(
    "response_fields",
    [
        {"status": "completed", "stop_reason": "STOP"},
        {"stop_reason": "end_turn"},
        {"stop_reason": "stop_sequence"},
    ],
)
def test_normal_provider_stop_reasons_remain_scorable(response_fields):
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    record = base_result_record(
        "test",
        SPEC,
        50,
        1,
        "assay",
        META,
        "prompt",
        2,
        data_bundle=DATA_BUNDLE,
        subset=subset,
    )
    record_response(
        record,
        {
            "text": '{"ranking":["M01","M02"]}',
            "response_model_id": "test-model-2026-07-14",
            **response_fields,
        },
        ["M01", "M02"],
        subset,
    )
    assert record["error"] is None
    assert record["parsed"] is True
    assert record["spearman"] == pytest.approx(1.0)


def test_live_truncation_stays_outside_final_results(tmp_path, monkeypatch):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(
        run.client,
        "chat",
        lambda *_args, **_kwargs: {
            "text": '{"ranking":["M01","M02"]}',
            "status": "incomplete",
            "incomplete_reason": "max_output_tokens",
            "error": None,
        },
    )
    final_dir = tmp_path / "results"
    attempt_dir = tmp_path / "attempts"
    run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        final_dir,
        attempt_dir,
        data_bundle=DATA_BUNDLE,
    )
    assert not (final_dir / "assay.json").exists()
    attempt = json.loads((attempt_dir / "assay.json").read_text())
    assert attempt["truncated"] is True
    assert attempt["raw_output"]
    assert attempt["attempt_started_at_utc"].endswith("Z")
    assert attempt["attempt_completed_at_utc"].endswith("Z")
