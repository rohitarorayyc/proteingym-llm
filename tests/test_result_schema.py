import hashlib
import json

import pytest

from src import client
from src.run import (
    RESULT_SCHEMA_VERSION,
    base_result_record,
    is_truncated,
    output_limit_evidence,
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
    "fitness_description": "canonical description",
}
DATA_BUNDLE = {
    "bundle_version": "test-v1",
    "manifest_sha256": "1" * 64,
    "selection": {"sizes": [50], "seeds": [1, 2, 3]},
}


def test_output_limit_evidence_is_provider_neutral():
    assert output_limit_evidence(
        {
            "status": "incomplete",
            "incomplete_reason": "max_output_tokens",
            "stop_reason": None,
            "stream_completed": None,
            "output_tokens": 127,
        },
        128,
    )
    assert output_limit_evidence(
        {"stop_reason": "length", "output_tokens": None, "stream_completed": False}, 128
    )
    assert not output_limit_evidence(
        {"status": "incomplete", "output_tokens": 12, "stream_completed": None}, 128
    )


def test_content_filter_incomplete_response_is_not_output_truncation():
    record = {
        "status": "incomplete",
        "incomplete_reason": "content_filter",
        "output_tokens": 12,
    }
    assert not is_truncated(record)


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
        "provider_response": {"id": "resp_1", "object": "response"},
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
        "traffic_type": "ON_DEMAND",
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
    assert saved["assay_description"] == "canonical description"
    assert "assay_description_source" not in saved
    assert saved["prompt_sha256"]
    assert saved["split_sha256"]
    assert saved["run_label"] == "canonical"
    assert saved["eval_bundle_manifest_sha256"] == "1" * 64
    assert saved["request_fingerprint"]
    assert saved["response_model_id"] == "test-model-2026-07-14"
    assert saved["provider_response_version"] == "2026-07-14"
    assert saved["provider_created_at"] == 1784020800
    assert saved["provider_response"] == {"id": "resp_1", "object": "response"}
    assert saved["traffic_type"] == "ON_DEMAND"
    assert saved["attempt_started_at_utc"].endswith("Z")
    assert saved["attempt_completed_at_utc"].endswith("Z")
    assert len(saved["runtime"]["code_sha256"]) == 64
    assert saved["runtime"]["python_version"]


@pytest.mark.parametrize(
    ("actual_tier", "reported_tier"),
    [("default", "default"), (None, "missing")],
)
def test_requested_service_tier_must_match_provider_response(actual_tier, reported_tier):
    ids = ["M01", "M02"]
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    spec = {**SPEC, "service_tier": "flex"}
    record = base_result_record(
        "test",
        spec,
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
        "response_model_id": "test-model",
        "provider_response": {"id": "response"},
        "service_tier": actual_tier,
        "status": "completed",
    }
    record_response(record, response, ids, subset)

    assert record["error"] == (
        f"unexpected provider service tier: requested flex, received {reported_tier}"
    )
    assert record["parsed"] is False
    assert record["spearman"] is None


def test_matching_requested_service_tier_can_finalize():
    ids = ["M01", "M02"]
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    spec = {**SPEC, "service_tier": "flex", "require_usage": True}
    record = base_result_record(
        "test",
        spec,
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
        "response_model_id": "test-model",
        "provider_response": {"id": "response", "service_tier": "flex"},
        "service_tier": "flex",
        "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "output_tokens": 2,
    }
    record_response(record, response, ids, subset)

    assert record["error"] is None
    assert record["parsed"] is True
    assert record["spearman"] == pytest.approx(1.0)


def test_resume_policy_retries_only_requested_failures(tmp_path):
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"error": "rate limit", "stop_reason": None}))
    assert not should_run(output)
    assert should_run(output, retry_errors=True)
    output.write_text(
        json.dumps(
            {
                "error": "APIError: Invalid prompt: we've limited access to this content "
                "for safety reasons.",
                "retryable": True,
            }
        )
    )
    assert not should_run(output, retry_errors=True)
    output.write_text(
        json.dumps(
            {
                "error": "APIError: You exceeded your current quota, please check your plan "
                "and billing details.",
                "retryable": True,
            }
        )
    )
    assert not should_run(output, retry_errors=True)
    output.write_text(
        json.dumps(
            {
                "error": "provider refusal: cannot comply",
                "failure_class": "provider_policy_block",
                "retryable": True,
            }
        )
    )
    assert not should_run(output, retry_errors=True)
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


@pytest.mark.parametrize(
    ("requirement", "response_fields", "message"),
    [
        (
            "require_usage",
            {"reasoning_text": "trace"},
            "missing required token usage metadata",
        ),
        (
            "require_reasoning",
            {"usage": {"completion_tokens": 2}, "output_tokens": 2},
            "missing required reasoning trace",
        ),
    ],
)
def test_required_audit_metadata_is_fail_closed(requirement, response_fields, message):
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    spec = dict(SPEC, **{requirement: True})
    record = base_result_record(
        "test",
        spec,
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
            "response_model_id": "test-model",
            "status": "completed",
            **response_fields,
        },
        ["M01", "M02"],
        subset,
    )
    assert record["error"] == f"provider response {message}"
    assert record["parsed"] is False
    assert record["spearman"] is None


@pytest.mark.parametrize(
    ("usage", "output_tokens"),
    [
        ({"completion_tokens": 3}, 2),
        ({"completion_tokens": True}, True),
        ({"output_tokens": None}, 2),
    ],
)
def test_required_usage_must_have_matching_integer_output_count(usage, output_tokens):
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    record = base_result_record(
        "test",
        dict(SPEC, require_usage=True),
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
            "response_model_id": "test-model",
            "status": "completed",
            "usage": usage,
            "output_tokens": output_tokens,
            "provider_response": {"id": "response"},
        },
        ["M01", "M02"],
        subset,
    )
    assert record["error"] == "provider response missing required token usage metadata"
    assert record["spearman"] is None


def test_complete_response_without_full_provider_payload_is_not_scorable():
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
            "response_model_id": "test-model",
            "status": "completed",
            "provider_response": None,
        },
        ["M01", "M02"],
        subset,
    )
    assert record["error"] == "provider response missing full provider payload"
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
            "provider_response": {"id": "response"},
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
    attempt_paths = list(attempt_dir.glob("assay.attempt-*.json"))
    assert len(attempt_paths) == 2
    attempts = [json.loads(path.read_text()) for path in attempt_paths]
    assert sorted(attempt["transport_attempt"] for attempt in attempts) == [1, 2]
    assert all(attempt["truncated"] is True for attempt in attempts)
    assert all(attempt["raw_output"] for attempt in attempts)
    assert all(attempt["attempt_started_at_utc"].endswith("Z") for attempt in attempts)
    assert all(attempt["attempt_completed_at_utc"].endswith("Z") for attempt in attempts)


def test_retries_preserve_every_failure_after_canonical_success(tmp_path, monkeypatch):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    responses = iter(
        [
            {"text": "partial first", "error": "rate limit one"},
            {"text": "partial second", "error": "rate limit two"},
            {
                "text": '{"ranking":["M01","M02"]}',
                "response_model_id": "test-model",
                "provider_response": {"id": "response"},
                "status": "completed",
                "error": None,
            },
        ]
    )
    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", lambda *_args, **_kwargs: next(responses))
    final_dir = tmp_path / "results"
    attempt_dir = tmp_path / "attempts"
    common = (
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        final_dir,
        attempt_dir,
    )

    run_assay(*common, data_bundle=DATA_BUNDLE)
    first_paths = list(attempt_dir.glob("assay.attempt-*.json"))
    assert len(first_paths) == 1
    first_payload = first_paths[0].read_bytes()

    unresolved = run_assay(*common, data_bundle=DATA_BUNDLE)
    assert unresolved["execution_state"] == "unresolved_existing_attempt"
    assert list(attempt_dir.glob("assay.attempt-*.json")) == first_paths

    run_assay(*common, data_bundle=DATA_BUNDLE, retry_errors=True)
    second_paths = list(attempt_dir.glob("assay.attempt-*.json"))
    assert len(second_paths) == 2
    assert first_paths[0].read_bytes() == first_payload
    preserved = {json.loads(path.read_text())["raw_output"] for path in second_paths}
    assert preserved == {"partial first", "partial second"}

    payloads_before_success = {path.name: path.read_bytes() for path in second_paths}
    run_assay(*common, data_bundle=DATA_BUNDLE, retry_errors=True)

    canonical = json.loads((final_dir / "assay.json").read_text())
    assert canonical["raw_output"] == '{"ranking":["M01","M02"]}'
    assert canonical["spearman"] == pytest.approx(1.0)
    attempts_after_success = list(attempt_dir.glob("assay.attempt-*.json"))
    assert len(attempts_after_success) == 3
    assert all(
        path.read_bytes() == payload
        for name, payload in payloads_before_success.items()
        for path in attempts_after_success
        if path.name == name
    )
    assert (
        sum(
            json.loads(path.read_text())["raw_output"] == '{"ranking":["M01","M02"]}'
            for path in attempts_after_success
        )
        == 1
    )
    assert run_assay(*common, data_bundle=DATA_BUNDLE) is None
    final_attempts = list(attempt_dir.glob("assay.attempt-*.json"))
    assert {path.name: path.read_bytes() for path in final_attempts} == {
        path.name: path.read_bytes() for path in attempts_after_success
    }


def test_runner_retries_each_transport_call_with_fresh_reservation_and_attempt(
    tmp_path, monkeypatch
):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    responses = iter(
        [
            {
                "text": "",
                "error": "429 overloaded",
                "retryable": True,
                "retry_after_s": 2.5,
            },
            {
                "text": "partial",
                "error": "stream interrupted",
                "retryable": True,
            },
            {
                "text": '{"ranking":["M01","M02"]}',
                "response_model_id": "test-model",
                "provider_response": {"id": "response"},
                "status": "completed",
                "error": None,
            },
        ]
    )
    calls = []
    sleeps = []

    class Limiter:
        def reserve(self, value):
            calls.append(value)

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(run.time, "sleep", sleeps.append)
    final_dir = tmp_path / "results"
    attempt_dir = tmp_path / "attempts"
    result = run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        final_dir,
        attempt_dir,
        data_bundle=DATA_BUNDLE,
        retries=3,
        token_limiter=Limiter(),
    )

    assert result["spearman"] == pytest.approx(1.0)
    assert len(calls) == 3
    assert len(set(calls)) == 1
    assert sleeps == [2.5]
    attempts = sorted(attempt_dir.glob("assay.attempt-*.json"))
    assert len(attempts) == 3
    assert [json.loads(path.read_text())["transport_attempt"] for path in attempts] == [1, 2, 3]
    assert (final_dir / "assay.json").is_file()


def test_runner_writes_attempt_before_call_and_durably_journals_stream(tmp_path, monkeypatch):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    attempt_dir = tmp_path / "attempts"
    observed = {}

    def fake_chat(*_args, **kwargs):
        attempt_paths = list(attempt_dir.glob("assay.attempt-*.json"))
        assert len(attempt_paths) == 1
        write_ahead = json.loads(attempt_paths[0].read_text())
        observed["path"] = attempt_paths[0]
        observed["write_ahead"] = write_ahead
        observed["client_request_id"] = kwargs["client_request_id"]
        sink = kwargs["event_sink"]
        sink(
            {
                "kind": "response.headers",
                "headers": {"x-request-id": "req_provider"},
            }
        )
        sink(
            {
                "kind": "response.event",
                "event": {"type": "response.created", "response": {"id": "resp_1"}},
            }
        )
        sink(
            {
                "kind": "response.event",
                "event": {"type": "response.completed", "response": {"id": "resp_1"}},
            }
        )
        return {
            "text": '{"ranking":["M01","M02"]}',
            "response_model_id": "test-model",
            "provider_response": {"id": "resp_1"},
            "provider_request_id": "req_provider",
            "stream_telemetry": {"transport": "responses-sse"},
            "stream_completed": True,
            "status": "completed",
            "error": None,
        }

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", fake_chat)
    result = run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        tmp_path / "results",
        attempt_dir,
        data_bundle=DATA_BUNDLE,
        retries=1,
    )

    write_ahead = observed["write_ahead"]
    assert write_ahead["attempt_state"] == "request_in_progress"
    assert write_ahead["attempt_completed_at_utc"] is None
    assert write_ahead["response_received_at_utc"] is None
    assert write_ahead["client_request_id"] == observed["client_request_id"]
    assert write_ahead["attempt_id"] == observed["client_request_id"]
    assert observed["client_request_id"].startswith("pgllm-")

    assert result["attempt_state"] == "finalized"
    assert result["provider_request_id"] == "req_provider"
    assert result["stream_telemetry"] == {"transport": "responses-sse"}
    assert result["event_journal_count"] == 3
    assert result["event_journal_kind_counts"] == {
        "response.headers": 1,
        "response.event": 2,
    }
    assert result["event_journal_event_type_counts"] == {
        "response.created": 1,
        "response.completed": 1,
    }
    assert result["response_received_at_utc"] == result["event_journal_first_response_at_utc"]
    assert result["response_returned_at_utc"].endswith("Z")
    assert observed["path"].name == result["attempt_artifact"]

    journal = attempt_dir / result["event_journal_artifact"]
    journal_bytes = journal.read_bytes()
    assert hashlib.sha256(journal_bytes).hexdigest() == result["event_journal_sha256"]
    lines = [json.loads(line) for line in journal_bytes.splitlines()]
    assert [line["sequence"] for line in lines] == [1, 2, 3]
    assert all(line["received_at_utc"].endswith("Z") for line in lines)


def test_runner_preserves_partial_stream_and_redacted_exception_chain(tmp_path, monkeypatch):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]

    def interrupted(*_args, **kwargs):
        kwargs["event_sink"](
            {
                "kind": "response.event",
                "event": {"type": "response.output_text.delta", "delta": "partial"},
            }
        )
        try:
            raise OSError("authorization=Bearer sk-secret-inner")
        except OSError as cause:
            raise RuntimeError("api_key=sk-secret-outer disconnected") from cause

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", interrupted)
    attempt_dir = tmp_path / "attempts"
    result = run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        tmp_path / "results",
        attempt_dir,
        data_bundle=DATA_BUNDLE,
        retries=1,
    )

    assert result["attempt_state"] == "finalized"
    assert result["stream_completed"] is False
    assert result["event_journal_count"] == 1
    assert result["response_received_at_utc"] == result["event_journal_first_response_at_utc"]
    assert [entry["type"] for entry in result["exception_chain"]] == [
        "RuntimeError",
        "OSError",
    ]
    serialized = json.dumps(result["exception_chain"])
    assert "sk-secret" not in serialized
    assert "<redacted" in serialized
    assert not (tmp_path / "results" / "assay.json").exists()


def test_runner_durably_records_policy_stream_error_without_retry(tmp_path, monkeypatch):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    calls = []
    provider_error = {
        "exception_type": "APIError",
        "code": "invalid_prompt",
        "type": "invalid_request_error",
        "body": {
            "code": "invalid_prompt",
            "message": "we've limited access to this content for safety reasons",
        },
    }

    def policy_block(*_args, **kwargs):
        calls.append(kwargs["client_request_id"])
        kwargs["event_sink"](
            {
                "kind": "response.event",
                "client_request_id": kwargs["client_request_id"],
                "event_type": "response.created",
                "event": {"type": "response.created", "response": {"id": "resp_blocked"}},
            }
        )
        kwargs["event_sink"](
            {
                "kind": "response.stream_error",
                "client_request_id": kwargs["client_request_id"],
                "error": "APIError: Invalid prompt: we've limited access to this content "
                "for safety reasons.",
                "failure_class": "provider_policy_block",
                "incomplete_reason": "provider_policy_block",
                "retryable": False,
                "provider_error": provider_error,
            }
        )
        return {
            "text": "",
            "error": "APIError: Invalid prompt: we've limited access to this content "
            "for safety reasons.",
            "retryable": False,
            "status": "failed",
            "incomplete_reason": "provider_policy_block",
            "stream_completed": False,
            "stream_terminal_event": None,
            "failure_class": "provider_policy_block",
            "provider_error": provider_error,
        }

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", policy_block)
    attempt_dir = tmp_path / "attempts"
    result = run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        tmp_path / "results",
        attempt_dir,
        data_bundle=DATA_BUNDLE,
        retries=2,
    )

    assert len(calls) == 1
    assert result["retryable"] is False
    assert result["failure_class"] == "provider_policy_block"
    assert result["provider_error"] == provider_error
    assert result["event_journal_kind_counts"] == {
        "response.event": 1,
        "response.stream_error": 1,
    }
    journal = attempt_dir / result["event_journal_artifact"]
    journal_rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert journal_rows[-1]["kind"] == "response.stream_error"
    assert journal_rows[-1]["failure_class"] == "provider_policy_block"
    assert journal_rows[-1]["retryable"] is False
    assert not (tmp_path / "results" / "assay.json").exists()


def test_runner_does_not_claim_a_response_when_transport_fails_before_first_event(
    tmp_path, monkeypatch
):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]

    def fail_before_response(*_args, **_kwargs):
        raise TimeoutError("no response arrived")

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", fail_before_response)
    result = run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        tmp_path / "results",
        tmp_path / "attempts",
        data_bundle=DATA_BUNDLE,
        retries=1,
    )

    assert result["event_journal_count"] == 0
    assert result["response_received_at_utc"] is None
    assert result["response_returned_at_utc"] is None


def test_each_retry_has_a_fresh_request_id_attempt_and_journal(tmp_path, monkeypatch):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    calls = []

    def fake_chat(*_args, **kwargs):
        calls.append(kwargs["client_request_id"])
        kwargs["event_sink"](
            {
                "kind": "response.event",
                "event": {"type": "response.completed", "call": len(calls)},
            }
        )
        if len(calls) == 1:
            return {"text": "", "error": "transient", "retryable": True}
        return {
            "text": '{"ranking":["M01","M02"]}',
            "response_model_id": "test-model",
            "provider_response": {"id": "response"},
            "status": "completed",
            "error": None,
        }

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", fake_chat)
    attempt_dir = tmp_path / "attempts"
    result = run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        tmp_path / "results",
        attempt_dir,
        data_bundle=DATA_BUNDLE,
        retries=2,
    )

    attempts = [json.loads(path.read_text()) for path in attempt_dir.glob("*.json")]
    assert result["spearman"] == pytest.approx(1.0)
    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert {attempt["client_request_id"] for attempt in attempts} == set(calls)
    assert len({attempt["attempt_artifact"] for attempt in attempts}) == 2
    assert len({attempt["event_journal_artifact"] for attempt in attempts}) == 2
    assert len(list(attempt_dir.glob("*.events.jsonl"))) == 2


def _successful_response(spec):
    response = {
        "text": '{"ranking":["M01","M02"]}',
        "response_model_id": "test-model",
        "provider_response": {"id": "response"},
        "status": "completed",
    }
    if spec.get("require_usage"):
        response.update(
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            output_tokens=2,
        )
    if spec.get("require_reasoning"):
        response["reasoning_text"] = "auditable reasoning"
    if spec.get("service_tier"):
        response["service_tier"] = spec["service_tier"]
    return response


@pytest.mark.parametrize(
    "failure_kind",
    [
        "malformed_ranking",
        "empty_response",
        "missing_model",
        "missing_usage",
        "missing_reasoning",
        "missing_payload",
        "model_mismatch",
        "tier_mismatch",
        "truncation",
    ],
)
def test_runner_retries_response_integrity_anomalies_once(tmp_path, monkeypatch, failure_kind):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    spec = dict(SPEC)
    if failure_kind == "missing_usage":
        spec["require_usage"] = True
    elif failure_kind == "missing_reasoning":
        spec["require_reasoning"] = True
    elif failure_kind == "model_mismatch":
        spec["response_model_ids"] = ["test-model"]
    elif failure_kind == "tier_mismatch":
        spec["service_tier"] = "flex"

    failure = _successful_response(spec)
    if failure_kind == "malformed_ranking":
        failure["text"] = '{"ranking":["M01"]}'
    elif failure_kind == "empty_response":
        failure["text"] = ""
    elif failure_kind == "missing_model":
        failure.pop("response_model_id")
    elif failure_kind == "missing_usage":
        failure.pop("usage")
        failure.pop("output_tokens")
    elif failure_kind == "missing_reasoning":
        failure.pop("reasoning_text")
    elif failure_kind == "missing_payload":
        failure["provider_response"] = None
    elif failure_kind == "model_mismatch":
        failure["response_model_id"] = "unexpected-model"
    elif failure_kind == "tier_mismatch":
        failure["service_tier"] = "default"
    elif failure_kind == "truncation":
        failure.update(
            status="incomplete",
            incomplete_reason="max_output_tokens",
            output_tokens=spec["max_tokens"],
        )

    responses = iter([failure, _successful_response(spec)])
    calls = []

    def fake_chat(*_args, **_kwargs):
        calls.append(None)
        return next(responses)

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", fake_chat)
    attempt_dir = tmp_path / "attempts"
    result = run_assay(
        "test",
        spec,
        50,
        1,
        "assay",
        {"assay": META},
        tmp_path / "results",
        attempt_dir,
        data_bundle=DATA_BUNDLE,
        retries=4,
    )

    assert result["spearman"] == pytest.approx(1.0)
    assert len(calls) == 2
    attempts = [json.loads(path.read_text()) for path in attempt_dir.glob("assay.attempt-*.json")]
    assert sorted(attempt["transport_attempt"] for attempt in attempts) == [1, 2]
    assert all(attempt["attempt_artifact"] for attempt in attempts)


@pytest.mark.parametrize(("retries", "expected_calls"), [(1, 1), (5, 2)])
def test_response_integrity_retry_respects_attempt_limit_and_one_shot_budget(
    tmp_path, monkeypatch, retries, expected_calls
):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    calls = []

    def malformed(*_args, **_kwargs):
        calls.append(None)
        response = _successful_response(SPEC)
        response["text"] = '{"ranking":["M01"]}'
        response["retryable"] = True
        return response

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", malformed)
    attempt_dir = tmp_path / "attempts"
    result = run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        tmp_path / "results",
        attempt_dir,
        data_bundle=DATA_BUNDLE,
        retries=retries,
    )

    assert result["error"] == "complete ranking not found"
    assert len(calls) == expected_calls
    assert len(list(attempt_dir.glob("assay.attempt-*.json"))) == expected_calls


@pytest.mark.parametrize(
    "response",
    [
        {
            "text": "blocked",
            "error": "provider policy block: invalid_prompt",
            "retryable": True,
            "status": "blocked",
        },
        {
            **_successful_response(SPEC),
            "stop_reason": "SAFETY",
            "retryable": True,
        },
        {
            "text": "bad request",
            "error": "invalid_request: unsupported option",
            "retryable": True,
        },
        {
            **_successful_response(SPEC),
            "status": "failed",
            "retryable": True,
        },
    ],
)
def test_runner_does_not_retry_deterministic_provider_failures(tmp_path, monkeypatch, response):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    calls = []

    def deterministic(*_args, **_kwargs):
        calls.append(None)
        return dict(response)

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", deterministic)
    attempt_dir = tmp_path / "attempts"
    run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        tmp_path / "results",
        attempt_dir,
        data_bundle=DATA_BUNDLE,
        retries=5,
    )

    assert len(calls) == 1
    assert len(list(attempt_dir.glob("assay.attempt-*.json"))) == 1


def test_runner_owns_reasoning_summary_compatibility_fallback(tmp_path, monkeypatch):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    responses = iter(
        [
            {
                "text": "",
                "error": "summary unsupported",
                "retryable": False,
                "summary_unsupported": True,
            },
            {
                "text": '{"ranking":["M01","M02"]}',
                "response_model_id": "test-model",
                "provider_response": {"id": "response"},
                "status": "completed",
                "error": None,
            },
        ]
    )
    summary_flags = []
    reservations = []

    class Limiter:
        def reserve(self, value):
            reservations.append(value)

    def fake_chat(*_args, **kwargs):
        summary_flags.append(kwargs["reasoning_summary"])
        return next(responses)

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(run.client, "chat", fake_chat)
    result = run_assay(
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        tmp_path / "results",
        tmp_path / "attempts",
        data_bundle=DATA_BUNDLE,
        retries=2,
        token_limiter=Limiter(),
    )

    assert result["spearman"] == pytest.approx(1.0)
    assert summary_flags == [True, False]
    assert len(reservations) == 2
    assert len(list((tmp_path / "attempts").glob("assay.attempt-*.json"))) == 2


def test_overwrite_archives_prior_canonical_by_content_hash(tmp_path, monkeypatch):
    from src import run

    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    rankings = iter(
        [
            '{"ranking":["M01","M02"]}',
            '{"ranking":["M02","M01"]}',
        ]
    )
    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(
        run.client,
        "chat",
        lambda *_args, **_kwargs: {
            "text": next(rankings),
            "response_model_id": "test-model",
            "provider_response": {"id": "response"},
            "status": "completed",
            "error": None,
        },
    )
    final_dir = tmp_path / "results"
    attempt_dir = tmp_path / "attempts"
    superseded = tmp_path / "superseded"
    common = (
        "test",
        SPEC,
        50,
        1,
        "assay",
        {"assay": META},
        final_dir,
        attempt_dir,
    )
    run_assay(*common, data_bundle=DATA_BUNDLE)
    old_bytes = (final_dir / "assay.json").read_bytes()
    old_hash = __import__("hashlib").sha256(old_bytes).hexdigest()

    run_assay(
        *common,
        data_bundle=DATA_BUNDLE,
        overwrite=True,
        superseded_dir=superseded,
    )

    archived = superseded / f"assay.{old_hash}.json"
    assert archived.read_bytes() == old_bytes
    replacement = json.loads((final_dir / "assay.json").read_text())
    assert replacement["ranking"] == ["M02", "M01"]
    assert replacement["supersedes_sha256"] == old_hash
