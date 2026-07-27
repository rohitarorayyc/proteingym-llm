import gzip
import hashlib
import json

import pytest

from src import export, prompt
from src.export import PublicationBundleError, export_publication_bundle
from src.run import RESULT_SCHEMA_VERSION, RUN_MANIFEST_VERSION, prompt_sha256, split_sha256

DATA_BUNDLE = {
    "bundle_version": "test-v1",
    "manifest_sha256": "1" * 64,
    "selection": {"sizes": [2], "seeds": [1, 2, 3]},
}
SUBSET = [("v1", "AAAA", 1.0), ("v2", "AAAT", 0.0)]
META = {
    "reference_sequence": "AAAA",
    "fitness_description": "test fitness",
}


def _condition():
    descriptor = {
        "accepted_response_model_ids": ["test-model"],
        "inference_options": {
            "stream": False,
            "required_response_metadata": {"usage": True, "reasoning": False},
        },
    }
    return {
        "model": "test-model",
        "size": 2,
        "provider": "openai-compatible",
        "provider_model_id": "test-model",
        "reasoning_effort": "max",
        "max_output_tokens": 128,
        "requested_service_tier": None,
        "prompt_token_estimator": "test",
        "runtime": {"code_sha256": "2" * 64},
        "request_descriptor": descriptor,
        "request_fingerprint": "a" * 64,
    }


def _write_result(root, *, assay="assay", complete=True):
    user, ids = prompt.build_user_prompt(META, META["reference_sequence"], SUBSET)
    condition = _condition()
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_label": "canonical",
        "model": "test-model",
        "provider": condition["provider"],
        "provider_model_id": condition["provider_model_id"],
        "response_model_id": "test-model",
        "provider_response_version": None,
        "response_id": "response-1",
        "assay": assay,
        "assay_description": META["fitness_description"],
        "seed": 1,
        "batch": 1,
        "size": 2,
        "n": 2,
        "spearman": 1.0,
        "parsed": complete,
        "ranking": ids if complete else ids[:1],
        "output_tokens": 10,
        "reasoning_tokens": 8,
        "reasoning_effort": "max",
        "reasoning_text": "trace",
        "reasoning_summary": "trace",
        "answer_text": json.dumps({"ranking": ids}, separators=(",", ":")),
        "raw_output": json.dumps({"ranking": ids}, separators=(",", ":")),
        "usage": {"completion_tokens": 10, "prompt_tokens": 4, "total_tokens": 14},
        "status": "completed",
        "stop_reason": "stop",
        "incomplete_reason": None,
        "stream_completed": None,
        "request_descriptor": condition["request_descriptor"],
        "request_fingerprint": condition["request_fingerprint"],
        "prompt_version": prompt.PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(prompt.SYSTEM_PROMPT, user),
        "split_sha256": split_sha256(SUBSET),
        "prompt_token_estimator": condition["prompt_token_estimator"],
        "eval_bundle_manifest_sha256": DATA_BUNDLE["manifest_sha256"],
        "eval_bundle_version": DATA_BUNDLE["bundle_version"],
        "max_output_tokens": condition["max_output_tokens"],
        "requested_service_tier": None,
        "runtime": condition["runtime"],
        "attempt_started_at_utc": "2026-07-16T20:00:00.000Z",
        "attempt_completed_at_utc": "2026-07-16T20:01:00.000Z",
        "error": None,
        "overflow": False,
        "truncated": False,
        "response_content": [{"type": "message"}],
        "provider_response": {"id": "response-1"},
    }
    target = root / "test-model" / "n2" / "b1" / f"{assay}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result), encoding="utf-8")
    return target


def _prepare(tmp_path, monkeypatch):
    root = tmp_path / "results"
    root.mkdir()
    manifest = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_label": "canonical",
        "prompt_version": prompt.PROMPT_VERSION,
        "data_bundle": DATA_BUNDLE,
        "created_at_utc": "2026-07-16T19:00:00.000Z",
        "updated_at_utc": "2026-07-16T21:00:00.000Z",
        "conditions": {"test-model/n2": _condition()},
    }
    (root / "_run.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(export, "authenticate_data_bundle", lambda: DATA_BUNDLE)
    monkeypatch.setattr(export, "load_assay_meta", lambda: {"assay": META})
    monkeypatch.setattr(export, "shared_subset", lambda *_args: SUBSET)
    return root


def test_export_is_deterministic_streamed_and_keeps_trace_without_provider_payload(
    tmp_path, monkeypatch
):
    root = _prepare(tmp_path, monkeypatch)
    _write_result(root)
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    kwargs = {
        "results_root": root,
        "models": ["test-model"],
        "sizes": [2],
        "seeds": [1],
        "assays": ["assay"],
    }
    export_publication_bundle(first, **kwargs)
    export_publication_bundle(second, **kwargs)

    assert first.read_bytes() == second.read_bytes()
    lines = [json.loads(line) for line in gzip.decompress(first.read_bytes()).splitlines()]
    assert lines[0]["coverage"] == {
        "expected_records": 1,
        "exported_records": 1,
        "missing_records": 0,
        "complete": True,
    }
    assert lines[0]["conditions"]["test-model/n2"] == _condition()
    record = lines[1]["record"]
    assert record["reasoning"] == "trace"
    assert "provider_response" not in record
    assert len(record["provider_payload_digest"]["canonical_sha256"]) == 64
    assert (
        record["source_sha256"]
        == hashlib.sha256((root / record["source_path"]).read_bytes()).hexdigest()
    )
    assert lines[-1]["records"] == 1


def test_export_rejects_invalid_score_or_packed_payload(tmp_path, monkeypatch):
    root = _prepare(tmp_path, monkeypatch)
    path = _write_result(root, complete=False)
    with pytest.raises(PublicationBundleError, match="failed scorer audit"):
        export_publication_bundle(
            tmp_path / "bad.jsonl.gz",
            results_root=root,
            seeds=[1],
            assays=["assay"],
        )

    payload = json.loads(path.read_text())
    payload["parsed"] = True
    payload["ranking"] = ["M01", "M02"]
    payload["provider_response"] = {
        "schema": "proteingym-llm-provider-payload",
        "version": 1,
        "encoding": "gzip+base64",
        "raw_sha256": "0" * 64,
        "compressed_sha256": "0" * 64,
        "raw_bytes": 1,
        "compressed_bytes": 1,
        "payload": "AA==",
    }
    payload["response_content"] = "provider_response.payload#/response_content"
    path.write_text(json.dumps(payload))
    with pytest.raises(PublicationBundleError, match="provider payload"):
        export_publication_bundle(
            tmp_path / "bad-payload.jsonl.gz",
            results_root=root,
            seeds=[1],
            assays=["assay"],
        )


def test_export_requires_complete_selected_coverage_by_default(tmp_path, monkeypatch):
    root = _prepare(tmp_path, monkeypatch)
    _write_result(root)
    with pytest.raises(PublicationBundleError, match="coverage is incomplete"):
        export_publication_bundle(
            tmp_path / "partial.jsonl.gz",
            results_root=root,
            seeds=[1, 2],
            assays=["assay"],
        )

    manifest = export_publication_bundle(
        tmp_path / "debug-partial.jsonl.gz",
        results_root=root,
        seeds=[1, 2],
        assays=["assay"],
        allow_incomplete=True,
    )
    assert manifest["coverage"]["complete"] is False
    assert manifest["coverage"]["missing_records"] == 1
