import argparse
import json
import sys

import pytest

from src import batch, run, verify_models

DATA_BUNDLE = {
    "bundle_version": "test-v1",
    "manifest_sha256": "1" * 64,
    "selection": {"sizes": [50], "seeds": [1]},
}


def _prepare_run_main(monkeypatch, tmp_path, outcome):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pgllm-run",
            "--models",
            "gpt-5.5",
            "--assays",
            "assay",
            "--sizes",
            "50",
            "--batches",
            "1",
            "--concurrency",
            "1",
        ],
    )
    monkeypatch.setattr(run, "authenticate_data_bundle", lambda: DATA_BUNDLE)
    monkeypatch.setattr(run, "ensure_run_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run, "load_assay_meta", lambda: {"assay": {}})
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "ATTEMPTS", tmp_path / "attempts")
    if isinstance(outcome, Exception):

        def fail(*_args, **_kwargs):
            raise outcome

        monkeypatch.setattr(run, "run_assay", fail)
    else:
        monkeypatch.setattr(run, "run_assay", lambda *_args, **_kwargs: dict(outcome))


@pytest.mark.parametrize(
    "outcome",
    [
        {"error": "provider failed", "spearman": None},
        {"truncated": True, "spearman": None},
        {"overflow": True, "spearman": None},
        {"parsed": False, "spearman": None},
    ],
)
def test_live_main_returns_nonzero_for_incomplete_selected_work(monkeypatch, tmp_path, outcome):
    _prepare_run_main(monkeypatch, tmp_path, outcome)
    assert run.main() == 1


def test_live_main_returns_nonzero_for_worker_exception(monkeypatch, tmp_path):
    _prepare_run_main(monkeypatch, tmp_path, RuntimeError("worker exploded"))
    assert run.main() == 1


def test_live_dry_run_success_returns_zero(monkeypatch, tmp_path):
    _prepare_run_main(
        monkeypatch,
        tmp_path,
        {"dry_run": True, "error": None, "truncated": False, "overflow": False},
    )
    sys.argv.append("--dry-run")
    assert run.main() == 0


def _batch_args(*, dry_run=True):
    return argparse.Namespace(
        models=["gpt-5.5"],
        assays=["assay"],
        sizes=[50],
        batches=[1],
        workers=1,
        dry_run=dry_run,
    )


def _prepare_batch_submit(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "authenticate_data_bundle", lambda: DATA_BUNDLE)
    monkeypatch.setattr(batch, "load_assay_meta", lambda: {"assay": {}})
    monkeypatch.setattr(batch, "ensure_run_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(batch, "JOBS", tmp_path / "jobs")


def test_batch_submit_aggregates_worker_failure(monkeypatch, tmp_path):
    _prepare_batch_submit(monkeypatch, tmp_path)

    def fail(*_args, **_kwargs):
        raise RuntimeError("submission failed")

    monkeypatch.setattr(batch, "_submit_cell", fail)
    assert batch.submit(_batch_args()) == 1


def test_batch_submit_dry_run_success_returns_zero(monkeypatch, tmp_path):
    _prepare_batch_submit(monkeypatch, tmp_path)
    monkeypatch.setattr(
        batch,
        "_submit_cell",
        lambda *_args, **_kwargs: batch.BatchSubmitOutcome("dry-run one request"),
    )
    assert batch.submit(_batch_args()) == 0


def test_batch_main_propagates_selected_command_status(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pgllm-batch", "submit", "--dry-run"])
    monkeypatch.setattr(batch, "submit", lambda _args: 1)
    assert batch.main() == 1


def test_batch_collect_returns_nonzero_for_failed_or_mismatched_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "authenticate_data_bundle", lambda: DATA_BUNDLE)
    monkeypatch.setattr(batch, "load_assay_meta", lambda: {})
    monkeypatch.setattr(batch, "JOBS", tmp_path / "jobs")
    batch.JOBS.mkdir(parents=True)
    (batch.JOBS / "wrong.json").write_text(
        '{"tag":"wrong","data_bundle":{"bundle_version":"other"}}'
    )
    assert batch.collect(argparse.Namespace(overwrite=False)) == 1


def test_batch_collect_ignores_old_failed_response_after_valid_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "authenticate_data_bundle", lambda: DATA_BUNDLE)
    monkeypatch.setattr(batch, "load_assay_meta", lambda: {"assay": {}})
    monkeypatch.setattr(batch, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(batch, "RESULTS", tmp_path / "results")
    batch.JOBS.mkdir(parents=True)
    job = {
        "tag": "original",
        "model": "gpt-5.5",
        "provider": "openai",
        "job_id": "batch-1",
        "model_spec": {"provider": "openai"},
        "size": 50,
        "batch": 1,
        "assays": ["assay"],
        "episodes": {"assay": {"present": True}},
        "data_bundle": DATA_BUNDLE,
        "submission_started_at_utc": "2026-07-14T12:00:00.000Z",
        "submitted_at_utc": "2026-07-14T12:00:01.000Z",
        "runtime": batch.runtime_provenance("openai"),
    }
    (batch.JOBS / "original.json").write_text(json.dumps(job))
    output = batch.RESULTS / "gpt-5.5" / "n50" / "b1" / "assay.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}")
    monkeypatch.setattr(batch, "_fetch_results", lambda _job: {"assay": {"error": "old"}})
    monkeypatch.setattr(batch, "_rebuild", lambda *_args: ("prompt", ["M01"], [("v", "A", 1.0)]))
    monkeypatch.setattr(batch, "base_result_record", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(batch, "should_run", lambda *_args, **_kwargs: False)
    assert batch.collect(argparse.Namespace(overwrite=False)) == 0


def test_batch_retry_distinguishes_complete_noop_from_uncollected_job(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "authenticate_data_bundle", lambda: DATA_BUNDLE)
    monkeypatch.setattr(batch, "load_assay_meta", lambda: {})
    monkeypatch.setattr(batch, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(batch, "FAILURES", batch.JOBS / "failures")
    monkeypatch.setattr(batch, "RESULTS", tmp_path / "results")
    batch.JOBS.mkdir(parents=True)
    job = {
        "tag": "job",
        "model": "gpt-5.5",
        "provider": "openai",
        "size": 50,
        "batch": 1,
        "attempt": 0,
        "assays": ["assay"],
        "data_bundle": DATA_BUNDLE,
        "submission_started_at_utc": "2026-07-14T12:00:00.000Z",
        "submitted_at_utc": "2026-07-14T12:00:01.000Z",
        "runtime": batch.runtime_provenance("openai"),
    }
    (batch.JOBS / "job.json").write_text(json.dumps(job))
    args = argparse.Namespace(jobs=["job"], dry_run=True)
    assert batch.retry(args) == 1

    final = batch.RESULTS / "gpt-5.5" / "n50" / "b1" / "assay.json"
    final.parent.mkdir(parents=True)
    final.write_text("{}")
    assert batch.retry(args) == 0


def test_model_probe_is_cheap_by_default_and_fails_closed(monkeypatch):
    seen = []

    def probe(spec, *_args, **_kwargs):
        seen.append(spec)
        return {"text": "", "error": "unreachable", "elapsed_s": 0}

    monkeypatch.setattr(verify_models.client, "chat", probe)
    monkeypatch.setattr(sys, "argv", ["pgllm-models", "--models", "gpt-5.5"])
    assert verify_models.main() == 1
    assert seen[0]["reasoning"] == "low"
    assert seen[0]["max_tokens"] == verify_models.PROBE_MAX_OUTPUT_TOKENS


def test_model_probe_canonical_opt_in_and_success(monkeypatch):
    seen = []

    def probe(spec, *_args, **_kwargs):
        seen.append(spec)
        return {"text": "OK", "error": None, "elapsed_s": 0.1}

    monkeypatch.setattr(verify_models.client, "chat", probe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pgllm-models", "--models", "gpt-5.5", "--canonical-settings"],
    )
    assert verify_models.main() == 0
    assert seen[0]["reasoning"] == "xhigh"
    assert seen[0]["max_tokens"] == 128_000


def test_model_probe_rejects_unexpected_or_incomplete_output(monkeypatch):
    responses = iter(
        [
            {"text": "not OK", "error": None, "elapsed_s": 0.1},
            {
                "text": "OK",
                "error": None,
                "elapsed_s": 0.1,
                "status": "incomplete",
                "incomplete_reason": "max_output_tokens",
            },
        ]
    )
    monkeypatch.setattr(
        verify_models.client,
        "chat",
        lambda *_args, **_kwargs: next(responses),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pgllm-models", "--models", "gpt-5.5", "gpt-5.6-sol"],
    )
    assert verify_models.main() == 1
