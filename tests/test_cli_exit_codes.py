import sys

import pytest

from src import run, verify_models

DATA_BUNDLE = {
    "bundle_version": "test-v1",
    "manifest_sha256": "1" * 64,
    "selection": {"sizes": [50], "seeds": [1]},
}
SPEC = {
    "provider": "openai-compatible",
    "api_style": "responses",
    "model_id": "internal-model",
    "api_key_env": "LAB_API_KEY",
    "base_url_env": "LAB_BASE_URL",
    "reasoning": "max",
    "send_reasoning": True,
    "max_tokens": 128_000,
    "ctx": 1_000_000,
    "display_name": "Lab model",
}


def _prepare_run_main(monkeypatch, tmp_path, outcome):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pgllm-run",
            "--registry",
            "models.json",
            "--models",
            "lab-model",
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
    monkeypatch.setattr(run, "load_model_registry", lambda _path: {"lab-model": SPEC})
    monkeypatch.setattr(run, "condition_record", lambda *_args, **_kwargs: {})
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


def _prepare_probe(monkeypatch, models=("lab-model",)):
    registry = {name: dict(SPEC, model_id=name) for name in models}
    monkeypatch.setattr(verify_models, "load_model_registry", lambda _path: registry)
    return registry


def test_model_probe_is_cheap_by_default_and_fails_closed(monkeypatch):
    seen = []
    _prepare_probe(monkeypatch)

    def probe(spec, *_args, **_kwargs):
        seen.append(spec)
        return {"text": "", "error": "unreachable", "elapsed_s": 0}

    monkeypatch.setattr(verify_models.client, "chat", probe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pgllm-models", "--registry", "models.json", "--models", "lab-model"],
    )
    assert verify_models.main() == 1
    assert seen[0]["reasoning"] == "low"
    assert seen[0]["max_tokens"] == verify_models.PROBE_MAX_OUTPUT_TOKENS


def test_model_probe_canonical_opt_in_and_success(monkeypatch):
    seen = []
    _prepare_probe(monkeypatch)

    def probe(spec, *_args, **_kwargs):
        seen.append(spec)
        return {"text": "OK", "error": None, "elapsed_s": 0.1}

    monkeypatch.setattr(verify_models.client, "chat", probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pgllm-models",
            "--registry",
            "models.json",
            "--models",
            "lab-model",
            "--canonical-settings",
        ],
    )
    assert verify_models.main() == 0
    assert seen[0]["reasoning"] == "max"
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
    _prepare_probe(monkeypatch, ("lab-one", "lab-two"))
    monkeypatch.setattr(
        verify_models.client,
        "chat",
        lambda *_args, **_kwargs: next(responses),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pgllm-models",
            "--registry",
            "models.json",
            "--models",
            "lab-one",
            "lab-two",
        ],
    )
    assert verify_models.main() == 1
