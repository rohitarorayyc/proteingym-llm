import json

from src import status


def test_status_reports_plan_completion_and_preserved_attempts(tmp_path, monkeypatch):
    root = tmp_path / "results"
    manifest = {
        "conditions": {
            "test-model/n50": {
                "model": "test-model",
                "size": 50,
            }
        }
    }
    root.mkdir()
    (root / "_run.json").write_text(json.dumps(manifest))
    canonical = root / "test-model/n50/b1/assay.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        json.dumps(
            {
                "spearman": 0.5,
                "parsed": True,
                "error": None,
                "truncated": False,
                "overflow": False,
                "attempt_started_at_utc": "2026-07-16T20:00:00.000Z",
                "attempt_completed_at_utc": "2026-07-16T20:01:00.000Z",
                "response_model_id": "test-model",
            }
        )
    )
    attempt = root / "_attempts/test-model/n50/b2/assay.attempt-test.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text(json.dumps({"error": "connection dropped"}))
    monkeypatch.setattr(status, "load_assay_meta", lambda: {"assay": {}})

    report = status.build_status(
        results_root=root,
        models=["test-model"],
        sizes=[50],
        seeds=[1, 2],
        assays=["assay"],
    )

    assert report["totals"]["expected"] == 2
    assert report["totals"]["complete"] == 1
    assert report["totals"]["missing"] == 1
    assert report["totals"]["attempts"] == {"all": 1, "error": 1}
    assert report["missing_cells"] == ["test-model/n50/b2/assay"]


def test_status_does_not_count_a_service_tier_mismatch_as_complete(tmp_path, monkeypatch):
    root = tmp_path / "results"
    manifest = {
        "conditions": {
            "test-model/n50": {
                "model": "test-model",
                "size": 50,
                "requested_service_tier": "flex",
            }
        }
    }
    root.mkdir()
    (root / "_run.json").write_text(json.dumps(manifest))
    canonical = root / "test-model/n50/b1/assay.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        json.dumps(
            {
                "spearman": 0.5,
                "parsed": True,
                "error": None,
                "truncated": False,
                "overflow": False,
                "attempt_started_at_utc": "2026-07-16T20:00:00.000Z",
                "attempt_completed_at_utc": "2026-07-16T20:01:00.000Z",
                "response_model_id": "test-model",
                "requested_service_tier": "flex",
                "service_tier": "default",
            }
        )
    )
    monkeypatch.setattr(status, "load_assay_meta", lambda: {"assay": {}})

    report = status.build_status(
        results_root=root,
        models=["test-model"],
        sizes=[50],
        seeds=[1],
        assays=["assay"],
    )

    assert report["totals"]["complete"] == 0
    assert report["totals"]["invalid"] == 1
    assert report["conditions"][0]["invalid_result"] == 1
