import json
import sys

import pytest

from src import analyze, prompt
from src.run import (
    base_result_record,
    condition_key,
    condition_record,
    ensure_run_manifest,
    record_response,
    write_result,
)

SPEC = {
    "provider": "openai",
    "model_id": "test-model",
    "reasoning": "max",
    "max_tokens": 128000,
    "ctx": 1000000,
}
META = {
    "reference_sequence": "WT",
    "fitness_description": "measured activity",
    "fitness_description_source": "Activity; assay",
    "fitness_description_repair": None,
    "seq_len": 2,
    "uniprot_id": "TEST",
    "function": "Activity",
    "taxon": "Human",
    "msa_category": "medium",
}
SUBSET = [("v1", "A", 1.0), ("v2", "B", 0.0)]
DATA_BUNDLE = {
    "bundle_version": "test-v1",
    "manifest_sha256": "1" * 64,
    "selection": {"sizes": [50], "seeds": [1, 2, 3]},
}


def _write_one_cell(root, *, batch=1, tamper_split=False):
    condition = condition_record("test", 50, SPEC)
    ensure_run_manifest(
        root,
        "canonical",
        {condition_key("test", 50): condition},
        data_bundle=DATA_BUNDLE,
    )
    user, ids = prompt.build_user_prompt(META, "WT", SUBSET)
    record = base_result_record(
        "test",
        SPEC,
        50,
        batch,
        "assay",
        META,
        user,
        len(ids),
        via="live",
        data_bundle=DATA_BUNDLE,
        subset=SUBSET,
    )
    record_response(
        record,
        {"text": '{"ranking":["M01","M02"]}', "status": "completed"},
        ids,
        SUBSET,
    )
    if tamper_split:
        record["split_sha256"] = "0" * 64
    write_result(root / "test" / "n50" / f"b{batch}" / "assay.json", record)


def _patch_analysis(monkeypatch, root):
    monkeypatch.setattr(analyze, "_result_root", lambda _label: (root, "canonical"))
    monkeypatch.setattr(analyze, "load_assay_meta", lambda: {"assay": META})
    monkeypatch.setattr(analyze, "shared_subset", lambda *_args: SUBSET)
    monkeypatch.setattr(analyze, "authenticate_data_bundle", lambda: DATA_BUNDLE)
    monkeypatch.setattr(sys, "argv", ["analyze", "--models", "test", "--sizes", "50"])


def test_analysis_refuses_missing_assays_or_seeds(tmp_path, monkeypatch):
    root = tmp_path / "results"
    _write_one_cell(root)
    _patch_analysis(monkeypatch, root)
    with pytest.raises(SystemExit, match="unmatched/incomplete coverage"):
        analyze.main()


def test_analysis_recomputes_and_rejects_episode_hashes(tmp_path, monkeypatch):
    root = tmp_path / "results"
    _write_one_cell(root, tamper_split=True)
    _patch_analysis(monkeypatch, root)
    with pytest.raises(SystemExit, match="provenance-incompatible"):
        analyze.main()


def test_analysis_rejects_a_different_authenticated_bundle(tmp_path, monkeypatch):
    root = tmp_path / "results"
    _write_one_cell(root)
    _patch_analysis(monkeypatch, root)
    changed = {**DATA_BUNDLE, "manifest_sha256": "2" * 64}
    monkeypatch.setattr(analyze, "authenticate_data_bundle", lambda: changed)

    with pytest.raises(SystemExit, match="run-manifest provenance mismatch"):
        analyze.main()


def test_analysis_accepts_complete_matched_three_seed_run(tmp_path, monkeypatch, capsys):
    root = tmp_path / "results"
    for batch in (1, 2, 3):
        _write_one_cell(root, batch=batch)
    _patch_analysis(monkeypatch, root)

    analyze.main()

    output = capsys.readouterr().out
    assert "test" in output
    assert "+1.000" in output
    assert "      3       1       0" in output


def test_run_manifest_freezes_each_model_size_condition(tmp_path):
    root = tmp_path / "results"
    condition = condition_record("test", 50, SPEC)
    ensure_run_manifest(
        root,
        "scan",
        {condition_key("test", 50): condition},
        data_bundle=DATA_BUNDLE,
    )
    changed = dict(condition)
    changed["reasoning_effort"] = "high"
    with pytest.raises(RuntimeError, match="different configuration"):
        ensure_run_manifest(
            root,
            "scan",
            {condition_key("test", 50): changed},
            data_bundle=DATA_BUNDLE,
        )

    changed_bundle = {**DATA_BUNDLE, "manifest_sha256": "2" * 64}
    with pytest.raises(RuntimeError, match="incompatible data_bundle"):
        ensure_run_manifest(
            root,
            "scan",
            {condition_key("test", 50): condition},
            data_bundle=changed_bundle,
        )

    manifest = json.loads((root / "_run.json").read_text())
    assert manifest["conditions"]["test/n50"]["reasoning_effort"] == "max"
    assert len(manifest["conditions"]["test/n50"]["runtime"]["code_sha256"]) == 64
    assert manifest["data_bundle"] == DATA_BUNDLE
    assert manifest["created_at_utc"].endswith("Z")
    assert manifest["updated_at_utc"].endswith("Z")
