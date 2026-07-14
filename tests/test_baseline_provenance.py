import json
import sys

import pytest

from src import analyze, baselines
from src.data_bundle import BundleError

BUNDLE = {
    "bundle_version": "test-v1",
    "manifest_sha256": "1" * 64,
    "selection": {"sizes": [2], "seeds": [1, 2, 3]},
}
META = {
    "assay": {
        "uniprot_id": "TEST",
        "function": "Activity",
    }
}


def _write_fixture(tmp_path, monkeypatch):
    splits = tmp_path / "data" / "splits"
    scores = tmp_path / "data" / "baselines" / "zero_shot_substitutions_scores"
    output = tmp_path / "results_baselines"
    assay_dir = splits / "assay"
    assay_dir.mkdir(parents=True)
    scores.mkdir(parents=True)
    episodes = {
        1: [("v1", "AA", 0.0), ("v2", "AB", 1.0)],
        2: [("v1", "AA", 0.0), ("v3", "AC", 2.0)],
        3: [("v2", "AB", 1.0), ("v3", "AC", 2.0)],
    }
    for seed, subset in episodes.items():
        split = {
            "assay": "assay",
            "size": 2,
            "batch": seed,
            "seed": seed,
            "n": 2,
            "variants": [{"id": mutant, "seq": sequence} for mutant, sequence, _ in subset],
        }
        labels = {mutant: label for mutant, _, label in subset}
        (assay_dir / f"n2_b{seed}.json").write_text(json.dumps(split), encoding="utf-8")
        (assay_dir / f"n2_b{seed}.labels.json").write_text(json.dumps(labels), encoding="utf-8")
    source = scores / "assay.csv"
    source.write_text(
        "mutant,mutated_sequence,Complete,Partial,DMS_score\n"
        "v1,AA,0,0,0\n"
        "v2,AB,1,1,1\n"
        "v3,AC,2,,2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baselines, "SPLITS", splits)
    monkeypatch.setattr(baselines, "SCORES", scores)
    monkeypatch.setattr(baselines, "OUT", output)
    monkeypatch.setattr(baselines, "authenticate_data_bundle", lambda: BUNDLE)
    monkeypatch.setattr(baselines, "load_assay_meta", lambda: META)
    monkeypatch.setattr(
        sys,
        "argv",
        ["baselines", "--assays", "assay", "--sizes", "2"],
    )
    return output, source


def test_baselines_require_full_authentication_before_reading_sources(tmp_path, monkeypatch):
    output = tmp_path / "results_baselines"
    monkeypatch.setattr(baselines, "OUT", output)
    monkeypatch.setattr(
        baselines,
        "authenticate_data_bundle",
        lambda: (_ for _ in ()).throw(BundleError("tampered")),
    )
    monkeypatch.setattr(
        baselines,
        "load_baseline_scores",
        lambda _assay: pytest.fail("source was read before bundle authentication"),
    )
    monkeypatch.setattr(sys, "argv", ["baselines"])

    with pytest.raises(SystemExit, match="failed authentication: tampered"):
        baselines.main()
    assert not output.exists()


def test_cells_require_complete_predictions_and_carry_full_provenance(tmp_path, monkeypatch):
    output, _ = _write_fixture(tmp_path, monkeypatch)
    baselines.main()

    summary = baselines.validate_summary(output / "summary.json", BUNDLE)
    complete = summary["baselines"]["Complete"]["n2"]
    partial = summary["baselines"]["Partial"]["n2"]
    assert complete["coverage"] == {
        "expected_cells": 3,
        "scored_cells": 3,
        "fraction_cells": 1.0,
        "expected_candidates": 6,
        "scored_candidates": 6,
        "predicted_candidates": 6,
    }
    assert partial["coverage"]["scored_cells"] == 1
    assert partial["coverage"]["scored_candidates"] == 2
    assert not (output / "Partial" / "n2" / "b2" / "assay.json").exists()

    cell = json.loads((output / "Complete" / "n2" / "b1" / "assay.json").read_text())
    assert cell["schema_version"] == baselines.BASELINE_CELL_SCHEMA_VERSION
    assert cell["data_bundle"] == BUNDLE
    assert cell["n"] == 2
    assert cell["coverage"] == {"expected": 2, "predicted": 2, "fraction": 1.0}
    assert len(cell["split_sha256"]) == 64
    assert len(cell["baseline_source"]["sha256"]) == 64


@pytest.mark.parametrize("tamper", ["source", "cell", "summary_bundle"])
def test_validation_rejects_stale_or_mismatched_artifacts(tmp_path, monkeypatch, tamper):
    output, source = _write_fixture(tmp_path, monkeypatch)
    baselines.main()
    summary_path = output / "summary.json"
    if tamper == "source":
        source.write_text(source.read_text() + "\n", encoding="utf-8")
    elif tamper == "cell":
        cell_path = output / "Complete" / "n2" / "b1" / "assay.json"
        cell = json.loads(cell_path.read_text())
        cell["spearman"] = -1.0
        cell_path.write_text(json.dumps(cell), encoding="utf-8")
    else:
        summary = json.loads(summary_path.read_text())
        summary["data_bundle"]["manifest_sha256"] = "2" * 64
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(baselines.BaselineProvenanceError):
        baselines.validate_summary(summary_path, BUNDLE)


def test_validation_recomputes_cell_score_even_if_artifact_digests_are_rewritten(
    tmp_path, monkeypatch
):
    output, _ = _write_fixture(tmp_path, monkeypatch)
    baselines.main()
    summary_path = output / "summary.json"
    cell_path = output / "Complete" / "n2" / "b1" / "assay.json"
    cell = json.loads(cell_path.read_text())
    cell["spearman"] = -1.0
    cell_path.write_text(json.dumps(cell), encoding="utf-8")

    summary = json.loads(summary_path.read_text())
    ref = summary["baselines"]["Complete"]["n2"]["cells"][0]
    ref["record_sha256"] = baselines._record_sha256(cell)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        baselines.BaselineProvenanceError,
        match="baseline score does not match source data",
    ):
        baselines.validate_summary(summary_path, BUNDLE)


def test_analyzer_omits_present_but_unauthenticated_summary(tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(analyze, "BASELINE_SUMMARY", summary)
    monkeypatch.setattr(
        analyze,
        "validate_summary",
        lambda *_args: (_ for _ in ()).throw(baselines.BaselineProvenanceError("stale")),
    )

    assert analyze._baseline_means(BUNDLE, assays={"assay"}, sizes={2}, seeds={1, 2, 3}) == {}
    assert "omitting unauthenticated baseline summary" in capsys.readouterr().err


def test_analyzer_never_compares_partial_baseline_coverage(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(analyze, "BASELINE_SUMMARY", summary_path)
    monkeypatch.setattr(
        analyze,
        "validate_summary",
        lambda *_args: {
            "selection": {"assays": ["assay"], "sizes": [2], "seeds": [1, 2, 3]},
            "baselines": {
                "Partial": {
                    "n2": {
                        "mean_rho_macro": 1.0,
                        "coverage": {
                            "expected_cells": 3,
                            "scored_cells": 2,
                            "expected_candidates": 6,
                            "scored_candidates": 4,
                        },
                    }
                }
            },
        },
    )

    assert analyze._baseline_means(BUNDLE, assays={"assay"}, sizes={2}, seeds={1, 2, 3}) == {}
    assert "without full assay and candidate coverage" in capsys.readouterr().err
