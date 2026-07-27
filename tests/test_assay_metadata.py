from __future__ import annotations

import csv

import pytest

from src import assays


@pytest.fixture(autouse=True)
def _clear_assay_caches():
    assays._runnable_assays.cache_clear()
    assays.load_assay_meta.cache_clear()
    yield
    assays._runnable_assays.cache_clear()
    assays.load_assay_meta.cache_clear()


def _write_bundle_metadata(tmp_path, description: str) -> None:
    reference = tmp_path / "reference" / "DMS_substitutions.csv"
    reference.parent.mkdir(parents=True)
    with reference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "DMS_id",
                "UniProt_ID",
                "source_organism",
                "target_seq",
                "fitness_description",
                "coarse_selection_type",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "DMS_id": "TOY",
                "UniProt_ID": "TOY_PROTEIN",
                "source_organism": "Test organism",
                "target_seq": "AAAA",
                "fitness_description": description,
                "coarse_selection_type": "Activity",
            }
        )

    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "manifest.csv").write_text("assay\nTOY\n", encoding="utf-8")


def _load_from(tmp_path, monkeypatch):
    monkeypatch.setattr(assays, "REFERENCE", tmp_path / "reference" / "DMS_substitutions.csv")
    monkeypatch.setattr(assays, "SPLITS_DIR", tmp_path / "splits")
    assays._runnable_assays.cache_clear()
    assays.load_assay_meta.cache_clear()
    return assays.load_assay_meta()


def test_loader_reads_the_canonical_description_directly(tmp_path, monkeypatch):
    _write_bundle_metadata(tmp_path, "Audited experimental endpoint.")
    meta = _load_from(tmp_path, monkeypatch)

    assert meta["TOY"]["fitness_description"] == "Audited experimental endpoint."
    assert set(meta["TOY"]) == {
        "reference_sequence",
        "target_name",
        "uniprot_id",
        "organism",
        "fitness_description",
        "function",
        "taxon",
        "msa_category",
        "seq_len",
        "multi",
    }


def test_loader_rejects_missing_canonical_description(tmp_path, monkeypatch):
    _write_bundle_metadata(tmp_path, "")
    with pytest.raises(ValueError, match="no fitness_description for TOY"):
        _load_from(tmp_path, monkeypatch)
