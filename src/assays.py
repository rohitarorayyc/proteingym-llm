"""ProteinGym assay metadata for the frozen evaluation bundle."""

from __future__ import annotations

import csv
from collections import Counter
from functools import lru_cache

from config.paths import DATA_ROOT

SPLITS_DIR = DATA_ROOT / "splits"
REFERENCE = DATA_ROOT / "reference" / "DMS_substitutions.csv"


@lru_cache(maxsize=1)
def _runnable_assays() -> set[str]:
    split_manifest = SPLITS_DIR / "manifest.csv"
    if split_manifest.is_file():
        with split_manifest.open(encoding="utf-8", newline="") as handle:
            return {
                (row.get("assay") or "").strip()
                for row in csv.DictReader(handle)
                if (row.get("assay") or "").strip()
            }
    return (
        {path.name for path in SPLITS_DIR.iterdir() if path.is_dir()}
        if SPLITS_DIR.exists()
        else set()
    )


@lru_cache(maxsize=1)
def load_assay_meta() -> dict[str, dict]:
    """Load metadata for assays in the frozen evaluation split."""
    if not REFERENCE.exists():
        return {}

    runnable = _runnable_assays()
    meta: dict[str, dict] = {}
    with REFERENCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    ids = [(row.get("DMS_id") or "").strip() for row in rows]
    if any(not assay for assay in ids):
        raise ValueError("Frozen evaluation metadata contains an empty DMS_id")
    duplicates = sorted(assay for assay, count in Counter(ids).items() if assay and count > 1)
    if duplicates:
        raise ValueError(f"Duplicate assays in frozen evaluation metadata: {', '.join(duplicates)}")
    available = {assay for assay in ids if assay}
    if available != runnable:
        missing = sorted(runnable - available)
        extra = sorted(available - runnable)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("Frozen evaluation metadata/split mismatch: " + "; ".join(details))

    for row in rows:
        assay_id = (row.get("DMS_id") or "").strip()
        wt = (row.get("target_seq") or "").strip()
        description = (row.get("fitness_description") or "").strip()
        if not description:
            raise ValueError(
                f"Frozen evaluation metadata has no fitness_description for {assay_id}"
            )
        meta[assay_id] = {
            "reference_sequence": wt,
            "target_name": (row.get("UniProt_ID") or assay_id).strip(),
            "uniprot_id": (row.get("UniProt_ID") or "").strip(),
            "organism": (row.get("source_organism") or row.get("taxon") or "").strip(),
            "fitness_description": description,
            "function": (row.get("coarse_selection_type") or "?").strip() or "?",
            "taxon": (row.get("taxon") or "?").strip() or "?",
            "msa_category": (row.get("MSA_Neff_L_category") or "?").strip() or "?",
            "seq_len": int(row.get("seq_len") or len(wt) or 0),
            "multi": str(row.get("includes_multiple_mutants", "")).strip().lower()
            in {"true", "1", "yes"},
        }
    return meta
