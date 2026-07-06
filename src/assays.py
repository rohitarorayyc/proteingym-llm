"""Assay metadata + CSV lookup (simple).

Builds {assay_id: meta} from the ProteinGym reference file for assays whose DMS
CSV is present under data/DMS/ (globbed recursively; the zip extracts into a
subdir). meta carries what the prompt needs (WT seq, name, organism, assay
description) plus fields for later stratification (function/taxon).
ProteinGym DMS_score is pre-oriented so higher = fitter for every assay, so
there is no per-assay direction to carry.
"""
from __future__ import annotations
import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DMS_DIR = ROOT / "data" / "DMS"
REFERENCE = ROOT / "data" / "reference" / "DMS_substitutions.csv"


@lru_cache(maxsize=1)
def _csv_map() -> dict:
    return {p.stem: p for p in DMS_DIR.rglob("*.csv")} if DMS_DIR.exists() else {}


def assay_csv(assay_id: str) -> Path:
    return _csv_map().get(assay_id, DMS_DIR / f"{assay_id}.csv")


@lru_cache(maxsize=1)
def load_assay_meta() -> dict:
    have = set(_csv_map())
    meta: dict = {}
    if not REFERENCE.exists():
        return meta
    for r in csv.DictReader(open(REFERENCE)):
        aid = (r.get("DMS_id") or "").strip()
        if aid not in have:
            continue
        wt = (r.get("target_seq") or "").strip()
        desc = "; ".join(b for b in [(r.get("coarse_selection_type") or "").strip(),
                                     (r.get("selection_assay") or "").strip()] if b)
        meta[aid] = {
            "reference_sequence": wt,
            "target_name": (r.get("UniProt_ID") or aid).strip(),
            "uniprot_id": (r.get("UniProt_ID") or "").strip(),      # nested-macro grouping key; blank -> dropped (oracle dropna)
            "organism": (r.get("source_organism") or r.get("taxon") or "").strip(),
            "fitness_description": desc or "deep mutational scan (fitness)",
            "function": (r.get("coarse_selection_type") or "?").strip() or "?",
            "taxon": (r.get("taxon") or "?").strip() or "?",
            "msa_category": (r.get("MSA_Neff_L_category") or "?").strip() or "?",
            "seq_len": int(r.get("seq_len") or len(wt) or 0),
            "multi": str(r.get("includes_multiple_mutants", "")).strip().lower() in ("true", "1", "yes"),
        }
    return meta
