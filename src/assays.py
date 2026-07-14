"""ProteinGym assay metadata and versioned prompt-description repairs."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from config.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
DMS_DIR = DATA_ROOT / "DMS"
SPLITS_DIR = DATA_ROOT / "splits"
REFERENCE = DATA_ROOT / "reference" / "DMS_substitutions.csv"
PROMPT_REPAIRS = ROOT / "config" / "assay_prompt_repairs_v1.json"
PROMPT_REPAIR_VERSION = "hard-prompt-repairs-v1"


def source_fitness_description(row: dict) -> str:
    """Reconstruct the generic description used by the original benchmark."""
    values = [
        (row.get("coarse_selection_type") or "").strip(),
        (row.get("selection_assay") or "").strip(),
    ]
    return "; ".join(value for value in values if value) or "deep mutational scan (fitness)"


@lru_cache(maxsize=1)
def load_prompt_repair_document() -> dict:
    with PROMPT_REPAIRS.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("repair_set") != PROMPT_REPAIR_VERSION:
        raise ValueError(f"Unexpected prompt-repair set: {document.get('repair_set')!r}")
    if not isinstance(document.get("repairs"), dict):
        raise ValueError("Prompt-repair manifest must contain a repairs mapping")
    return document


def apply_prompt_repair(assay_id: str, source_description: str) -> tuple[str, str | None]:
    """Return ``(description, repair_version)`` for one assay.

    The recorded before-text must match byte-for-byte.  This fails loudly if a
    future ProteinGym release changes source metadata beneath a stale repair.
    """
    repair = load_prompt_repair_document()["repairs"].get(assay_id)
    if not repair:
        return source_description, None
    if repair.get("before") != source_description:
        raise ValueError(
            f"Stale prompt repair for {assay_id}: expected {repair.get('before')!r}, "
            f"found {source_description!r}"
        )
    return repair["after"], PROMPT_REPAIR_VERSION


@lru_cache(maxsize=1)
def _csv_map() -> dict[str, Path]:
    return {path.stem: path for path in DMS_DIR.rglob("*.csv")} if DMS_DIR.exists() else {}


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
    assays = set(_csv_map())
    if SPLITS_DIR.exists():
        assays.update(path.name for path in SPLITS_DIR.iterdir() if path.is_dir())
    return assays


def assay_csv(assay_id: str) -> Path:
    return _csv_map().get(assay_id, DMS_DIR / f"{assay_id}.csv")


@lru_cache(maxsize=1)
def load_assay_meta() -> dict[str, dict]:
    """Load metadata for assays backed by a DMS table or a frozen split."""
    if not REFERENCE.exists():
        return {}

    runnable = _runnable_assays()
    meta: dict[str, dict] = {}
    with REFERENCE.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            assay_id = (row.get("DMS_id") or "").strip()
            if assay_id not in runnable:
                continue
            wt = (row.get("target_seq") or "").strip()
            source_description = source_fitness_description(row)
            description, repair_version = apply_prompt_repair(assay_id, source_description)
            meta[assay_id] = {
                "reference_sequence": wt,
                "target_name": (row.get("UniProt_ID") or assay_id).strip(),
                "uniprot_id": (row.get("UniProt_ID") or "").strip(),
                "organism": (row.get("source_organism") or row.get("taxon") or "").strip(),
                "fitness_description": description,
                "fitness_description_source": source_description,
                "fitness_description_repair": repair_version,
                "function": (row.get("coarse_selection_type") or "?").strip() or "?",
                "taxon": (row.get("taxon") or "?").strip() or "?",
                "msa_category": (row.get("MSA_Neff_L_category") or "?").strip() or "?",
                "seq_len": int(row.get("seq_len") or len(wt) or 0),
                "multi": str(row.get("includes_multiple_mutants", "")).strip().lower()
                in {"true", "1", "yes"},
            }
    return meta
