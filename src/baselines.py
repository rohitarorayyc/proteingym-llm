"""Score published ProteinGym predictors on authenticated frozen episodes.

Baseline artifacts are deliberately self-authenticating: each complete cell is
bound to the pinned evaluation bundle, the exact ordered split (including held-
out labels), and the digest of the source prediction CSV.  Partial prediction
vectors are never scored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.models import SIZES  # noqa: E402
from config.paths import BASELINE_RESULTS_ROOT, DATA_ROOT  # noqa: E402
from src.aggregate import macro_headline, nested_macro  # noqa: E402
from src.assays import load_assay_meta  # noqa: E402
from src.data_bundle import BundleError, sha256_file  # noqa: E402
from src.prompt import spearman  # noqa: E402
from src.run import authenticate_data_bundle, split_sha256  # noqa: E402

SPLITS = DATA_ROOT / "splits"
SCORES = DATA_ROOT / "baselines" / "zero_shot_substitutions_scores"
OUT = BASELINE_RESULTS_ROOT
NON_MODEL = {"mutant", "mutated_sequence"}
BASELINE_CELL_SCHEMA_VERSION = 1
BASELINE_SUMMARY_SCHEMA_VERSION = 1
CELL_KIND = "proteingym_llm_baseline_cell"
SUMMARY_KIND = "proteingym_llm_baseline_summary"


class BaselineProvenanceError(RuntimeError):
    """A baseline artifact cannot be authenticated against its recorded inputs."""


def _is_model_col(column: str) -> bool:
    return column not in NON_MODEL and not column.startswith("DMS_")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _record_sha256(record: dict) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _baseline_csv(assay: str) -> Path | None:
    path = SCORES / f"{assay}.csv"
    return path if path.is_file() and not path.is_symlink() else None


def load_baseline_scores(assay: str) -> tuple[dict[str, dict], list[str], dict] | None:
    """Load one prediction source and bind it to its byte-level digest."""
    path = _baseline_csv(assay)
    if path is None:
        return None
    rows: dict[str, dict] = {}
    columns: list[str] | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "mutant" not in reader.fieldnames:
            raise BaselineProvenanceError(f"baseline source lacks mutant column: {path}")
        for row in reader:
            if columns is None:
                columns = [column for column in row if _is_model_col(column)]
            mutant = row.get("mutant", "")
            if not mutant or mutant in rows:
                raise BaselineProvenanceError(
                    f"empty or duplicate mutant {mutant!r} in baseline source: {path}"
                )
            rows[mutant] = row
    source = {
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    return rows, (columns or []), source


def load_split(assay: str, size: int, seed: int) -> list[tuple[str, str, float]] | None:
    split_path = SPLITS / assay / f"n{size}_b{seed}.json"
    label_path = SPLITS / assay / f"n{size}_b{seed}.labels.json"
    if not split_path.is_file() or not label_path.is_file():
        return None
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        variants = split["variants"]
        subset = [(item["id"], item["seq"], float(labels[item["id"]])) for item in variants]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BaselineProvenanceError(f"invalid frozen split: {split_path}") from error
    if (
        split.get("assay") != assay
        or split.get("size") != size
        or split.get("batch") != seed
        or split.get("seed") != seed
        or split.get("n") != len(subset)
        or len({item[0] for item in subset}) != len(subset)
        or any(not math.isfinite(item[2]) for item in subset)
    ):
        raise BaselineProvenanceError(f"inconsistent frozen split: {split_path}")
    return subset


def _complete_column_score(
    rows: dict[str, dict], column: str, subset: list[tuple[str, str, float]]
) -> float | None:
    """Return a score only when *column* covers every candidate with a finite value."""
    predictions: list[float] = []
    for mutant, _, _ in subset:
        try:
            prediction = float(rows[mutant][column])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(prediction):
            return None
        predictions.append(prediction)
    if len(predictions) != len(subset):
        return None
    rho = spearman([label for _, _, label in subset], predictions)
    return rho if rho is not None and math.isfinite(rho) else None


def score_cell(
    assay: str,
    size: int,
    seed: int,
    *,
    source_data: tuple[dict[str, dict], list[str], dict] | None = None,
    only_baselines: set[str] | None = None,
) -> tuple[dict[str, float], list[tuple[str, str, float]], dict] | None:
    """Score only predictors with one finite prediction for every candidate."""
    subset = load_split(assay, size, seed)
    source_data = source_data if source_data is not None else load_baseline_scores(assay)
    if subset is None or source_data is None:
        return None
    rows, columns, source = source_data
    if only_baselines:
        columns = [column for column in columns if column in only_baselines]
    scores: dict[str, float] = {}
    for column in columns:
        rho = _complete_column_score(rows, column, subset)
        if rho is not None:
            scores[column] = rho
    return scores, subset, source


def _safe_baseline_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _summary_entry(
    baseline: str,
    size: int,
    seed_maps: dict[int, dict[str, float]],
    cell_refs: list[dict],
    *,
    data_bundle: dict,
    meta: dict,
    expected_cells: int,
    expected_candidates: int,
) -> dict:
    seed_scores = {
        seed: macro_headline(nested_macro(assays, meta, "function"))
        for seed, assays in seed_maps.items()
    }
    seed_scores = {seed: rho for seed, rho in seed_scores.items() if rho is not None}
    assay_values: dict[str, list[float]] = defaultdict(list)
    for assay_scores in seed_maps.values():
        for assay, rho in assay_scores.items():
            assay_values[assay].append(rho)
    assay_rho = {assay: st.mean(values) for assay, values in assay_values.items()}
    by_selection = nested_macro(assay_rho, meta, "function")
    scored_candidates = sum(cell["n"] for cell in cell_refs)
    scored_cells = len(cell_refs)
    macro_values = list(seed_scores.values())
    return {
        "schema_version": BASELINE_SUMMARY_SCHEMA_VERSION,
        "kind": "proteingym_llm_baseline_summary_entry",
        "baseline": baseline,
        "size": size,
        "n": size,
        "data_bundle": data_bundle,
        "mean_rho_macro": st.mean(macro_values) if macro_values else None,
        "seed_rho": {str(seed): rho for seed, rho in sorted(seed_scores.items())},
        "mean_rho_flat": st.mean(assay_rho.values()) if assay_rho else None,
        "n_assays": len(assay_rho),
        "n_uniprot": len(
            {meta[a]["uniprot_id"] for a in assay_rho if a in meta and meta[a]["uniprot_id"]}
        ),
        "by_selection": {
            group: values["mean_rho"] for group, values in sorted(by_selection.items())
        },
        "coverage": {
            "expected_cells": expected_cells,
            "scored_cells": scored_cells,
            "fraction_cells": scored_cells / expected_cells if expected_cells else 0.0,
            "expected_candidates": expected_candidates,
            "scored_candidates": scored_candidates,
            "predicted_candidates": scored_candidates,
        },
        "cells": sorted(cell_refs, key=lambda cell: (cell["assay"], cell["seed"])),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineProvenanceError(message)


def validate_summary(path: Path, data_bundle: dict) -> dict:
    """Fully validate summary, cells, splits, prediction sources, and aggregates."""
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineProvenanceError(f"cannot read baseline summary: {path}") from error
    _require(
        summary.get("schema_version") == BASELINE_SUMMARY_SCHEMA_VERSION, "stale summary schema"
    )
    _require(summary.get("kind") == SUMMARY_KIND, "invalid summary kind")
    _require(summary.get("data_bundle") == data_bundle, "baseline summary bundle mismatch")
    selection = summary.get("selection")
    _require(isinstance(selection, dict), "baseline summary selection is missing")
    assays = selection.get("assays")
    sizes = selection.get("sizes")
    seeds = selection.get("seeds")
    _require(
        isinstance(assays, list) and len(assays) == len(set(assays)), "invalid assay selection"
    )
    _require(
        isinstance(sizes, list) and set(sizes) <= set(data_bundle["selection"]["sizes"]),
        "invalid size selection",
    )
    _require(
        isinstance(seeds, list) and set(seeds) <= set(data_bundle["selection"]["seeds"]),
        "invalid seed selection",
    )
    baselines = summary.get("baselines")
    _require(isinstance(baselines, dict), "baseline summaries are missing")

    meta = load_assay_meta()
    split_cache: dict[tuple[str, int, int], list[tuple[str, str, float]]] = {}
    source_cache: dict[str, tuple[dict[str, dict], list[str], dict]] = {}
    for baseline, by_size in baselines.items():
        _require(isinstance(by_size, dict), f"invalid summary for {baseline}")
        for size_key, entry in by_size.items():
            _require(size_key == f"n{entry.get('size')}", f"size key mismatch for {baseline}")
            size = entry.get("size")
            _require(
                size in sizes and entry.get("n") == size, f"invalid N for {baseline}/{size_key}"
            )
            _require(
                entry.get("schema_version") == BASELINE_SUMMARY_SCHEMA_VERSION,
                "stale summary entry schema",
            )
            _require(entry.get("baseline") == baseline, f"baseline name mismatch for {baseline}")
            _require(
                entry.get("data_bundle") == data_bundle,
                f"bundle mismatch for {baseline}/{size_key}",
            )
            refs = entry.get("cells")
            _require(isinstance(refs, list), f"cell manifest missing for {baseline}/{size_key}")
            seen: set[tuple[str, int]] = set()
            seed_maps: dict[int, dict[str, float]] = defaultdict(dict)
            normalized_refs: list[dict] = []
            for ref in refs:
                assay, seed = ref.get("assay"), ref.get("seed")
                _require(assay in assays and seed in seeds, "cell lies outside summary selection")
                _require((assay, seed) not in seen, "duplicate baseline cell reference")
                seen.add((assay, seed))
                relative = Path(ref.get("path", ""))
                _require(
                    not relative.is_absolute() and ".." not in relative.parts, "unsafe cell path"
                )
                cell_path = (path.parent / relative).resolve()
                _require(
                    path.parent.resolve() in cell_path.parents, "cell path escapes baseline root"
                )
                try:
                    record = json.loads(cell_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise BaselineProvenanceError(
                        f"cannot read baseline cell: {cell_path}"
                    ) from error
                _require(
                    _record_sha256(record) == ref.get("record_sha256"),
                    "baseline cell digest mismatch",
                )
                key = (assay, size, seed)
                if key not in split_cache:
                    subset = load_split(*key)
                    _require(subset is not None, f"missing authenticated split {key}")
                    split_cache[key] = subset
                subset = split_cache[key]
                split_hash = split_sha256(subset)
                if assay not in source_cache:
                    source_data = load_baseline_scores(assay)
                    _require(source_data is not None, f"missing baseline source for {assay}")
                    source_cache[assay] = source_data
                rows, columns, source = source_cache[assay]
                _require(baseline in columns, f"baseline column missing from source: {baseline}")
                recorded_rho = record.get("spearman")
                _require(
                    isinstance(recorded_rho, (int, float))
                    and not isinstance(recorded_rho, bool)
                    and math.isfinite(recorded_rho),
                    "invalid baseline Spearman value",
                )
                expected_rho = _complete_column_score(rows, baseline, subset)
                _require(expected_rho is not None, "baseline source has incomplete cell coverage")
                _require(recorded_rho == expected_rho, "baseline score does not match source data")
                coverage = {"expected": len(subset), "predicted": len(subset), "fraction": 1.0}
                expected_fields = {
                    "schema_version": BASELINE_CELL_SCHEMA_VERSION,
                    "kind": CELL_KIND,
                    "baseline": baseline,
                    "assay": assay,
                    "size": size,
                    "batch": seed,
                    "seed": seed,
                    "n": len(subset),
                    "coverage": coverage,
                    "split_sha256": split_hash,
                    "baseline_source": source,
                    "data_bundle": data_bundle,
                    "spearman": expected_rho,
                }
                _require(
                    record == expected_fields, f"baseline cell provenance mismatch: {cell_path}"
                )
                _require(ref.get("n") == len(subset), "baseline cell N mismatch")
                _require(ref.get("coverage") == coverage, "baseline cell coverage mismatch")
                _require(ref.get("split_sha256") == split_hash, "baseline split digest mismatch")
                _require(
                    ref.get("baseline_source_sha256") == source["sha256"],
                    "baseline source digest mismatch",
                )
                seed_maps[seed][assay] = expected_rho
                normalized_refs.append(ref)

            expected_candidates = 0
            for assay in assays:
                for seed in seeds:
                    key = (assay, size, seed)
                    if key not in split_cache:
                        subset = load_split(*key)
                        _require(subset is not None, f"missing authenticated split {key}")
                        split_cache[key] = subset
                    expected_candidates += len(split_cache[key])
            expected = _summary_entry(
                baseline,
                size,
                seed_maps,
                normalized_refs,
                data_bundle=data_bundle,
                meta=meta,
                expected_cells=len(assays) * len(seeds),
                expected_candidates=expected_candidates,
            )
            _require(entry == expected, f"baseline aggregate mismatch for {baseline}/{size_key}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assays", nargs="*")
    parser.add_argument("--baselines", nargs="*", help="subset of baseline columns")
    parser.add_argument("--sizes", nargs="*", type=int, default=list(SIZES))
    args = parser.parse_args()

    try:
        data_bundle = authenticate_data_bundle()
    except BundleError as error:
        raise SystemExit(f"evaluation data failed authentication: {error}") from error
    sizes = list(dict.fromkeys(args.sizes))
    seeds = list(data_bundle["selection"]["seeds"])
    unsupported = sorted(set(sizes) - set(data_bundle["selection"]["sizes"]))
    if unsupported:
        parser.error(f"sizes outside the authenticated bundle: {unsupported}")
    assays = args.assays or sorted(path.name for path in SPLITS.iterdir() if path.is_dir())
    if len(assays) != len(set(assays)):
        parser.error("duplicate assays are not allowed")
    only = set(args.baselines) if args.baselines else None
    meta = load_assay_meta()
    unknown = sorted(set(assays) - set(meta))
    if unknown:
        parser.error("unknown assay(s): " + ", ".join(unknown))

    OUT.mkdir(parents=True, exist_ok=True)
    seed_maps: dict[tuple[str, int], dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    cell_refs: dict[tuple[str, int], list[dict]] = defaultdict(list)
    expected_candidates: dict[int, int] = defaultdict(int)
    for assay in assays:
        source_data = load_baseline_scores(assay)
        for size in sizes:
            for seed in seeds:
                subset = load_split(assay, size, seed)
                if subset is None:
                    raise SystemExit(f"authenticated bundle is missing {assay}/n{size}/b{seed}")
                expected_candidates[size] += len(subset)
                result = score_cell(
                    assay,
                    size,
                    seed,
                    source_data=source_data,
                    only_baselines=only,
                )
                if result is None:
                    continue
                scores, scored_subset, source = result
                split_hash = split_sha256(scored_subset)
                for baseline, rho in scores.items():
                    record = {
                        "schema_version": BASELINE_CELL_SCHEMA_VERSION,
                        "kind": CELL_KIND,
                        "baseline": baseline,
                        "assay": assay,
                        "size": size,
                        "batch": seed,
                        "seed": seed,
                        "n": len(scored_subset),
                        "coverage": {
                            "expected": len(scored_subset),
                            "predicted": len(scored_subset),
                            "fraction": 1.0,
                        },
                        "split_sha256": split_hash,
                        "baseline_source": source,
                        "data_bundle": data_bundle,
                        "spearman": rho,
                    }
                    relative = (
                        Path(_safe_baseline_name(baseline))
                        / f"n{size}"
                        / f"b{seed}"
                        / f"{assay}.json"
                    )
                    _atomic_json(OUT / relative, record)
                    seed_maps[(baseline, size)][seed][assay] = rho
                    cell_refs[(baseline, size)].append(
                        {
                            "path": relative.as_posix(),
                            "record_sha256": _record_sha256(record),
                            "assay": assay,
                            "seed": seed,
                            "n": len(scored_subset),
                            "coverage": record["coverage"],
                            "split_sha256": split_hash,
                            "baseline_source_sha256": source["sha256"],
                        }
                    )

    summaries: dict[str, dict] = {}
    for (baseline, size), maps in sorted(seed_maps.items()):
        summaries.setdefault(baseline, {})[f"n{size}"] = _summary_entry(
            baseline,
            size,
            maps,
            cell_refs[(baseline, size)],
            data_bundle=data_bundle,
            meta=meta,
            expected_cells=len(assays) * len(seeds),
            expected_candidates=expected_candidates[size],
        )
    summary = {
        "schema_version": BASELINE_SUMMARY_SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "data_bundle": data_bundle,
        "selection": {"assays": assays, "sizes": sizes, "seeds": seeds},
        "baselines": summaries,
    }
    _atomic_json(OUT / "summary.json", summary)

    if sizes:
        largest = max(sizes)
        board = sorted(
            (
                (name, values[f"n{largest}"])
                for name, values in summaries.items()
                if f"n{largest}" in values
            ),
            key=lambda item: -(item[1]["mean_rho_macro"] or -9),
        )
        print(f"\nBaseline nested-macro Spearman @ n{largest} (complete cells only):")
        for name, entry in board[:15]:
            coverage = entry["coverage"]
            print(
                f"  {name:34s} {entry['mean_rho_macro']:+.3f}  "
                f"[{coverage['scored_cells']}/{coverage['expected_cells']} cells]"
            )
    print(f"\n-> {OUT}/<baseline>/n<size>/b<seed>/<assay>.json + summary.json")


if __name__ == "__main__":
    main()
