"""Inspect benchmark coverage and preserved transport attempts without calling a provider."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config.models import N_BATCHES
from config.paths import RESULTS_ROOT
from src.assays import load_assay_meta
from src.integrity import service_tier_matches_condition
from src.run import ATTEMPTS, _record_succeeded, _result_root, is_truncated


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def build_status(
    *,
    results_root: Path = RESULTS_ROOT,
    run_label: str = "canonical",
    models: list[str] | None = None,
    sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    assays: list[str] | None = None,
) -> dict:
    source_root = results_root if run_label == "canonical" else results_root / "_runs" / run_label
    manifest_path = source_root / "_run.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"valid run manifest missing: {manifest_path}")
    conditions = manifest.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("run manifest conditions are malformed")
    selected_conditions = sorted(
        (
            condition
            for condition in conditions.values()
            if (not models or condition.get("model") in models)
            and (not sizes or condition.get("size") in sizes)
        ),
        key=lambda condition: (condition["model"], int(condition["size"])),
    )
    if not selected_conditions:
        raise ValueError("selection matched no run-manifest conditions")
    meta = load_assay_meta()
    selected_assays = sorted(assays or meta)
    unknown_assays = sorted(set(selected_assays) - set(meta))
    if unknown_assays:
        raise ValueError("unknown assay(s): " + ", ".join(unknown_assays))
    selected_seeds = sorted(seeds or range(1, N_BATCHES + 1))

    attempt_root = ATTEMPTS if run_label == "canonical" else ATTEMPTS / run_label
    # Tests and callers with an alternate results root get its adjacent attempt tree.
    if results_root != RESULTS_ROOT:
        attempt_root = results_root / "_attempts"
        if run_label != "canonical":
            attempt_root = attempt_root / run_label

    rows = []
    all_missing: list[str] = []
    total_attempts = Counter()
    for condition in selected_conditions:
        model, size = condition["model"], int(condition["size"])
        counts = Counter()
        missing: list[str] = []
        for seed in selected_seeds:
            for assay in selected_assays:
                key = f"{model}/n{size}/b{seed}/{assay}"
                path = source_root / f"{model}/n{size}/b{seed}/{assay}.json"
                record = _read_json(path)
                if record is None:
                    counts["missing"] += 1
                    missing.append(key)
                    all_missing.append(key)
                elif _record_succeeded(record) and service_tier_matches_condition(
                    record, condition
                ):
                    counts["complete"] += 1
                elif is_truncated(record):
                    counts["truncated_result"] += 1
                else:
                    counts["invalid_result"] += 1

                attempt_dir = attempt_root / model / f"n{size}" / f"b{seed}"
                for attempt in attempt_dir.glob(f"{assay}.attempt-*.json"):
                    attempt_record = _read_json(attempt)
                    counts["attempts"] += 1
                    total_attempts["all"] += 1
                    if attempt_record is None:
                        counts["malformed_attempts"] += 1
                        total_attempts["malformed"] += 1
                    elif _record_succeeded(attempt_record) and service_tier_matches_condition(
                        attempt_record, condition
                    ):
                        counts["successful_attempts"] += 1
                        total_attempts["successful"] += 1
                    elif is_truncated(attempt_record):
                        counts["truncated_attempts"] += 1
                        total_attempts["truncated"] += 1
                    elif _record_succeeded(attempt_record):
                        counts["invalid_attempts"] += 1
                        total_attempts["invalid"] += 1
                    else:
                        counts["error_attempts"] += 1
                        total_attempts["error"] += 1
        rows.append(
            {
                "model": model,
                "size": size,
                "expected": len(selected_seeds) * len(selected_assays),
                **counts,
                "missing_cells": missing,
            }
        )
    superseded = len(list((source_root / "_superseded").glob("*/n*/b*/*.json")))
    return {
        "run_label": run_label,
        "manifest": str(manifest_path),
        "seeds": selected_seeds,
        "assays": len(selected_assays),
        "conditions": rows,
        "totals": {
            "expected": sum(row["expected"] for row in rows),
            "complete": sum(row.get("complete", 0) for row in rows),
            "missing": len(all_missing),
            "invalid": sum(row.get("invalid_result", 0) for row in rows),
            "truncated": sum(row.get("truncated_result", 0) for row in rows),
            "attempts": dict(total_attempts),
            "superseded": superseded,
        },
        "missing_cells": all_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show planned coverage, canonical completion, and immutable attempts"
    )
    parser.add_argument("--run-label")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--sizes", nargs="+", type=_positive_int)
    parser.add_argument("--seeds", "--batches", dest="seeds", nargs="+", type=_positive_int)
    parser.add_argument("--assays", nargs="+")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--show-missing", action="store_true")
    args = parser.parse_args()
    try:
        _, run_label = _result_root(args.run_label)
        report = build_status(
            run_label=run_label,
            models=args.models,
            sizes=args.sizes,
            seeds=args.seeds,
            assays=args.assays,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.as_json:
        print(json.dumps(report, indent=2))
        return 0
    print(
        f"run={report['run_label']} assays={report['assays']} seeds={report['seeds']} "
        f"complete={report['totals']['complete']}/{report['totals']['expected']}"
    )
    for row in report["conditions"]:
        print(
            f"{row['model']:24s} n{row['size']:<4} "
            f"complete={row.get('complete', 0):4d}/{row['expected']:<4d} "
            f"missing={row.get('missing', 0):4d} attempts={row.get('attempts', 0):4d} "
            f"errors={row.get('error_attempts', 0):3d} "
            f"invalid={row.get('invalid_result', 0):3d} "
            f"truncated={row.get('truncated_result', 0):3d}"
        )
    if args.show_missing:
        for key in report["missing_cells"]:
            print(f"MISSING {key}")
    if any(report["totals"][field] for field in ("missing", "invalid", "truncated")):
        print("Next: inspect attempts, then resume with --retry-errors and/or --retry-truncated.")
    else:
        print("Coverage complete. Run pgllm-score for the full provenance and score audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
