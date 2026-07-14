from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.models import N_BATCHES, SIZES  # noqa: E402
from config.paths import DATA_ROOT  # noqa: E402
from src import subsample  # noqa: E402
from src.assays import assay_csv, load_assay_meta  # noqa: E402

DEFAULT_SIZES = SIZES
DEFAULT_BATCHES = list(range(1, N_BATCHES + 1))
DEFAULT_STRATA = 10
SPLITS = DATA_ROOT / "splits"


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_json(path: Path):
    return json.loads(path.read_text())


def check_manifest(expected_assays: list[str], sizes: list[int]) -> list[str]:
    path = SPLITS / "manifest.csv"
    if not path.exists():
        return ["data/splits/manifest.csv is missing"]
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    warnings: list[str] = []
    if len(rows) != len(expected_assays):
        warnings.append(f"manifest row count is {len(rows)}, expected {len(expected_assays)}")
    cols = set(rows[0]) if rows else set()
    missing_cols = [f"n_at_{s}" for s in sizes if f"n_at_{s}" not in cols]
    if missing_cols:
        warnings.append(f"manifest missing columns: {', '.join(missing_cols)}")
    return warnings


def verify(args: argparse.Namespace) -> int:
    meta = load_assay_meta()
    assays = args.assays or sorted(meta)
    sizes = args.sizes
    batches = args.batches

    missing = []
    bad_meta = []
    bad_labels = []
    duplicates = []
    deterministic_mismatches = []
    missing_sources = []
    source_checked = 0
    checked = 0

    for assay in assays:
        if assay not in meta:
            bad_meta.append((assay, "assay missing from metadata"))
            continue
        source_path = assay_csv(assay)
        rows = subsample.load_variants(source_path) if source_path.exists() else None
        if rows is None:
            missing_sources.append(assay)
        for size in sizes:
            for batch in batches:
                split_path = SPLITS / assay / f"n{size}_b{batch}.json"
                label_path = SPLITS / assay / f"n{size}_b{batch}.labels.json"
                if not split_path.exists() or not label_path.exists():
                    missing.append((assay, size, batch, split_path.exists(), label_path.exists()))
                    continue

                split = load_json(split_path)
                labels = load_json(label_path)
                variants = split.get("variants", [])
                variant_ids = [v.get("id") for v in variants]

                if (
                    split.get("assay") != assay
                    or split.get("size") != size
                    or split.get("batch") != batch
                    or split.get("seed") != batch
                    or split.get("n") != len(variants)
                ):
                    bad_meta.append(
                        (
                            assay,
                            size,
                            batch,
                            split.get("assay"),
                            split.get("size"),
                            split.get("batch"),
                            split.get("seed"),
                            split.get("n"),
                            len(variants),
                        )
                    )

                if len(set(variant_ids)) != len(variant_ids):
                    duplicates.append((assay, size, batch))

                if set(variant_ids) != set(labels):
                    bad_labels.append(
                        (
                            assay,
                            size,
                            batch,
                            len(variant_ids),
                            len(labels),
                            len(set(variant_ids) ^ set(labels)),
                        )
                    )

                if rows is not None:
                    expected = subsample.stratified_sample(rows, size, args.strata, seed=batch)
                    expected_variants = [
                        {"id": variant, "seq": sequence} for variant, sequence, _ in expected
                    ]
                    expected_labels = {variant: score for variant, _, score in expected}
                    if variants != expected_variants or labels != expected_labels:
                        deterministic_mismatches.append((assay, size, batch))
                    source_checked += 1

                checked += 1

    warnings = check_manifest(assays, sizes)
    hard_errors = missing + bad_meta + bad_labels + duplicates + deterministic_mismatches
    if args.require_source:
        hard_errors += missing_sources

    print(f"assays {len(assays)}")
    print(f"sizes {sizes}")
    print(f"batches {batches}")
    print(f"strata {args.strata}")
    print(f"split_files {sum(1 for p in SPLITS.rglob('*') if p.is_file())}")
    print(f"expected_files_including_manifest {len(assays) * len(sizes) * len(batches) * 2 + 1}")
    print(f"checked_cells {checked}")
    print(f"source_recomputed_cells {source_checked}")
    print(f"assays_without_raw_source {len(missing_sources)}")
    print(f"missing {len(missing)}")
    print(f"bad_meta {len(bad_meta)}")
    print(f"bad_labels {len(bad_labels)}")
    print(f"duplicate_variant_cells {len(duplicates)}")
    print(f"deterministic_mismatches {len(deterministic_mismatches)}")

    for warning in warnings:
        print(f"WARNING {warning}")

    for sample in args.sha_samples:
        path = SPLITS / sample
        if path.exists():
            print(f"sha256_16 data/splits/{sample} {sha16(path)}")
        else:
            print(f"WARNING missing sha sample: data/splits/{sample}")

    if hard_errors:
        print("FAIL split verification failed")
        return 1
    if warnings and args.strict_manifest:
        print("FAIL split verification failed strict manifest check")
        return 1
    print("OK split verification passed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assays", nargs="*")
    parser.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--batches", nargs="*", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--strata", type=int, default=DEFAULT_STRATA)
    parser.add_argument("--strict-manifest", action="store_true")
    parser.add_argument(
        "--require-source",
        action="store_true",
        help="also require raw DMS tables and recompute every deterministic split",
    )
    parser.add_argument(
        "--sha-samples",
        nargs="*",
        default=[
            "TPMT_HUMAN_Matreyek_2018/n10_b1.json",
            "TPMT_HUMAN_Matreyek_2018/n50_b2.json",
            "TPMT_HUMAN_Matreyek_2018/n100_b3.json",
            "manifest.csv",
        ],
    )
    raise SystemExit(verify(parser.parse_args()))


if __name__ == "__main__":
    main()
