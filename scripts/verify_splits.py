from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.data_bundle import EVAL_SEEDS, EVAL_SIZES  # noqa: E402
from config.paths import DATA_ROOT  # noqa: E402
from src.assays import load_assay_meta  # noqa: E402

DEFAULT_SIZES = list(EVAL_SIZES)
DEFAULT_BATCHES = list(EVAL_SEEDS)
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
    checked = 0

    for assay in assays:
        if assay not in meta:
            bad_meta.append((assay, "assay missing from metadata"))
            continue
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

                checked += 1

    warnings = check_manifest(assays, sizes)
    hard_errors = missing + bad_meta + bad_labels + duplicates

    print(f"assays {len(assays)}")
    print(f"sizes {sizes}")
    print(f"batches {batches}")
    print(f"split_files {sum(1 for p in SPLITS.rglob('*') if p.is_file())}")
    print(f"expected_files_including_manifest {len(assays) * len(sizes) * len(batches) * 2 + 1}")
    print(f"checked_cells {checked}")
    print(f"missing {len(missing)}")
    print(f"bad_meta {len(bad_meta)}")
    print(f"bad_labels {len(bad_labels)}")
    print(f"duplicate_variant_cells {len(duplicates)}")

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
    parser.add_argument(
        "--seeds", "--batches", dest="batches", nargs="*", type=int, default=DEFAULT_BATCHES
    )
    parser.add_argument("--strict-manifest", action="store_true")
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
