#!/usr/bin/env python3
"""Submit targeted OpenAI batch retries for retryable result errors.

The normal batch submitter creates one manifest per model/size/batch and skips
cells that already have a manifest. This helper is intentionally narrower: it
scans existing result JSON files, selects only retryable error records, submits
one new OpenAI batch per affected cell, and writes a uniquely named manifest
whose `model` remains the original model. Collection therefore overwrites only
the retried assay result files in the standard results tree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.models import MODELS  # noqa: E402
from src import batch as batch_mod  # noqa: E402
from src.assays import load_assay_meta  # noqa: E402
from src.run import RESULTS  # noqa: E402


DEFAULT_RETRY_CATEGORIES = {
    "timeout_batch_expired",
    "cancelled_unexecuted",
    "server_internal_error",
    "server_overloaded",
    "provider_failed_unspecified",
}


def classify_error(message: str) -> str:
    s = (message or "").lower()
    if "batch_expired" in s or "completion window expired" in s:
        return "timeout_batch_expired"
    if "batch_cancelled" in s or "batch was cancelled" in s or "cancelled" in s or "canceled" in s:
        return "cancelled_unexecuted"
    if "limited access to this content" in s or "safety" in s:
        return "safety_block"
    if "overloaded" in s:
        return "server_overloaded"
    if "server had an error" in s or "error occurred while processing" in s or "internal" in s:
        return "server_internal_error"
    if s.strip() == "failed" or "'failed'" in s:
        return "provider_failed_unspecified"
    if "empty response" in s:
        return "empty_response"
    return "other_error"


def safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip())
    return cleaned.strip("-") or "retry"


def retry_tag(model: str, size: int, batch: int, label: str) -> str:
    return f"zz-retry-{safe_label(label)}__{model}__n{size}__b{batch}"


def selected_assays(model: str, categories: set[str], sizes: set[int] | None,
                    batches: set[int] | None) -> dict[tuple[int, int], list[tuple[str, str]]]:
    selected: dict[tuple[int, int], list[tuple[str, str]]] = {}
    root = RESULTS / model
    for path in sorted(root.glob("n*/b*/*.json")):
        size = int(path.parts[-3][1:])
        batch = int(path.parts[-2][1:])
        if sizes is not None and size not in sizes:
            continue
        if batches is not None and batch not in batches:
            continue
        rec = json.loads(path.read_text())
        error = rec.get("error")
        if not error:
            continue
        category = classify_error(str(error))
        if category in categories:
            selected.setdefault((size, batch), []).append((path.stem, category))
    return selected


def submit_cell(model: str, size: int, batch: int, assays_with_categories: list[tuple[str, str]],
                label: str, dry_run: bool) -> str:
    spec = MODELS[model]
    assays = [assay for assay, _ in assays_with_categories]
    meta = load_assay_meta()
    reqs, keyed = batch_mod._build_requests(model, spec, size, batch, assays, meta)
    tag = retry_tag(model, size, batch, label)
    manifest = batch_mod.JOBS / f"{tag}.json"
    if manifest.exists():
        return f"skip {tag}: manifest already exists"
    counts: dict[str, int] = {}
    for _, category in assays_with_categories:
        counts[category] = counts.get(category, 0) + 1
    if dry_run:
        return f"dry-run {tag}: {len(reqs)} requests {counts}"
    job_id, extra = batch_mod._submit_provider(spec["provider"], reqs, tag)
    manifest.write_text(json.dumps({
        "tag": tag,
        "model": model,
        "provider": spec["provider"],
        "size": size,
        "batch": batch,
        "job_id": job_id,
        "assays": list(keyed),
        "retry_of": batch_mod._tag(model, size, batch),
        "retry_label": label,
        "retry_categories": sorted(set(counts)),
        "retry_category_counts": counts,
        **extra,
    }, indent=2))
    return f"submitted {tag}: {len(reqs)} requests -> job {job_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--categories", nargs="*", default=sorted(DEFAULT_RETRY_CATEGORIES))
    parser.add_argument("--sizes", nargs="*", type=int)
    parser.add_argument("--batches", nargs="*", type=int)
    parser.add_argument("--label", default=datetime.now(timezone.utc).strftime("retry-%Y%m%d-%H%M%S"))
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--submit", action="store_true", help="actually upload and submit retry batches")
    args = parser.parse_args()

    if args.model not in MODELS:
        raise SystemExit(f"unknown model: {args.model}")
    if MODELS[args.model]["provider"] != "openai":
        raise SystemExit("this helper only supports OpenAI models")

    categories = set(args.categories)
    sizes = set(args.sizes) if args.sizes else None
    batches = set(args.batches) if args.batches else None
    selected = selected_assays(args.model, categories, sizes, batches)

    total = sum(len(v) for v in selected.values())
    print(f"model={args.model} label={args.label} submit={args.submit}")
    print(f"categories={sorted(categories)}")
    print(f"affected_cells={len(selected)} total_requests={total}")
    for (size, batch), assays in sorted(selected.items()):
        counts: dict[str, int] = {}
        for _, category in assays:
            counts[category] = counts.get(category, 0) + 1
        print(f"  n{size}/b{batch}: {len(assays)} {counts}")

    if not selected:
        return

    batch_mod.JOBS.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(selected)))
    dry_run = not args.submit
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(submit_cell, args.model, size, batch, assays, args.label, dry_run)
            for (size, batch), assays in sorted(selected.items())
        ]
        for fut in as_completed(futures):
            print(fut.result())


if __name__ == "__main__":
    main()
