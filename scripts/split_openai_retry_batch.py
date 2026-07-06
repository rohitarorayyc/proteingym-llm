#!/usr/bin/env python3
"""Split an existing OpenAI retry batch manifest into smaller replacement jobs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.models import MODELS  # noqa: E402
from src import batch as batch_mod, client  # noqa: E402
from src.assays import load_assay_meta  # noqa: E402


def chunks(items: list[str], parts: int) -> list[list[str]]:
    base, extra = divmod(len(items), parts)
    out = []
    start = 0
    for idx in range(parts):
        n = base + (1 if idx < extra else 0)
        out.append(items[start:start + n])
        start += n
    return out


def cancel_openai_batch(job_id: str) -> dict:
    from openai import OpenAI

    cli = OpenAI(api_key=client._key("OPENAI_API_KEY"))
    batch = cli.batches.retrieve(job_id)
    if batch.status in {"validating", "in_progress", "finalizing"}:
        batch = cli.batches.cancel(job_id)
    return {
        "id": batch.id,
        "status": batch.status,
        "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="existing manifest tag without .json")
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument("--label", default="split")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    manifest_path = batch_mod.JOBS / f"{args.tag}.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    model = manifest["model"]
    spec = MODELS[model]
    if spec["provider"] != "openai":
        raise SystemExit("only OpenAI retry manifests are supported")

    assays = list(manifest["assays"])
    parts = [part for part in chunks(assays, args.parts) if part]
    print(f"tag={args.tag} model={model} n={manifest['size']} b={manifest['batch']} assays={len(assays)}")
    for idx, part in enumerate(parts, 1):
        print(f"  part{idx}={len(part)}")
    if not args.submit:
        print("dry-run only")
        return

    cancel_state = cancel_openai_batch(manifest["job_id"])
    archive_dir = batch_mod.JOBS / "archived-split"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_manifest = dict(manifest)
    archived_manifest["archived_reason"] = "split_retry_batch"
    archived_manifest["archived_at"] = datetime.now(timezone.utc).isoformat()
    archived_manifest["cancel_state_at_archive"] = cancel_state
    archive_path = archive_dir / f"{args.tag}.json"
    archive_path.write_text(json.dumps(archived_manifest, indent=2))
    manifest_path.unlink()
    print(f"archived {args.tag}: cancel_state={cancel_state}")

    meta = load_assay_meta()
    for idx, part in enumerate(parts, 1):
        part_tag = f"{args.tag}__part{idx}of{len(parts)}"
        reqs, keyed = batch_mod._build_requests(model, spec, manifest["size"], manifest["batch"], part, meta)
        job_id, extra = batch_mod._submit_provider(spec["provider"], reqs, part_tag)
        part_manifest = {
            "tag": part_tag,
            "model": model,
            "provider": spec["provider"],
            "size": manifest["size"],
            "batch": manifest["batch"],
            "job_id": job_id,
            "assays": list(keyed),
            "retry_of": manifest.get("retry_of"),
            "retry_label": manifest.get("retry_label"),
            "split_of": args.tag,
            "split_part": idx,
            "split_parts": len(parts),
            **extra,
        }
        (batch_mod.JOBS / f"{part_tag}.json").write_text(json.dumps(part_manifest, indent=2))
        print(f"submitted {part_tag}: {len(reqs)} requests -> job {job_id}")


if __name__ == "__main__":
    main()
