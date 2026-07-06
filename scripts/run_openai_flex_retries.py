#!/usr/bin/env python3
"""Run targeted OpenAI recoverable-result retries through service_tier=flex.

This is intentionally separate from the Batch API path. It scans existing result
files, selects retryable OpenAI errors, calls /v1/responses with Flex processing,
and overwrites only successful/permanent retry outcomes in the normal results
tree. 429/resource-unavailable responses are not written; they stay pending and
can be retried after the configured delay.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.models import MODELS  # noqa: E402
from scripts.submit_openai_retries import DEFAULT_RETRY_CATEGORIES, classify_error  # noqa: E402
from src import client, prompt  # noqa: E402
from src.assays import load_assay_meta  # noqa: E402
from src.run import RESULTS, shared_subset  # noqa: E402

STATE_DIR = RESULTS / "_flex_retries"


def selected_tasks(
    model: str,
    categories: set[str],
    sizes: set[int] | None,
    batches: set[int] | None,
) -> list[tuple[int, int, str, str]]:
    tasks: list[tuple[int, int, str, str]] = []
    for path in sorted((RESULTS / model).glob("n*/b*/*.json")):
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
            tasks.append((size, batch, path.stem, category))
    return tasks


def is_429(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    low = f"{type(exc).__name__}: {exc}".lower()
    return status == 429 or "429" in low or "resource unavailable" in low


def is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    low = f"{type(exc).__name__}: {exc}".lower()
    return (
        status in {408, 409, 500, 502, 503, 504}
        or any(s in low for s in ("timeout", "overload", "server", "internal", "connection"))
    )


def model_dump(obj: Any) -> dict | None:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return None


def write_cell(model: str, size: int, batch: int, assay: str, rec: dict) -> None:
    cell_dir = RESULTS / model / f"n{size}" / f"b{batch}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / f"{assay}.json").write_text(json.dumps(rec, indent=2))


def run_one(model: str, spec: dict, size: int, batch: int, assay: str, meta: dict, timeout: int) -> dict:
    if assay not in meta:
        return {"status": "missing_meta", "assay": assay, "size": size, "batch": batch}
    sub = shared_subset(assay, size, batch)
    if not sub:
        return {"status": "missing_split", "assay": assay, "size": size, "batch": batch}

    user, ids = prompt.build_user_prompt(meta[assay], meta[assay]["reference_sequence"], sub)
    ntok = client.estimate_tokens(prompt.SYSTEM_PROMPT) + len(user) // 4
    base = {
        "model": model,
        "assay": assay,
        "size": size,
        "batch": batch,
        "n": len(ids),
        "prompt_tokens_est": ntok,
        "via": "openai_flex",
        "service_tier": "flex",
    }
    if ntok > spec["ctx"] - spec["max_tokens"]:
        rec = {**base, "overflow": True, "spearman": None}
        write_cell(model, size, batch, assay, rec)
        return {"status": "overflow", "assay": assay, "size": size, "batch": batch}

    from openai import OpenAI

    t0 = time.time()
    cli = OpenAI(api_key=client._key("OPENAI_API_KEY"), timeout=timeout, max_retries=0)
    try:
        response = cli.responses.create(
            model=spec["model_id"],
            instructions=prompt.SYSTEM_PROMPT,
            input=user,
            reasoning={"effort": spec.get("reasoning", "high")},
            max_output_tokens=spec["max_tokens"],
            service_tier="flex",
        )
    except Exception as exc:  # noqa: BLE001 - classify provider errors by status/body
        elapsed = round(time.time() - t0, 1)
        msg = f"{type(exc).__name__}: {exc}"
        if is_429(exc):
            return {"status": "rate_limited_429", "assay": assay, "size": size, "batch": batch,
                    "elapsed_s": elapsed, "error": msg}
        if is_transient(exc):
            return {"status": "transient_error", "assay": assay, "size": size, "batch": batch,
                    "elapsed_s": elapsed, "error": msg}
        rec = {**base, "overflow": False, "spearman": None, "parsed": False,
               "ranking": None, "raw_output": "", "error": msg, "elapsed_s": elapsed}
        write_cell(model, size, batch, assay, rec)
        return {"status": "terminal_error", "assay": assay, "size": size, "batch": batch,
                "elapsed_s": elapsed, "error": msg}

    elapsed = round(time.time() - t0, 1)
    text = response.output_text or ""
    err = "empty response" if not text.strip() else None
    ranking = None if err else prompt.parse_ranking(text, ids)
    rho = prompt.score_ranking(ranking, ids, sub) if ranking else None
    rec = {
        **base,
        "overflow": False,
        "spearman": rho,
        "parsed": ranking is not None,
        "error": err,
        "elapsed_s": elapsed,
        "ranking": ranking,
        "raw_output": text[:4000],
        "response_id": getattr(response, "id", None),
        "actual_service_tier": getattr(response, "service_tier", None),
        "usage": model_dump(getattr(response, "usage", None)),
    }
    write_cell(model, size, batch, assay, rec)
    return {"status": "ok" if err is None else "empty_response", "assay": assay,
            "size": size, "batch": batch, "elapsed_s": elapsed, "spearman": rho}


def write_state(label: str, payload: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{label}.json").write_text(json.dumps(payload, indent=2))


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(label: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock = STATE_DIR / f"{label}.lock"
    if lock.exists():
        try:
            payload = json.loads(lock.read_text())
        except Exception:  # noqa: BLE001
            payload = {}
        pid = payload.get("pid")
        if isinstance(pid, int) and process_alive(pid):
            raise SystemExit(f"another flex retry process is active: pid={pid} lock={lock}")
        lock.unlink()
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as fh:
        json.dump({"pid": os.getpid(), "created_at": ts()}, fh)
    atexit.register(lambda: lock.unlink(missing_ok=True))


def run_round(args, round_idx: int, meta: dict) -> tuple[Counter, list[dict]]:
    categories = set(args.categories)
    sizes = set(args.sizes) if args.sizes else None
    batches = set(args.batches) if args.batches else None
    tasks = selected_tasks(args.model, categories, sizes, batches)
    print(f"{ts()} round={round_idx} selected={len(tasks)} concurrency={args.concurrency}")
    if args.dry_run or not tasks:
        return Counter({"selected": len(tasks)}), []

    spec = MODELS[args.model]
    counts: Counter = Counter()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = [
            ex.submit(run_one, args.model, spec, size, batch, assay, meta, args.timeout)
            for size, batch, assay, _category in tasks
        ]
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            counts[row["status"]] += 1
            print(f"{ts()} {row['status']:16s} n{row['size']}/b{row['batch']} {row['assay']}")
    counts["selected"] = len(tasks)
    return counts, rows


def ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--categories", nargs="*", default=sorted(DEFAULT_RETRY_CATEGORIES))
    parser.add_argument("--sizes", nargs="*", type=int)
    parser.add_argument("--batches", nargs="*", type=int)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--retry-delay-minutes", type=float, default=30)
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="0 means keep retrying until no selected recoverable errors remain")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--label", default="flex-recoverable-gpt55-20260624")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.model not in MODELS:
        raise SystemExit(f"unknown model: {args.model}")
    if MODELS[args.model]["provider"] != "openai":
        raise SystemExit("this runner only supports OpenAI Responses models")

    if not args.dry_run:
        acquire_lock(args.label)

    meta = load_assay_meta()
    round_idx = 0
    history: list[dict] = []
    while True:
        round_idx += 1
        counts, rows = run_round(args, round_idx, meta)
        snapshot = {
            "label": args.label,
            "model": args.model,
            "updated_at": ts(),
            "concurrency": args.concurrency,
            "retry_delay_minutes": args.retry_delay_minutes,
            "round": round_idx,
            "latest_counts": dict(counts),
            "history": history[-20:] + [{"round": round_idx, "counts": dict(counts), "updated_at": ts()}],
            "latest_rows": rows[-100:],
        }
        history = snapshot["history"]
        write_state(args.label, snapshot)
        print(f"{ts()} round={round_idx} counts={dict(counts)}")

        pending = counts.get("rate_limited_429", 0) + counts.get("transient_error", 0)
        if args.dry_run or counts.get("selected", 0) == 0:
            return
        if args.max_rounds and round_idx >= args.max_rounds:
            return
        if pending == 0:
            remaining = selected_tasks(
                args.model,
                set(args.categories),
                set(args.sizes) if args.sizes else None,
                set(args.batches) if args.batches else None,
            )
            if not remaining:
                return
        delay = max(0, args.retry_delay_minutes * 60)
        print(f"{ts()} sleeping {delay / 60:.1f} minutes before retrying pending recoverable cells")
        time.sleep(delay)


if __name__ == "__main__":
    main()
