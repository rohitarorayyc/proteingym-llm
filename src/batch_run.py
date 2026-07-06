"""Anthropic Message Batches path for pg-agent (50% cheaper, async).

Submits cells (default: claude-sonnet-5 N=50/N=100) as batch requests, polls, and
writes result files in the SAME schema/paths as ``src.run`` so they are drop-in
for analysis. Prompt + subsample are deterministic per (assay, size, batch), so
retrieval rebuilds ids/sub and scores LOCALLY -- any local parse/score/write bug is
fixable by re-running ``--mode poll`` on the retained batch results (no resubmit,
no extra cost). Only a genuinely wrong prompt would waste spend, and the prompt is
the identical, already-proven ``src.run`` prompt.

One batch per (size, batch) grid cell (<=217 requests each, well under the API's
100k / 256MB limits). custom_id = ``c_<size>_<batch>_<assay>`` (<=51 chars).
Resume-safe: skips cells whose result file already exists; transient batch errors
are left unwritten so the next ``--mode submit`` picks them up.

    python -m src.batch_run --mode submit --sizes 50 100 --batches 1 2 3
    python -m src.batch_run --mode poll
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import client, prompt                         # noqa: E402
from src.assays import load_assay_meta                 # noqa: E402
from src.run import shared_subset, RESULTS             # noqa: E402  (identical subsample)
from config.models import MODELS                       # noqa: E402

CID = re.compile(r"^c_(\d+)_(\d+)_(.+)$")
_META = None


def meta():
    global _META
    if _META is None:
        _META = load_assay_meta()
    return _META


def _anth():
    import anthropic
    return anthropic.Anthropic(api_key=client._key("ANTHROPIC_API_KEY"))


def _state_path(model):
    return RESULTS / model / "_batch" / "state.json"


def load_state(model):
    p = _state_path(model)
    return json.loads(p.read_text()) if p.exists() else {"model": model, "batches": []}


def save_state(model, state):
    p = _state_path(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def _prompt_for(m, assay, size, batch):
    """(user, ids, sub, ntok) exactly as src.run would build them, or None."""
    if assay not in m:
        return None
    sub = shared_subset(assay, size, batch)
    if not sub:
        return None
    user, ids = prompt.build_user_prompt(m[assay], m[assay]["reference_sequence"], sub)
    ntok = client.estimate_tokens(prompt.SYSTEM_PROMPT) + len(user) // 4
    return user, ids, sub, ntok


def _write_rec(cell_dir, assay, rec):
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / f"{assay}.json").write_text(json.dumps(rec, indent=2))


def build_cell_requests(model, size, batch):
    """Batch requests for one (size, batch) cell, skipping existing files. Overflow
    cells (prompt too big for ctx) are written directly, never submitted."""
    spec = MODELS[model]
    m = meta()
    cell_dir = RESULTS / model / f"n{size}" / f"b{batch}"
    reqs, n_over, n_skip = [], 0, 0
    for assay in sorted(m):
        if (cell_dir / f"{assay}.json").exists():
            n_skip += 1
            continue
        built = _prompt_for(m, assay, size, batch)
        if built is None:
            continue
        user, ids, _sub, ntok = built
        if ntok > spec["ctx"] - spec["max_tokens"]:            # mirror src.run pre-check
            _write_rec(cell_dir, assay, {"model": model, "assay": assay, "size": size,
                                         "batch": batch, "n": len(ids),
                                         "prompt_tokens_est": ntok, "overflow": True,
                                         "spearman": None})
            n_over += 1
            continue
        reqs.append({"custom_id": f"c_{size}_{batch}_{assay}",
                     "params": {"model": spec["model_id"], "max_tokens": spec["max_tokens"],
                                "system": prompt.SYSTEM_PROMPT,
                                "messages": [{"role": "user", "content": user}],
                                "thinking": {"type": "adaptive", "display": "summarized"}}})
    return reqs, n_over, n_skip


def process_result(res, model, size, batch, assay):
    """Return a result rec for a SUCCEEDED batch item (mirrors src.run's non-overflow
    rec), or None for a transient failure (errored/canceled/expired) so it is left
    unwritten and re-submitted next round."""
    m = meta()
    built = _prompt_for(m, assay, size, batch)
    if built is None:
        return None
    _user, ids, sub, ntok = built
    rec = {"model": model, "assay": assay, "size": size, "batch": batch,
           "n": len(ids), "prompt_tokens_est": ntok}
    r = res.result
    if r.type != "succeeded":
        return None
    msg = r.message
    stop = getattr(msg, "stop_reason", None)
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if stop == "refusal":                                  # safety refusal (empty text)
        err = "anthropic refusal (stop_reason=refusal)"
    elif not text.strip():
        err = "max_tokens (empty)" if stop == "max_tokens" else "empty response"
    else:
        err = None
    ranking = None if err else prompt.parse_ranking(text, ids)
    rho = prompt.score_ranking(ranking, ids, sub) if ranking else None
    rec.update({"overflow": False, "spearman": rho, "parsed": ranking is not None,
                "error": err, "elapsed_s": None, "ranking": ranking,
                "raw_output": text[:4000], "stop_reason": stop, "via": "batch"})
    return rec


def submit(model, sizes, batches):
    cli = _anth()
    state = load_state(model)
    for size in sizes:
        for batch in batches:
            reqs, n_over, n_skip = build_cell_requests(model, size, batch)
            if not reqs:
                print(f"n{size} b{batch}: nothing to submit "
                      f"(skip {n_skip} existing, {n_over} overflow written)")
                continue
            mb = cli.messages.batches.create(requests=reqs)
            state["batches"].append({"id": mb.id, "size": size, "batch": batch,
                                     "n": len(reqs), "done": False})
            save_state(model, state)
            print(f"n{size} b{batch}: submitted {len(reqs)} -> {mb.id}  "
                  f"(skip {n_skip} existing, {n_over} overflow)")
    print("submitted. poll with: python -m src.batch_run --mode poll")


def poll(model):
    cli = _anth()
    state = load_state(model)
    pending = 0
    for e in state["batches"]:
        if e.get("done"):
            continue
        mb = cli.messages.batches.retrieve(e["id"])
        rc = mb.request_counts
        print(f"{e['id']}  n{e['size']} b{e['batch']}  {mb.processing_status:12s} "
              f"proc={rc.processing} ok={rc.succeeded} err={rc.errored} "
              f"cancel={rc.canceled} exp={rc.expired}")
        if mb.processing_status != "ended":
            pending += 1
            continue
        w = skip = fail = 0
        for res in cli.messages.batches.results(e["id"]):
            mt = CID.match(res.custom_id)
            if not mt:
                continue
            size, batch, assay = int(mt.group(1)), int(mt.group(2)), mt.group(3)
            cell_dir = RESULTS / model / f"n{size}" / f"b{batch}"
            if (cell_dir / f"{assay}.json").exists():
                skip += 1
                continue
            rec = process_result(res, model, size, batch, assay)
            if rec is None:
                fail += 1
                continue
            _write_rec(cell_dir, assay, rec)
            w += 1
        e["done"] = True
        save_state(model, state)
        print(f"  -> wrote {w}, skipped {skip} (existing), left {fail} (transient; re-submit)")
    print("STATUS: all ended" if pending == 0 else f"STATUS: {pending} batch(es) still processing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["submit", "poll"], required=True)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--sizes", nargs="*", type=int, default=[50, 100])
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    if args.mode == "submit":
        submit(args.model, args.sizes, args.batches)
    else:
        poll(args.model)


if __name__ == "__main__":
    main()
