"""Re-attach to an already-submitted Opus reasoning-audit batch and collect it
when it ends. Network-robust polling (a dropped connection won't kill it).
Does NOT re-submit. Reads the batch id from disk.

  python -m src.collect_audit_batch
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import anthropic
from src.client import _key
from src.assays import load_assay_meta
from src.run_reasoning_audit import build, score, OUT, SIZE, BATCH, PROVIDER_MODEL

MODEL = PROVIDER_MODEL["anthropic"]


def main():
    bid = (OUT / MODEL / "batch_id.txt").read_text().strip()
    cli = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))
    meta = load_assay_meta()
    odir = OUT / MODEL / f"n{SIZE}" / f"b{BATCH}"; odir.mkdir(parents=True, exist_ok=True)
    idmap = {}
    for a in sorted(meta):
        u, ids, sub = build(a, meta)
        if u:
            idmap[a] = (ids, sub)
    print(f"attached to batch {bid}; polling...")
    while True:
        try:
            b = cli.messages.batches.retrieve(bid)
        except Exception as e:  # noqa: BLE001 — survive network blips
            print(f"  poll error ({type(e).__name__}); retrying in 30s"); time.sleep(30); continue
        c = b.request_counts
        print(f"  status={b.processing_status} succeeded={c.succeeded} errored={c.errored} processing={c.processing}")
        if b.processing_status == "ended":
            break
        time.sleep(30)
    n_ok = 0
    for attempt in range(5):
        try:
            for res in cli.messages.batches.results(bid):
                a = res.custom_id; ids, sub = idmap.get(a, (None, None))
                rec = {"model": MODEL, "assay": a, "size": SIZE, "batch": BATCH,
                       "provider": "anthropic", "n": len(ids) if ids else None}
                if res.result.type == "succeeded":
                    msg = res.result.message
                    ans = "".join(getattr(x, "text", "") for x in msg.content if getattr(x, "type", "") == "text")
                    think = "".join(getattr(x, "thinking", "") for x in msg.content if getattr(x, "type", "") == "thinking")
                    ranking, rho = score(ans, ids, sub)
                    rec.update({"spearman": rho, "parsed": ranking is not None, "ranking": ranking,
                                "answer_text": ans, "reasoning_text": think, "reasoning_len": len(think),
                                "error": None})
                    if think:
                        n_ok += 1
                else:
                    rec.update({"spearman": None, "parsed": False, "error": f"batch_{res.result.type}"})
                (odir / f"{a}.json").write_text(json.dumps(rec, indent=2))
            break
        except Exception as e:  # noqa: BLE001
            print(f"  results fetch error ({type(e).__name__}); retry {attempt+1}/5 in 20s"); time.sleep(20)
    print(f"collected; {n_ok} cells with thinking captured")


if __name__ == "__main__":
    main()
