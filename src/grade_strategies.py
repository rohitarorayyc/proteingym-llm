"""Grade each raw reasoning trace against the faceted strategy taxonomy
(paper/strategy/grading_prompt.md). Single judge (gemini-3.5-flash), multi-label,
blind to accuracy. Reads raw reasoning_text from results_reasoning_audit/.

  python -m src.grade_strategies --test           # 3 traces incl. a p53 inverter, prints
  python -m src.grade_strategies                  # full ~597, saves paper/strategy/grades/
"""
from __future__ import annotations
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.models import MODELS
from src import client
from src.assays import load_assay_meta

META = load_assay_meta()
JUDGE = MODELS["gemini-3.5-flash"]
MODELS_AUDIT = ["gpt-5.5", "gemini-3.5-flash", "claude-opus-4.8"]
AUDIT = ROOT / "results_reasoning_audit"
OUT = ROOT / "paper" / "strategy" / "grades"
PROMPT = ROOT / "paper" / "strategy" / "grading_prompt.md"
AXES = ["biophysical_severity", "directionality", "structural_grounding",
        "knowledge_reliance", "beneficial_modeling", "output_form", "target_class", "quality_flags"]


def _block(header):
    txt = PROMPT.read_text()
    m = re.search(rf"## {header}\s*\n```\n(.*?)\n```", txt, re.DOTALL)
    if not m:
        raise SystemExit(f"could not extract {header} block from grading_prompt.md")
    return m.group(1)


SYSTEM = _block("SYSTEM")
USER_T = _block("USER")


def get_reasoning(model, assay):
    f = AUDIT / model / "n50" / "b1" / f"{assay}.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    rt = (d.get("reasoning_text") or "").strip()
    return (rt if len(rt) >= 300 else None), d.get("spearman")


def build_user(model, assay, trace):
    m = META.get(assay, {})
    prot = f"{m.get('target_name', assay)} ({m.get('organism', '')})"
    u = USER_T.replace("<<REAL_PROTEIN_NAME_AND_ORGANISM>>", prot)
    u = u.replace("<<fitness_description>>", m.get("fitness_description", ""))
    u = u.replace("<<gpt-5.5 | gemini-3.5-flash | claude-opus-4.8>>", model)
    u = u.replace("<<PASTE THE FULL reasoning_text HERE>>", trace[:30000])
    return u


def parse(txt):
    m = re.search(r"\{.*\}", txt or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj if all(k in obj for k in AXES) else None


def grade_one(model, assay, save=True):
    out = OUT / model / f"{assay}.json"
    if save and out.exists():
        return None
    trace, rho = get_reasoning(model, assay)
    if not trace:
        return {"_skip": f"{model}/{assay}"}
    user = build_user(model, assay, trace)
    labels = None
    for attempt in range(2):
        r = client.chat(JUDGE, SYSTEM, user, timeout=120, retries=2)
        labels = parse(r.get("text") or "")
        if labels:
            break
        time.sleep(2)
    rec = {"model": model, "assay": assay, "spearman": rho, "labels": labels,
           "error": None if labels else "parse_fail"}
    if save and labels:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
    return rec


def corpus():
    items = []
    for m in MODELS_AUDIT:
        d = AUDIT / m / "n50" / "b1"
        if d.exists():
            for f in sorted(d.glob("*.json")):
                t, _ = get_reasoning(m, f.stem)
                if t:
                    items.append((m, f.stem))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.test:
        p53 = next((f.stem for f in (AUDIT / "gpt-5.5" / "n50" / "b1").glob("P53_HUMAN*")), None)
        picks = [("gpt-5.5", p53)] if p53 else []
        gem = next((f.stem for f in (AUDIT / "gemini-3.5-flash" / "n50" / "b1").glob("P53_HUMAN*")), None)
        if gem:
            picks.append(("gemini-3.5-flash", gem))
        picks.append(("claude-opus-4.8", next(f.stem for f in (AUDIT / "claude-opus-4.8" / "n50" / "b1").glob("*.json"))))
        for m, asy in picks:
            o = grade_one(m, asy, save=False)
            print("\n" + "=" * 80)
            print(f"{m} | {asy} | rho={round(o.get('spearman') or 0, 3)}")
            print(json.dumps(o.get("labels"), indent=2) if o.get("labels") else f"  {o}")
        return
    items = corpus()
    print(f"grading {len(items)} traces with {JUDGE['model_id']}...")
    done = ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(grade_one, m, asy): (m, asy) for m, asy in items}
        for f in as_completed(futs):
            r = f.result(); done += 1
            if r and r.get("labels"):
                ok += 1
            if done % 50 == 0:
                print(f"  {done}/{len(items)} ({ok} labeled)")
    print(f"done: {done} processed, {ok} labeled")


if __name__ == "__main__":
    main()
