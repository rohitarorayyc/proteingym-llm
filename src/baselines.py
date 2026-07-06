"""Score the published ProteinGym baselines on the EXACT SAME frozen subsamples
the LLMs see — apples-to-apples.

For each (assay, size, batch) frozen split we look up each baseline model's
per-variant score (data/baselines/zero_shot_substitutions_scores/<assay>.csv),
restrict to the split's variant ids, and compute Spearman vs the held-out DMS
scores — the identical metric used for the LLM rankings. A baseline column with
missing/empty values on a subset is skipped for that cell.

Output (mirrors results/ for the LLMs):
  results_baselines/<baseline>/n<size>/b<batch>/<assay>.json  -> {spearman, n}
  results_baselines/summary.json                              -> per-baseline mean rho by size

  python -m src.baselines
  python -m src.baselines --baselines GEMME "ESM2 (650M)" --sizes 10 50
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prompt import spearman                                         # noqa: E402
from src.assays import load_assay_meta                                  # noqa: E402
from src.aggregate import nested_macro, macro_headline, SELECTION_TYPES # noqa: E402
from config.models import SIZES, N_BATCHES                              # noqa: E402

SPLITS = ROOT / "data" / "splits"
SCORES = ROOT / "data" / "baselines" / "zero_shot_substitutions_scores"
OUT = ROOT / "results_baselines"
# columns in the baseline CSVs that are not model scores. Any DMS_*-prefixed
# column is an experimental label (incl. the binarized label, which one assay
# mis-names "DMS_bin_score") and must never be treated as a predictor.
NON_MODEL = {"mutant", "mutated_sequence"}


def _is_model_col(c):
    return c not in NON_MODEL and not c.startswith("DMS_")


def _baseline_csv(assay):
    p = SCORES / f"{assay}.csv"
    return p if p.exists() else None


def load_baseline_scores(assay):
    """-> (rows_by_mutant, model_columns). rows_by_mutant[id][col] = float|None."""
    p = _baseline_csv(assay)
    if not p:
        return None, []
    rows, cols = {}, None
    for r in csv.DictReader(open(p)):
        if cols is None:
            cols = [c for c in r if _is_model_col(c)]
        rows[r["mutant"]] = r
    return rows, (cols or [])


def load_split(assay, size, batch):
    sp = SPLITS / assay / f"n{size}_b{batch}.json"
    lp = SPLITS / assay / f"n{size}_b{batch}.labels.json"
    if not sp.exists() or not lp.exists():
        return None
    ids = [v["id"] for v in json.loads(sp.read_text())["variants"]]
    labels = json.loads(lp.read_text())
    return ids, labels


def score_cell(assay, size, batch, only_baselines=None):
    """-> {baseline: (rho, n_scored)} for one (assay,size,batch) on the matched subset.

    n_scored = variants with a usable baseline value (a baseline missing some of the
    subset's variants is scored on the rest — the same dropna treatment ProteinGym's
    own per-assay Spearman uses); recorded per cell so coverage is visible."""
    split = load_split(assay, size, batch)
    rows, cols = load_baseline_scores(assay)
    if not split or rows is None:
        return {}
    ids, labels = split
    cols = [c for c in cols if (not only_baselines or c in only_baselines)]
    dms = [labels[i] for i in ids]
    out = {}
    for col in cols:
        vals, ref = [], []
        for i, d in zip(ids, dms):
            cell = rows.get(i, {}).get(col, "")
            try:
                fval = float(cell)
            except (TypeError, ValueError):
                continue                              # missing baseline value -> drop
            if not math.isfinite(fval):               # nan/inf -> drop (keeps vals/ref aligned)
                continue
            vals.append(fval); ref.append(d)
        if len(vals) >= 2:
            rho = spearman(ref, vals)
            if rho is not None:
                out[col] = (rho, len(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="*")
    ap.add_argument("--baselines", nargs="*", help="subset of baseline columns")
    ap.add_argument("--sizes", nargs="*", type=int, default=SIZES)
    args = ap.parse_args()

    assays = args.assays or sorted(p.name for p in SPLITS.iterdir() if p.is_dir())
    only = set(args.baselines) if args.baselines else None
    meta = load_assay_meta()
    OUT.mkdir(parents=True, exist_ok=True)
    # agg[(baseline, size)][assay] = list of rho across batches (one per batch)
    agg = defaultdict(lambda: defaultdict(list))
    n_cells = 0
    for assay in assays:
        for size in args.sizes:
            for batch in range(1, N_BATCHES + 1):
                res = score_cell(assay, size, batch, only)
                if not res:
                    continue
                n_cells += 1
                for bl, (rho, n_scored) in res.items():
                    safe = bl.replace("/", "_").replace(" ", "_")
                    d = OUT / safe / f"n{size}" / f"b{batch}"
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{assay}.json").write_text(json.dumps(
                        {"baseline": bl, "assay": assay, "size": size,
                         "batch": batch, "spearman": rho, "n": n_scored}))
                    agg[(bl, size)][assay].append(rho)

    # Aggregate with the official ProteinGym nested macro (UniProt -> selection-type
    # -> mean of 5), same as analyze.py, so model vs baseline is apples-to-apples.
    # Flat mean over assays is kept alongside for reference.
    summary = {}
    for (bl, size), assay_map in sorted(agg.items()):
        assay_rho = {a: st.mean(v) for a, v in assay_map.items() if v}
        by_sel = nested_macro(assay_rho, meta, "function")
        macro = macro_headline(by_sel)
        n_uni = len({meta[a]["uniprot_id"] for a in assay_rho
                     if a in meta and meta[a]["uniprot_id"]})
        summary.setdefault(bl, {})[f"n{size}"] = {
            "mean_rho_macro": round(macro, 4) if macro is not None else None,
            "mean_rho_flat": round(st.mean(assay_rho.values()), 4) if assay_rho else None,
            "n_assays": len(assay_rho), "n_uniprot": n_uni,
            "by_selection": {g: round(v["mean_rho"], 4) for g, v in by_sel.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # print a compact leaderboard at the largest requested size (by nested macro)
    if args.sizes:
        big = max(args.sizes)
        board = sorted(((bl, v.get(f"n{big}", {}).get("mean_rho_macro"))
                        for bl, v in summary.items()
                        if v.get(f"n{big}")), key=lambda x: -(x[1] if x[1] is not None else -9))
        print(f"\nBaseline nested-macro Spearman @ n{big} (matched subsamples, {n_cells} cells scored):")
        for bl, rho in board[:15]:
            print(f"  {bl:34s} {rho:+.3f}" if rho is not None else f"  {bl:34s}   -- ")
    print(f"\n-> {OUT}/<baseline>/n<size>/b<batch>/<assay>.json  + summary.json")


if __name__ == "__main__":
    main()
