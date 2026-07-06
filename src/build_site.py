"""Regenerate docs/data.json (the static site's data) from the results tree.

Computes, per primary (model, size): one assay score averaged across seeds, then
overall nested-macro rho + +/- SEM, and nested-macro breakdowns by function,
taxon, and mutation depth. Sensitivity and nonuniform runs are kept in separate
JSON sections so they are not mixed into the primary leaderboard.

    python -m src.build_site            # uses results/ + results_baselines/

GitHub Pages / Vercel serves docs/.
"""
from __future__ import annotations
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aggregate import nested_macro, macro_headline      # noqa: E402
from src.assays import load_assay_meta, assay_csv           # noqa: E402
from src.subsample import load_split as _load_split         # noqa: E402
from src.metrics import roc_auc, mcc as _mcc, recall_topk   # noqa: E402
from src.prompt import spearman                             # noqa: E402

DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
NONUNIFORM_RESULTS = ROOT / "results_nonuniform" / "top5_or_top50"
SCORES = ROOT / "data" / "baselines" / "zero_shot_substitutions_scores"
SPLITS = ROOT / "data" / "splits"
SIZES = [10, 50, 100]
BATCHES = [1, 2, 3]
PRIMARY_MODELS = [
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "claude-opus-4.8",
    "claude-opus-4.7",
    "claude-sonnet-5",
    "glm-5.2",
    "gemini-3.5-flash",
    "gemini-3.1-pro",
    "gemini-3.1-flash-lite",
]
GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.1-pro", "gemini-3.1-flash-lite"]
EFFORT_GPT55 = ["gpt-5.5-low", "gpt-5.5-medium", "gpt-5.5", "gpt-5.5-xhigh"]
EFFORT_GEMINI35_FLASH = [
    "gemini-3.5-flash-low",
    "gemini-3.5-flash-medium",
    "gemini-3.5-flash",
]
MODEL_DETAILS = {
    "gpt-5.5": ("OpenAI", "high reasoning", "https://platform.openai.com/docs/models"),
    "gpt-5.4-mini": ("OpenAI", "high reasoning", "https://platform.openai.com/docs/models"),
    "gpt-5.4-nano": ("OpenAI", "high reasoning", "https://platform.openai.com/docs/models"),
    "claude-opus-4.8": ("Anthropic", "high reasoning", "https://docs.anthropic.com/en/docs/about-claude/models/overview"),
    "claude-opus-4.7": ("Anthropic", "high reasoning", "https://docs.anthropic.com/en/docs/about-claude/models/overview"),
    "claude-sonnet-5": ("Anthropic", "high reasoning", "https://docs.anthropic.com/en/docs/about-claude/models/overview"),
    "glm-5.2": ("DeepInfra", "GLM 5.2, high reasoning", "https://deepinfra.com/models/text-generation"),
    "gemini-3.5-flash": ("Google", "high thinking", "https://ai.google.dev/gemini-api/docs/models"),
    "gemini-3.1-pro": ("Google", "high thinking", "https://ai.google.dev/gemini-api/docs/models"),
    "gemini-3.1-flash-lite": ("Google", "high thinking", "https://ai.google.dev/gemini-api/docs/models"),
    "gpt-5.5-low": ("OpenAI", "low reasoning", "https://platform.openai.com/docs/models"),
    "gpt-5.5-medium": ("OpenAI", "medium reasoning", "https://platform.openai.com/docs/models"),
    "gpt-5.5-xhigh": ("OpenAI", "xhigh reasoning", "https://platform.openai.com/docs/models"),
    "gemini-3.5-flash-low": ("Google", "low thinking", "https://ai.google.dev/gemini-api/docs/models"),
    "gemini-3.5-flash-medium": ("Google", "medium thinking", "https://ai.google.dev/gemini-api/docs/models"),
}
# the three breakdown axes -> meta field each groups by
AXES = {"function": "function", "taxon": "taxon", "depth": "depth"}
# the four per-assay metrics reported on the leaderboard (ProteinGym parity):
# rank correlation + ROC AUC + Matthews corr + recall of the true top-10%.
METRICS = ["spearman", "auc", "mcc", "recall10"]
_BIN_CACHE: dict[str, dict] = {}   # assay -> {variant_id: 0/1}  (ProteinGym DMS_score_bin)


def _f(x):
    try:
        v = float(x)
        return round(v, 4) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _meta_with_depth():
    meta = load_assay_meta()
    for m in meta.values():
        m["depth"] = "Multi-mutant" if m.get("multi") else "Single-mutant"
    return meta


def _breakdowns(assay_rho, meta):
    """-> {overall, by_function, by_taxon, by_depth} of macro ρ."""
    out = {}
    for axis, key in AXES.items():
        groups = nested_macro(assay_rho, meta, key)        # {group: {mean_rho,...}}
        out[axis] = {g: _f(v["mean_rho"]) for g, v in groups.items()}
    out["overall"] = _f(macro_headline(nested_macro(assay_rho, meta, "function")))
    return out


def _metric_block(am, per_batch, meta, metric):
    """The full leaderboard block for ONE metric of ONE entity:
    {overall, function, taxon, depth, sem, seed_sem, n_assays}. `am` is
    {assay: {metric: batch-mean}}; `per_batch` is {batch: {assay: {metric: val}}}
    (drives the across-seed SE). None if the metric has no scored assay."""
    assay_map = {a: m[metric] for a, m in am.items() if m.get(metric) is not None}
    if not assay_map:
        return None
    bd = _breakdowns(assay_map, meta)
    vals = list(assay_map.values())
    bd["sem"] = _f(st.pstdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else None
    bd["n_assays"] = len(assay_map)
    seed = []
    for _batch, amap in sorted(per_batch.items()):
        bm = {a: mm[metric] for a, mm in amap.items() if mm.get(metric) is not None}
        o = _breakdowns(bm, meta)["overall"] if bm else None
        if o is not None:
            seed.append(o)
    bd["seed_sem"] = _f(st.stdev(seed) / math.sqrt(len(seed))) if len(seed) > 1 else None
    bd["seed_sd"] = _f(st.stdev(seed)) if len(seed) > 1 else None
    return bd


def _bin_map(assay):
    """{variant_id: 0/1} from the assay CSV's DMS_score_bin column (cached)."""
    if assay not in _BIN_CACHE:
        import csv
        p = assay_csv(assay)
        m = {}
        if p.exists():
            for r in csv.DictReader(open(p)):
                try:
                    m[r["mutant"]] = int(float(r.get("DMS_score_bin", "")))
                except (TypeError, ValueError):
                    pass
        _BIN_CACHE[assay] = m
    return _BIN_CACHE[assay]


def _metrics_from_pred(pred, y_cont, y_bin):
    """The three classification/retrieval metrics from aligned per-variant arrays
    (pred higher = predicted fitter). Spearman is carried separately."""
    pairs = [(p, b) for p, b in zip(pred, y_bin) if b is not None]
    auc = mcc = None
    if pairs:
        pv = [p for p, _ in pairs]
        bv = [b for _, b in pairs]
        auc = roc_auc(bv, pv)
        mcc = _mcc(bv, pv)
    return {"auc": auc, "mcc": mcc, "recall10": recall_topk(y_cont, pred)}


def _llm_cell_metrics(data):
    """All four metrics for one scored LLM cell (uses the stored ranking + the
    frozen split to recover each variant's DMS score/bin). Falls back to
    spearman-only if the split is unavailable (e.g. sensitivity runs)."""
    out = {"spearman": float(data["spearman"])}
    ranking = data.get("ranking")
    sub = _load_split(data["assay"], data["size"], data["batch"]) if ranking else None
    if not sub:
        return out
    n = len(sub)
    pos = {mid: i for i, mid in enumerate(ranking)}
    ids = [f"M{i + 1:02d}" for i in range(n)]
    pred = [-pos.get(ids[i], n) for i in range(n)]        # higher = predicted fitter
    y_cont = [sub[i][2] for i in range(n)]
    bmap = _bin_map(data["assay"])
    y_bin = [bmap.get(sub[i][0]) for i in range(n)]
    out.update(_metrics_from_pred(pred, y_cont, y_bin))
    return out


def _record_pa(pa, size, assay_metrics, name, kind, meta):
    """Stash one entity's per-assay metrics into the individual-view accumulator."""
    if pa is None:
        return
    d = pa.setdefault(str(size), {})
    for a, m in assay_metrics.items():
        if a not in meta:
            continue
        d.setdefault(a, {})[name] = {
            "s": _f(m.get("spearman")), "a": _f(m.get("auc")),
            "m": _f(m.get("mcc")), "r": _f(m.get("recall10")), "k": kind}


def _read_json(path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"_invalid_json": True}
    return data if isinstance(data, dict) else {"_invalid_json": True}


def _cell_status(data):
    if data is None:
        return "missing"
    if data.get("_invalid_json"):
        return "invalid_json"
    if data.get("overflow"):
        return "overflow"
    if data.get("error"):
        return "error"
    if data.get("spearman") is None:
        return "no_spearman"
    return "ok"


def _llm_assay_metrics(model, size, meta, *, root=RESULTS, batches=BATCHES, reconstruct=True):
    """({assay: {metric: batch-mean}}, cell_counts, {batch: {assay: {metric: val}}})
    for one model/size. reconstruct=False keeps only spearman (used for sensitivity /
    non-uniform runs whose frozen split does not match the standard subsample)."""
    by_assay = defaultdict(lambda: defaultdict(list))       # assay -> metric -> [vals]
    per_batch = defaultdict(dict)                            # batch -> {assay: {metric: val}}
    counts = Counter()
    counts["expected"] = len(meta) * len(batches)
    for batch in batches:
        cell = root / model / f"n{size}" / f"b{batch}"
        for assay in meta:
            data = _read_json(cell / f"{assay}.json")
            status = _cell_status(data)
            counts[status] += 1
            if status != "missing":
                counts["present"] += 1
            if status == "ok":
                cm = _llm_cell_metrics(data) if reconstruct else {"spearman": float(data["spearman"])}
                per_batch[batch][assay] = {k: v for k, v in cm.items() if v is not None}
                for k, v in cm.items():
                    if v is not None:
                        by_assay[assay][k].append(v)
    out = {a: {k: st.mean(vs) for k, vs in md.items() if vs} for a, md in by_assay.items()}
    return out, dict(counts), dict(per_batch)


def _entry_from_metrics(am, per_batch, meta):
    """Shared LLM/baseline assembly: nested per-metric blocks + top-level spearman
    fields (kept flat for back-compat). None if no spearman survives."""
    metrics = {mt: _metric_block(am, per_batch, meta, mt) for mt in METRICS}
    metrics = {mt: b for mt, b in metrics.items() if b}
    if "spearman" not in metrics:
        return None
    bd = dict(metrics["spearman"])          # overall/function/taxon/depth/sem/seed_sem
    bd["metrics"] = metrics
    return bd


def _model_size_entry(model, size, meta, *, root=RESULTS, batches=BATCHES,
                      pa=None, kind="llm", reconstruct=True):
    am, counts, per_batch = _llm_assay_metrics(
        model, size, meta, root=root, batches=batches, reconstruct=reconstruct)
    bd = _entry_from_metrics(am, per_batch, meta)
    if bd is None:
        return None
    bd["batches"] = batches
    bd["cell_counts"] = counts
    provider, desc, url = MODEL_DETAILS.get(model, ("", "", ""))
    bd["provider"] = provider
    bd["description"] = desc
    bd["model_url"] = url
    _record_pa(pa, size, am, model, kind, meta)
    return bd


def _model_series(models, sizes, meta, *, root=RESULTS, batches=BATCHES,
                  pa=None, reconstruct=True):
    out = {}
    for model in models:
        per_size = {}
        for size in sizes:
            entry = _model_size_entry(model, size, meta, root=root, batches=batches,
                                      pa=pa, reconstruct=reconstruct)
            if entry:
                per_size[str(size)] = entry
        if per_size:
            out[model] = per_size
    return out


def main():
    meta = _meta_with_depth()

    # assay counts per group (so the site can show n=X under each column)
    fn_c = Counter(m["function"] for m in meta.values())
    tax_c = Counter(m["taxon"] for m in meta.values())
    dep_c = Counter(m["depth"].split("-")[0] for m in meta.values())  # Single / Multi

    data = {"metric": "ProteinGym nested-macro Spearman", "metrics": METRICS,
            "batches": BATCHES, "sizes": SIZES,
            "n_assays": len(meta), "models": {}, "baselines": {}, "best_baseline": {},
            "subanalyses": {},
            "function_counts": dict(fn_c), "taxon_counts": dict(tax_c),
            "depth_counts": dict(dep_c)}

    pa = {}   # per-assay individual-view accumulator: {size: {assay: {entity: {...}}}}

    # ---- LLMs ----
    data["models"] = _model_series(PRIMARY_MODELS, SIZES, meta, pa=pa)

    # ---- subanalyses (spearman only — their subsamples don't match the standard
    #      frozen split, so the classification metrics aren't reconstructable) ----
    data["subanalyses"]["gemini_n500_existing"] = _model_series(
        GEMINI_MODELS, [500], meta, batches=[1], reconstruct=False)
    data["subanalyses"]["effort_sensitivity"] = {
        "gpt-5.5": _model_series(EFFORT_GPT55, [50], meta, batches=[1], reconstruct=False),
        "gemini-3.5-flash": _model_series(EFFORT_GEMINI35_FLASH, [50], meta, batches=[1], reconstruct=False),
    }
    data["subanalyses"]["gemini35_flash_nonuniform_top5_or_top50"] = _model_series(
        ["gemini-3.5-flash"], [50], meta, root=NONUNIFORM_RESULTS, batches=BATCHES, reconstruct=False)

    # ---- baselines: recompute breakdowns on OUR frozen splits, then attach
    #      ProteinGym's published description + reference (joined by model name) ----
    data["baselines"] = _baseline_breakdowns(meta, pa=pa)
    _attach_pg_metadata(data["baselines"])
    _attach_baseline_grp(data["baselines"])

    # best baseline per size (by overall macro) for the summary chart
    for s in SIZES:
        best = None
        for bl, entry in data["baselines"].items():
            size_entry = entry["sizes"].get(str(s)) or entry["sizes"].get(s, {})
            v = size_entry.get("overall")
            if v is not None and (best is None or v > best[1]):
                best = (bl, v)
        if best:
            data["best_baseline"][s] = {"name": best[0], "macro": best[1]}

    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(data, indent=2))
    print(f"wrote {DOCS/'data.json'} — {len(data['models'])} models, "
          f"{len(data['baselines'])} baselines, axes={list(AXES)}")

    # ---- per-assay individual view (lazy-loaded by the benchmark page) ----
    pa_meta = {a: {"name": meta[a]["target_name"], "uniprot": meta[a]["uniprot_id"],
                   "function": meta[a]["function"], "taxon": meta[a]["taxon"],
                   "depth": meta[a]["depth"], "msa": meta[a]["msa_category"],
                   "n_mut": len(_bin_map(a))}
               for a in {a for sz in pa.values() for a in sz}}
    per_assay = {"metrics": METRICS, "sizes": SIZES, "keys": {"s": "spearman",
                 "a": "auc", "m": "mcc", "r": "recall10"},
                 "assays": pa_meta, "data": pa}
    (DOCS / "per_assay.json").write_text(json.dumps(per_assay, separators=(",", ":")))
    kb = (DOCS / "per_assay.json").stat().st_size / 1024
    print(f"wrote {DOCS/'per_assay.json'} — {len(pa_meta)} assays, {kb:.0f} KB")


def _baseline_breakdowns(meta, pa=None):
    """For each baseline column, per size: overall + by function/taxon/depth macro ρ,
    scored on the frozen splits (identical subsamples to the LLMs), averaged over
    the same primary batches before nested-macro aggregation.

    Assay-outer so each per-variant CSV is read once: per assay, load the CSV +
    the splits, then score every baseline column at every size from memory.
    """
    from src.baselines import load_baseline_scores, load_split

    assays = [p.name for p in SPLITS.iterdir() if p.is_dir()] if SPLITS.exists() else []
    if not assays or not SCORES.exists():
        return {}
    # met[bl][size][assay][metric] = [value per batch]; per_batch drives seed SE
    met = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    per_batch = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # bl->size->batch->{assay:{metric}}
    for a in sorted(assays):
        if a not in meta:
            continue
        rows, cols = load_baseline_scores(a)
        if rows is None:
            continue
        bmap = _bin_map(a)
        for s in SIZES:
            for batch in BATCHES:
                split = load_split(a, s, batch)
                if not split:
                    continue
                ids, labels = split
                for bl in cols:
                    ref, val, binl = [], [], []
                    for i in ids:
                        try:
                            v = float(rows.get(i, {}).get(bl, ""))
                            if math.isfinite(v):
                                val.append(v); ref.append(labels[i]); binl.append(bmap.get(i))
                        except (TypeError, ValueError):
                            pass
                    if len(val) < 2:
                        continue
                    r = spearman(ref, val)
                    if r is None or not math.isfinite(r):
                        continue
                    cm = {"spearman": r, **_metrics_from_pred(val, ref, binl)}  # baseline value = predictor
                    per_batch[bl][s][batch][a] = {k: v for k, v in cm.items() if v is not None}
                    for k, v in cm.items():
                        if v is not None:
                            met[bl][s][a][k].append(v)

    out = {}
    for bl, by_size in met.items():
        sizes = {}
        for s, assay_map in by_size.items():
            am = {a: {k: st.mean(vs) for k, vs in md.items() if vs}
                  for a, md in assay_map.items()}
            bd = _entry_from_metrics(am, per_batch[bl][s], meta)
            if bd is None:
                continue
            sizes[str(s)] = bd
            _record_pa(pa, s, am, bl, "base", meta)
        out[bl] = {"sizes": sizes, "description": "", "ref_url": "", "type": ""}
    return out


def _norm(s):
    import re
    return re.sub(r"[()\s_\-]+", "", s.lower())


# stragglers our column names -> ProteinGym display names (normalization misses these)
_PG_ALIAS = {
    "prosst-20": "ProSST (K=20)", "prosst-128": "ProSST (K=128)", "prosst-512": "ProSST (K=512)",
    "prosst-1024": "ProSST (K=1024)", "prosst-2048": "ProSST (K=2048)", "prosst-4096": "ProSST (K=4096)",
    "progen2_small": "Progen2 S", "progen2_medium": "Progen2 M", "progen2_large": "Progen2 L",
    "progen2_xlarge": "Progen2 XL", "mulan_small": "MULAN", "unirep_evotune": "Unirep evotuned",
    "saprot_650m_af2": "SaProt (650M)", "saprot_35m_af2": "SaProt (35M)", "poet": "PoET (200M)",
    "esm3": "ESM3 open (1.4B)",
    "protssn_k10_h512": "ProtSSN (k=10, h=512)", "protssn_k10_h768": "ProtSSN (k=10, h=768)",
    "protssn_k10_h1280": "ProtSSN (k=10, h=1280)", "protssn_k20_h512": "ProtSSN (k=20, h=512)",
    "protssn_k20_h768": "ProtSSN (k=20, h=768)", "protssn_k20_h1280": "ProtSSN (k=20, h=1280)",
    "protssn_k30_h512": "ProtSSN (k=30, h=512)", "protssn_k30_h768": "ProtSSN (k=30, h=768)",
    "protssn_k30_h1280": "ProtSSN (k=30, h=1280)",
}


def _attach_pg_metadata(baselines):
    """Join ProteinGym's published description/reference/type onto our recomputed
    baselines (matched by normalized name, with an alias map for stragglers)."""
    pg_path = DOCS / "pg_baselines.json"
    if not pg_path.exists():
        print("  (no docs/pg_baselines.json — run src.scrape_proteingym; skipping metadata)")
        return
    pg = json.loads(pg_path.read_text())
    by_norm = {_norm(b["name"]): b for b in pg}
    by_name = {b["name"]: b for b in pg}
    hit = 0
    for bl, entry in baselines.items():
        match = by_norm.get(_norm(bl)) or by_name.get(_PG_ALIAS.get(bl.lower(), ""))
        if match:
            entry["description"] = match.get("description", "")
            entry["ref_url"] = match.get("ref_url", "")
            entry["type"] = match.get("type", "")
            hit += 1
    print(f"  attached ProteinGym metadata to {hit}/{len(baselines)} baselines")


def _attach_baseline_grp(baselines):
    """Attach the paper's single_seq / msa_struct class to each baseline (from
    baseline_types.json, the same split Figure 1 uses; falls back to the
    ProteinGym type string if that artifact is absent)."""
    bt_path = ROOT / "paper" / "analysis" / "out" / "baseline_types.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else {}
    for name, entry in baselines.items():
        grp = (bt.get(name) or {}).get("grp")
        if not grp:
            grp = "single_seq" if (entry.get("type") or "").strip() == "Single sequence" else "msa_struct"
        entry["grp"] = grp
    print(f"  attached single_seq/msa_struct class to {len(baselines)} baselines")


if __name__ == "__main__":
    main()
