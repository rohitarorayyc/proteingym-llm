"""Scrape the official ProteinGym DMS-substitution zero-shot leaderboard.

Downloads `Summary_performance_DMS_substitutions_Spearman.csv` (the exact table
proteingym.org/benchmarks renders) and writes docs/pg_baselines.json with the
columns our benchmark page mirrors: overall Spearman, by-function (5), by-taxon
(4), mutation depth (mapped to Single = Depth_1, Multi = mean of Depth 2..5+),
model type / input modalities, description, references.

These are ProteinGym's PUBLISHED FULL-ASSAY numbers (size-independent) — shown as
reference rows alongside our subsampled LLM scores.

    python -m src.scrape_proteingym
"""
from __future__ import annotations
import csv
import json
import re
import ssl
import urllib.request
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
URL = ("https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/"
       "DMS_zero_shot/substitutions/Spearman/Summary_performance_DMS_substitutions_Spearman.csv")

FUNCTION = {"Activity": "Function_Activity", "Binding": "Function_Binding",
            "Expression": "Function_Expression", "OrganismalFitness": "Function_OrganismalFitness",
            "Stability": "Function_Stability"}
TAXON = {"Human": "Taxa_Human", "Other Eukaryote": "Taxa_Other_Eukaryote",
         "Prokaryote": "Taxa_Prokaryote", "Virus": "Taxa_Virus"}
MULTI_COLS = ["Depth_2", "Depth_3", "Depth_4", "Depth_5+"]


def _f(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


def _ref_url(cell):
    """Pull just the first href out of ProteinGym's <a href='...'>citation</a>."""
    m = re.search(r"href=['\"]([^'\"]+)['\"]", cell or "")
    return m.group(1) if m else ""


def main():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(URL, headers={"User-Agent": "pg-agent"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        text = r.read().decode("utf-8", "replace")

    out = []
    for row in csv.DictReader(StringIO(text)):
        multi = [_f(row.get(c)) for c in MULTI_COLS]
        multi = [v for v in multi if v is not None]
        out.append({
            "name": (row.get("Model_name") or "").strip(),
            "type": (row.get("Model type") or "").strip(),
            "overall": _f(row.get("Average_Spearman")),
            "function": {k: _f(row.get(col)) for k, col in FUNCTION.items()},
            "taxon": {k: _f(row.get(col)) for k, col in TAXON.items()},
            "depth": {"Single": _f(row.get("Depth_1")),
                      "Multi": round(sum(multi) / len(multi), 4) if multi else None},
            "description": (row.get("Model details") or "").strip(),
            "ref_url": _ref_url(row.get("References")),
        })
    DOCS.mkdir(exist_ok=True)
    (DOCS / "pg_baselines.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {DOCS/'pg_baselines.json'} — {len(out)} ProteinGym baselines")


if __name__ == "__main__":
    main()
