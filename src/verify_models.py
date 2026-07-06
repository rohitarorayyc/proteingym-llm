"""Connectivity check: send a tiny prompt to each model and report OK/FAIL.
Validates keys + native model_ids + provider paths. Run before a big run.

    python -m src.verify_models
    python -m src.verify_models --models gemini-3.5-flash gpt-5.5
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import client            # noqa: E402
from config.models import MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    for k in (args.models or list(MODELS)):
        spec = MODELS[k]
        tag = f"{spec['provider']}/{spec['model_id']}"
        r = client.chat(spec, "You are a connectivity test.",
                        "Reply with the single word: OK", timeout=180, retries=1)
        if r["error"]:
            print(f"FAIL  {k:22s} {tag:42s} {r['error'][:110]}")
        else:
            print(f"OK    {k:22s} {tag:42s} ({r['elapsed_s']}s) {r['text'][:30]!r}")


if __name__ == "__main__":
    main()
