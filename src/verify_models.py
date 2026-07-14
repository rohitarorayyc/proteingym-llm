"""Connectivity check: send a tiny prompt to each model and report OK/FAIL.
Validates keys + native model_ids + provider paths. Run before a big run.

    python -m src.verify_models
    python -m src.verify_models --models gemini-3.5-flash gpt-5.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.models import PRIMARY_MODELS, load_model_registry  # noqa: E402
from src import client  # noqa: E402

PROBE_MAX_OUTPUT_TOKENS = 256
PROBE_REASONING_EFFORT = "low"


def _probe_spec(spec: dict, *, canonical_settings: bool) -> dict:
    """Bound connectivity checks without changing the provider request shape."""
    if canonical_settings:
        return dict(spec)
    probe = dict(spec)
    probe["reasoning"] = PROBE_REASONING_EFFORT
    probe["max_tokens"] = min(PROBE_MAX_OUTPUT_TOKENS, spec["max_tokens"])
    return probe


def _probe_error(response: dict) -> str | None:
    if response.get("error"):
        return str(response["error"])
    status_values = (
        str(response.get("status") or "").lower(),
        str(response.get("incomplete_reason") or "").lower(),
        str(response.get("stop_reason") or "").lower(),
    )
    if any(
        marker in value
        for value in status_values
        for marker in ("incomplete", "max_tokens", "max_output_tokens", "length")
    ):
        return "probe response was truncated or incomplete"
    answer = str(response.get("text") or "").strip()
    if answer != "OK":
        return f"unexpected probe response: {answer[:80]!r}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--registry")
    ap.add_argument(
        "--canonical-settings",
        action="store_true",
        help="probe with the full benchmark reasoning effort and output ceiling",
    )
    args = ap.parse_args()
    try:
        registry = load_model_registry(args.registry)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        ap.error(f"invalid model registry: {error}")
    failures = 0
    for k in args.models or PRIMARY_MODELS:
        if k not in registry:
            ap.error(f"unknown model: {k}")
        spec = _probe_spec(registry[k], canonical_settings=args.canonical_settings)
        tag = f"{spec['provider']}/{spec['model_id']}"
        r = client.chat(
            spec,
            "You are a connectivity test.",
            "Reply with the single word: OK",
            timeout=180,
            retries=1,
        )
        error = _probe_error(r)
        if error:
            failures += 1
            print(f"FAIL  {k:22s} {tag:42s} {error[:110]}")
        else:
            print(f"OK    {k:22s} {tag:42s} ({r['elapsed_s']}s) {r['text'][:30]!r}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
