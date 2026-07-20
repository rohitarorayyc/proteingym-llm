"""Send a bounded connectivity probe to a user-supplied model endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.models import load_model_registry  # noqa: E402
from src import client  # noqa: E402

PROBE_MAX_OUTPUT_TOKENS = 256
PROBE_REASONING_EFFORT = "low"


def _probe_spec(spec: dict, *, canonical_settings: bool) -> dict:
    """Bound connectivity checks without changing the provider request shape."""
    if canonical_settings:
        return dict(spec)
    probe = dict(spec)
    probe["reasoning"] = spec.get("probe_reasoning", PROBE_REASONING_EFFORT)
    probe["max_tokens"] = min(
        spec.get("probe_max_tokens", PROBE_MAX_OUTPUT_TOKENS), spec["max_tokens"]
    )
    return probe


def _probe_error(response: dict, spec: dict) -> str | None:
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
        for marker in (
            "incomplete",
            "max_tokens",
            "max_output_tokens",
            "max_completion_tokens",
            "length",
        )
    ):
        return "probe response was truncated or incomplete"
    answer = str(response.get("text") or "").strip()
    if answer != "OK":
        return f"unexpected probe response: {answer[:80]!r}"
    if spec.get("require_usage") and (
        not isinstance(response.get("usage"), dict)
        or not isinstance(response.get("output_tokens"), int)
    ):
        return "probe response omitted required token usage metadata"
    if spec.get("require_reasoning") and not str(response.get("reasoning_text") or "").strip():
        return "probe response omitted required reasoning trace"
    accepted_model_ids = spec.get("response_model_ids") or []
    if accepted_model_ids and response.get("response_model_id") not in accepted_model_ids:
        return f"unexpected provider model identity: {response.get('response_model_id')!r}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument(
        "--canonical-settings",
        action="store_true",
        help="probe with the full benchmark reasoning effort and output limit",
    )
    args = ap.parse_args()
    try:
        registry = load_model_registry(args.registry)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        ap.error(f"invalid model registry: {error}")
    failures = 0
    for k in args.models:
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
        error = _probe_error(r, spec)
        if error:
            failures += 1
            print(f"FAIL  {k:22s} {tag:42s} {error[:110]}")
        else:
            returned_model = r.get("response_model_id") or "unreported"
            stop_reason = r.get("stop_reason") or r.get("status") or "unreported"
            usage = "yes" if r.get("usage") is not None else "no"
            reasoning = "yes" if r.get("reasoning_text") else "no"
            print(
                f"OK    {k:22s} {tag:42s} ({r['elapsed_s']}s) "
                f"returned={returned_model!r} stop={stop_reason!r} "
                f"usage={usage} reasoning={reasoning}"
            )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
