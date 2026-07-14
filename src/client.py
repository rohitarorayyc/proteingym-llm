"""OpenAI-compatible endpoint client with a lossless response schema."""

from __future__ import annotations

import hashlib
import json
import os
import time
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

from config.models import validate_model_spec
from config.paths import WORK_ROOT

ENV_FILE = WORK_ROOT / ".env"
PUBLIC_REQUEST_DESCRIPTOR_VERSION = 1


@lru_cache(maxsize=1)
def _env() -> dict[str, str]:
    values: dict[str, str] = {}
    if ENV_FILE.exists():
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(os.environ)
    return values


def _key(name: str) -> str:
    value = _env().get(name)
    if not value:
        raise RuntimeError(f"{name} not set (export it or add it to {ENV_FILE.name})")
    return value


def _base_url(spec: dict) -> str:
    env_name = spec["base_url_env"]
    value = _key(env_name).rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"{env_name} must contain a valid HTTP(S) URL")
    return value


def _endpoint(spec: dict) -> str:
    path = "responses" if spec["api_style"] == "responses" else "chat/completions"
    return f"{_base_url(spec)}/{path}"


def public_request_descriptor(spec: dict) -> dict:
    """Describe the request without exposing credentials or endpoint text."""
    normalized = validate_model_spec(spec)
    style = normalized["api_style"]
    options = (
        {
            "send_reasoning": normalized.get("send_reasoning", True),
            "reasoning_summary": "auto_with_unsupported_fallback",
        }
        if style == "responses"
        else {"send_reasoning_effort": normalized.get("send_reasoning_effort", True)}
    )
    return {
        "descriptor_version": PUBLIC_REQUEST_DESCRIPTOR_VERSION,
        "provider": "openai-compatible",
        "model_id": normalized["model_id"],
        "api_style": style,
        "reasoning_effort": normalized["reasoning"],
        "max_output_tokens": normalized["max_tokens"],
        "context_window": normalized["ctx"],
        "service_tier": normalized.get("service_tier"),
        "tokenizer_encoding": normalized.get("tokenizer_encoding"),
        "inference_options": options,
        "api_key_env": normalized["api_key_env"],
        "base_url_env": normalized["base_url_env"],
        "endpoint_sha256": hashlib.sha256(_endpoint(normalized).encode()).hexdigest(),
    }


def fingerprint_request_descriptor(descriptor: dict) -> str:
    payload = json.dumps(
        descriptor,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def request_provenance(spec: dict) -> dict:
    descriptor = public_request_descriptor(spec)
    return {
        "request_descriptor": descriptor,
        "request_fingerprint": fingerprint_request_descriptor(descriptor),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    try:
        return value.model_dump(mode="json")
    except Exception:
        return str(value)


def _responses(spec: dict, system: str, user: str, timeout: int) -> dict:
    from openai import OpenAI

    api = OpenAI(
        api_key=_key(spec["api_key_env"]),
        base_url=_base_url(spec),
        timeout=timeout,
    )
    kwargs = {
        "model": spec["model_id"],
        "instructions": system,
        "input": user,
        "max_output_tokens": spec["max_tokens"],
    }
    if spec.get("service_tier"):
        kwargs["service_tier"] = spec["service_tier"]
    if spec.get("send_reasoning", True):
        reasoning = {"effort": spec["reasoning"], "summary": "auto"}
        try:
            response = api.responses.create(reasoning=reasoning, **kwargs)
        except Exception as error:
            if "summary" not in str(error).lower():
                raise
            reasoning.pop("summary")
            response = api.responses.create(reasoning=reasoning, **kwargs)
    else:
        response = api.responses.create(**kwargs)

    usage = getattr(response, "usage", None)
    output_details = getattr(usage, "output_tokens_details", None)
    input_details = getattr(usage, "input_tokens_details", None)
    summaries = [
        getattr(summary, "text", "")
        for item in (getattr(response, "output", None) or [])
        if getattr(item, "type", "") == "reasoning"
        for summary in (getattr(item, "summary", None) or [])
        if getattr(summary, "text", "")
    ]
    return {
        "text": getattr(response, "output_text", None) or "",
        "reasoning_text": "\n\n".join(summaries) or None,
        "response_content": [_jsonable(item) for item in (getattr(response, "output", None) or [])],
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "reasoning_tokens": getattr(output_details, "reasoning_tokens", None),
            "cached_tokens": getattr(input_details, "cached_tokens", None),
        },
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", None),
        "response_id": getattr(response, "id", None),
        "response_model_id": getattr(response, "model", None),
        "provider_response_version": None,
        "provider_created_at": getattr(response, "created_at", None),
        "status": getattr(response, "status", None),
        "incomplete_reason": getattr(getattr(response, "incomplete_details", None), "reason", None),
        "stop_reason": None,
        "service_tier": getattr(response, "service_tier", None),
    }


def _chat_completions(spec: dict, system: str, user: str, timeout: int) -> dict:
    from openai import OpenAI

    api = OpenAI(
        api_key=_key(spec["api_key_env"]),
        base_url=_base_url(spec),
        timeout=timeout,
    )
    kwargs = {
        "model": spec["model_id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": spec["max_tokens"],
    }
    if spec.get("send_reasoning_effort", True):
        kwargs["reasoning_effort"] = spec["reasoning"]
    response = api.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "text": message.content or "",
        "reasoning_text": getattr(message, "reasoning_content", None),
        "response_content": [_jsonable(message)],
        "usage": _jsonable(usage),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "response_id": getattr(response, "id", None),
        "response_model_id": getattr(response, "model", None),
        "provider_response_version": None,
        "provider_created_at": getattr(response, "created", None),
        "status": None,
        "incomplete_reason": None,
        "stop_reason": getattr(choice, "finish_reason", None),
        "service_tier": None,
    }


def _call(spec: dict, system: str, user: str, timeout: int) -> dict:
    if spec["api_style"] == "responses":
        return _responses(spec, system, user, timeout)
    return _chat_completions(spec, system, user, timeout)


CALLERS = {"openai-compatible": _call}


def _non_retryable(message: str) -> bool:
    lowered = message.lower()
    fatal = (
        "safety",
        "policy",
        "content_filter",
        "invalid_request",
        "permission",
        "not found",
        "does not exist",
    )
    return any(token in lowered for token in fatal) and not any(
        token in lowered for token in ("rate", "overload")
    )


def chat(
    spec: dict,
    system: str,
    user: str,
    *,
    timeout: int = 900,
    retries: int = 3,
) -> dict:
    """Call one model and return a normalized, lossless response."""
    spec = validate_model_spec(spec)
    last_error = ""
    for attempt in range(retries):
        started = time.time()
        try:
            result = _call(spec, system, user, timeout)
            result.update({"error": None, "elapsed_s": round(time.time() - started, 1)})
            return result
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            if _non_retryable(last_error):
                break
            if attempt + 1 < retries:
                time.sleep(min(2 ** (attempt + 1), 30))
    return {"text": "", "error": last_error, "elapsed_s": 0}


def token_estimator(spec: dict) -> str:
    encoding = spec.get("tokenizer_encoding")
    return f"tiktoken:{encoding}" if encoding else "utf8-bytes-upper-bound"


def estimate_tokens(text: str, spec: dict) -> int:
    encoding = spec.get("tokenizer_encoding")
    if encoding:
        import tiktoken

        return len(tiktoken.get_encoding(encoding).encode(text))
    return len(text.encode("utf-8"))
