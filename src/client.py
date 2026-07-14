"""Native provider clients with one lossless response schema.

Every successful call returns the answer, provider-visible reasoning summary,
serialized response blocks, token usage, response identifier, and stop status.
No OpenRouter or cross-provider translation layer is used.
"""

from __future__ import annotations

import base64
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

OAI_COMPAT = {
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "key": "DEEPINFRA_API_KEY",
    },
}

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


def _configured_base_url(spec: dict) -> str:
    """Resolve a custom base URL without ever returning it in public metadata."""
    provider_config = OAI_COMPAT.get(spec["provider"], {})
    if spec.get("base_url_env"):
        env_name = spec["base_url_env"]
        value = _key(env_name)
    else:
        env_name = None
        value = provider_config.get("base_url")
    if not value:
        suffix = f" ({env_name})" if env_name else ""
        raise RuntimeError(f"base URL is required for provider {spec['provider']}{suffix}")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        label = env_name or f"the {spec['provider']} endpoint"
        raise RuntimeError(f"{label} must contain a valid HTTP(S) URL")
    return value.rstrip("/")


def _google_project_info() -> dict:
    try:
        info = json.loads(base64.b64decode(_key("GCP_KEY_JSON")).decode())
    except Exception as error:
        raise RuntimeError(
            "GCP_KEY_JSON must contain base64-encoded service-account JSON"
        ) from error
    if not isinstance(info, dict) or not isinstance(info.get("project_id"), str):
        raise RuntimeError("GCP_KEY_JSON service-account JSON requires project_id")
    return info


def _resolved_request_endpoint(spec: dict) -> str:
    """Return the exact inference endpoint for hashing or request dispatch."""
    provider = spec["provider"]
    if provider == "openai":
        return "https://api.openai.com/v1/responses"
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/messages"
    if provider == "google":
        project_id = _google_project_info()["project_id"]
        return (
            f"https://aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/global/publishers/google/models/{spec['model_id']}:generateContent"
        )
    base_url = _configured_base_url(spec)
    path = "responses" if spec.get("api_style") == "responses" else "chat/completions"
    return f"{base_url}/{path}"


def public_request_descriptor(spec: dict) -> dict:
    """Describe an inference request without exposing credentials or endpoint text.

    The endpoint is resolved before hashing.  Consequently a custom endpoint's
    ``base_url_env`` must be set for live provenance, but neither its value nor
    any credential is included in the returned object.
    """
    normalized = validate_model_spec(spec)
    provider = normalized["provider"]
    api_style = (
        normalized["api_style"]
        if provider == "openai-compatible"
        else {
            "openai": "responses",
            "anthropic": "messages",
            "google": "generateContent",
            "deepinfra": "chat",
        }[provider]
    )
    api_key_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GCP_KEY_JSON",
        "deepinfra": OAI_COMPAT["deepinfra"]["key"],
    }.get(provider, normalized.get("api_key_env"))

    inference_options: dict[str, Any] = {}
    if provider == "openai" or (provider == "openai-compatible" and api_style == "responses"):
        inference_options = {
            "send_reasoning": normalized.get("send_reasoning", True),
            "reasoning_summary": "auto_with_unsupported_fallback",
        }
    elif provider == "anthropic":
        inference_options = {
            "thinking_type": "adaptive",
            "thinking_display": "summarized",
        }
    elif provider == "google":
        inference_options = {"temperature": 1, "include_thoughts": True}
    else:
        inference_options = {"send_reasoning_effort": normalized.get("send_reasoning_effort", True)}

    endpoint = _resolved_request_endpoint(normalized)
    return {
        "descriptor_version": PUBLIC_REQUEST_DESCRIPTOR_VERSION,
        "provider": provider,
        "model_id": normalized["model_id"],
        "api_style": api_style,
        "reasoning_effort": normalized["reasoning"],
        "max_output_tokens": normalized["max_tokens"],
        "context_window": normalized["ctx"],
        "service_tier": normalized.get("service_tier"),
        "supports_batch": normalized["supports_batch"],
        "tokenizer_encoding": normalized.get("tokenizer_encoding"),
        "inference_options": inference_options,
        "api_key_env": api_key_env,
        "base_url_env": normalized.get("base_url_env"),
        "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
    }


def fingerprint_request_descriptor(descriptor: dict) -> str:
    """Hash a public descriptor using canonical, cross-process JSON."""
    payload = json.dumps(
        descriptor,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def request_provenance(
    spec: dict,
    *,
    delivery_mode: str = "live",
    delivery_region: str | None = None,
) -> dict:
    """Return the safe public request descriptor and its stable fingerprint."""
    if delivery_mode not in {"live", "batch"}:
        raise ValueError(f"unsupported delivery mode: {delivery_mode!r}")
    if delivery_region is not None and (
        not isinstance(delivery_region, str) or not delivery_region.strip()
    ):
        raise ValueError("delivery region must be a non-empty string")
    descriptor = public_request_descriptor(spec)
    descriptor["delivery_mode"] = delivery_mode
    descriptor["delivery_region"] = delivery_region
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


def _openai(spec: dict, system: str, user: str, timeout: int) -> dict:
    from openai import OpenAI

    key_name = spec.get("api_key_env", "OPENAI_API_KEY")
    base_url = (
        _configured_base_url(spec)
        if spec["provider"] == "openai-compatible"
        else "https://api.openai.com/v1"
    )
    client = OpenAI(api_key=_key(key_name), base_url=base_url, timeout=timeout)
    kwargs = {
        "model": spec["model_id"],
        "instructions": system,
        "input": user,
        "max_output_tokens": spec["max_tokens"],
    }
    if spec.get("service_tier"):
        kwargs["service_tier"] = spec["service_tier"]

    if spec.get("send_reasoning", True):
        reasoning = {"effort": spec.get("reasoning", "high"), "summary": "auto"}
        try:
            response = client.responses.create(reasoning=reasoning, **kwargs)
        except Exception as error:
            if "summary" not in str(error).lower():
                raise
            reasoning.pop("summary")
            response = client.responses.create(reasoning=reasoning, **kwargs)
    else:
        response = client.responses.create(**kwargs)

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
        "text": response.output_text or "",
        "reasoning_text": "\n\n".join(summaries) or None,
        "response_content": [_jsonable(item) for item in (response.output or [])],
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


def normalize_anthropic_message(message: Any) -> dict:
    """Normalize a synchronous or Batch API Anthropic message."""
    blocks = list(getattr(message, "content", None) or [])
    answer = "".join(
        getattr(block, "text", "") for block in blocks if getattr(block, "type", "") == "text"
    )
    thinking = "\n\n".join(
        text
        for block in blocks
        if getattr(block, "type", "") == "thinking"
        for text in [getattr(block, "thinking", "") or getattr(block, "text", "")]
        if text
    )
    usage = getattr(message, "usage", None)
    return {
        "text": answer,
        "reasoning_text": thinking or None,
        "response_content": [_jsonable(block) for block in blocks],
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        },
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": None,
        "response_id": getattr(message, "id", None),
        "response_model_id": getattr(message, "model", None),
        "provider_response_version": None,
        "provider_created_at": getattr(message, "created_at", None),
        "status": None,
        "incomplete_reason": None,
        "stop_reason": getattr(message, "stop_reason", None),
        "stop_sequence": getattr(message, "stop_sequence", None),
        "service_tier": None,
    }


def _anthropic(spec: dict, system: str, user: str, timeout: int) -> dict:
    import anthropic

    client = anthropic.Anthropic(
        api_key=_key("ANTHROPIC_API_KEY"), base_url="https://api.anthropic.com"
    )
    message = client.messages.create(
        model=spec["model_id"],
        max_tokens=spec["max_tokens"],
        system=system,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive", "display": "summarized"},
        extra_body={"output_config": {"effort": spec.get("reasoning", "high")}},
        timeout=timeout,
    )
    return normalize_anthropic_message(message)


def normalize_google_payload(payload: dict) -> dict:
    """Normalize a live or Vertex Batch generateContent response."""
    candidate = (payload.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    answer = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    thinking = "\n\n".join(
        part.get("text", "") for part in parts if part.get("thought") and part.get("text")
    )
    usage = payload.get("usageMetadata") or {}
    answer_tokens = usage.get("candidatesTokenCount") or 0
    thought_tokens = usage.get("thoughtsTokenCount") or 0
    return {
        "text": answer,
        "reasoning_text": thinking or None,
        "response_content": parts,
        "usage": usage,
        "output_tokens": (answer_tokens + thought_tokens) or None,
        "reasoning_tokens": thought_tokens or None,
        "response_id": payload.get("responseId"),
        "response_model_id": payload.get("model") or payload.get("modelVersion"),
        "provider_response_version": payload.get("modelVersion"),
        "provider_created_at": payload.get("createTime"),
        "status": None,
        "incomplete_reason": None,
        "stop_reason": candidate.get("finishReason"),
        "service_tier": None,
    }


def _google(spec: dict, system: str, user: str, timeout: int) -> dict:
    import requests
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    info = _google_project_info()
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    url = _resolved_request_endpoint(spec)
    body = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "maxOutputTokens": spec["max_tokens"],
            "temperature": 1,
            "thinkingConfig": {
                "thinkingLevel": spec.get("reasoning", "high"),
                "includeThoughts": True,
            },
        },
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {credentials.token}"},
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    return normalize_google_payload(response.json())


def _openai_compatible(spec: dict, system: str, user: str, timeout: int) -> dict:
    from openai import OpenAI

    provider = spec["provider"]
    provider_config = OAI_COMPAT.get(provider, {})
    key_name = spec.get("api_key_env") or provider_config.get("key")
    if not key_name:
        raise RuntimeError(f"api_key_env is required for provider {provider}")
    base_url = _configured_base_url(spec)
    client = OpenAI(
        api_key=_key(key_name),
        base_url=base_url,
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
    if provider == "deepinfra" or spec.get("send_reasoning_effort", True):
        kwargs["reasoning_effort"] = spec.get("reasoning", "high")
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(message, "reasoning_content", None)
    return {
        "text": message.content or "",
        "reasoning_text": reasoning or None,
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


def _custom_openai(spec: dict, system: str, user: str, timeout: int) -> dict:
    if spec.get("api_style", "responses") == "responses":
        return _openai(spec, system, user, timeout)
    return _openai_compatible(spec, system, user, timeout)


CALLERS = {
    "openai": _openai,
    "anthropic": _anthropic,
    "google": _google,
    "deepinfra": _openai_compatible,
    "openai-compatible": _custom_openai,
}


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
    """Call one model and return a normalized, lossless provider response."""
    spec = validate_model_spec(spec)
    caller = CALLERS.get(spec["provider"])
    if caller is None:
        return {"text": "", "error": f"unknown provider {spec['provider']}", "elapsed_s": 0}

    last_error = ""
    for attempt in range(retries):
        started = time.time()
        try:
            result = caller(spec, system, user, timeout)
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
    """Name the frozen tokenizer or conservative fallback used for preflight."""
    encoding = spec.get("tokenizer_encoding")
    return f"tiktoken:{encoding}" if encoding else "utf8-bytes-upper-bound"


def estimate_tokens(text: str, spec: dict) -> int:
    """Estimate prompt tokens with a frozen tokenizer or a safe byte bound."""
    encoding = spec.get("tokenizer_encoding")
    if encoding:
        import tiktoken

        return len(tiktoken.get_encoding(encoding).encode(text))
    # Every token represents at least one encoded byte, so this is a safe upper
    # bound for endpoints whose tokenizer is not declared.
    return len(text.encode("utf-8"))
