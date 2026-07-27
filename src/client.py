"""Native provider clients with a lossless, publication-safe response schema."""

from __future__ import annotations

import base64
import email.utils
import hashlib
import json
import math
import os
import re
import socket
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

from config.models import validate_model_spec
from config.paths import WORK_ROOT

ENV_FILE = WORK_ROOT / ".env"
PUBLIC_REQUEST_DESCRIPTOR_VERSION = 2
RESPONSES_SSE_DESCRIPTOR_VERSION = 3


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
    if spec["provider"] == "google-vertex":
        project = _google_project(spec)
        location = spec["location"]
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        return (
            f"https://{host}/{spec['api_version']}/projects/{project}/locations/{location}"
            f"/publishers/google/models/{spec['model_id']}:generateContent"
        )
    path = "responses" if spec["api_style"] == "responses" else "chat/completions"
    return f"{_base_url(spec)}/{path}"


def public_request_descriptor(spec: dict) -> dict:
    """Describe the request without exposing credentials or endpoint text."""
    normalized = validate_model_spec(spec)
    if normalized["provider"] == "google-vertex":
        options = {
            "transport": "vertex-generate-content-sync",
            "temperature": normalized.get("temperature", "provider_default"),
            "thinking": {
                "level": normalized["reasoning"],
                "include_thoughts": normalized["include_thoughts"],
            },
            "required_response_metadata": {
                "usage": normalized["require_usage"],
                "reasoning": normalized["require_reasoning"],
            },
        }
        return {
            "descriptor_version": PUBLIC_REQUEST_DESCRIPTOR_VERSION,
            "provider": "google-vertex",
            "model_id": normalized["model_id"],
            "api_style": normalized["api_style"],
            "api_version": normalized["api_version"],
            "location": normalized["location"],
            "reasoning_effort": normalized["reasoning"],
            "max_output_tokens": normalized["max_tokens"],
            "context_window": normalized["ctx"],
            "service_tier": None,
            "tokenizer_encoding": normalized.get("tokenizer_encoding"),
            "inference_options": options,
            "accepted_response_model_ids": normalized["response_model_ids"],
            "project_env": normalized["project_env"],
            "credentials_env": normalized["credentials_env"],
            "endpoint_sha256": hashlib.sha256(_endpoint(normalized).encode()).hexdigest(),
        }
    style = normalized["api_style"]
    options = (
        {
            "send_reasoning": normalized.get("send_reasoning", True),
            "reasoning_summary": "auto_with_unsupported_fallback",
            "store": False,
            "background": False,
        }
        if style == "responses"
        else {
            "send_reasoning_effort": normalized.get("send_reasoning_effort", True),
            "output_token_field": normalized["chat_output_token_field"],
            "stream": normalized["stream"],
            "stream_include_usage": normalized["stream"],
        }
    )
    descriptor_version = PUBLIC_REQUEST_DESCRIPTOR_VERSION
    if style == "responses" and normalized["stream"]:
        descriptor_version = RESPONSES_SSE_DESCRIPTOR_VERSION
        options["transport"] = "responses-sse"
        options["stream"] = True
    options["required_response_metadata"] = {
        "usage": normalized["require_usage"],
        "reasoning": normalized["require_reasoning"],
    }
    return {
        "descriptor_version": descriptor_version,
        "provider": "openai-compatible",
        "model_id": normalized["model_id"],
        "api_style": style,
        "reasoning_effort": normalized["reasoning"],
        "max_output_tokens": normalized["max_tokens"],
        "context_window": normalized["ctx"],
        "service_tier": normalized.get("service_tier"),
        "tokenizer_encoding": normalized.get("tokenizer_encoding"),
        "inference_options": options,
        "accepted_response_model_ids": normalized["response_model_ids"],
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


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _tcp_keepalive_socket_options() -> list[tuple[int, int, int]]:
    """Return portable httpcore socket options for long-lived SSE connections."""
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle_option = getattr(socket, "TCP_KEEPIDLE", None)
    if idle_option is None:
        idle_option = getattr(socket, "TCP_KEEPALIVE", None)
    for option, value in (
        (idle_option, 30),
        (getattr(socket, "TCP_KEEPINTVL", None), 10),
        (getattr(socket, "TCP_KEEPCNT", None), 3),
    ):
        if option is not None:
            options.append((socket.IPPROTO_TCP, option, value))
    return options


def _responses_http_client(timeout: int):
    """Build the owned HTTP client used for a single Responses SSE request."""
    import httpx

    transport = httpx.HTTPTransport(
        retries=0,
        socket_options=_tcp_keepalive_socket_options(),
    )
    return httpx.Client(transport=transport, timeout=timeout)


_SAFE_RESPONSE_HEADERS = {
    "cf-ray",
    "content-type",
    "date",
    "openai-processing-ms",
    "openai-version",
    "request-id",
    "retry-after",
    "server",
    "x-client-request-id",
    "x-envoy-upstream-service-time",
    "x-request-id",
}


class EventSinkError(RuntimeError):
    """Raised when durable stream-event persistence fails."""


def _safe_response_headers(headers: Any) -> dict[str, str]:
    """Keep operational response headers while excluding cookies and credentials."""
    if headers is None:
        return {}
    try:
        items = headers.items()
    except AttributeError:
        return {}
    safe: dict[str, str] = {}
    for name, value in items:
        lowered = str(name).lower()
        if lowered in _SAFE_RESPONSE_HEADERS or lowered.startswith("x-ratelimit-"):
            safe[lowered] = str(value)
    return safe


def _emit_response_stream_record(event_sink: Any, record: dict, spec: dict) -> None:
    if event_sink is not None:
        try:
            # Redact before the record is durably journaled: a hostile or buggy
            # provider can echo the configured credential/endpoint into a header
            # value or stream event, and the final result-level redaction happens
            # too late to scrub the append-only transport journal.
            event_sink(_redact_payload(record, spec))
        except EventSinkError:
            raise
        except Exception as error:  # noqa: BLE001
            raise EventSinkError("Responses event sink failed") from error


def _responses_error_message(response: Any) -> str | None:
    provider_error = _field(response, "error", None)
    if provider_error is None:
        return None
    code = _field(provider_error, "code", None)
    message = _field(provider_error, "message", None)
    if code and message:
        return f"{code}: {message}"
    if message:
        return str(message)
    return json.dumps(_jsonable(provider_error), sort_keys=True, ensure_ascii=False)


_POLICY_BLOCK_MARKERS = (
    "content_filter",
    "content filter",
    "provider policy block",
    "safety reasons",
    "safety_skip",
    "safety skip",
    "we've limited access to this content",
)
_QUOTA_EXHAUSTED_MARKERS = (
    "credit balance is too low",
    "exceeded your current quota",
    "exceeded_current_quota",
    "insufficient balance",
    "insufficient_quota",
    "please recharge",
)


def _provider_failure_class(message: str, provider_error: Any = None) -> str | None:
    """Classify deterministic provider stops that require human intervention."""
    details = message
    if provider_error is not None:
        try:
            details += " " + json.dumps(
                _jsonable(provider_error),
                sort_keys=True,
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            details += f" {provider_error}"
    lowered = details.lower()
    if any(marker in lowered for marker in _QUOTA_EXHAUSTED_MARKERS):
        return "quota_exhausted"
    if any(marker in lowered for marker in _POLICY_BLOCK_MARKERS):
        return "provider_policy_block"
    return None


def _normalize_responses_response(
    response: Any,
    *,
    stream_completed: bool | None = None,
    terminal_event_type: str | None = None,
    error: str | None = None,
    incomplete_reason: str | None = None,
    retryable: bool | None = None,
) -> dict:
    """Normalize a complete or terminal Responses object without dropping fields."""
    usage = _field(response, "usage", None)
    output_details = _field(usage, "output_tokens_details", None)
    input_details = _field(usage, "input_tokens_details", None)
    output = _field(response, "output", None) or []
    summaries = [
        _field(summary, "text", "")
        for item in output
        if _field(item, "type", "") == "reasoning"
        for summary in (_field(item, "summary", None) or [])
        if _field(summary, "text", "")
    ]
    response_status = _field(response, "status", None)
    provider_incomplete_reason = _field(
        _field(response, "incomplete_details", None), "reason", None
    )
    if incomplete_reason is None:
        incomplete_reason = provider_incomplete_reason
    if error is None and terminal_event_type == "response.failed":
        error = _responses_error_message(response) or "provider emitted response.failed"
    elif error is None and response_status == "failed":
        error = _responses_error_message(response) or "provider response status is failed"
    elif error is None and terminal_event_type == "response.incomplete":
        reason = incomplete_reason or "unspecified"
        error = f"provider emitted response.incomplete: {reason}"
    provider_error = _jsonable(_field(response, "error", None))
    failure_class = _provider_failure_class(error or "", provider_error)
    if failure_class is not None:
        incomplete_reason = failure_class
        retryable = False
    if retryable is None:
        retryable = bool(error) and not _non_retryable(error)

    result = {
        "text": _field(response, "output_text", None) or "",
        "reasoning_text": "\n\n".join(str(value) for value in summaries) or None,
        "response_content": [_jsonable(item) for item in output],
        "usage": {
            "input_tokens": _field(usage, "input_tokens", None),
            "output_tokens": _field(usage, "output_tokens", None),
            "total_tokens": _field(usage, "total_tokens", None),
            "reasoning_tokens": _field(output_details, "reasoning_tokens", None),
            "cached_tokens": _field(input_details, "cached_tokens", None),
        },
        "output_tokens": _field(usage, "output_tokens", None),
        "reasoning_tokens": _field(output_details, "reasoning_tokens", None),
        "response_id": _field(response, "id", None),
        "response_model_id": _field(response, "model", None),
        "provider_response_version": None,
        "provider_created_at": _field(response, "created_at", None),
        "status": response_status,
        "incomplete_reason": incomplete_reason,
        "stop_reason": None,
        "service_tier": _field(response, "service_tier", None),
        "provider_response": _jsonable(response),
        "provider_error": provider_error,
        "error": error,
        "retryable": retryable,
        "failure_class": failure_class,
    }
    if stream_completed is not None:
        result["stream_completed"] = stream_completed
        result["stream_terminal_event"] = terminal_event_type
    return result


def _consume_responses_stream(
    stream: Any,
    *,
    spec: dict,
    event_sink: Any = None,
    client_request_id: str | None = None,
) -> dict:
    """Consume Responses SSE, accepting only ``response.completed`` as success."""
    response = getattr(stream, "response", None)
    headers = _safe_response_headers(getattr(response, "headers", None))
    _emit_response_stream_record(
        event_sink,
        {
            "kind": "response.headers",
            "client_request_id": client_request_id,
            "headers": headers,
        },
        spec,
    )

    terminal_event_type = None
    terminal_response = None
    last_response = None
    error_event = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    seen_events = 0
    stream_error = None
    stream_failure_class = None
    stream_retryable = None
    provider_error = None
    try:
        for event in stream:
            event_payload = _jsonable(event)
            event_type = str(_field(event, "type", "") or "")
            _emit_response_stream_record(
                event_sink,
                {
                    "kind": "response.event",
                    "client_request_id": client_request_id,
                    "event_type": event_type or None,
                    "sequence_number": _field(event, "sequence_number", None),
                    "event": event_payload,
                },
                spec,
            )
            seen_events += 1
            event_response = _field(event, "response", None)
            if event_response is not None:
                last_response = event_response
            if event_type == "response.output_text.delta":
                delta = _field(event, "delta", None)
                if delta:
                    text_parts.append(str(delta))
            elif event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            }:
                delta = _field(event, "delta", None)
                if delta:
                    reasoning_parts.append(str(delta))
            if event_type in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }:
                if terminal_event_type is None:
                    terminal_event_type = event_type
                    terminal_response = event_response
                elif terminal_event_type != event_type:
                    stream_error = (
                        "stream emitted conflicting terminal events: "
                        f"{terminal_event_type}, {event_type}"
                    )
            elif event_type == "error":
                terminal_event_type = terminal_event_type or "error"
                error_event = event_payload
    except EventSinkError:
        raise
    except Exception as error:  # noqa: BLE001
        stream_error = _redact_error(f"{type(error).__name__}: {error}", spec)
        provider_error = _redact_payload(_provider_error_payload(error), spec)
        stream_failure_class = _provider_failure_class(stream_error, provider_error)
        stream_retryable = bool(
            stream_failure_class is None
            and (_error_is_retryable(error, stream_error) or not _non_retryable(stream_error))
        )
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                # Stream state is derived entirely from the observed events. A
                # close() failure must never discard an already-observed terminal
                # event and force a completed, billed request to be retried.
                pass

    if terminal_event_type == "response.completed" and terminal_response is not None:
        if stream_error is None:
            return _normalize_responses_response(
                terminal_response,
                stream_completed=True,
                terminal_event_type=terminal_event_type,
                retryable=False,
            )
        # The complete provider response is still authoritative once its typed
        # terminal event has been durably observed.  Preserve any later local
        # stream-close error without downgrading the completed response.
        result = _normalize_responses_response(
            terminal_response,
            stream_completed=True,
            terminal_event_type=terminal_event_type,
            retryable=False,
        )
        result["post_completion_stream_error"] = stream_error
        return result

    if terminal_response is not None:
        return _normalize_responses_response(
            terminal_response,
            stream_completed=False,
            terminal_event_type=terminal_event_type,
        )

    if error_event is not None:
        message = _field(error_event, "message", None) or "provider emitted error event"
        code = _field(error_event, "code", None)
        error = f"{code}: {message}" if code else str(message)
        provider_error = _jsonable(error_event)
        failure_class = _provider_failure_class(error, provider_error)
        return {
            "text": "".join(text_parts),
            "reasoning_text": "".join(reasoning_parts) or None,
            "response_content": [],
            "usage": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "response_id": _field(last_response, "id", None),
            "response_model_id": _field(last_response, "model", None),
            "provider_response_version": None,
            "provider_created_at": _field(last_response, "created_at", None),
            "status": "failed",
            "incomplete_reason": failure_class or "response_error_event",
            "stop_reason": None,
            "service_tier": _field(last_response, "service_tier", None),
            "stream_completed": False,
            "stream_terminal_event": "error",
            "provider_response": error_event,
            "provider_error": provider_error,
            "error": error,
            "retryable": failure_class is None and not _non_retryable(error),
            "failure_class": failure_class,
        }

    if stream_error is not None:
        incomplete_reason = stream_failure_class or "stream_interrupted"
        error = stream_error
        failure_class = stream_failure_class or (
            "transport_error" if stream_retryable else "stream_exception"
        )
        retryable = bool(stream_retryable)
    elif seen_events == 0:
        incomplete_reason = "empty_stream"
        error = "Responses stream ended without events"
        failure_class = "stream_protocol_error"
        retryable = True
    else:
        incomplete_reason = "missing_terminal_event"
        error = "Responses stream ended without a terminal event"
        failure_class = "stream_protocol_error"
        retryable = True
    _emit_response_stream_record(
        event_sink,
        {
            "kind": "response.stream_error",
            "client_request_id": client_request_id,
            "error": error,
            "failure_class": failure_class,
            "incomplete_reason": incomplete_reason,
            "retryable": retryable,
            "provider_error": provider_error,
        },
        spec,
    )
    return {
        "text": "".join(text_parts),
        "reasoning_text": "".join(reasoning_parts) or None,
        "response_content": [],
        "usage": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "response_id": _field(last_response, "id", None),
        "response_model_id": _field(last_response, "model", None),
        "provider_response_version": None,
        "provider_created_at": _field(last_response, "created_at", None),
        "status": (
            "failed"
            if failure_class in {"provider_policy_block", "quota_exhausted"}
            else "incomplete"
        ),
        "incomplete_reason": incomplete_reason,
        "stop_reason": None,
        "service_tier": _field(last_response, "service_tier", None),
        "stream_completed": False,
        "stream_terminal_event": None,
        "provider_response": _jsonable(last_response) if last_response is not None else None,
        "provider_error": provider_error,
        "error": error,
        "retryable": retryable,
        "failure_class": failure_class,
    }


def _responses(
    spec: dict,
    system: str,
    user: str,
    timeout: int,
    *,
    reasoning_summary: bool = True,
    client_request_id: str | None = None,
    event_sink: Any = None,
) -> dict:
    from openai import OpenAI

    owned_http_client = _responses_http_client(timeout) if spec["stream"] else None
    api_kwargs = {
        "api_key": _key(spec["api_key_env"]),
        "base_url": _base_url(spec),
        "timeout": timeout,
        "max_retries": 0,
    }
    if owned_http_client is not None:
        api_kwargs["http_client"] = owned_http_client
    api = OpenAI(
        **api_kwargs,
    )
    kwargs = {
        "model": spec["model_id"],
        "instructions": system,
        "input": user,
        "max_output_tokens": spec["max_tokens"],
        "store": False,
        "background": False,
    }
    if spec["stream"]:
        kwargs["stream"] = True
    if client_request_id is not None:
        kwargs["extra_headers"] = {"X-Client-Request-Id": client_request_id}
    if spec.get("service_tier"):
        kwargs["service_tier"] = spec["service_tier"]
    try:
        if spec.get("send_reasoning", True):
            reasoning = {"effort": spec["reasoning"]}
            if reasoning_summary:
                reasoning["summary"] = "auto"
            # Exactly one provider request is made here.  A summary-compatibility
            # fallback is a distinct runner attempt so it receives a fresh rate-limit
            # reservation and an immutable attempt artifact.
            response = api.responses.create(reasoning=reasoning, **kwargs)
        else:
            response = api.responses.create(**kwargs)
        if spec["stream"]:
            return _consume_responses_stream(
                response,
                spec=spec,
                event_sink=event_sink,
                client_request_id=client_request_id,
            )
        return _normalize_responses_response(response)
    finally:
        if owned_http_client is not None:
            try:
                owned_http_client.close()
            except Exception:  # noqa: BLE001
                # Never let closing the owned HTTP client mask the request's real
                # outcome (a completed stream or the original transport error).
                pass


def _consume_chat_stream(stream: Any) -> dict:
    """Losslessly assemble a Chat Completions stream after the SDK observes [DONE]."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    chunks: list[Any] = []
    usage = None
    response_id = None
    response_model_id = None
    provider_created_at = None
    stop_reason = None
    service_tier = None
    stream_error = None

    try:
        for chunk in stream:
            chunks.append(_jsonable(chunk))
            response_id = _field(chunk, "id", response_id)
            response_model_id = _field(chunk, "model", response_model_id)
            provider_created_at = _field(chunk, "created", provider_created_at)
            service_tier = _field(chunk, "service_tier", service_tier)
            choices = _field(chunk, "choices", None) or []
            choice = choices[0] if choices else None
            if choice is not None:
                delta = _field(choice, "delta", None)
                content = _field(delta, "content", None)
                reasoning = _field(delta, "reasoning_content", None)
                if content:
                    text_parts.append(str(content))
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                stop_reason = _field(choice, "finish_reason", None) or stop_reason
                usage = _field(choice, "usage", None) or usage
            usage = _field(chunk, "usage", None) or usage
    except Exception as error:  # noqa: BLE001
        if not chunks:
            raise
        stream_error = f"{type(error).__name__}: {error}"

    usage_json = _jsonable(usage) if usage is not None else None
    output_tokens = _field(usage, "completion_tokens", None)
    details = _field(usage, "completion_tokens_details", None)
    has_terminal_finish = isinstance(stop_reason, str) and bool(stop_reason.strip())
    completed = stream_error is None and bool(chunks) and has_terminal_finish
    if stream_error is not None:
        incomplete_reason = "stream_interrupted"
        error = stream_error
    elif not chunks:
        incomplete_reason = "empty_stream"
        error = "stream ended without response chunks"
    elif not has_terminal_finish:
        incomplete_reason = "missing_terminal_finish_reason"
        error = "stream ended without terminal finish_reason"
    else:
        incomplete_reason = None
        error = None
    return {
        "text": "".join(text_parts),
        "reasoning_text": "".join(reasoning_parts) or None,
        "response_content": chunks,
        "usage": usage_json,
        "output_tokens": output_tokens,
        "reasoning_tokens": _field(details, "reasoning_tokens", None),
        "response_id": response_id,
        "response_model_id": response_model_id,
        "provider_response_version": None,
        "provider_created_at": provider_created_at,
        "status": "completed" if completed else "incomplete",
        "incomplete_reason": incomplete_reason,
        "stop_reason": stop_reason,
        "service_tier": service_tier,
        "stream_completed": completed,
        "provider_response": {"stream_completed": completed, "chunks": chunks},
        "error": error,
        "retryable": error is not None,
    }


def _chat_completions(spec: dict, system: str, user: str, timeout: int) -> dict:
    from openai import OpenAI

    api = OpenAI(
        api_key=_key(spec["api_key_env"]),
        base_url=_base_url(spec),
        timeout=timeout,
        max_retries=0,
    )
    kwargs = {
        "model": spec["model_id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    kwargs[spec.get("chat_output_token_field", "max_tokens")] = spec["max_tokens"]
    if spec.get("send_reasoning_effort", True):
        kwargs["reasoning_effort"] = spec["reasoning"]
    if spec.get("stream", False):
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        return _consume_chat_stream(api.chat.completions.create(**kwargs))
    response = api.chat.completions.create(**kwargs)
    choices = getattr(response, "choices", None) or []
    if not choices:
        # A 2xx body with no choices is an unusable-but-transient provider
        # response: preserve it and mark it retryable instead of raising an
        # IndexError that would be classified non-retryable and lose the body.
        return {
            "text": "",
            "reasoning_text": None,
            "response_content": [],
            "usage": _jsonable(getattr(response, "usage", None)),
            "output_tokens": None,
            "reasoning_tokens": None,
            "response_id": getattr(response, "id", None),
            "response_model_id": getattr(response, "model", None),
            "provider_response_version": None,
            "provider_created_at": getattr(response, "created", None),
            "status": "incomplete",
            "incomplete_reason": "no_choices",
            "stop_reason": None,
            "service_tier": None,
            "stream_completed": None,
            "provider_response": _jsonable(response),
            "error": "chat completion returned no choices",
            "retryable": True,
        }
    choice = choices[0]
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
        "stream_completed": None,
        "provider_response": _jsonable(response),
    }


_GOOGLE_REFRESH_LOCK = threading.Lock()


def _google_service_account_info(spec: dict) -> dict | None:
    """Decode the configured service account entirely in memory, when present."""
    encoded = _env().get(spec["credentials_env"])
    if not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
        info = json.loads(decoded)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{spec['credentials_env']} must be base64-encoded service-account JSON"
        ) from error
    if not isinstance(info, dict) or info.get("type") != "service_account":
        raise RuntimeError(
            f"{spec['credentials_env']} must contain a Google service-account object"
        )
    if not isinstance(info.get("project_id"), str) or not info["project_id"].strip():
        raise RuntimeError(f"{spec['credentials_env']} is missing project_id")
    return info


def _google_project(spec: dict) -> str:
    """Resolve the Vertex project without exposing service-account contents."""
    configured = _env().get(spec["project_env"])
    info = _google_service_account_info(spec)
    credential_project = info.get("project_id") if info else None
    if configured and credential_project and configured != credential_project:
        raise RuntimeError(
            f"{spec['project_env']} does not match the project in {spec['credentials_env']}"
        )
    project = configured or credential_project
    if not project:
        raise RuntimeError(
            f"set {spec['project_env']} or provide {spec['credentials_env']} with project_id"
        )
    return project


@lru_cache(maxsize=8)
def _google_credentials(credentials_env: str, project_env: str):
    """Load service-account credentials once, with ADC as a safe fallback."""
    spec = {"credentials_env": credentials_env, "project_env": project_env}
    info = _google_service_account_info(spec)
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    if info is not None:
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_info(info, scopes=scopes)

    import google.auth

    credentials, adc_project = google.auth.default(scopes=scopes)
    configured_project = _env().get(project_env)
    if configured_project and adc_project and configured_project != adc_project:
        raise RuntimeError(f"{project_env} does not match the active ADC project")
    return credentials


def _google_authorized_session(spec: dict):
    """Create an authenticated session without writing credentials to disk."""
    from google.auth.transport.requests import AuthorizedSession, Request

    credentials = _google_credentials(spec["credentials_env"], spec["project_env"])
    if not credentials.valid:
        with _GOOGLE_REFRESH_LOCK:
            if not credentials.valid:
                credentials.refresh(Request())
    return AuthorizedSession(credentials)


def _google_usage(usage_metadata: Any) -> tuple[dict | None, int | None, int | None, str | None]:
    """Normalize billing-relevant counters while retaining every provider field."""
    if not isinstance(usage_metadata, dict):
        return None, None, None, None
    usage = dict(usage_metadata)
    prompt_tokens = usage_metadata.get("promptTokenCount")
    candidate_tokens = usage_metadata.get("candidatesTokenCount")
    thought_tokens = usage_metadata.get("thoughtsTokenCount")
    total_tokens = usage_metadata.get("totalTokenCount")
    cached_tokens = usage_metadata.get("cachedContentTokenCount")
    known_outputs = [
        value
        for value in (candidate_tokens, thought_tokens)
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    output_tokens = sum(known_outputs) if known_outputs else None
    if output_tokens is None and all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (total_tokens, prompt_tokens)
    ):
        output_tokens = max(total_tokens - prompt_tokens, 0)
    usage.update(
        {
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": thought_tokens,
            "cached_tokens": cached_tokens,
        }
    )
    return usage, output_tokens, thought_tokens, usage_metadata.get("trafficType")


def _google_visible_reasoning(answer_text: str) -> str | None:
    """Recover provider-visible rationale when Gemini combines it with the answer.

    Gemini normally emits visible thought summaries in parts marked
    ``thought=true``. Some model versions instead put the rationale and final
    ranking in one ordinary text part while returning only an opaque thought
    signature for the hidden chain. In that case the prose preceding the final
    ranking is still provider-visible reasoning and can be preserved without
    claiming that the encrypted chain was exposed.
    """
    text = answer_text.strip()
    if not text:
        return None

    for match in reversed(list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.I))):
        if '"ranking"' in match.group(1):
            visible = f"{text[: match.start()]}\n{text[match.end() :]}".strip()
            return visible or None

    marker = text.rfind('{"ranking"')
    if marker >= 0:
        candidate = text[marker:].strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict) and isinstance(parsed.get("ranking"), list):
                visible = text[:marker].strip()
                return visible or None
    return None


def _google_vertex(
    spec: dict,
    system: str,
    user: str,
    timeout: int,
    *,
    reasoning_summary: bool = True,
) -> dict:
    """Make exactly one synchronous native Vertex ``generateContent`` request."""
    del reasoning_summary  # Vertex thought visibility is frozen by the registry.
    generation_config = {
        "maxOutputTokens": spec["max_tokens"],
        "thinkingConfig": {
            "thinkingLevel": spec["reasoning"].upper(),
            "includeThoughts": spec["include_thoughts"],
        },
    }
    if "temperature" in spec:
        generation_config["temperature"] = spec["temperature"]
    request_body = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": generation_config,
    }
    response = _google_authorized_session(spec).post(
        _endpoint(spec),
        json=request_body,
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        body = response.json()
    except ValueError as parse_error:
        error = ValueError("Vertex generateContent returned invalid JSON")
        error.response = response
        raise error from parse_error
    if not isinstance(body, dict):
        error = ValueError("Vertex generateContent returned a non-object JSON payload")
        error.response = response
        raise error

    candidates = body.get("candidates") or []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else None
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else []
    parts = parts if isinstance(parts, list) else []
    answer_parts: list[str] = []
    thought_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            continue
        (thought_parts if part.get("thought") is True else answer_parts).append(part["text"])

    finish_reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
    usage, output_tokens, reasoning_tokens, traffic_type = _google_usage(body.get("usageMetadata"))
    truncated = str(finish_reason or "").upper() == "MAX_TOKENS"
    prompt_feedback = body.get("promptFeedback")
    block_reason = prompt_feedback.get("blockReason") if isinstance(prompt_feedback, dict) else None
    if candidate is None and block_reason:
        error = f"provider blocked prompt: {block_reason}"
        retryable = False
        status = "blocked"
    elif candidate is None:
        error = "Vertex generateContent response contained no candidates"
        retryable = True
        status = "incomplete"
    else:
        error = None
        retryable = False
        status = "incomplete" if truncated else "completed"

    model_version = body.get("modelVersion")
    answer_text = "".join(answer_parts)
    reasoning_text = "\n\n".join(thought_parts) or _google_visible_reasoning(answer_text)
    return {
        "text": answer_text,
        "reasoning_text": reasoning_text,
        "response_content": candidates,
        "usage": usage,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "response_id": body.get("responseId"),
        "response_model_id": model_version,
        "provider_response_version": model_version,
        "provider_created_at": body.get("createTime"),
        "status": status,
        "incomplete_reason": "MAX_TOKENS" if truncated else None,
        "stop_reason": finish_reason,
        "service_tier": None,
        "traffic_type": traffic_type,
        "stream_completed": None,
        "provider_response": body,
        "error": error,
        "retryable": retryable,
    }


def _call(
    spec: dict,
    system: str,
    user: str,
    timeout: int,
    *,
    reasoning_summary: bool = True,
    client_request_id: str | None = None,
    event_sink: Any = None,
) -> dict:
    if spec["api_style"] == "responses":
        return _responses(
            spec,
            system,
            user,
            timeout,
            reasoning_summary=reasoning_summary,
            client_request_id=client_request_id,
            event_sink=event_sink,
        )
    return _chat_completions(spec, system, user, timeout)


CALLERS = {
    "openai-compatible": _call,
    "google-vertex": _google_vertex,
}


def _non_retryable(message: str) -> bool:
    if _provider_failure_class(message) is not None:
        return True
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


def _error_is_retryable(error: Exception, message: str) -> bool:
    """Retry only authoritative transient HTTP or transport failures."""
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or isinstance(status_code, bool):
        status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        if status_code in {408, 409, 425, 429} or status_code >= 500:
            return True
        # A successful HTTP response with an unusable body is a transient
        # provider-response failure.  All other raised HTTP statuses are
        # deterministic for this exact request.
        return 200 <= status_code < 300

    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    transport_classes = {
        "google.auth.exceptions.TransportError",
        "httpx.ConnectError",
        "httpx.ConnectTimeout",
        "httpx.NetworkError",
        "httpx.PoolTimeout",
        "httpx.ReadError",
        "httpx.ReadTimeout",
        "httpx.RemoteProtocolError",
        "httpx.TransportError",
        "httpx.WriteError",
        "httpx.WriteTimeout",
        "openai.APIConnectionError",
        "requests.exceptions.ChunkedEncodingError",
        "requests.exceptions.ConnectionError",
        "requests.exceptions.ContentDecodingError",
        "requests.exceptions.Timeout",
    }
    return any(
        f"{error_class.__module__}.{error_class.__name__}" in transport_classes
        for error_class in type(error).__mro__
    )


def _retry_after_seconds(error: Exception) -> float | None:
    """Return a bounded Retry-After delay exposed by an SDK exception."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    if raw is None:
        raw = getattr(error, "retry_after", None)
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        try:
            target = email.utils.parsedate_to_datetime(str(raw))
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        seconds = (target - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(seconds):
        # A non-finite Retry-After (e.g. "NaN"/"inf") is not a usable delay and
        # must not be persisted as a non-standard JSON token.
        return None
    return min(max(seconds, 0.0), 3600.0)


def _summary_is_unsupported(message: str) -> bool:
    lowered = message.lower()
    return "summary" in lowered and any(
        marker in lowered
        for marker in ("unsupported", "not supported", "unknown", "unrecognized", "invalid")
    )


def _private_replacements(spec: dict) -> list[tuple[str, str]]:
    environment = _env()
    if spec["provider"] == "google-vertex":
        replacements: set[tuple[str, str]] = set()
        encoded = environment.get(spec["credentials_env"])
        project = environment.get(spec["project_env"])
        if encoded:
            replacements.add((encoded, "<redacted-google-credentials>"))
            try:
                info = _google_service_account_info(spec)
            except RuntimeError:
                info = None
            credential_project = info.get("project_id") if info else None
            if credential_project:
                replacements.add((credential_project, "<redacted-google-project>"))
        if project:
            replacements.add((project, "<redacted-google-project>"))
        return sorted(replacements, key=lambda item: len(item[0]), reverse=True)
    else:
        api_key = environment.get(spec["api_key_env"])
        base_url = environment.get(spec["base_url_env"])
        candidates = {
            (api_key or "", "<redacted-api-key>"),
            (base_url or "", "<redacted-base-url>"),
            ((base_url or "").rstrip("/"), "<redacted-base-url>"),
        }
    return sorted(
        ((value, marker) for value, marker in candidates if len(value) >= 8),
        key=lambda item: len(item[0]),
        reverse=True,
    )


def _redact_text(value: str, spec: dict) -> str:
    for private_value, marker in _private_replacements(spec):
        value = value.replace(private_value, marker)
    return value


def _redact_payload(value: Any, spec: dict) -> Any:
    """Recursively remove configured credentials and endpoint text before persistence."""
    if isinstance(value, str):
        return _redact_text(value, spec)
    if isinstance(value, list):
        return [_redact_payload(item, spec) for item in value]
    if isinstance(value, dict):
        return {
            _redact_text(key, spec) if isinstance(key, str) else key: _redact_payload(item, spec)
            for key, item in value.items()
        }
    return value


def _redact_error(message: str, spec: dict) -> str:
    return _redact_text(message, spec)


def _provider_error_payload(error: Exception) -> Any:
    """Keep a provider's structured error body without persisting HTTP headers."""
    direct_body = getattr(error, "body", None)
    direct_fields = {
        name: getattr(error, name, None)
        for name in ("code", "type", "param")
        if getattr(error, name, None) is not None
    }
    if direct_body is not None or direct_fields:
        return {
            "exception_type": type(error).__name__,
            **direct_fields,
            "body": _jsonable(direct_body),
        }
    response = getattr(error, "response", None)
    if response is None:
        return None
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = getattr(response, "text", None)
    if isinstance(body, (dict, list)) and body:
        return body
    if isinstance(body, str) and body:
        return {"raw_body": body}
    return None


def chat(
    spec: dict,
    system: str,
    user: str,
    *,
    timeout: int = 900,
    retries: int = 1,
    reasoning_summary: bool = True,
    client_request_id: str | None = None,
    event_sink: Any = None,
) -> dict:
    """Make exactly one provider request and return a normalized response.

    ``retries`` remains as a compatibility parameter, but transport retries are
    deliberately owned by :mod:`src.run`.  Keeping them there ensures every
    request gets a fresh TPM/RPM reservation and its own immutable provenance.
    """
    spec = validate_model_spec(spec)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries <= 0:
        raise ValueError("retries must be a positive integer")
    if client_request_id is not None and (
        not isinstance(client_request_id, str)
        or not client_request_id
        or len(client_request_id) > 512
        or any(character in client_request_id for character in "\r\n")
    ):
        raise ValueError("client_request_id must be a non-empty, single-line string")
    if event_sink is not None and not callable(event_sink):
        raise ValueError("event_sink must be callable")
    started = time.time()
    try:
        caller_kwargs = {"reasoning_summary": reasoning_summary}
        if spec["provider"] == "openai-compatible":
            caller_kwargs.update(
                client_request_id=client_request_id,
                event_sink=event_sink,
            )
        result = CALLERS[spec["provider"]](
            spec,
            system,
            user,
            timeout,
            **caller_kwargs,
        )
        result.setdefault("error", None)
        result.setdefault("retryable", bool(result.get("error")))
        if client_request_id is not None:
            result["client_request_id"] = client_request_id
        result["elapsed_s"] = round(time.time() - started, 1)
        return _redact_payload(result, spec)
    except EventSinkError:
        raise
    except Exception as error:  # noqa: BLE001
        message = _redact_error(f"{type(error).__name__}: {error}", spec)
        result = {
            "text": "",
            "error": message,
            "elapsed_s": round(time.time() - started, 1),
            "retryable": _error_is_retryable(error, message),
            "retry_after_s": _retry_after_seconds(error),
            "summary_unsupported": bool(
                spec["provider"] == "openai-compatible"
                and spec["api_style"] == "responses"
                and reasoning_summary
                and _summary_is_unsupported(message)
            ),
        }
        if client_request_id is not None:
            result["client_request_id"] = client_request_id
        if event_sink is not None and spec["api_style"] == "responses" and spec["stream"]:
            provider_payload = _redact_payload(_provider_error_payload(error), spec)
            failure_class = _provider_failure_class(message, provider_payload)
            if failure_class is not None:
                result["failure_class"] = failure_class
                result["retryable"] = False
            _emit_response_stream_record(
                event_sink,
                {
                    "kind": "response.transport_error",
                    "client_request_id": client_request_id,
                    "error": message,
                    "retryable": result["retryable"],
                    "failure_class": failure_class or "transport_error",
                    "provider_error": provider_payload,
                },
                spec,
            )
        else:
            provider_payload = _redact_payload(_provider_error_payload(error), spec)
        if provider_payload is not None:
            result["provider_response"] = provider_payload
            result["provider_error"] = provider_payload
        return _redact_payload(result, spec)


def token_estimator(spec: dict) -> str:
    encoding = spec.get("tokenizer_encoding")
    return f"tiktoken:{encoding}" if encoding else "utf8-bytes-upper-bound"


def estimate_tokens(text: str, spec: dict) -> int:
    encoding = spec.get("tokenizer_encoding")
    if encoding:
        import tiktoken

        return len(tiktoken.get_encoding(encoding).encode(text))
    return len(text.encode("utf-8"))
