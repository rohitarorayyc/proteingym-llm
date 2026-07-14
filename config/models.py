"""Validation for user-supplied OpenAI-compatible model endpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path

N_BATCHES = 3
PRIMARY_SIZE = 50
SET_SIZE_REASONING = "high"
MODEL_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_FIELD = re.compile(
    r"(?:api[_-]?key|token|secret|password|credential|base[_-]?url|endpoint)", re.I
)
_URL_VALUE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SECRET_VALUE = re.compile(
    r"^(?:sk-(?:ant-)?|gh[pousr]_|github_pat_|xox[baprs]-|AIza|Bearer\s+|eyJ[^.]+\.)",
    re.I,
)
_FIELDS = {
    "display_name",
    "provider",
    "api_style",
    "model_id",
    "api_key_env",
    "base_url_env",
    "reasoning",
    "max_tokens",
    "ctx",
    "tokenizer_encoding",
    "send_reasoning",
    "send_reasoning_effort",
    "service_tier",
}


def _reject_embedded_secrets(name: str, spec: dict, allowed: set[str]) -> None:
    for field, value in spec.items():
        if field not in allowed and _SENSITIVE_FIELD.search(field):
            raise ValueError(
                f"model {name!r} forbids embedded secret/URL field {field!r}; "
                "reference an environment variable instead"
            )
        if field in {"api_key_env", "base_url_env"}:
            continue
        if isinstance(value, str) and (_URL_VALUE.match(value) or _SECRET_VALUE.match(value)):
            raise ValueError(
                f"model {name!r} field {field!r} looks like an embedded secret or URL; "
                "reference an environment variable instead"
            )


def validate_model_spec(spec: dict, *, name: str = "request") -> dict:
    """Validate and normalize one endpoint definition."""
    if not MODEL_SLUG.fullmatch(name):
        raise ValueError(f"invalid model slug {name!r}")
    if not isinstance(spec, dict):
        raise ValueError(f"model {name!r} must map to an object")
    normalized = dict(spec)
    _reject_embedded_secrets(name, normalized, _FIELDS)
    unknown = sorted(set(normalized) - _FIELDS)
    if unknown:
        raise ValueError(f"model {name!r} has unknown config fields: {', '.join(unknown)}")
    if normalized.get("provider") != "openai-compatible":
        raise ValueError(f"model {name!r} provider must be 'openai-compatible'")
    for field in ("model_id", "reasoning"):
        if not isinstance(normalized.get(field), str) or not normalized[field].strip():
            raise ValueError(f"model {name!r} requires non-empty {field}")
    for field in ("max_tokens", "ctx"):
        value = normalized.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"model {name!r} requires a positive integer {field}")
    if normalized["max_tokens"] >= normalized["ctx"]:
        raise ValueError(f"model {name!r} output ceiling must be below its context window")

    normalized.setdefault("display_name", name)
    style = normalized.setdefault("api_style", "responses")
    if style not in {"responses", "chat"}:
        raise ValueError(f"model {name!r} api_style must be 'responses' or 'chat'")
    normalized.setdefault("api_key_env", "PGLLM_API_KEY")
    normalized.setdefault("base_url_env", "PGLLM_BASE_URL")
    for field in ("api_key_env", "base_url_env"):
        if not isinstance(normalized[field], str) or not ENV_NAME.fullmatch(normalized[field]):
            raise ValueError(f"model {name!r} {field} must be an environment variable name")
    if "tokenizer_encoding" in normalized and (
        not isinstance(normalized["tokenizer_encoding"], str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", normalized["tokenizer_encoding"])
    ):
        raise ValueError(f"model {name!r} tokenizer_encoding must be a safe non-empty name")
    if "service_tier" in normalized and (
        style != "responses"
        or not isinstance(normalized["service_tier"], str)
        or not normalized["service_tier"]
    ):
        raise ValueError(f"model {name!r} service_tier requires a non-empty Responses value")
    if style == "responses":
        if "send_reasoning_effort" in normalized:
            raise ValueError(
                f"model {name!r} send_reasoning_effort is only valid for api_style='chat'"
            )
        normalized.setdefault("send_reasoning", True)
    else:
        if "send_reasoning" in normalized or "service_tier" in normalized:
            field = "send_reasoning" if "send_reasoning" in normalized else "service_tier"
            raise ValueError(f"model {name!r} {field} is only valid for api_style='responses'")
        normalized.setdefault("send_reasoning_effort", True)
    for field in ("send_reasoning", "send_reasoning_effort"):
        if field in normalized and not isinstance(normalized[field], bool):
            raise ValueError(f"model {name!r} {field} must be boolean")
    return normalized


def load_model_registry(path: str | Path | None) -> dict[str, dict]:
    """Load an explicit, secret-free model registry."""
    if path is None:
        raise ValueError("a model registry is required")
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("model registry must be a JSON object")
    _reject_embedded_secrets("registry", document, {"models"})
    unknown = sorted(set(document) - {"models"})
    if unknown:
        raise ValueError(f"model registry has unknown top-level fields: {', '.join(unknown)}")
    models = document.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("model registry requires a non-empty 'models' object")
    return {name: validate_model_spec(spec, name=name) for name, spec in models.items()}


def benchmark_spec(
    model: str,
    size: int,
    *,
    reasoning: str | None = None,
    max_output_tokens: int | None = None,
    service_tier: str | None = None,
    registry: dict[str, dict] | None = None,
) -> dict:
    """Return the request settings for one frozen benchmark condition."""
    if registry is None or model not in registry:
        raise ValueError(f"model {model!r} is absent from the supplied registry")
    spec = dict(registry[model])
    spec["reasoning"] = (
        reasoning
        if reasoning is not None
        else spec["reasoning"]
        if size == PRIMARY_SIZE
        else SET_SIZE_REASONING
    )
    if max_output_tokens is not None:
        spec["max_tokens"] = max_output_tokens
    if service_tier is not None:
        spec["service_tier"] = service_tier
    return validate_model_spec(spec, name=model)
