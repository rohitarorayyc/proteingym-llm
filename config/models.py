"""Validation for secret-free model endpoint registries."""

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
    "credentials_env",
    "project_env",
    "location",
    "api_version",
    "temperature",
    "include_thoughts",
    "reasoning",
    "max_tokens",
    "ctx",
    "tokenizer_encoding",
    "send_reasoning",
    "send_reasoning_effort",
    "chat_output_token_field",
    "stream",
    "probe_reasoning",
    "probe_max_tokens",
    "allowed_sizes",
    "require_usage",
    "require_reasoning",
    "response_model_ids",
    "service_tier",
    "leaderboard_preset",
}


def _value_has_embedded_secret(value: object) -> bool:
    """Scan a field value, including list items (e.g. response_model_ids), for a secret.

    Lists are recursed so a secret hidden in a list entry cannot bypass the check;
    dicts are not, because per-field validation (including the api/base/credential
    env-name exemption) is applied to each spec field individually by the caller.
    """
    if isinstance(value, str):
        return bool(_URL_VALUE.match(value) or _SECRET_VALUE.match(value))
    if isinstance(value, list):
        return any(_value_has_embedded_secret(item) for item in value)
    return False


def _reject_embedded_secrets(name: str, spec: dict, allowed: set[str]) -> None:
    for field, value in spec.items():
        if field not in allowed and _SENSITIVE_FIELD.search(field):
            raise ValueError(
                f"model {name!r} forbids embedded secret/URL field {field!r}; "
                "reference an environment variable instead"
            )
        if field in {"api_key_env", "base_url_env", "credentials_env", "project_env"}:
            continue
        if _value_has_embedded_secret(value):
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
    provider = normalized.get("provider")
    if provider not in {"openai-compatible", "google-vertex"}:
        raise ValueError(f"model {name!r} provider must be 'openai-compatible' or 'google-vertex'")
    for field in ("model_id", "reasoning"):
        if not isinstance(normalized.get(field), str) or not normalized[field].strip():
            raise ValueError(f"model {name!r} requires non-empty {field}")
    if "probe_reasoning" in normalized and (
        not isinstance(normalized["probe_reasoning"], str)
        or not normalized["probe_reasoning"].strip()
    ):
        raise ValueError(f"model {name!r} probe_reasoning must be a non-empty string")
    if "allowed_sizes" in normalized:
        allowed_sizes = normalized["allowed_sizes"]
        if (
            not isinstance(allowed_sizes, list)
            or not allowed_sizes
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in allowed_sizes
            )
            or len(set(allowed_sizes)) != len(allowed_sizes)
        ):
            raise ValueError(
                f"model {name!r} allowed_sizes must be a non-empty list of unique positive integers"
            )
    if "response_model_ids" in normalized:
        response_model_ids = normalized["response_model_ids"]
        if (
            not isinstance(response_model_ids, list)
            or any(not isinstance(value, str) or not value.strip() for value in response_model_ids)
            or len(set(response_model_ids)) != len(response_model_ids)
        ):
            raise ValueError(
                f"model {name!r} response_model_ids must be a list of unique non-empty strings"
            )
    for field in ("max_tokens", "ctx"):
        value = normalized.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"model {name!r} requires a positive integer {field}")
    if "probe_max_tokens" in normalized:
        probe_max_tokens = normalized["probe_max_tokens"]
        if (
            not isinstance(probe_max_tokens, int)
            or isinstance(probe_max_tokens, bool)
            or probe_max_tokens <= 0
            or probe_max_tokens > normalized["max_tokens"]
        ):
            raise ValueError(
                f"model {name!r} probe_max_tokens must be a positive integer no larger "
                "than max_tokens"
            )
    if normalized["max_tokens"] >= normalized["ctx"]:
        raise ValueError(f"model {name!r} output limit must be below its context window")

    normalized.setdefault("display_name", name)
    style = normalized.setdefault(
        "api_style", "responses" if provider == "openai-compatible" else "generate-content"
    )
    if provider == "openai-compatible":
        if style not in {"responses", "chat"}:
            raise ValueError(f"model {name!r} api_style must be 'responses' or 'chat'")
        normalized.setdefault("api_key_env", "PGLLM_API_KEY")
        normalized.setdefault("base_url_env", "PGLLM_BASE_URL")
        for field in ("api_key_env", "base_url_env"):
            if not isinstance(normalized[field], str) or not ENV_NAME.fullmatch(normalized[field]):
                raise ValueError(f"model {name!r} {field} must be an environment variable name")
    else:
        if style != "generate-content":
            raise ValueError(f"model {name!r} google-vertex api_style must be 'generate-content'")
        forbidden = sorted(
            field
            for field in (
                "api_key_env",
                "base_url_env",
                "send_reasoning",
                "send_reasoning_effort",
                "chat_output_token_field",
                "stream",
                "service_tier",
            )
            if field in normalized
        )
        if forbidden:
            raise ValueError(
                f"model {name!r} google-vertex forbids OpenAI-only field(s): "
                + ", ".join(forbidden)
            )
        normalized.setdefault("project_env", "GOOGLE_CLOUD_PROJECT")
        normalized.setdefault("credentials_env", "GCP_KEY_JSON")
        for field in ("project_env", "credentials_env"):
            if not isinstance(normalized[field], str) or not ENV_NAME.fullmatch(normalized[field]):
                raise ValueError(f"model {name!r} {field} must be an environment variable name")
        location = normalized.setdefault("location", "global")
        if not isinstance(location, str) or not re.fullmatch(r"[a-z0-9-]+", location):
            raise ValueError(f"model {name!r} location must be a safe Vertex region")
        api_version = normalized.setdefault("api_version", "v1")
        if not isinstance(api_version, str) or not re.fullmatch(
            r"v[0-9]+(?:beta[0-9]+)?", api_version
        ):
            raise ValueError(f"model {name!r} api_version must be a safe Google API version")
        temperature = normalized.setdefault("temperature", 1.0)
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not 0.0 < float(temperature) <= 2.0
        ):
            raise ValueError(f"model {name!r} temperature must be in (0, 2]")
        normalized["temperature"] = float(temperature)
        normalized.setdefault("include_thoughts", True)
        if not isinstance(normalized["include_thoughts"], bool):
            raise ValueError(f"model {name!r} include_thoughts must be boolean")
        if normalized["reasoning"] not in {"minimal", "low", "medium", "high"}:
            raise ValueError(
                f"model {name!r} google-vertex reasoning must be minimal, low, medium, or high"
            )
    if "tokenizer_encoding" in normalized and (
        not isinstance(normalized["tokenizer_encoding"], str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", normalized["tokenizer_encoding"])
    ):
        raise ValueError(f"model {name!r} tokenizer_encoding must be a safe non-empty name")
    if "service_tier" in normalized and (
        provider != "openai-compatible"
        or style != "responses"
        or not isinstance(normalized["service_tier"], str)
        or not normalized["service_tier"]
    ):
        raise ValueError(f"model {name!r} service_tier requires a non-empty Responses value")
    if provider == "google-vertex":
        pass
    elif style == "responses":
        if "send_reasoning_effort" in normalized:
            raise ValueError(
                f"model {name!r} send_reasoning_effort is only valid for api_style='chat'"
            )
        if "chat_output_token_field" in normalized:
            raise ValueError(
                f"model {name!r} chat_output_token_field is only valid for api_style='chat'"
            )
        normalized.setdefault("send_reasoning", True)
        normalized.setdefault("stream", False)
    else:
        if "send_reasoning" in normalized or "service_tier" in normalized:
            field = "send_reasoning" if "send_reasoning" in normalized else "service_tier"
            raise ValueError(f"model {name!r} {field} is only valid for api_style='responses'")
        normalized.setdefault("send_reasoning_effort", True)
        normalized.setdefault("stream", False)
        output_field = normalized.setdefault("chat_output_token_field", "max_tokens")
        if output_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                f"model {name!r} chat_output_token_field must be "
                "'max_tokens' or 'max_completion_tokens'"
            )
    for field in (
        "send_reasoning",
        "send_reasoning_effort",
        "require_usage",
        "require_reasoning",
        "stream",
        "leaderboard_preset",
    ):
        if field in normalized and not isinstance(normalized[field], bool):
            raise ValueError(f"model {name!r} {field} must be boolean")
    normalized.setdefault("require_usage", False)
    normalized.setdefault("require_reasoning", False)
    normalized.setdefault("response_model_ids", [])
    normalized.setdefault("leaderboard_preset", False)
    if (
        provider == "google-vertex"
        and normalized["require_reasoning"]
        and not normalized["include_thoughts"]
    ):
        raise ValueError(f"model {name!r} require_reasoning=true requires include_thoughts=true")
    if normalized["leaderboard_preset"] and (
        not normalized["require_usage"] or not normalized["response_model_ids"]
    ):
        raise ValueError(
            f"model {name!r} leaderboard_preset requires require_usage=true and "
            "at least one explicit response_model_ids value"
        )
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
    if "allowed_sizes" in spec and size not in spec["allowed_sizes"]:
        allowed = ", ".join(str(value) for value in spec["allowed_sizes"])
        raise ValueError(f"model {model!r} is restricted to set size(s): {allowed}")
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
