"""Canonical model registry for the ProteinGym-LLM benchmark.

There is one entry per leaderboard model.  The registry records the reasoning
level used by the N=50 primary benchmark.  The N=10/N=100 set-size controls use
uniform ``high`` effort; other sensitivity runs override the setting at the
command line rather than creating duplicate model aliases.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MODELS: dict[str, dict] = {
    "gpt-5.6-sol": {
        "display_name": "GPT-5.6 Sol",
        "provider": "openai",
        "model_id": "gpt-5.6-sol",
        "reasoning": "max",
        "max_tokens": 128_000,
        "ctx": 400_000,
        "tokenizer_encoding": "cl100k_base",
        "supports_batch": False,
    },
    "gpt-5.5": {
        "display_name": "GPT-5.5",
        "provider": "openai",
        "model_id": "gpt-5.5",
        "reasoning": "xhigh",
        "max_tokens": 128_000,
        "ctx": 1_050_000,
        "tokenizer_encoding": "cl100k_base",
        "supports_batch": True,
    },
    "gpt-5.4-mini": {
        "display_name": "GPT-5.4 mini",
        "provider": "openai",
        "model_id": "gpt-5.4-mini",
        "reasoning": "xhigh",
        "max_tokens": 128_000,
        "ctx": 400_000,
        "tokenizer_encoding": "cl100k_base",
        "service_tier": "flex",
        "supports_batch": True,
    },
    "gpt-5.4-nano": {
        "display_name": "GPT-5.4 nano",
        "provider": "openai",
        "model_id": "gpt-5.4-nano",
        "reasoning": "xhigh",
        "max_tokens": 128_000,
        "ctx": 400_000,
        "tokenizer_encoding": "cl100k_base",
        "service_tier": "flex",
        "supports_batch": True,
    },
    "claude-opus-4.8": {
        "display_name": "Claude Opus 4.8",
        "provider": "anthropic",
        "model_id": "claude-opus-4-8",
        "reasoning": "max",
        "max_tokens": 128_000,
        "ctx": 1_000_000,
        "supports_batch": True,
    },
    "claude-opus-4.7": {
        "display_name": "Claude Opus 4.7",
        "provider": "anthropic",
        "model_id": "claude-opus-4-7",
        "reasoning": "max",
        "max_tokens": 128_000,
        "ctx": 1_000_000,
        "supports_batch": True,
    },
    "gemini-3.5-flash": {
        "display_name": "Gemini 3.5 Flash",
        "provider": "google",
        "model_id": "gemini-3.5-flash",
        "reasoning": "high",
        "max_tokens": 64_000,
        "ctx": 1_000_000,
        "supports_batch": True,
    },
    "gemini-3.1-pro": {
        "display_name": "Gemini 3.1 Pro",
        "provider": "google",
        "model_id": "gemini-3.1-pro-preview",
        "reasoning": "high",
        "max_tokens": 64_000,
        "ctx": 1_000_000,
        "supports_batch": True,
    },
    "gemini-3.1-flash-lite": {
        "display_name": "Gemini 3.1 Flash-Lite",
        "provider": "google",
        "model_id": "gemini-3.1-flash-lite-preview",
        "reasoning": "high",
        "max_tokens": 64_000,
        "ctx": 1_000_000,
        "supports_batch": True,
    },
    "glm-5.2": {
        "display_name": "GLM-5.2",
        "provider": "deepinfra",
        "model_id": "zai-org/GLM-5.2",
        "reasoning": "high",
        "max_tokens": 64_000,
        "ctx": 1_048_576,
        "supports_batch": False,
    },
}

PRIMARY_MODELS = list(MODELS)
PILOT_MODELS = ["gpt-5.6-sol", "claude-opus-4.8", "gemini-3.5-flash", "glm-5.2"]
SIZES = [10, 50, 100]
N_BATCHES = 3
PRIMARY_SIZE = 50
SET_SIZE_REASONING = "high"
MODEL_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_PROVIDERS = {"openai", "anthropic", "google", "deepinfra", "openai-compatible"}

COMMON_FIELDS = {
    "display_name",
    "provider",
    "model_id",
    "reasoning",
    "max_tokens",
    "ctx",
    "supports_batch",
    "tokenizer_encoding",
}
PROVIDER_FIELDS = {
    "openai": {"send_reasoning", "service_tier"},
    "anthropic": set(),
    "google": set(),
    "deepinfra": set(),
    "openai-compatible": {
        "api_style",
        "api_key_env",
        "base_url_env",
        "send_reasoning",
        "send_reasoning_effort",
        "service_tier",
    },
}
_SENSITIVE_FIELD = re.compile(
    r"(?:api[_-]?key|token|secret|password|credential|base[_-]?url|endpoint)", re.I
)
_URL_VALUE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SECRET_VALUE = re.compile(
    r"^(?:sk-(?:ant-)?|gh[pousr]_|github_pat_|xox[baprs]-|AIza|Bearer\s+|eyJ[^.]+\.)",
    re.I,
)


def _reject_embedded_secrets(name: str, spec: dict, allowed: set[str]) -> None:
    """Reject credential/endpoint values; registries may name env vars only."""
    for field, value in spec.items():
        if field not in allowed and _SENSITIVE_FIELD.search(field):
            hint = " use base_url_env instead" if "url" in field.lower() else ""
            raise ValueError(
                f"model {name!r} forbids embedded secret/URL field {field!r};{hint}".rstrip(";")
            )
        if field in {"api_key_env", "base_url_env"}:
            continue
        if isinstance(value, str) and (_URL_VALUE.match(value) or _SECRET_VALUE.match(value)):
            raise ValueError(
                f"model {name!r} field {field!r} looks like an embedded secret or URL; "
                "reference an environment variable instead"
            )


def _validate_model(name: str, raw: dict) -> dict:
    if not MODEL_SLUG.fullmatch(name):
        raise ValueError(f"invalid model slug {name!r}")
    if not isinstance(raw, dict):
        raise ValueError(f"model {name!r} must map to an object")
    spec = dict(raw)
    provider = spec.get("provider")
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"model {name!r} has unsupported provider {provider!r}")
    allowed = COMMON_FIELDS | PROVIDER_FIELDS[provider]
    _reject_embedded_secrets(name, spec, allowed)
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(f"model {name!r} has unknown config fields: {', '.join(unknown)}")
    for field in ("model_id", "reasoning"):
        if not isinstance(spec.get(field), str) or not spec[field]:
            raise ValueError(f"model {name!r} requires non-empty {field}")
    for field in ("max_tokens", "ctx"):
        value = spec.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"model {name!r} requires a positive integer {field}")
    if spec["max_tokens"] >= spec["ctx"]:
        raise ValueError(f"model {name!r} output ceiling must be below its context window")
    display_name = spec.setdefault("display_name", name)
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError(f"model {name!r} requires a non-empty display_name")
    supports_batch = spec.setdefault("supports_batch", False)
    if not isinstance(supports_batch, bool):
        raise ValueError(f"model {name!r} supports_batch must be boolean")
    for field in ("send_reasoning", "send_reasoning_effort"):
        if field in spec and not isinstance(spec[field], bool):
            raise ValueError(f"model {name!r} {field} must be boolean")
    if "service_tier" in spec and (
        not isinstance(spec["service_tier"], str) or not spec["service_tier"]
    ):
        raise ValueError(f"model {name!r} service_tier must be a non-empty string")
    if "tokenizer_encoding" in spec and (
        not isinstance(spec["tokenizer_encoding"], str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", spec["tokenizer_encoding"])
    ):
        raise ValueError(f"model {name!r} tokenizer_encoding must be a safe non-empty name")
    if provider == "openai":
        spec.setdefault("send_reasoning", True)
    if provider == "openai-compatible":
        style = spec.setdefault("api_style", "responses")
        if style not in {"responses", "chat"}:
            raise ValueError(f"model {name!r} api_style must be 'responses' or 'chat'")
        spec.setdefault("api_key_env", "PGLLM_API_KEY")
        spec.setdefault("base_url_env", "PGLLM_BASE_URL")
        for field in ("api_key_env", "base_url_env"):
            if not isinstance(spec[field], str) or not ENV_NAME.fullmatch(spec[field]):
                raise ValueError(f"model {name!r} {field} must be an environment variable name")
        if supports_batch:
            raise ValueError(f"model {name!r} openai-compatible endpoints do not support batch")
        if style == "responses":
            if "send_reasoning_effort" in spec:
                raise ValueError(
                    f"model {name!r} send_reasoning_effort is only valid for api_style='chat'"
                )
            spec.setdefault("send_reasoning", True)
        else:
            if "send_reasoning" in spec or "service_tier" in spec:
                field = "send_reasoning" if "send_reasoning" in spec else "service_tier"
                raise ValueError(f"model {name!r} {field} is only valid for api_style='responses'")
            spec.setdefault("send_reasoning_effort", True)
        spec["supports_batch"] = False
    return spec


def validate_model_spec(spec: dict, *, name: str = "request") -> dict:
    """Validate and normalize one model specification without a registry file."""
    return _validate_model(name, spec)


def load_model_registry(path: str | Path | None = None) -> dict[str, dict]:
    """Return built-ins plus an optional secret-free JSON registry."""
    registry = {name: _validate_model(name, spec) for name, spec in MODELS.items()}
    if path is None:
        return registry
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("custom registry must be a JSON object")
    _reject_embedded_secrets("registry", document, {"models"})
    unknown = sorted(set(document) - {"models"})
    if unknown:
        raise ValueError(f"custom registry has unknown top-level fields: {', '.join(unknown)}")
    additions = document.get("models") if isinstance(document, dict) else None
    if not isinstance(additions, dict) or not additions:
        raise ValueError("custom registry must contain a non-empty 'models' object")
    collisions = sorted(set(additions) & set(registry))
    if collisions:
        raise ValueError(f"custom registry cannot replace built-ins: {', '.join(collisions)}")
    registry.update({name: _validate_model(name, spec) for name, spec in additions.items()})
    return registry


def benchmark_spec(
    model: str,
    size: int,
    *,
    reasoning: str | None = None,
    max_output_tokens: int | None = None,
    service_tier: str | None = None,
    registry: dict[str, dict] | None = None,
) -> dict:
    """Return the immutable model request for one benchmark condition."""
    spec = dict((registry or MODELS)[model])
    if reasoning is None:
        spec["reasoning"] = spec["reasoning"] if size == PRIMARY_SIZE else SET_SIZE_REASONING
    else:
        spec["reasoning"] = reasoning
    if max_output_tokens is not None:
        spec["max_tokens"] = max_output_tokens
    if service_tier is not None:
        spec["service_tier"] = service_tier
    return _validate_model(model, spec)
