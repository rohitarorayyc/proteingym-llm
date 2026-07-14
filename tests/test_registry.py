import json

import pytest

from config.models import benchmark_spec, load_model_registry
from src import client


def _document(name="lab-model"):
    return {
        "models": {
            name: {
                "provider": "openai-compatible",
                "model_id": "internal-model",
                "api_key_env": "LAB_API_KEY",
                "base_url_env": "LAB_BASE_URL",
                "reasoning": "max",
                "max_tokens": 128000,
                "ctx": 1000000,
            }
        }
    }


def test_registry_loads_user_model_and_applies_set_size_protocol(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_document()))
    registry = load_model_registry(path)

    assert set(registry) == {"lab-model"}
    assert registry["lab-model"]["api_style"] == "responses"
    assert benchmark_spec("lab-model", 50, registry=registry)["reasoning"] == "max"
    assert benchmark_spec("lab-model", 10, registry=registry)["reasoning"] == "high"
    assert "openai-compatible" in client.CALLERS


def test_shipped_endpoint_template_is_valid():
    registry = load_model_registry("examples/internal_model.json")
    assert set(registry) == {"lab-model"}
    assert registry["lab-model"]["reasoning"] == "max"


def test_registry_requires_explicit_file_and_safe_model_slug(tmp_path):
    with pytest.raises(ValueError, match="registry is required"):
        load_model_registry(None)

    path = tmp_path / "models.json"
    path.write_text(json.dumps(_document("../escape")))
    with pytest.raises(ValueError, match="invalid model slug"):
        load_model_registry(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("api_key", "sk-do-not-store-this", "forbids embedded secret/URL field"),
        ("base_url", "https://private.example/v1", "forbids embedded secret/URL field"),
        ("unexpected", "value", "unknown config fields"),
        ("base_url_env", "https://private.example/v1", "environment variable name"),
        ("provider", "anthropic", "provider must be 'openai-compatible'"),
    ],
)
def test_registry_rejects_secrets_urls_unknown_fields_and_other_providers(
    tmp_path, field, value, message
):
    document = _document()
    document["models"]["lab-model"][field] = value
    path = tmp_path / "models.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        load_model_registry(path)


def test_registry_rejects_top_level_secrets_and_unknown_fields(tmp_path):
    path = tmp_path / "models.json"
    document = _document()
    document["api_key"] = "sk-do-not-store-this"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="forbids embedded secret/URL field"):
        load_model_registry(path)

    document = _document()
    document["notes"] = "not part of the schema"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unknown top-level fields"):
        load_model_registry(path)


def test_chat_style_rejects_responses_only_options(tmp_path):
    document = _document()
    spec = document["models"]["lab-model"]
    spec["api_style"] = "chat"
    spec["send_reasoning"] = True
    path = tmp_path / "models.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="send_reasoning.*only valid.*responses"):
        load_model_registry(path)


def test_request_provenance_is_stable_and_secret_free(monkeypatch, tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_document()))
    spec = benchmark_spec("lab-model", 50, registry=load_model_registry(path))
    private_url = "https://inference.private.example/v1"
    secret = "sk-private-credential"
    environment = {"LAB_BASE_URL": private_url, "LAB_API_KEY": secret}
    monkeypatch.setattr(client, "_env", lambda: environment)

    first = client.request_provenance(spec)
    second = client.request_provenance(dict(reversed(list(spec.items()))))
    serialized = json.dumps(first, sort_keys=True)

    assert first == second
    assert len(first["request_fingerprint"]) == 64
    assert private_url not in serialized
    assert secret not in serialized
    descriptor = first["request_descriptor"]
    assert descriptor["base_url_env"] == "LAB_BASE_URL"
    assert descriptor["api_key_env"] == "LAB_API_KEY"
    assert len(descriptor["endpoint_sha256"]) == 64
    assert descriptor["reasoning_effort"] == "max"
    assert descriptor["max_output_tokens"] == 128000

    environment["LAB_BASE_URL"] = "https://other.private.example/v1"
    changed = client.request_provenance(spec)
    assert changed["request_fingerprint"] != first["request_fingerprint"]


def test_request_provenance_requires_resolved_endpoint(monkeypatch, tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_document()))
    spec = benchmark_spec("lab-model", 50, registry=load_model_registry(path))
    monkeypatch.setattr(client, "_env", lambda: {})
    with pytest.raises(RuntimeError, match="LAB_BASE_URL not set"):
        client.request_provenance(spec)
