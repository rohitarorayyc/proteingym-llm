import base64
import json

import pytest

from config.models import MODELS, benchmark_spec, load_model_registry
from src import client


def _custom_document(name="lab-model"):
    return {
        "models": {
            name: {
                "provider": "openai-compatible",
                "api_style": "responses",
                "model_id": "internal-model",
                "api_key_env": "LAB_API_KEY",
                "base_url_env": "LAB_BASE_URL",
                "reasoning": "max",
                "max_tokens": 128000,
                "ctx": 1000000,
            }
        }
    }


def test_custom_registry_adds_internal_model_without_secrets(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_custom_document()))
    registry = load_model_registry(path)
    assert registry["lab-model"]["provider"] == "openai-compatible"
    assert benchmark_spec("lab-model", 50, registry=registry)["reasoning"] == "max"
    assert benchmark_spec("lab-model", 10, registry=registry)["reasoning"] == "high"
    assert "openai-compatible" in client.CALLERS


def test_custom_registry_rejects_collision_or_unsafe_slug(tmp_path):
    for document, message in (
        (_custom_document("gpt-5.5"), "cannot replace built-ins"),
        (_custom_document("../escape"), "invalid model slug"),
    ):
        path = tmp_path / "models.json"
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match=message):
            load_model_registry(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("api_key", "sk-do-not-store-this", "forbids embedded secret/URL field"),
        ("base_url", "https://private.example/v1", "forbids embedded secret/URL field"),
        ("unexpected", "value", "unknown config fields"),
        ("base_url_env", "https://private.example/v1", "environment variable name"),
    ],
)
def test_registry_rejects_embedded_secrets_urls_and_unknown_fields(tmp_path, field, value, message):
    document = _custom_document()
    document["models"]["lab-model"][field] = value
    path = tmp_path / "models.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        load_model_registry(path)


def test_registry_rejects_provider_irrelevant_or_secret_looking_values(tmp_path):
    document = _custom_document()
    document["models"]["lab-model"]["model_id"] = "sk-do-not-store-this"
    path = tmp_path / "models.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="looks like an embedded secret"):
        load_model_registry(path)

    anthropic = dict(MODELS["claude-opus-4.8"], service_tier="flex")
    path.write_text(json.dumps({"models": {"internal-claude": anthropic}}))
    with pytest.raises(ValueError, match="unknown config fields: service_tier"):
        load_model_registry(path)


def test_registry_rejects_top_level_secrets_and_unknown_fields(tmp_path):
    path = tmp_path / "models.json"
    document = _custom_document()
    document["api_key"] = "sk-do-not-store-this"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="forbids embedded secret/URL field"):
        load_model_registry(path)

    document = _custom_document()
    document["notes"] = "not part of the registry schema"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unknown top-level fields: notes"):
        load_model_registry(path)


def test_chat_style_rejects_responses_only_options(tmp_path):
    document = _custom_document()
    spec = document["models"]["lab-model"]
    spec["api_style"] = "chat"
    spec["send_reasoning"] = True
    with pytest.raises(ValueError, match="send_reasoning.*only valid.*responses"):
        path = tmp_path / "models.json"
        path.write_text(json.dumps(document))
        load_model_registry(path)


def test_custom_request_provenance_is_stable_and_secret_free(monkeypatch, tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_custom_document()))
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
    assert descriptor["inference_options"]["send_reasoning"] is True

    environment["LAB_BASE_URL"] = "https://other.private.example/v1"
    changed = client.request_provenance(spec)
    assert changed["request_descriptor"]["endpoint_sha256"] != descriptor["endpoint_sha256"]
    assert changed["request_fingerprint"] != first["request_fingerprint"]


def test_custom_fingerprint_requires_resolved_endpoint(monkeypatch, tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_custom_document()))
    spec = benchmark_spec("lab-model", 50, registry=load_model_registry(path))
    monkeypatch.setattr(client, "_env", lambda: {})
    with pytest.raises(RuntimeError, match="LAB_BASE_URL not set"):
        client.request_provenance(spec)


def test_delivery_mode_and_region_are_frozen_into_request_fingerprint():
    spec = benchmark_spec("gpt-5.5", 50)
    live = client.request_provenance(spec)
    batch = client.request_provenance(
        spec,
        delivery_mode="batch",
        delivery_region="us-central1",
    )

    assert live["request_fingerprint"] != batch["request_fingerprint"]
    assert live["request_descriptor"]["delivery_mode"] == "live"
    assert batch["request_descriptor"]["delivery_mode"] == "batch"
    assert batch["request_descriptor"]["delivery_region"] == "us-central1"


def test_request_provenance_supports_every_builtin(monkeypatch):
    gcp = base64.b64encode(json.dumps({"project_id": "test-project"}).encode()).decode()
    monkeypatch.setattr(client, "_env", lambda: {"GCP_KEY_JSON": gcp})

    fingerprints = set()
    for model in MODELS:
        spec = benchmark_spec(model, 50)
        provenance = client.request_provenance(spec)
        descriptor = provenance["request_descriptor"]
        assert descriptor["provider"] == spec["provider"]
        assert descriptor["model_id"] == spec["model_id"]
        assert descriptor["endpoint_sha256"]
        fingerprints.add(provenance["request_fingerprint"])
    assert len(fingerprints) == len(MODELS)
