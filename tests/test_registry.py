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


def _google_document(name="gemini-test"):
    return {
        "models": {
            name: {
                "provider": "google-vertex",
                "api_style": "generate-content",
                "model_id": "gemini-test",
                "credentials_env": "GCP_KEY_JSON",
                "project_env": "GOOGLE_CLOUD_PROJECT",
                "location": "global",
                "api_version": "v1",
                "reasoning": "high",
                "max_tokens": 65535,
                "ctx": 1048576,
                "temperature": 1.0,
                "include_thoughts": True,
                "require_usage": True,
                "require_reasoning": True,
                "response_model_ids": ["gemini-test"],
                "allowed_sizes": [50],
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
    registry = load_model_registry("examples/model.json")
    assert set(registry) == {"lab-model"}
    assert registry["lab-model"]["reasoning"] == "max"


def test_shipped_chat_template_is_valid():
    registry = load_model_registry("examples/chat_model.json")
    spec = benchmark_spec("chat-model", 50, registry=registry)

    assert set(registry) == {"chat-model"}
    assert spec["api_style"] == "chat"
    assert spec["model_id"] == "replace-with-model-id"
    assert spec["reasoning"] == "max"
    assert spec["chat_output_token_field"] == "max_completion_tokens"
    assert spec["stream"] is True
    assert spec["require_usage"] is True
    assert spec["leaderboard_preset"] is True
    assert spec["require_reasoning"] is False
    assert spec["response_model_ids"] == ["replace-with-provider-returned-model-id"]
    assert spec["max_tokens"] == 128000
    assert spec["ctx"] == 1000000


def test_google_vertex_registry_freezes_native_generate_content_contract(tmp_path):
    path = tmp_path / "google.json"
    path.write_text(json.dumps(_google_document()))
    registry = load_model_registry(path)
    spec = benchmark_spec("gemini-test", 50, registry=registry)

    assert spec["provider"] == "google-vertex"
    assert spec["api_style"] == "generate-content"
    assert spec["credentials_env"] == "GCP_KEY_JSON"
    assert spec["project_env"] == "GOOGLE_CLOUD_PROJECT"
    assert spec["location"] == "global"
    assert spec["api_version"] == "v1"
    assert spec["temperature"] == 1.0
    assert spec["include_thoughts"] is True
    assert "google-vertex" in client.CALLERS
    with pytest.raises(ValueError, match="restricted to set size.*50"):
        benchmark_spec("gemini-test", 10, registry=registry)


def test_google_registry_rejects_hidden_required_reasoning(tmp_path):
    document = _google_document()
    document["models"]["gemini-test"]["include_thoughts"] = False
    path = tmp_path / "google.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="require_reasoning=true requires include_thoughts=true"):
        load_model_registry(path)


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


def test_registry_rejects_secret_hidden_in_response_model_ids_list(tmp_path):
    path = tmp_path / "models.json"
    document = _document()
    document["models"]["lab-model"]["response_model_ids"] = [
        "internal-model",
        "sk-live-should-not-be-stored",
    ]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="looks like an embedded secret or URL"):
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


def test_registry_validates_chat_output_field_and_allowed_sizes(tmp_path):
    path = tmp_path / "models.json"
    document = _document()
    spec = document["models"]["lab-model"]
    spec.update(
        {
            "api_style": "chat",
            "chat_output_token_field": "unsupported",
            "allowed_sizes": [50],
        }
    )
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="chat_output_token_field"):
        load_model_registry(path)

    spec["chat_output_token_field"] = "max_completion_tokens"
    spec["allowed_sizes"] = [50, 50]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="allowed_sizes"):
        load_model_registry(path)


def test_leaderboard_preset_requires_usage_and_explicit_response_identity(tmp_path):
    path = tmp_path / "models.json"
    document = _document()
    document["models"]["lab-model"]["leaderboard_preset"] = True
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="leaderboard_preset requires require_usage"):
        load_model_registry(path)

    document["models"]["lab-model"].update(
        {"require_usage": True, "response_model_ids": ["internal-model"]}
    )
    path.write_text(json.dumps(document))
    assert load_model_registry(path)["lab-model"]["leaderboard_preset"] is True


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
    assert descriptor["descriptor_version"] == 2

    environment["LAB_BASE_URL"] = "https://other.private.example/v1"
    changed = client.request_provenance(spec)
    assert changed["request_fingerprint"] != first["request_fingerprint"]


def test_google_request_provenance_is_stable_and_hides_project(monkeypatch, tmp_path):
    path = tmp_path / "google.json"
    path.write_text(json.dumps(_google_document()))
    spec = benchmark_spec("gemini-test", 50, registry=load_model_registry(path))
    project = "private-research-project"
    monkeypatch.setattr(client, "_env", lambda: {"GOOGLE_CLOUD_PROJECT": project})

    first = client.request_provenance(spec)
    second = client.request_provenance(dict(reversed(list(spec.items()))))
    serialized = json.dumps(first, sort_keys=True)

    assert first == second
    assert project not in serialized
    descriptor = first["request_descriptor"]
    assert descriptor["provider"] == "google-vertex"
    assert descriptor["api_style"] == "generate-content"
    assert descriptor["api_version"] == "v1"
    assert descriptor["location"] == "global"
    assert descriptor["credentials_env"] == "GCP_KEY_JSON"
    assert descriptor["project_env"] == "GOOGLE_CLOUD_PROJECT"
    assert descriptor["inference_options"]["thinking"] == {
        "level": "high",
        "include_thoughts": True,
    }
    assert len(descriptor["endpoint_sha256"]) == 64
    assert descriptor["max_output_tokens"] == 65535
    assert descriptor["descriptor_version"] == 2


def test_request_provenance_requires_resolved_endpoint(monkeypatch, tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_document()))
    spec = benchmark_spec("lab-model", 50, registry=load_model_registry(path))
    monkeypatch.setattr(client, "_env", lambda: {})
    with pytest.raises(RuntimeError, match="LAB_BASE_URL not set"):
        client.request_provenance(spec)
