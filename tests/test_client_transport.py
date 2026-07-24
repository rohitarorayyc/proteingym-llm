import base64
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from config.models import benchmark_spec, load_model_registry
from src import client


class Dumpable(SimpleNamespace):
    def model_dump(self, mode="json"):
        assert mode == "json"
        return _deep_dump(dict(self.__dict__))


def _deep_dump(value):
    if isinstance(value, SimpleNamespace):
        return {key: _deep_dump(item) for key, item in value.__dict__.items()}
    if isinstance(value, list):
        return [_deep_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _deep_dump(item) for key, item in value.items()}
    return value


def _google_spec(reasoning="high"):
    return {
        "provider": "google-vertex",
        "api_style": "generate-content",
        "model_id": "gemini-3.5-flash",
        "credentials_env": "GCP_KEY_JSON",
        "project_env": "GOOGLE_CLOUD_PROJECT",
        "location": "global",
        "api_version": "v1",
        "reasoning": reasoning,
        "max_tokens": 65535,
        "ctx": 1048576,
        "include_thoughts": True,
        "require_usage": True,
        "require_reasoning": True,
        "response_model_ids": ["gemini-3.5-flash"],
    }


class FakeGoogleResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            error = RuntimeError(f"{self.status_code} Client Error")
            error.response = self
            raise error


def test_google_vertex_native_request_and_response_are_lossless(monkeypatch):
    sent = {}
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"thought": True, "text": "Compare the substitutions."},
                        {
                            "text": '{"ranking":["M01","M02"]}',
                            "thoughtSignature": "opaque-signature",
                        },
                    ],
                },
                "finishReason": "STOP",
                "safetyRatings": [],
            }
        ],
        "createTime": "2026-07-17T12:34:56.123456Z",
        "modelVersion": "gemini-3.5-flash",
        "responseId": "vertex-response-1",
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 12,
            "thoughtsTokenCount": 40,
            "totalTokenCount": 152,
            "cachedContentTokenCount": 5,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 100}],
            "trafficType": "ON_DEMAND",
        },
    }

    class Session:
        @staticmethod
        def post(url, **kwargs):
            sent["url"] = url
            sent.update(kwargs)
            return FakeGoogleResponse(payload)

    monkeypatch.setattr(client, "_google_authorized_session", lambda _spec: Session())
    monkeypatch.setattr(client, "_env", lambda: {"GOOGLE_CLOUD_PROJECT": "private-project"})

    result = client.chat(_google_spec(), "system", "user", timeout=45, retries=3)

    assert sent["url"].endswith(
        "/v1/projects/private-project/locations/global/publishers/google/models/"
        "gemini-3.5-flash:generateContent"
    )
    assert sent["timeout"] == 45
    assert sent["json"] == {
        "contents": [{"role": "user", "parts": [{"text": "user"}]}],
        "systemInstruction": {"parts": [{"text": "system"}]},
        "generationConfig": {
            "maxOutputTokens": 65535,
            "thinkingConfig": {"thinkingLevel": "HIGH", "includeThoughts": True},
        },
    }
    descriptor = client.public_request_descriptor(_google_spec())
    assert descriptor["reasoning_effort"] == "high"
    assert descriptor["inference_options"]["temperature"] == "provider_default"
    assert descriptor["inference_options"]["thinking"]["level"] == "high"
    assert result["text"] == '{"ranking":["M01","M02"]}'
    assert result["reasoning_text"] == "Compare the substitutions."
    assert result["output_tokens"] == 52
    assert result["reasoning_tokens"] == 40
    assert result["usage"]["input_tokens"] == 100
    assert result["usage"]["output_tokens"] == 52
    assert result["usage"]["promptTokensDetails"] == [{"modality": "TEXT", "tokenCount": 100}]
    assert result["traffic_type"] == "ON_DEMAND"
    assert result["response_id"] == "vertex-response-1"
    assert result["response_model_id"] == "gemini-3.5-flash"
    assert result["provider_response_version"] == "gemini-3.5-flash"
    assert result["provider_created_at"] == "2026-07-17T12:34:56.123456Z"
    assert result["stop_reason"] == "STOP"
    assert result["status"] == "completed"
    assert result["provider_response"] == payload
    assert result["response_content"] == payload["candidates"]
    assert (
        result["provider_response"]["candidates"][0]["content"]["parts"][1]["thoughtSignature"]
        == "opaque-signature"
    )


def test_google_vertex_max_tokens_is_explicitly_truncated(monkeypatch):
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"thought": True, "text": "unfinished thought"}],
                },
                "finishReason": "MAX_TOKENS",
            }
        ],
        "modelVersion": "gemini-3.5-flash",
        "responseId": "vertex-capped",
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 0,
            "thoughtsTokenCount": 65535,
            "totalTokenCount": 65635,
            "trafficType": "ON_DEMAND",
        },
    }

    class Session:
        @staticmethod
        def post(*_args, **_kwargs):
            return FakeGoogleResponse(payload)

    monkeypatch.setattr(client, "_google_authorized_session", lambda _spec: Session())
    monkeypatch.setattr(client, "_env", lambda: {"GOOGLE_CLOUD_PROJECT": "project"})

    result = client.chat(_google_spec(), "system", "user")

    assert result["status"] == "incomplete"
    assert result["incomplete_reason"] == "MAX_TOKENS"
    assert result["stop_reason"] == "MAX_TOKENS"
    assert result["output_tokens"] == 65535
    assert result["reasoning_text"] == "unfinished thought"
    assert result["provider_response"] == payload


def test_google_vertex_http_errors_preserve_payload_and_retry_class(monkeypatch):
    monkeypatch.setattr(client, "_env", lambda: {"GOOGLE_CLOUD_PROJECT": "project"})

    def response_for(status_code):
        class Session:
            @staticmethod
            def post(*_args, **_kwargs):
                return FakeGoogleResponse(
                    {"error": {"code": status_code, "message": "provider message"}},
                    status_code=status_code,
                    headers={"retry-after": "7"},
                )

        monkeypatch.setattr(client, "_google_authorized_session", lambda _spec: Session())
        return client.chat(_google_spec(), "system", "user")

    overloaded = response_for(429)
    invalid = response_for(400)

    assert overloaded["retryable"] is True
    assert overloaded["retry_after_s"] == 7.0
    assert overloaded["provider_response"] == {
        "error": {"code": 429, "message": "provider message"}
    }
    assert invalid["retryable"] is False
    assert invalid["provider_response"] == {"error": {"code": 400, "message": "provider message"}}


def test_google_vertex_malformed_body_is_preserved(monkeypatch):
    class Response:
        status_code = 200
        headers = {}
        text = "malformed-provider-body"

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            raise ValueError("not JSON")

    class Session:
        @staticmethod
        def post(*_args, **_kwargs):
            return Response()

    monkeypatch.setattr(client, "_google_authorized_session", lambda _spec: Session())
    monkeypatch.setattr(client, "_env", lambda: {"GOOGLE_CLOUD_PROJECT": "project"})

    result = client.chat(_google_spec(), "system", "user")

    assert result["error"] == "ValueError: Vertex generateContent returned invalid JSON"
    assert result["retryable"] is True
    assert result["provider_response"] == {"raw_body": "malformed-provider-body"}


def test_google_vertex_errors_redact_project_identity(monkeypatch):
    spec = _google_spec()
    project = "abc123"
    monkeypatch.setattr(client, "_env", lambda: {"GOOGLE_CLOUD_PROJECT": project})

    payload = {
        "error": f"request to projects/{project}/locations/global failed",
        "provider_response": {"resource": f"projects/{project}"},
    }
    redacted = client._redact_payload(payload, spec)

    assert project not in str(redacted)
    assert "<redacted-google-project>" in str(redacted)


def test_google_vertex_uses_base64_service_account_in_memory(monkeypatch):
    from google.oauth2 import service_account

    project = "credential-project"
    private_key = "private-key-material-that-must-never-be-persisted"
    info = {
        "type": "service_account",
        "project_id": project,
        "client_email": "runner@example.invalid",
        "private_key": private_key,
    }
    encoded = base64.b64encode(json.dumps(info).encode()).decode()
    monkeypatch.setattr(client, "_env", lambda: {"GCP_KEY_JSON": encoded})

    seen = {}
    credentials = SimpleNamespace(valid=True)

    def from_info(value, *, scopes):
        seen["info"] = value
        seen["scopes"] = scopes
        return credentials

    monkeypatch.setattr(
        service_account.Credentials,
        "from_service_account_info",
        staticmethod(from_info),
    )
    client._google_credentials.cache_clear()
    try:
        spec = _google_spec()
        assert client._google_project(spec) == project
        assert client._google_credentials("GCP_KEY_JSON", "GOOGLE_CLOUD_PROJECT") is credentials
        assert seen["info"] == info
        assert seen["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]

        descriptor = client.public_request_descriptor(spec)
        serialized = json.dumps(descriptor, sort_keys=True)
        assert encoded not in serialized
        assert private_key not in serialized
        assert project not in serialized
        assert descriptor["credentials_env"] == "GCP_KEY_JSON"
        assert descriptor["project_env"] == "GOOGLE_CLOUD_PROJECT"

        redacted = client._redact_payload({"encoded": encoded, "project": project}, spec)
        assert encoded not in json.dumps(redacted)
        assert project not in json.dumps(redacted)
    finally:
        client._google_credentials.cache_clear()


def test_google_vertex_rejects_conflicting_project_sources(monkeypatch):
    info = {"type": "service_account", "project_id": "credential-project"}
    encoded = base64.b64encode(json.dumps(info).encode()).decode()
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "GCP_KEY_JSON": encoded,
            "GOOGLE_CLOUD_PROJECT": "different-project",
        },
    )

    try:
        client._google_project(_google_spec())
    except RuntimeError as error:
        assert "does not match" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("conflicting project sources were accepted")


def test_chat_request_and_response_are_preserved(monkeypatch):
    sent = {}
    usage = Dumpable(
        prompt_tokens=100,
        completion_tokens=42,
        total_tokens=142,
        cached_tokens=80,
    )
    chunks = [
        Dumpable(
            id="chatcmpl-test",
            model="replace-with-provider-returned-model-id",
            created=1784174400,
            choices=[
                Dumpable(
                    delta=Dumpable(reasoning_content="I compared ", content=None),
                    finish_reason=None,
                    usage=None,
                )
            ],
            usage=None,
        ),
        Dumpable(
            id="chatcmpl-test",
            model="replace-with-provider-returned-model-id",
            created=1784174400,
            choices=[
                Dumpable(
                    delta=Dumpable(reasoning_content="the substitutions.", content=None),
                    finish_reason=None,
                    usage=None,
                )
            ],
            usage=None,
        ),
        Dumpable(
            id="chatcmpl-test",
            model="replace-with-provider-returned-model-id",
            created=1784174400,
            choices=[
                Dumpable(
                    delta=Dumpable(
                        reasoning_content=None,
                        content='{"ranking":["M01","M02"]}',
                    ),
                    finish_reason=None,
                    usage=None,
                )
            ],
            usage=None,
        ),
        Dumpable(
            id="chatcmpl-test",
            model="replace-with-provider-returned-model-id",
            created=1784174400,
            choices=[
                Dumpable(
                    delta=Dumpable(reasoning_content=None, content=None),
                    finish_reason="stop",
                    usage=usage,
                )
            ],
            usage=None,
        ),
    ]

    class Completions:
        @staticmethod
        def create(**kwargs):
            sent.update(kwargs)
            return iter(chunks)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            sent["client"] = kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-only-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )
    spec = benchmark_spec(
        "chat-model",
        50,
        registry=load_model_registry("examples/chat_model.json"),
    )

    result = client._chat_completions(spec, "system", "user", timeout=30)

    assert sent["client"]["base_url"] == "https://inference.test/v1"
    assert sent["client"]["max_retries"] == 0
    assert sent["model"] == "replace-with-model-id"
    assert sent["reasoning_effort"] == "max"
    assert sent["max_completion_tokens"] == 128000
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert "max_tokens" not in sent
    assert result["text"] == '{"ranking":["M01","M02"]}'
    assert result["reasoning_text"] == "I compared the substitutions."
    assert result["usage"]["cached_tokens"] == 80
    assert result["output_tokens"] == 42
    assert result["reasoning_tokens"] is None
    assert result["stop_reason"] == "stop"
    assert result["stream_completed"] is True
    assert result["response_model_id"] == "replace-with-provider-returned-model-id"
    assert result["response_id"] == "chatcmpl-test"
    assert result["provider_response"]["stream_completed"] is True
    assert len(result["provider_response"]["chunks"]) == 4
    assert result["response_content"][0]["choices"][0]["delta"]["reasoning_content"]


def test_chat_empty_choices_is_retryable_and_preserves_body(monkeypatch):
    class EmptyResponse:
        choices: list = []
        usage = None
        id = "chatcmpl-empty"
        model = "provider-model"
        created = 1784174400

    class Completions:
        @staticmethod
        def create(**_kwargs):
            return EmptyResponse()

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {"LAB_API_KEY": "test-only-key", "LAB_BASE_URL": "https://inference.test/v1"},
    )
    spec = {
        "provider": "openai-compatible",
        "api_style": "chat",
        "model_id": "m",
        "api_key_env": "LAB_API_KEY",
        "base_url_env": "LAB_BASE_URL",
        "reasoning": "high",
        "send_reasoning_effort": True,
        "stream": False,
        "chat_output_token_field": "max_tokens",
        "max_tokens": 100,
        "ctx": 1000,
        "require_usage": False,
        "require_reasoning": False,
        "response_model_ids": [],
    }

    result = client._chat_completions(spec, "system", "user", timeout=30)

    assert result["error"] == "chat completion returned no choices"
    assert result["retryable"] is True
    assert result["provider_response"] is not None
    assert result["response_model_id"] == "provider-model"


def test_provider_errors_redact_configured_key_and_endpoint(monkeypatch):
    spec = benchmark_spec(
        "chat-model",
        50,
        registry=load_model_registry("examples/chat_model.json"),
    )
    environment = {
        "LAB_API_KEY": "secret-test-key",
        "LAB_BASE_URL": "https://private.example/v1/",
    }
    monkeypatch.setattr(client, "_env", lambda: environment)

    payload = {
        "error": "request failed with secret-test-key",
        "provider_response": {
            "endpoint": "https://private.example/v1/chat/completions",
        },
    }
    redacted = client._redact_payload(payload, spec)

    serialized = str(redacted)
    assert environment["LAB_API_KEY"] not in serialized
    assert environment["LAB_BASE_URL"].rstrip("/") not in serialized
    assert "<redacted-api-key>" in serialized
    assert "<redacted-base-url>" in serialized


def test_retry_classification_distinguishes_fatal_from_transient_errors():
    assert client._non_retryable("Invalid_Request: unsupported parameter") is True
    assert client._non_retryable("permission denied") is True
    assert client._non_retryable("rate limit exceeded") is False
    assert client._non_retryable("provider overloaded") is False


@pytest.mark.parametrize("error", [RuntimeError("missing configuration"), ValueError("bad value")])
def test_unknown_statusless_errors_are_not_retried(error):
    assert client._error_is_retryable(error, str(error)) is False


@pytest.mark.parametrize(
    "error", [ConnectionError("permission denied"), TimeoutError("invalid_request")]
)
def test_transport_types_override_misleading_error_text(error):
    assert client._error_is_retryable(error, str(error)) is True


@pytest.mark.parametrize(
    ("status_code", "expected"), [(200, True), (302, False), (400, False), (429, True), (503, True)]
)
def test_direct_http_status_controls_retry_class(status_code, expected):
    error = RuntimeError("provider response")
    error.status_code = status_code
    assert client._error_is_retryable(error, str(error)) is expected


def _responses_spec(*, send_reasoning=True):
    return {
        "provider": "openai-compatible",
        "api_style": "responses",
        "model_id": "test-model",
        "api_key_env": "LAB_API_KEY",
        "base_url_env": "LAB_BASE_URL",
        "reasoning": "max",
        "send_reasoning": send_reasoning,
        "max_tokens": 128,
        "ctx": 4096,
    }


def _completed_responses_object():
    return Dumpable(
        output_text='{"ranking":["M01","M02"]}',
        output=[
            Dumpable(
                type="reasoning",
                summary=[Dumpable(text="Compared substitutions.")],
            )
        ],
        usage=Dumpable(
            input_tokens=100,
            output_tokens=42,
            total_tokens=142,
            output_tokens_details=Dumpable(reasoning_tokens=30),
            input_tokens_details=Dumpable(cached_tokens=10),
        ),
        id="resp-sse",
        model="test-model",
        created_at=1784174400,
        status="completed",
        incomplete_details=None,
        error=None,
        service_tier="default",
    )


@pytest.mark.parametrize("send_reasoning", [True, False])
def test_responses_explicitly_disable_storage_and_background(monkeypatch, send_reasoning):
    sent = {}
    response = Dumpable(
        output_text="OK",
        output=[],
        usage=Dumpable(input_tokens=1, output_tokens=1, total_tokens=2),
        id="resp-test",
        model="test-model",
        created_at=1,
        status="completed",
        incomplete_details=None,
        service_tier=None,
    )

    class Responses:
        @staticmethod
        def create(**kwargs):
            sent.update(kwargs)
            return response

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = Responses()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )
    spec = _responses_spec(send_reasoning=send_reasoning)

    client.chat(spec, "system", "user")

    assert sent["store"] is False
    assert sent["background"] is False
    descriptor = client.public_request_descriptor(spec)
    assert descriptor["inference_options"]["store"] is False
    assert descriptor["inference_options"]["background"] is False


def test_responses_sse_is_opt_in_and_versions_only_the_new_transport(monkeypatch):
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )
    legacy = client.public_request_descriptor(_responses_spec())
    streamed = client.public_request_descriptor({**_responses_spec(), "stream": True})

    assert legacy["descriptor_version"] == 2
    assert "transport" not in legacy["inference_options"]
    assert "stream" not in legacy["inference_options"]
    assert streamed["descriptor_version"] == 3
    assert streamed["inference_options"]["transport"] == "responses-sse"
    assert streamed["inference_options"]["stream"] is True


def test_responses_sse_journals_headers_events_and_requires_completed(monkeypatch):
    sent = {}
    records = []
    terminal = _completed_responses_object()
    events = [
        Dumpable(
            type="response.output_text.delta",
            sequence_number=1,
            delta='{"ranking":',
        ),
        Dumpable(
            type="response.completed",
            sequence_number=2,
            response=terminal,
        ),
    ]

    class FakeStream:
        response = SimpleNamespace(
            headers={
                "X-Request-ID": "req-provider",
                "X-RateLimit-Remaining-Tokens": "1000",
                "Set-Cookie": "must-not-persist",
            }
        )

        def __iter__(self):
            return iter(events)

        def close(self):
            return None

    class Responses:
        @staticmethod
        def create(**kwargs):
            sent.update(kwargs)
            return FakeStream()

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = Responses()

    class FakeHTTPClient:
        def close(self):
            return None

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(client, "_responses_http_client", lambda _timeout: FakeHTTPClient())
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )

    result = client.chat(
        {**_responses_spec(), "stream": True},
        "system",
        "user",
        client_request_id="canary-request-1",
        event_sink=records.append,
    )

    assert sent["stream"] is True
    assert sent["store"] is False
    assert sent["background"] is False
    assert sent["extra_headers"] == {"X-Client-Request-Id": "canary-request-1"}
    assert records[0] == {
        "kind": "response.headers",
        "client_request_id": "canary-request-1",
        "headers": {
            "x-request-id": "req-provider",
            "x-ratelimit-remaining-tokens": "1000",
        },
    }
    assert [record["event_type"] for record in records[1:]] == [
        "response.output_text.delta",
        "response.completed",
    ]
    assert result["stream_completed"] is True
    assert result["stream_terminal_event"] == "response.completed"
    assert result["status"] == "completed"
    assert result["provider_response"] == terminal.model_dump(mode="json")
    assert result["reasoning_text"] == "Compared substitutions."
    assert result["reasoning_tokens"] == 30
    assert result["client_request_id"] == "canary-request-1"


def test_responses_sse_interruption_preserves_partial_events(monkeypatch):
    records = []

    class BrokenStream:
        response = SimpleNamespace(headers={"X-Request-ID": "req-interrupted"})

        def __iter__(self):
            yield Dumpable(
                type="response.output_text.delta",
                sequence_number=1,
                delta="partial answer",
            )
            raise TimeoutError("idle connection closed")

        def close(self):
            return None

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = SimpleNamespace(create=lambda **_kwargs: BrokenStream())

    class FakeHTTPClient:
        def close(self):
            return None

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(client, "_responses_http_client", lambda _timeout: FakeHTTPClient())
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )

    result = client.chat(
        {**_responses_spec(), "stream": True},
        "system",
        "user",
        event_sink=records.append,
    )

    assert result["stream_completed"] is False
    assert result["status"] == "incomplete"
    assert result["incomplete_reason"] == "stream_interrupted"
    assert result["text"] == "partial answer"
    assert "TimeoutError: idle connection closed" in result["error"]
    assert result["retryable"] is True
    assert result["failure_class"] == "transport_error"
    assert records[-2]["event_type"] == "response.output_text.delta"
    assert records[-1] == {
        "kind": "response.stream_error",
        "client_request_id": None,
        "error": "TimeoutError: idle connection closed",
        "failure_class": "transport_error",
        "incomplete_reason": "stream_interrupted",
        "retryable": True,
        "provider_error": None,
    }


@pytest.mark.parametrize(
    ("message", "body", "failure_class", "incomplete_reason"),
    [
        (
            "Invalid prompt: we've limited access to this content for safety reasons. "
            "This type of information may be used to benefit or to harm people.",
            {
                "code": "invalid_prompt",
                "type": "invalid_request_error",
                "message": "we've limited access to this content for safety reasons",
            },
            "provider_policy_block",
            "provider_policy_block",
        ),
        (
            "You exceeded your current quota, please check your plan and billing details.",
            {
                "code": "insufficient_quota",
                "type": "insufficient_quota",
                "message": "You exceeded your current quota",
            },
            "quota_exhausted",
            "quota_exhausted",
        ),
    ],
)
def test_responses_sse_api_error_is_terminal_classified_and_journaled(
    monkeypatch, message, body, failure_class, incomplete_reason
):
    records = []
    in_progress = Dumpable(
        id="resp_blocked",
        model="test-model",
        status="in_progress",
        output=[],
        usage=None,
        service_tier=None,
    )

    class BrokenStream:
        response = SimpleNamespace(headers={"X-Request-ID": "req-blocked"})

        def __iter__(self):
            yield Dumpable(
                type="response.created",
                sequence_number=0,
                response=in_progress,
            )
            yield Dumpable(
                type="response.in_progress",
                sequence_number=1,
                response=in_progress,
            )
            raise openai.APIError(
                message,
                httpx.Request("POST", "https://inference.test/v1/responses"),
                body=body,
            )

        def close(self):
            return None

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = SimpleNamespace(create=lambda **_kwargs: BrokenStream())

    class FakeHTTPClient:
        def close(self):
            return None

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(client, "_responses_http_client", lambda _timeout: FakeHTTPClient())
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )

    result = client.chat(
        {**_responses_spec(), "stream": True},
        "system",
        "user",
        client_request_id="blocked-request",
        event_sink=records.append,
    )

    assert result["status"] == "failed"
    assert result["stream_completed"] is False
    assert result["stream_terminal_event"] is None
    assert result["retryable"] is False
    assert result["failure_class"] == failure_class
    assert result["incomplete_reason"] == incomplete_reason
    assert result["provider_error"] == {
        "exception_type": "APIError",
        "code": body["code"],
        "type": body["type"],
        "body": body,
    }
    assert [record.get("event_type") for record in records[1:-1]] == [
        "response.created",
        "response.in_progress",
    ]
    assert records[-1]["kind"] == "response.stream_error"
    assert records[-1]["failure_class"] == failure_class
    assert records[-1]["retryable"] is False
    assert records[-1]["provider_error"] == result["provider_error"]


def test_responses_sse_event_sink_failure_is_fatal(monkeypatch):
    class FakeStream:
        response = SimpleNamespace(headers={"X-Request-ID": "req-journal"})

        def __iter__(self):
            return iter([])

        def close(self):
            return None

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = SimpleNamespace(create=lambda **_kwargs: FakeStream())

    class FakeHTTPClient:
        def close(self):
            return None

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(client, "_responses_http_client", lambda _timeout: FakeHTTPClient())
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )

    def broken_sink(_record):
        raise OSError("disk full")

    with pytest.raises(client.EventSinkError, match="event sink failed"):
        client.chat(
            {**_responses_spec(), "stream": True},
            "system",
            "user",
            event_sink=broken_sink,
        )


@pytest.mark.parametrize("field", ["store", "background"])
def test_responses_reject_persistence_mode_overrides(field):
    spec = {**_responses_spec(), field: True}
    with pytest.raises(ValueError, match="unknown config fields"):
        client.chat(spec, "system", "user")


def test_responses_transport_does_not_hide_summary_fallback_request(monkeypatch):
    calls = []

    class Responses:
        @staticmethod
        def create(**kwargs):
            calls.append(kwargs)
            raise ValueError("reasoning summary is not supported")

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = Responses()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )
    spec = {
        "provider": "openai-compatible",
        "api_style": "responses",
        "model_id": "test-model",
        "api_key_env": "LAB_API_KEY",
        "base_url_env": "LAB_BASE_URL",
        "reasoning": "max",
        "max_tokens": 128,
        "ctx": 4096,
    }

    result = client.chat(spec, "system", "user", retries=5)

    assert len(calls) == 1
    assert calls[0]["reasoning"] == {"effort": "max", "summary": "auto"}
    assert result["summary_unsupported"] is True
    assert result["retryable"] is False


def test_interrupted_stream_preserves_partial_trace_without_retry(monkeypatch):
    calls = 0
    partial = Dumpable(
        id="chatcmpl-partial",
        model="replace-with-provider-returned-model-id",
        created=1784174400,
        choices=[
            Dumpable(
                delta=Dumpable(reasoning_content="partial trace", content=None),
                finish_reason=None,
                usage=None,
            )
        ],
        usage=None,
    )

    class BrokenStream:
        def __iter__(self):
            yield partial
            raise RuntimeError("connection dropped at https://private.example/v1")

    class Completions:
        @staticmethod
        def create(**_kwargs):
            nonlocal calls
            calls += 1
            return BrokenStream()

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "secret-test-key",
            "LAB_BASE_URL": "https://private.example/v1",
        },
    )
    spec = benchmark_spec(
        "chat-model",
        50,
        registry=load_model_registry("examples/chat_model.json"),
    )

    result = client.chat(spec, "system", "user", timeout=30, retries=3)

    assert calls == 1
    assert result["stream_completed"] is False
    assert result["reasoning_text"] == "partial trace"
    assert result["status"] == "incomplete"
    assert "<redacted-base-url>" in result["error"]
    assert "https://private.example/v1" not in str(result)


def test_stream_eof_without_terminal_finish_preserves_partial_payload(monkeypatch):
    calls = 0
    usage = Dumpable(
        prompt_tokens=100,
        completion_tokens=42,
        total_tokens=142,
    )
    chunks = [
        Dumpable(
            id="chatcmpl-eof",
            model="replace-with-provider-returned-model-id",
            created=1784174400,
            choices=[
                Dumpable(
                    delta=Dumpable(reasoning_content="partial trace", content=None),
                    finish_reason=None,
                    usage=None,
                )
            ],
            usage=None,
        ),
        Dumpable(
            id="chatcmpl-eof",
            model="replace-with-provider-returned-model-id",
            created=1784174400,
            choices=[
                Dumpable(
                    delta=Dumpable(
                        reasoning_content=None,
                        content='{"ranking":["M01","M02"]}',
                    ),
                    finish_reason=None,
                    usage=usage,
                )
            ],
            usage=None,
        ),
    ]

    class Completions:
        @staticmethod
        def create(**_kwargs):
            nonlocal calls
            calls += 1
            return iter(chunks)

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        client,
        "_env",
        lambda: {
            "LAB_API_KEY": "secret-test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )
    spec = benchmark_spec(
        "chat-model",
        50,
        registry=load_model_registry("examples/chat_model.json"),
    )

    result = client.chat(spec, "system", "user", timeout=30, retries=3)

    assert calls == 1
    assert result["stream_completed"] is False
    assert result["status"] == "incomplete"
    assert result["incomplete_reason"] == "missing_terminal_finish_reason"
    assert result["error"] == "stream ended without terminal finish_reason"
    assert result["reasoning_text"] == "partial trace"
    assert result["text"] == '{"ranking":["M01","M02"]}'
    assert result["usage"]["completion_tokens"] == 42
    assert result["output_tokens"] == 42
    assert len(result["response_content"]) == 2
    assert result["provider_response"]["chunks"] == result["response_content"]
