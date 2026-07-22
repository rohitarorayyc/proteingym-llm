import base64
import copy
import gzip
import hashlib
import json

import pytest

from src.payloads import (
    PAYLOAD_SCHEMA,
    PAYLOAD_VERSION,
    PayloadValidationError,
    pack_response_payload,
    unpack_response_payload,
    validate_response_payload,
)


def _large_values():
    chunks = [
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "reasoning-" + ("x" * 4000),
                        "content": None,
                    }
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "delta": {
                        "reasoning_content": None,
                        "content": '{"ranking":["M01","M02"]}',
                    }
                }
            ],
        },
    ]
    return chunks, {"stream_completed": True, "chunks": chunks}


def _rebuild_envelope(envelope, document):
    raw = json.dumps(
        document,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    envelope.update(
        {
            "raw_bytes": len(raw),
            "compressed_bytes": len(compressed),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "payload": base64.b64encode(compressed).decode("ascii"),
        }
    )


def test_small_payload_stays_inline_unchanged():
    response_content = [{"delta": "hello"}]
    provider_response = {"id": "response-1"}

    packed_content, packed_response = pack_response_payload(
        response_content,
        provider_response,
    )

    assert packed_content is response_content
    assert packed_response is provider_response
    unpacked_content, unpacked_response = unpack_response_payload(
        packed_content,
        packed_response,
    )
    assert unpacked_content is response_content
    assert unpacked_response is provider_response
    assert validate_response_payload(packed_content, packed_response) is None


def test_boundary_payload_is_deterministically_compressed_and_round_trips():
    response_content, provider_response = _large_values()
    raw = json.dumps(
        {
            "response_content": response_content,
            "provider_response": provider_response,
        },
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")

    packed = pack_response_payload(
        response_content,
        provider_response,
        threshold=len(raw),
    )
    repeated = pack_response_payload(
        response_content,
        provider_response,
        threshold=len(raw),
    )

    assert packed == repeated
    reference, envelope = packed
    assert envelope["schema"] == PAYLOAD_SCHEMA
    assert envelope["version"] == PAYLOAD_VERSION
    assert envelope["raw_bytes"] == len(raw)
    assert reference["raw_sha256"] == envelope["raw_sha256"]
    assert len(base64.b64decode(envelope["payload"])) == envelope["compressed_bytes"]
    assert unpack_response_payload(*packed) == (response_content, provider_response)


def test_default_threshold_compresses_a_one_megabyte_payload():
    response_content = [{"content": "x" * 1_000_000}]
    provider_response = {"stream_completed": True}

    reference, envelope = pack_response_payload(response_content, provider_response)

    assert "$ref" in reference
    assert envelope["raw_bytes"] >= 1_000_000
    assert unpack_response_payload(reference, envelope) == (
        response_content,
        provider_response,
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("media_type", "text/plain", "media type"),
        ("content_encoding", "base64", "encoding"),
        ("raw_bytes", 1, "raw payload size"),
        ("compressed_bytes", 1, "compressed payload size"),
        ("raw_sha256", "0" * 64, "reference hash"),
        ("compressed_sha256", "0" * 64, "compressed payload hash"),
    ],
)
def test_envelope_metadata_tampering_fails_closed(field, replacement, message):
    values = _large_values()
    reference, envelope = pack_response_payload(*values, threshold=0)
    envelope = copy.deepcopy(envelope)
    envelope[field] = replacement

    with pytest.raises(PayloadValidationError, match=message):
        unpack_response_payload(reference, envelope)


def test_payload_and_reference_tampering_fail_closed():
    values = _large_values()
    reference, envelope = pack_response_payload(*values, threshold=0)

    broken_payload = copy.deepcopy(envelope)
    broken_payload["payload"] = "!" + broken_payload["payload"][1:]
    with pytest.raises(PayloadValidationError, match="base64"):
        unpack_response_payload(reference, broken_payload)

    broken_reference = copy.deepcopy(reference)
    broken_reference["raw_sha256"] = "0" * 64
    with pytest.raises(PayloadValidationError, match="reference hash"):
        unpack_response_payload(broken_reference, envelope)

    wrong_target = copy.deepcopy(reference)
    wrong_target["$ref"] = "somewhere-else"
    with pytest.raises(PayloadValidationError, match="payload reference"):
        unpack_response_payload(wrong_target, envelope)


def test_decoded_json_shape_and_canonical_form_are_validated():
    values = _large_values()
    reference, envelope = pack_response_payload(*values, threshold=0)
    envelope = copy.deepcopy(envelope)

    malformed_document = {"response_content": values[0]}
    _rebuild_envelope(envelope, malformed_document)
    reference["raw_sha256"] = envelope["raw_sha256"]
    with pytest.raises(PayloadValidationError, match="decoded payload shape"):
        unpack_response_payload(reference, envelope)

    document = {
        "response_content": values[0],
        "provider_response": values[1],
    }
    raw = json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    envelope.update(
        {
            "raw_bytes": len(raw),
            "compressed_bytes": len(compressed),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "payload": base64.b64encode(compressed).decode("ascii"),
        }
    )
    reference["raw_sha256"] = envelope["raw_sha256"]
    with pytest.raises(PayloadValidationError, match="not canonical JSON"):
        unpack_response_payload(reference, envelope)


def test_schema_shape_and_threshold_validation_fail_closed():
    values = _large_values()
    reference, envelope = pack_response_payload(*values, threshold=0)

    extra_field = copy.deepcopy(envelope)
    extra_field["unexpected"] = True
    with pytest.raises(PayloadValidationError, match="envelope shape"):
        unpack_response_payload(reference, extra_field)

    unsupported = copy.deepcopy(envelope)
    unsupported["version"] += 1
    with pytest.raises(PayloadValidationError, match="schema or version"):
        unpack_response_payload(reference, unsupported)

    with pytest.raises(ValueError, match="threshold"):
        pack_response_payload(*values, threshold=True)


def test_inline_response_with_envelope_named_field_round_trips():
    # A legitimate small provider response whose own JSON contains a field named
    # like an envelope key must stay inline, not be misdetected as a packed
    # envelope and rejected by unpack.
    response_content = [{"type": "message"}]
    for provider_response in (
        {"id": "r", "content_encoding": "utf-8"},
        {"id": "r", "payload": {"text": "hi"}},
        {"id": "r", "schema": "openai", "compressed_sha256": "not-a-real-hash"},
    ):
        packed_content, packed_response = pack_response_payload(response_content, provider_response)
        assert packed_content is response_content
        assert packed_response is provider_response
        assert unpack_response_payload(packed_content, packed_response) == (
            response_content,
            provider_response,
        )


def test_reference_and_envelope_must_appear_together():
    values = _large_values()
    reference, envelope = pack_response_payload(*values, threshold=0)

    with pytest.raises(PayloadValidationError, match="appear together"):
        unpack_response_payload(reference, {"id": "inline"})
    with pytest.raises(PayloadValidationError, match="appear together"):
        unpack_response_payload([{"inline": True}], envelope)
