"""Deterministic, self-validating storage for large provider response payloads."""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import re
import zlib
from typing import Any

DEFAULT_COMPRESSION_THRESHOLD = 1_000_000
PAYLOAD_SCHEMA = "proteingym-llm.response-payload"
PAYLOAD_VERSION = 1
PAYLOAD_MEDIA_TYPE = "application/json"
PAYLOAD_ENCODING = "gzip+base64"
RESPONSE_CONTENT_REFERENCE = "provider_response.payload#/response_content"

_ENVELOPE_FIELDS = {
    "schema",
    "version",
    "media_type",
    "content_encoding",
    "raw_bytes",
    "compressed_bytes",
    "raw_sha256",
    "compressed_sha256",
    "payload",
}
_REFERENCE_FIELDS = {"$ref", "raw_sha256"}
_PAYLOAD_FIELDS = {"response_content", "provider_response"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class PayloadValidationError(ValueError):
    """Raised when a packed provider payload cannot be validated exactly."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PayloadValidationError(f"provider payload is not canonical JSON: {error}") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _payload_document(response_content: Any, provider_response: Any) -> dict[str, Any]:
    return {
        "response_content": response_content,
        "provider_response": provider_response,
    }


def pack_response_payload(
    response_content: Any,
    provider_response: Any,
    *,
    threshold: int = DEFAULT_COMPRESSION_THRESHOLD,
) -> tuple[Any, Any]:
    """Return inline values or a deterministic reference/envelope pair.

    The size boundary applies to the canonical UTF-8 JSON representation of
    both values together. Payloads exactly at the boundary are compressed.
    """
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("compression threshold must be a non-negative integer")

    document = _payload_document(response_content, provider_response)
    raw = _canonical_json(document)
    if len(raw) < threshold:
        return response_content, provider_response

    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    raw_sha256 = _sha256(raw)
    compressed_sha256 = _sha256(compressed)
    reference = {
        "$ref": RESPONSE_CONTENT_REFERENCE,
        "raw_sha256": raw_sha256,
    }
    envelope = {
        "schema": PAYLOAD_SCHEMA,
        "version": PAYLOAD_VERSION,
        "media_type": PAYLOAD_MEDIA_TYPE,
        "content_encoding": PAYLOAD_ENCODING,
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "raw_sha256": raw_sha256,
        "compressed_sha256": compressed_sha256,
        "payload": base64.b64encode(compressed).decode("ascii"),
    }
    return reference, envelope


def _looks_like_reference(value: Any) -> bool:
    return isinstance(value, dict) and bool(set(value) & _REFERENCE_FIELDS)


def _looks_like_envelope(value: Any) -> bool:
    markers = {
        "schema",
        "content_encoding",
        "raw_bytes",
        "compressed_bytes",
        "compressed_sha256",
        "payload",
    }
    return isinstance(value, dict) and bool(set(value) & markers)


def _require_exact_shape(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PayloadValidationError(f"invalid {label} shape")
    return value


def _require_count(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PayloadValidationError(f"invalid {field}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PayloadValidationError(f"invalid {field}")
    return value


def unpack_response_payload(
    response_content: Any,
    provider_response: Any,
) -> tuple[Any, Any]:
    """Validate and unpack a packed payload, or validate and return inline values.

    Every envelope field, reference, size, digest, encoding layer, decoded JSON
    shape, and canonical serialization is checked before values are returned.
    """
    reference_like = _looks_like_reference(response_content)
    envelope_like = _looks_like_envelope(provider_response)
    if not reference_like and not envelope_like:
        _canonical_json(_payload_document(response_content, provider_response))
        return response_content, provider_response
    if not reference_like or not envelope_like:
        raise PayloadValidationError("payload reference and envelope must appear together")

    reference = _require_exact_shape(response_content, _REFERENCE_FIELDS, "payload reference")
    envelope = _require_exact_shape(provider_response, _ENVELOPE_FIELDS, "payload envelope")
    if reference["$ref"] != RESPONSE_CONTENT_REFERENCE:
        raise PayloadValidationError("invalid response_content payload reference")
    if envelope["schema"] != PAYLOAD_SCHEMA or envelope["version"] != PAYLOAD_VERSION:
        raise PayloadValidationError("unsupported payload schema or version")
    if envelope["media_type"] != PAYLOAD_MEDIA_TYPE:
        raise PayloadValidationError("unsupported payload media type")
    if envelope["content_encoding"] != PAYLOAD_ENCODING:
        raise PayloadValidationError("unsupported payload encoding")

    raw_bytes = _require_count(envelope["raw_bytes"], "raw_bytes")
    compressed_bytes = _require_count(envelope["compressed_bytes"], "compressed_bytes")
    raw_sha256 = _require_sha256(envelope["raw_sha256"], "raw_sha256")
    compressed_sha256 = _require_sha256(envelope["compressed_sha256"], "compressed_sha256")
    reference_sha256 = _require_sha256(reference["raw_sha256"], "reference raw_sha256")
    if reference_sha256 != raw_sha256:
        raise PayloadValidationError("payload reference hash does not match envelope")
    if not isinstance(envelope["payload"], str):
        raise PayloadValidationError("payload must be a base64 string")

    try:
        compressed = base64.b64decode(envelope["payload"].encode("ascii"), validate=True)
    except (binascii.Error, UnicodeError, ValueError) as error:
        raise PayloadValidationError("invalid base64 provider payload") from error
    if len(compressed) != compressed_bytes:
        raise PayloadValidationError("compressed payload size mismatch")
    if _sha256(compressed) != compressed_sha256:
        raise PayloadValidationError("compressed payload hash mismatch")
    try:
        raw = gzip.decompress(compressed)
    except (gzip.BadGzipFile, EOFError, OSError, zlib.error) as error:
        raise PayloadValidationError("invalid gzip provider payload") from error
    if len(raw) != raw_bytes:
        raise PayloadValidationError("raw payload size mismatch")
    if _sha256(raw) != raw_sha256:
        raise PayloadValidationError("raw payload hash mismatch")

    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PayloadValidationError("provider payload is not valid UTF-8 JSON") from error
    _require_exact_shape(document, _PAYLOAD_FIELDS, "decoded payload")
    if _canonical_json(document) != raw:
        raise PayloadValidationError("decoded payload is not canonical JSON")
    return document["response_content"], document["provider_response"]


def validate_response_payload(response_content: Any, provider_response: Any) -> None:
    """Raise :class:`PayloadValidationError` unless the payload is valid."""
    unpack_response_payload(response_content, provider_response)
