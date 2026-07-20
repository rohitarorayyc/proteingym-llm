"""Download, authenticate, install, and verify the frozen evaluation data.

The archive is treated as untrusted input.  Its checksum is verified before it
is opened, every member is checked against a per-file manifest, and extraction
refuses links, special files, duplicate members, and paths outside ``data_root``.

Typical use::

    python -m src.data_bundle
    python -m src.data_bundle --verify-only
    python -m src.data_bundle --url URL --sha256 HEX --data-root /scratch/eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.parse import urlsplit

from config.data_bundle import (
    EVAL_BUNDLE_FILENAME,
    EVAL_BUNDLE_MANIFEST,
    EVAL_BUNDLE_MANIFEST_SHA256,
    EVAL_BUNDLE_PROVENANCE,
    EVAL_BUNDLE_SCHEMA_VERSION,
    EVAL_BUNDLE_SHA256,
    EVAL_BUNDLE_URL,
)
from config.paths import DATA_ROOT

DEFAULT_DATA_ROOT = DATA_ROOT
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
DOWNLOAD_CHUNK_SIZE = 1 << 20
GITHUB_RELEASE_TAG_PATH = "/releases/tags/"
GITHUB_API_HOST = "api.github.com"
GITHUB_REPOSITORY_API_PREFIX = "/repos/rohitarorayyc/proteingym-llm/"


class BundleError(RuntimeError):
    """The bundle is missing, corrupt, unsafe, or incompatible."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of *path* without loading it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha256(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise BundleError(
            "expected SHA-256 must be exactly 64 hexadecimal characters; "
            "pass --sha256 or publish config/data_bundle.py"
        )
    return value.lower()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_stream_copy(source: BinaryIO, destination: Path) -> str:
    """Copy a stream to *destination* atomically and return its SHA-256 digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as output:
            while chunk := source.read(DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def default_cache_path() -> Path:
    """Use XDG_CACHE_HOME when set, otherwise the conventional user cache."""
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "proteingym-llm" / EVAL_BUNDLE_FILENAME


def _select_release_asset_api(release: dict, filename: str) -> str:
    """Select one exact GitHub release asset without trusting browser redirects."""
    assets = release.get("assets") if isinstance(release, dict) else None
    matches = [asset for asset in assets or [] if asset.get("name") == filename]
    if len(matches) != 1:
        raise BundleError(f"GitHub release must contain exactly one asset named {filename!r}")
    api_url = matches[0].get("url") or matches[0].get("api_url")
    if not isinstance(api_url, str) or not api_url.startswith(
        "https://api.github.com/repos/rohitarorayyc/proteingym-llm/releases/assets/"
    ):
        raise BundleError("GitHub release returned an unsafe asset API URL")
    return api_url


def _is_trusted_github_api_url(url: str) -> bool:
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == GITHUB_API_HOST
        and port in {None, 443}
        and parsed.path.startswith(GITHUB_REPOSITORY_API_PREFIX)
    )


def _download_headers(url: str, *, binary: bool) -> dict[str, str]:
    headers = {
        "User-Agent": "proteingym-llm/1.0",
        "Accept": "application/octet-stream" if binary else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    github_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if github_token and _is_trusted_github_api_url(url):
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward authorization when an HTTP redirect changes origin."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None:
            old = urlsplit(request.full_url)
            new = urlsplit(newurl)
            if (old.scheme, old.hostname, old.port) != (new.scheme, new.hostname, new.port):
                redirected.remove_header("Authorization")
        return redirected


def _open_url(request: urllib.request.Request, timeout: int):
    return urllib.request.build_opener(_SafeRedirectHandler()).open(request, timeout=timeout)


def _resolve_bundle_url(url: str, timeout: int) -> str:
    if GITHUB_RELEASE_TAG_PATH not in url or not _is_trusted_github_api_url(url):
        return url
    request = urllib.request.Request(url, headers=_download_headers(url, binary=False))
    with _open_url(request, timeout) as response:
        try:
            release = json.load(response)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BundleError("GitHub release metadata is not valid JSON") from error
    return _select_release_asset_api(release, EVAL_BUNDLE_FILENAME)


def download_bundle(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    force: bool = False,
    timeout: int = 3600,
) -> Path:
    """Download *url* atomically, accepting only the expected content digest."""
    expected = _expected_sha256(expected_sha256)
    destination = Path(destination)
    if destination.exists():
        if destination.is_file() and sha256_file(destination) == expected:
            return destination
        if not force:
            raise BundleError(
                f"cached archive does not match {expected}: {destination}; use --force"
            )

    try:
        resolved_url = _resolve_bundle_url(url, timeout)
    except urllib.error.HTTPError as error:
        hint = (
            " Set GH_TOKEN if the repository is private." if error.code in {401, 403, 404} else ""
        )
        raise BundleError(f"bundle release lookup failed with HTTP {error.code}.{hint}") from error
    except urllib.error.URLError as error:
        raise BundleError(f"bundle release lookup failed: {error.reason}") from error
    request = urllib.request.Request(
        resolved_url, headers=_download_headers(resolved_url, binary=True)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        with (
            _open_url(request, timeout) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise BundleError(f"bundle checksum mismatch: expected {expected}, downloaded {actual}")
        os.replace(temporary, destination)
    except urllib.error.HTTPError as error:
        hint = (
            " Set GH_TOKEN if the repository is private." if error.code in {401, 403, 404} else ""
        )
        raise BundleError(f"bundle download failed with HTTP {error.code}.{hint}") from error
    except urllib.error.URLError as error:
        raise BundleError(f"bundle download failed: {error.reason}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _safe_relative_path(name: str) -> Path:
    """Convert an archive/manifest name to a safe platform-native relative path."""
    posix = PurePosixPath(name)
    if not name or name.startswith("/") or posix.is_absolute():
        raise BundleError(f"unsafe absolute or empty bundle path: {name!r}")
    if "\\" in name or any(part in {"", ".", ".."} for part in posix.parts):
        raise BundleError(f"unsafe bundle path: {name!r}")
    return Path(*posix.parts)


def _safe_target(data_root: Path, relative: Path) -> Path:
    """Resolve a destination while refusing pre-existing escaping symlinks."""
    root = data_root.resolve()
    target = root.joinpath(relative)
    resolved = target.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise BundleError(f"bundle target escapes data root: {relative.as_posix()!r}")
    return target


def _parse_manifest(payload: bytes) -> dict:
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"invalid {EVAL_BUNDLE_MANIFEST}: {error}") from error
    if not isinstance(manifest, dict):
        raise BundleError("bundle manifest must be a JSON object")
    if manifest.get("schema_version") != EVAL_BUNDLE_SCHEMA_VERSION:
        raise BundleError(f"unsupported bundle manifest schema: {manifest.get('schema_version')!r}")
    if not isinstance(manifest.get("bundle_version"), str):
        raise BundleError("bundle manifest is missing bundle_version")
    if manifest.get("provenance") != EVAL_BUNDLE_PROVENANCE:
        raise BundleError("bundle manifest has missing or incompatible provenance/terms")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise BundleError("bundle manifest is missing selection")
    for dimension in ("sizes", "seeds"):
        values = selection.get(dimension)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, int) or isinstance(value, bool) for value in values)
            or len(values) != len(set(values))
        ):
            raise BundleError(f"invalid bundle selection {dimension}: {values!r}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleError("bundle manifest has no files")
    for name, record in files.items():
        _safe_relative_path(name)
        if name == EVAL_BUNDLE_MANIFEST:
            raise BundleError("bundle manifest must not list itself")
        if not isinstance(record, dict):
            raise BundleError(f"invalid manifest record for {name!r}")
        record["sha256"] = _expected_sha256(str(record.get("sha256", "")))
        size = record.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BundleError(f"invalid manifest size for {name!r}: {size!r}")
    return manifest


def _manifest_bytes(manifest: dict) -> bytes:
    """Return the exact normalized representation installed on disk."""
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _inspect_archive(
    archive: Path, expected_manifest_sha256: str
) -> tuple[dict, dict[str, tarfile.TarInfo], bytes]:
    """Validate archive topology and return its manifest and regular members."""
    try:
        handle = tarfile.open(archive, mode="r:*")
    except (tarfile.TarError, OSError) as error:
        raise BundleError(f"cannot open bundle {archive}: {error}") from error

    with handle:
        files: dict[str, tarfile.TarInfo] = {}
        manifest_payload: bytes | None = None
        for member in handle.getmembers():
            relative = _safe_relative_path(member.name)
            name = relative.as_posix()
            if member.issym() or member.islnk():
                raise BundleError(f"links are forbidden in evaluation bundles: {name!r}")
            if member.isdir():
                continue
            if not member.isfile():
                raise BundleError(f"special tar member is forbidden: {name!r}")
            if name in files or (name == EVAL_BUNDLE_MANIFEST and manifest_payload is not None):
                raise BundleError(f"duplicate bundle member: {name!r}")
            extracted = handle.extractfile(member)
            if extracted is None:
                raise BundleError(f"cannot read bundle member: {name!r}")
            if name == EVAL_BUNDLE_MANIFEST:
                manifest_payload = extracted.read()
            else:
                files[name] = member

        if manifest_payload is None:
            raise BundleError(f"bundle is missing {EVAL_BUNDLE_MANIFEST}")
        manifest = _parse_manifest(manifest_payload)
        installed_manifest_payload = _manifest_bytes(manifest)
        expected_manifest = _expected_sha256(expected_manifest_sha256)
        actual_manifest = _sha256_bytes(installed_manifest_payload)
        if actual_manifest != expected_manifest:
            raise BundleError(
                "bundle manifest checksum mismatch: "
                f"expected {expected_manifest}, found {actual_manifest}"
            )
        expected_names = set(manifest["files"])
        actual_names = set(files)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise BundleError(f"archive/manifest file mismatch; missing={missing}, extra={extra}")
        return manifest, files, installed_manifest_payload


def extract_bundle(
    archive: Path,
    data_root: Path,
    *,
    expected_manifest_sha256: str = EVAL_BUNDLE_MANIFEST_SHA256,
    force: bool = False,
) -> dict:
    """Verify all members, then install them with atomic per-file replacements."""
    archive = Path(archive)
    data_root = Path(data_root)
    expected_manifest = _expected_sha256(expected_manifest_sha256)
    manifest, members, manifest_payload = _inspect_archive(archive, expected_manifest)
    data_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".eval-bundle-", dir=data_root) as stage_name:
        stage = Path(stage_name)
        with tarfile.open(archive, mode="r:*") as handle:
            for name in sorted(members):
                member = handle.getmember(members[name].name)
                source = handle.extractfile(member)
                if source is None:
                    raise BundleError(f"cannot read bundle member: {name!r}")
                staged = stage / _safe_relative_path(name)
                actual_sha = _atomic_stream_copy(source, staged)
                record = manifest["files"][name]
                if staged.stat().st_size != record["size"] or actual_sha != record["sha256"]:
                    raise BundleError(f"manifest checksum/size mismatch for {name!r}")

        manifest_staged = stage / EVAL_BUNDLE_MANIFEST
        manifest_staged.write_bytes(manifest_payload)

        install_names = [*sorted(members), EVAL_BUNDLE_MANIFEST]
        for name in install_names:
            relative = _safe_relative_path(name)
            destination = _safe_target(data_root, relative)
            staged = stage / relative
            if destination.exists() or destination.is_symlink():
                if destination.is_file() and not destination.is_symlink():
                    if sha256_file(destination) == sha256_file(staged):
                        continue
                if not force:
                    raise BundleError(f"refusing to replace {destination}; use --force")

        for name in install_names:
            relative = _safe_relative_path(name)
            destination = _safe_target(data_root, relative)
            staged = stage / relative
            if destination.exists() and destination.is_file() and not destination.is_symlink():
                if sha256_file(destination) == sha256_file(staged):
                    continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, destination)

    return verify_data_bundle(data_root, expected_manifest_sha256=expected_manifest)


def _authenticated_installed_manifest(
    data_root: Path,
    expected_manifest_sha256: str,
) -> tuple[dict, str]:
    data_root = Path(data_root)
    expected = _expected_sha256(expected_manifest_sha256)
    manifest_path = _safe_target(data_root, Path(EVAL_BUNDLE_MANIFEST))
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise BundleError(f"missing installed bundle manifest: {manifest_path}")
    manifest_payload = manifest_path.read_bytes()
    actual = _sha256_bytes(manifest_payload)
    if actual != expected:
        raise BundleError(
            f"installed bundle manifest checksum mismatch: expected {expected}, found {actual}"
        )
    return _parse_manifest(manifest_payload), actual


def _identity_report(manifest: dict, manifest_sha256: str) -> dict:
    return {
        "bundle_version": manifest["bundle_version"],
        "manifest_sha256": manifest_sha256,
        "selection": {
            "sizes": list(manifest["selection"]["sizes"]),
            "seeds": list(manifest["selection"]["seeds"]),
        },
    }


def bundle_identity(
    data_root: Path,
    *,
    expected_manifest_sha256: str = EVAL_BUNDLE_MANIFEST_SHA256,
) -> dict:
    """Authenticate the installed manifest and return its recordable identity.

    This intentionally verifies only the small, config-pinned manifest.  Use
    :func:`verify_data_bundle` when every installed payload file must also be
    rehashed.  Custom bundles must pass their own expected manifest digest
    explicitly.
    """
    manifest, actual = _authenticated_installed_manifest(data_root, expected_manifest_sha256)
    return _identity_report(manifest, actual)


def verify_data_bundle(
    data_root: Path,
    *,
    expected_manifest_sha256: str = EVAL_BUNDLE_MANIFEST_SHA256,
) -> dict:
    """Verify every installed file against ``eval_bundle_manifest.json``."""
    data_root = Path(data_root)
    manifest, actual_manifest = _authenticated_installed_manifest(
        data_root, expected_manifest_sha256
    )
    identity = _identity_report(manifest, actual_manifest)
    total_bytes = 0
    for name, record in sorted(manifest["files"].items()):
        path = _safe_target(data_root, _safe_relative_path(name))
        if not path.is_file() or path.is_symlink():
            raise BundleError(f"missing or non-regular bundle file: {name!r}")
        size = path.stat().st_size
        actual_sha = sha256_file(path)
        if size != record["size"] or actual_sha != record["sha256"]:
            raise BundleError(
                f"installed file does not match manifest: {name!r} "
                f"(size={size}, sha256={actual_sha})"
            )
        total_bytes += size
    return {
        **identity,
        "file_count": len(manifest["files"]),
        "total_bytes": total_bytes,
        "data_root": str(data_root.resolve()),
    }


def install_bundle(
    archive: Path,
    expected_sha256: str,
    data_root: Path,
    *,
    expected_manifest_sha256: str = EVAL_BUNDLE_MANIFEST_SHA256,
    force: bool = False,
) -> dict:
    """Authenticate a local archive and safely install it."""
    expected = _expected_sha256(expected_sha256)
    actual = sha256_file(Path(archive))
    if actual != expected:
        raise BundleError(f"bundle checksum mismatch: expected {expected}, found {actual}")
    return extract_bundle(
        Path(archive),
        Path(data_root),
        expected_manifest_sha256=expected_manifest_sha256,
        force=force,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the frozen ProteinGym-LLM evaluation data."
    )
    parser.add_argument("--url", default=EVAL_BUNDLE_URL, help="immutable bundle URL")
    parser.add_argument(
        "--sha256", default=EVAL_BUNDLE_SHA256, help="expected full archive SHA-256"
    )
    parser.add_argument(
        "--manifest-sha256",
        default=EVAL_BUNDLE_MANIFEST_SHA256,
        help="expected installed eval_bundle_manifest.json SHA-256",
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="installation directory"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="use a local archive instead of downloading (still SHA-verified)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=default_cache_path(),
        help="download cache path",
    )
    parser.add_argument(
        "--timeout", type=_positive_int, default=3600, help="download timeout in seconds"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace a stale cache or installed files"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="verify installed data without downloading"
    )
    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify_only:
            report = verify_data_bundle(
                args.data_root, expected_manifest_sha256=args.manifest_sha256
            )
        else:
            archive = args.archive
            if archive is None:
                archive = download_bundle(
                    args.url,
                    args.cache,
                    args.sha256,
                    force=args.force,
                    timeout=args.timeout,
                )
            report = install_bundle(
                archive,
                args.sha256,
                args.data_root,
                expected_manifest_sha256=args.manifest_sha256,
                force=args.force,
            )
    except BundleError as error:
        _parser().error(str(error))
    print(
        f"OK eval bundle {report['bundle_version']}: {report['file_count']} files, "
        f"{report['total_bytes']} bytes -> {report['data_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
