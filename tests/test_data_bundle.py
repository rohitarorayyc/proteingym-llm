from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from urllib.request import Request

import pytest

from config.data_bundle import (
    EVAL_BUNDLE_MANIFEST,
    EVAL_BUNDLE_PROVENANCE,
    EVAL_BUNDLE_SCHEMA_VERSION,
    EVAL_SEEDS,
    EVAL_SIZES,
)
from scripts.package_eval_bundle import build_eval_bundle
from src.data_bundle import (
    BundleError,
    _download_headers,
    _SafeRedirectHandler,
    _select_release_asset_api,
    bundle_identity,
    install_bundle,
    sha256_file,
    verify_data_bundle,
)


def _source_data(root: Path) -> Path:
    data = root / "source"
    reference = data / "reference" / "DMS_substitutions.csv"
    reference.parent.mkdir(parents=True)
    reference.write_text("DMS_id,target_seq,fitness_description\nTOY_ASSAY,AAAA,Toy activity\n")
    assay = data / "splits" / "TOY_ASSAY"
    assay.mkdir(parents=True)
    (data / "splits" / "manifest.csv").write_text("assay,n_at_10\nTOY_ASSAY,10\n")
    for size in EVAL_SIZES:
        for seed in EVAL_SEEDS:
            (assay / f"n{size}_b{seed}.json").write_text(
                json.dumps({"assay": "TOY_ASSAY", "size": size, "seed": seed})
            )
            (assay / f"n{size}_b{seed}.labels.json").write_text('{"v": 1.0}')
    # Plausible-looking noncanonical files must never leak into the release.
    (assay / "n500_b1.json").write_text("excluded")
    (assay / "n50_b4.labels.json").write_text("excluded")
    return data


def _tar_with_member(path: Path, name: str, *, kind: bytes = tarfile.REGTYPE) -> None:
    manifest = {
        "schema_version": EVAL_BUNDLE_SCHEMA_VERSION,
        "bundle_version": "test",
        "selection": {"sizes": [50], "seeds": [1]},
        "provenance": EVAL_BUNDLE_PROVENANCE,
        "files": {"safe.txt": {"sha256": hashlib.sha256(b"safe").hexdigest(), "size": 4}},
    }
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(EVAL_BUNDLE_MANIFEST)
        payload = json.dumps(manifest).encode()
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        unsafe = tarfile.TarInfo(name)
        unsafe.type = kind
        if kind == tarfile.REGTYPE:
            unsafe.size = 4
            archive.addfile(unsafe, io.BytesIO(b"safe"))
        else:
            unsafe.linkname = "safe.txt"
            archive.addfile(unsafe)


def _archive_manifest_sha256(path: Path) -> str:
    with tarfile.open(path, "r:gz") as archive:
        source = archive.extractfile(EVAL_BUNDLE_MANIFEST)
        assert source is not None
        return hashlib.sha256(source.read()).hexdigest()


def test_deterministic_package_round_trip_and_manifest_verification(tmp_path: Path):
    source = _source_data(tmp_path)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_report = build_eval_bundle(source, first, version="test-v1")
    second_report = build_eval_bundle(source, second, version="test-v1")

    assert first_report["sha256"] == second_report["sha256"]
    assert first_report["manifest_sha256"] == _archive_manifest_sha256(first)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        names = set(archive.getnames())
        manifest_source = archive.extractfile(EVAL_BUNDLE_MANIFEST)
        assert manifest_source is not None
        embedded_manifest = json.load(manifest_source)
    assert embedded_manifest["provenance"] == EVAL_BUNDLE_PROVENANCE
    assert "splits/TOY_ASSAY/n500_b1.json" not in names
    assert "splits/TOY_ASSAY/n50_b4.labels.json" not in names
    assert len(names) == 21  # reference + split manifest + 18 cells + bundle manifest

    installed = tmp_path / "installed"
    manifest_sha256 = _archive_manifest_sha256(first)
    report = install_bundle(
        first,
        sha256_file(first),
        installed,
        expected_manifest_sha256=manifest_sha256,
    )
    assert report["bundle_version"] == "test-v1"
    assert report["manifest_sha256"] == manifest_sha256
    assert report["selection"] == {"sizes": [10, 50, 100], "seeds": [1, 2, 3]}
    assert report["file_count"] == 20
    assert verify_data_bundle(installed, expected_manifest_sha256=manifest_sha256) == report
    assert bundle_identity(installed, expected_manifest_sha256=manifest_sha256) == {
        "bundle_version": "test-v1",
        "manifest_sha256": manifest_sha256,
        "selection": {"sizes": [10, 50, 100], "seeds": [1, 2, 3]},
    }

    (installed / "splits" / "TOY_ASSAY" / "n50_b1.json").write_text("tampered")
    with pytest.raises(BundleError, match="does not match manifest"):
        verify_data_bundle(installed, expected_manifest_sha256=manifest_sha256)


def test_package_requires_canonical_assay_descriptions(tmp_path: Path):
    source = _source_data(tmp_path)
    reference = source / "reference" / "DMS_substitutions.csv"
    reference.write_text("DMS_id,target_seq\nTOY_ASSAY,AAAA\n")

    with pytest.raises(BundleError, match="missing columns: fitness_description"):
        build_eval_bundle(source, tmp_path / "invalid.tar.gz", version="test-v1")


def test_release_asset_resolution_is_exact_and_uses_only_the_api_url():
    release = {
        "assets": [
            {
                "name": "bundle.tar.gz",
                "url": (
                    "https://api.github.com/repos/rohitarorayyc/proteingym-llm/releases/assets/123"
                ),
                "browser_download_url": "https://attacker.invalid/file",
            }
        ]
    }
    assert _select_release_asset_api(release, "bundle.tar.gz").endswith("/123")
    with pytest.raises(BundleError, match="exactly one asset"):
        _select_release_asset_api(release, "missing.tar.gz")


def test_github_token_is_never_sent_to_arbitrary_download_url(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "sensitive-test-token")
    assert "Authorization" not in _download_headers(
        "https://downloads.example.org/eval.tar.gz", binary=True
    )
    trusted = _download_headers(
        "https://api.github.com/repos/rohitarorayyc/proteingym-llm/releases/assets/123",
        binary=True,
    )
    assert trusted["Authorization"] == "Bearer sensitive-test-token"

    request = Request(
        "https://api.github.com/repos/rohitarorayyc/proteingym-llm/releases/assets/123",
        headers=trusted,
    )
    redirected = _SafeRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://release-assets.githubusercontent.com/archive.tar.gz",
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") is None


def test_default_pin_rejects_custom_manifest(tmp_path: Path):
    source = _source_data(tmp_path)
    archive = tmp_path / "eval.tar.gz"
    build_eval_bundle(source, archive, version="test-v1")

    with pytest.raises(BundleError, match="bundle manifest checksum mismatch"):
        install_bundle(archive, sha256_file(archive), tmp_path / "installed")

    installed = tmp_path / "custom-installed"
    install_bundle(
        archive,
        sha256_file(archive),
        installed,
        expected_manifest_sha256=_archive_manifest_sha256(archive),
    )
    with pytest.raises(BundleError, match="installed bundle manifest checksum mismatch"):
        verify_data_bundle(installed)


def test_installed_manifest_tampering_is_rejected(tmp_path: Path):
    source = _source_data(tmp_path)
    archive = tmp_path / "eval.tar.gz"
    build_eval_bundle(source, archive, version="test-v1")
    manifest_sha256 = _archive_manifest_sha256(archive)
    installed = tmp_path / "installed"
    install_bundle(
        archive,
        sha256_file(archive),
        installed,
        expected_manifest_sha256=manifest_sha256,
    )

    manifest_path = installed / EVAL_BUNDLE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["bundle_version"] = "forged-v2"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(BundleError, match="installed bundle manifest checksum mismatch"):
        bundle_identity(installed, expected_manifest_sha256=manifest_sha256)
    with pytest.raises(BundleError, match="installed bundle manifest checksum mismatch"):
        verify_data_bundle(installed, expected_manifest_sha256=manifest_sha256)


def test_archive_checksum_mismatch_is_rejected_before_install(tmp_path: Path):
    source = _source_data(tmp_path)
    archive = tmp_path / "eval.tar.gz"
    build_eval_bundle(source, archive, version="test-v1")

    with pytest.raises(BundleError, match="bundle checksum mismatch"):
        install_bundle(archive, "0" * 64, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("../escape.txt", tarfile.REGTYPE, "unsafe bundle path"),
        ("safe.txt", tarfile.SYMTYPE, "links are forbidden"),
    ],
)
def test_unsafe_tar_members_are_rejected(tmp_path: Path, name: str, kind: bytes, message: str):
    archive = tmp_path / "unsafe.tar.gz"
    _tar_with_member(archive, name, kind=kind)

    with pytest.raises(BundleError, match=message):
        install_bundle(
            archive,
            sha256_file(archive),
            tmp_path / "installed",
            expected_manifest_sha256=_archive_manifest_sha256(archive),
        )
    assert not (tmp_path / "escape.txt").exists()
