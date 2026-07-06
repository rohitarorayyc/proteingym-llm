"""Download real ProteinGym data into data/.

  reference -> data/reference/DMS_substitutions.csv  (assay metadata: WT seq,
               function category, taxon, length, #mutants, ...)
  dms       -> data/DMS/...                          (217 substitution assays,
               full mutated sequences + DMS_score)

Canonical ProteinGym sources. The DMS zip is ~1GB; it is streamed to disk then
extracted. Use --what reference|dms|all. As a fallback for an environment with no
network, --from-local <dir> copies an existing ProteinGym folder instead.

    python -m src.download --what reference
    python -m src.download --what all
    python -m src.download --from-local /path/to/ProteinGym
"""
from __future__ import annotations
import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DMS = DATA / "DMS"
REF = DATA / "reference"
BASE = DATA / "baselines"

PG = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3"
# v1.3 canonical reference (assay metadata: function, taxon, WT seq, #mutants, ...)
REFERENCE_URL = f"{PG}/DMS_substitutions.csv"
DMS_ZIP_URLS = [f"{PG}/DMS_ProteinGym_substitutions.zip"]
# per-variant zero-shot scores (1.9 GB) -> recompute baseline Spearman on matched subsets
BASELINE_SCORES_ZIP_URLS = [f"{PG}/zero_shot_substitutions_scores.zip"]
# aggregated published DMS-level Spearman per baseline (small, sanity-check)
BASELINE_SPEARMAN_URLS = [
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/"
    "DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv",
]
HEADERS = {"User-Agent": "pg-agent/1.0"}


def _contexts():
    """Verified context first (certifi if available), then an unverified one as
    fallback — some data hosts (e.g. marks.hms.harvard.edu) serve an incomplete
    cert chain that even certifi can't verify in this environment."""
    import ssl
    ctxs = []
    try:
        import certifi
        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:  # noqa
        ctxs.append(ssl.create_default_context())
    unverified = ssl.create_default_context()
    unverified.check_hostname = False
    unverified.verify_mode = ssl.CERT_NONE
    ctxs.append(unverified)
    return ctxs


def _download_to(url: str, dest: Path, timeout: int) -> int:
    last = None
    for i, ctx in enumerate(_contexts()):
        req = urllib.request.Request(url, headers=HEADERS)
        total = 0
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)            # 1 MB
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    print(f"\r  {dest.name}: {total/1e6:8.1f} MB", end="", flush=True)
            print()
            return total
        except urllib.error.URLError as e:
            last = e
            print(f"  ({'verified' if i == 0 else 'unverified'}) failed: {e}; "
                  f"{'retrying unverified' if i == 0 else 'giving up'}")
    raise last


def _safe_extractall(z: zipfile.ZipFile, dest: Path) -> None:
    """extractall with Zip-Slip protection: refuse any member that would resolve
    outside dest (e.g. names containing '../' or absolute paths)."""
    dest = dest.resolve()
    for member in z.namelist():
        target = (dest / member).resolve()
        if target != dest and dest not in target.parents:
            raise RuntimeError(f"unsafe path in zip (zip-slip): {member!r}")
    z.extractall(dest)


def fetch_reference(timeout: int) -> None:
    REF.mkdir(parents=True, exist_ok=True)
    n = _download_to(REFERENCE_URL, REF / "DMS_substitutions.csv", timeout)
    print(f"reference: {n} bytes -> {REF/'DMS_substitutions.csv'}")


def fetch_dms(timeout: int) -> None:
    DMS.mkdir(parents=True, exist_ok=True)
    tmp = DATA / "_dms.zip"
    for url in DMS_ZIP_URLS:
        try:
            print(f"downloading DMS zip: {url}")
            _download_to(url, tmp, timeout)
            with zipfile.ZipFile(tmp) as z:
                _safe_extractall(z, DMS)
            tmp.unlink(missing_ok=True)
            n = len(list(DMS.rglob("*.csv")))
            print(f"DMS: extracted {n} CSVs -> {DMS}")
            return
        except Exception as e:  # noqa
            print(f"  failed: {type(e).__name__}: {e}")
            tmp.unlink(missing_ok=True)
    sys.exit("all DMS sources failed (try --from-local)")


def fetch_baselines(timeout: int) -> None:
    """Published baselines: small aggregated Spearman CSV + the large per-variant
    zero-shot scores zip (extracted under data/baselines/zero_shot_substitutions_scores/,
    which is exactly where baselines.py reads each assay's <assay>.csv)."""
    BASE.mkdir(parents=True, exist_ok=True)
    for url in BASELINE_SPEARMAN_URLS:
        try:
            _download_to(url, BASE / "DMS_substitutions_Spearman_DMS_level.csv", timeout)
            print("baseline summary Spearman: ok")
            break
        except Exception as e:  # noqa
            print(f"  summary failed: {e}")
    tmp = DATA / "_baselines.zip"
    scores_dir = BASE / "zero_shot_substitutions_scores"   # baselines.py reads here
    for url in BASELINE_SCORES_ZIP_URLS:
        try:
            print(f"downloading per-variant baseline scores (1.9 GB): {url}")
            _download_to(url, tmp, timeout)
            # The zip's top-level folder name isn't guaranteed (it may wrap the
            # CSVs in its own dir, or ship them flat). Extract to a staging area,
            # then flatten the per-assay CSVs into scores_dir so the layout always
            # matches what baselines.py expects regardless of the zip's structure.
            stage = BASE / "_scores_stage"
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp) as z:
                _safe_extractall(z, stage)
            # Start clean so a re-download never leaves stale CSVs behind.
            if scores_dir.exists():
                shutil.rmtree(scores_dir)
            scores_dir.mkdir(parents=True, exist_ok=True)
            n, seen = 0, set()
            for csv in stage.rglob("*.csv"):
                if csv.name in seen:            # two CSVs share a basename across subdirs
                    print(f"  warning: duplicate baseline CSV name, overwriting: {csv.name}")
                seen.add(csv.name)
                shutil.move(str(csv), str(scores_dir / csv.name))
                n += 1
            shutil.rmtree(stage, ignore_errors=True)
            tmp.unlink(missing_ok=True)
            print(f"baseline per-variant scores: {n} CSVs -> {scores_dir}")
            return
        except Exception as e:  # noqa
            print(f"  failed: {type(e).__name__}: {e}")
            tmp.unlink(missing_ok=True)
    print("per-variant baseline scores unavailable (summary Spearman may still be present)")


def from_local(src: str) -> None:
    """Copy a local ProteinGym folder: its *.csv assays -> data/DMS/, and a
    DMS_substitutions.csv (anywhere under it) -> data/reference/."""
    src_path = Path(src)
    if not src_path.exists():
        sys.exit(f"not found: {src_path}")
    DMS.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)
    ref = next((p for p in src_path.rglob("DMS_substitutions.csv")), None)
    if ref:
        shutil.copy2(ref, REF / "DMS_substitutions.csv")
        print(f"reference <- {ref}")
    n = 0
    for csv in src_path.rglob("*.csv"):
        if csv.name == "DMS_substitutions.csv":
            continue
        shutil.copy2(csv, DMS / csv.name)
        n += 1
    print(f"DMS: copied {n} CSVs -> {DMS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["reference", "dms", "baselines", "all"], default="all")
    ap.add_argument("--from-local")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    if args.from_local:
        from_local(args.from_local)
        return
    if args.what in ("reference", "all"):
        fetch_reference(args.timeout)
    if args.what in ("dms", "all"):
        fetch_dms(args.timeout)
    if args.what in ("baselines", "all"):
        fetch_baselines(args.timeout)


if __name__ == "__main__":
    main()
