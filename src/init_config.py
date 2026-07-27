"""Create a private local environment file without overwriting credentials."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from config.paths import WORK_ROOT

ENV_FILE = WORK_ROOT / ".env"
TEMPLATE = "LAB_API_KEY=\nLAB_BASE_URL=https://your-endpoint.example/v1\n"


def initialize_env(path: Path = ENV_FILE) -> tuple[Path, bool]:
    """Create *path* with mode 0600, or secure an existing regular file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"refusing non-regular environment path: {path}")
    else:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(TEMPLATE)
        created = True
    path.chmod(0o600)
    return path, created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a private local environment file without replacing credentials"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=ENV_FILE,
        help=f"environment file to create (default: {ENV_FILE})",
    )
    args = parser.parse_args(argv)
    path, created = initialize_env(args.path)
    action = "created" if created else "kept existing"
    print(f"{action} private environment file: {path}")
    print("Add LAB_API_KEY locally; never commit or paste the key into model JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
