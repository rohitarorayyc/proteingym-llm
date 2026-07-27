import stat

import pytest

from src.init_config import TEMPLATE, initialize_env, main


def test_initialize_env_is_private_and_never_overwrites(tmp_path):
    path = tmp_path / ".env"
    initialized, created = initialize_env(path)

    assert initialized == path
    assert created is True
    assert path.read_text() == TEMPLATE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    path.write_text("LAB_API_KEY=already-set\n")
    path.chmod(0o644)
    initialized, created = initialize_env(path)

    assert created is False
    assert path.read_text() == "LAB_API_KEY=already-set\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_initialize_env_refuses_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("do-not-touch")
    link = tmp_path / ".env"
    link.symlink_to(target)

    with pytest.raises(RuntimeError, match="non-regular"):
        initialize_env(link)


def test_help_has_no_filesystem_side_effect(tmp_path):
    path = tmp_path / ".env"
    with pytest.raises(SystemExit) as exit_info:
        main(["--help", "--path", str(path)])
    assert exit_info.value.code == 0
    assert not path.exists()
