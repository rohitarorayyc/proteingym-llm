import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_COMMANDS = {
    "pgllm-data",
    "pgllm-export",
    "pgllm-init",
    "pgllm-models",
    "pgllm-run",
    "pgllm-score",
    "pgllm-status",
    "pgllm-verify-splits",
}


def test_public_repository_excludes_internal_campaign_material():
    for name in ("artifacts", "campaigns"):
        assert not (ROOT / name).exists()

    for name in (
        "anthropic_batch.py",
        "effort_analysis.py",
        "effort_inputs.py",
        "effort_policy.py",
        "effort_reconcile.py",
        "effort_source.py",
        "incremental_campaign.py",
        "reconcile.py",
    ):
        assert not (ROOT / "src" / name).exists()


def test_readme_stays_focused_and_has_no_internal_recovery_language():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    assert len(readme.splitlines()) <= 200
    for phrase in (
        "campaign",
        "repair",
    ):
        assert phrase not in lowered

    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", readme):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative_path = target.split("#", 1)[0]
        assert (ROOT / relative_path).exists(), f"README link is broken: {target}"


def test_console_script_surface_contains_only_public_workflow_commands():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = re.search(
        r"^\[project\.scripts\]\n(?P<body>.*?)(?=^\[)",
        pyproject,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert scripts is not None
    names = {
        line.split("=", 1)[0].strip() for line in scripts.group("body").splitlines() if "=" in line
    }
    assert names == PUBLIC_COMMANDS
