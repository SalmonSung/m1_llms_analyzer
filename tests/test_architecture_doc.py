"""architecture.md must describe every module. Documentation rot fails the build.

The user asked for architecture.md to be kept current on every change; this test
is the mechanism that enforces it rather than relying on memory.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = REPO_ROOT / "architecture.md"
IGNORED = {"__pycache__"}


def _tracked_files() -> list[Path]:
    files = []
    for directory in ("src", "scripts", "tests", "notebooks", "docs"):
        root = REPO_ROOT / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() or any(part in IGNORED for part in path.parts):
                continue
            if path.suffix in {".py", ".ipynb", ".md"}:
                files.append(path.relative_to(REPO_ROOT))
    return files


def test_architecture_doc_exists():
    assert ARCHITECTURE.exists(), "architecture.md is required and must be kept up to date."


@pytest.mark.parametrize("relative", _tracked_files(), ids=str)
def test_every_file_is_documented(relative):
    """Each source file's name must appear somewhere in architecture.md."""
    text = ARCHITECTURE.read_text(encoding="utf-8")
    assert relative.name in text, (
        f"{relative} is not mentioned in architecture.md. Update the structure tree and "
        "its responsibility line whenever you add, rename, or remove a file."
    )


def test_design_docs_exist():
    assert (REPO_ROOT / "docs" / "design_decisions.md").exists()
    assert (REPO_ROOT / "docs" / "edge_cases.md").exists()
