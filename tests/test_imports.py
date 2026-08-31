"""Guards on how the suite imports things.

A test module that does `from tests.conftest import X` only works when the repo root is
on `sys.path`. That is true for `python -m pytest` (which prepends the cwd) but NOT for
the `pytest` console script -- which is what the README documents and what most people
type. The suite passed under one and collapsed at collection under the other.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "path", sorted(TESTS_DIR.glob("test_*.py")), ids=lambda p: p.name
)
def test_no_test_module_imports_the_tests_package(path):
    """Shared constants belong in `m1_analyzer.testing`, which is always importable."""
    offenders = {m for m in _imported_modules(path) if m == "tests" or m.startswith("tests.")}
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. That resolves only when the repo root "
        "is on sys.path, so `pytest` (console script) fails at collection even though "
        "`python -m pytest` passes. Import shared constants from m1_analyzer.testing, or "
        "use a fixture."
    )


def test_shape_constants_come_from_the_package():
    """The tiny-model shape has one home, so tests and builder cannot disagree."""
    from m1_analyzer.testing import TINY_HIDDEN, TINY_LAYERS, TINY_MAX_POSITIONS

    assert (TINY_LAYERS, TINY_HIDDEN, TINY_MAX_POSITIONS) == (4, 16, 32)
