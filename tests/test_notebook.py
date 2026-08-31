"""The Colab entrypoint notebook must be valid and correctly line-separated.

This exists because of a real bug: the notebook was generated with `source` arrays
whose elements had no trailing newline. `nbformat` still validated it and the JSON
still parsed, but Colab concatenates `source` elements *verbatim* -- so every cell
rendered as one enormous line and no code cell could run.

The check that missed it joined the source with "\\n" instead of "". These tests join
the way a notebook reader actually does, which is the only join that proves anything.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_entrypoint.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is the Colab entrypoint and must exist."
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    """Join exactly as nbformat/Colab do: straight concatenation, no separator."""
    return "".join(cell["source"])


def _cells(notebook: dict, kind: str | None = None) -> list[tuple[int, dict]]:
    return [
        (i, c)
        for i, c in enumerate(notebook["cells"])
        if kind is None or c["cell_type"] == kind
    ]


def test_notebook_is_valid_nbformat(notebook):
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.reads(json.dumps(notebook), as_version=4))


def test_notebook_has_cells(notebook):
    assert len(notebook["cells"]) > 10


def test_every_source_line_keeps_its_newline(notebook):
    """The exact bug: only the final element of `source` may lack a trailing newline."""
    offenders = []
    for i, cell in _cells(notebook):
        lines = cell["source"]
        for j, line in enumerate(lines[:-1]):
            if not line.endswith("\n"):
                offenders.append(f"cell {i} element {j}: {line[:60]!r}")
    assert not offenders, (
        "These `source` elements are missing a trailing newline. Colab concatenates "
        "`source` verbatim, so the cell will render as a single unusable line:\n  "
        + "\n  ".join(offenders[:10])
    )


def test_multiline_cells_actually_contain_newlines(notebook):
    """A cell built from several elements must survive the join as several lines."""
    for i, cell in _cells(notebook):
        if len(cell["source"]) > 1:
            assert "\n" in _source(cell), (
                f"cell {i} has {len(cell['source'])} source elements but joins to a "
                "single line -- the newlines were lost."
            )


def test_every_code_cell_compiles(notebook):
    """Parse the joined source, the way the kernel receives it."""
    for i, cell in _cells(notebook, "code"):
        source = _source(cell)
        # Colab's #@title / #@param magics are comments, so plain Python parsing applies.
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"code cell {i} is not valid Python: {exc}\n{source[:400]}")


def test_no_secrets_are_hardcoded(notebook):
    """Tokens must come from Colab secrets at runtime, never be baked into the file."""
    text = json.dumps(notebook)
    for marker in ("ghp_", "github_pat_", "hf_ey", "x-access-token:gh"):
        assert marker not in text, f"possible hardcoded credential ({marker!r}) in the notebook"


def test_bootstrap_scrubs_the_clone_token(notebook):
    """The clone URL carries a PAT; it must not be left behind in .git/config."""
    bootstrap = next(
        _source(c) for _, c in _cells(notebook, "code") if "git" in _source(c) and "clone" in _source(c)
    )
    assert "x-access-token" in bootstrap, "expected a token-authenticated clone URL"
    assert "set-url" in bootstrap, (
        "the bootstrap cell must run `git remote set-url` to remove the token from "
        ".git/config after cloning"
    )


def test_outputs_are_not_committed(notebook):
    """Executed outputs bloat diffs and can leak data; the committed notebook is clean."""
    for i, cell in _cells(notebook, "code"):
        assert not cell.get("outputs"), f"code cell {i} has saved outputs; clear them before committing"
        assert cell.get("execution_count") is None, f"code cell {i} has an execution_count"
