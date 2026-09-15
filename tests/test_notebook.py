"""The Colab notebooks must be valid and correctly line-separated.

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
import re
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = ("colab_entrypoint.ipynb", "experiment_1b.ipynb", "experiment_9a.ipynb", "experiment_9b.ipynb")
NOTEBOOK = NOTEBOOKS_DIR / NOTEBOOKS[0]


@pytest.fixture(scope="module", params=NOTEBOOKS)
def notebook(request) -> dict:
    """Each check runs against every notebook; all must clone via the same bootstrap."""
    path = NOTEBOOKS_DIR / request.param
    assert path.exists(), f"{path} is a Colab notebook and must exist."
    return json.loads(path.read_text(encoding="utf-8"))


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
    """Tokens must come from Colab secrets at runtime, never be baked into the file.

    Matched on shape (prefix + a long body), not on the bare prefix: the bootstrap cell
    legitimately mentions `ghp_`, `github_pat_` etc. as literals when naming a token type.
    """
    text = json.dumps(notebook)
    patterns = {
        "GitHub classic PAT": r"gh[pousr]_[A-Za-z0-9]{20,}",
        "GitHub fine-grained PAT": r"github_pat_[A-Za-z0-9_]{20,}",
        "Hugging Face token": r"hf_[A-Za-z0-9]{20,}",
    }
    for label, pattern in patterns.items():
        assert not re.search(pattern, text), f"possible hardcoded {label} in the notebook"


def test_bootstrap_never_puts_the_token_in_a_url(notebook):
    """Auth goes in a request header, so no credential can reach .git/config or a log.

    Embedding a PAT in the clone URL leaves it in `.git/config` on the runtime's disk
    until something scrubs it; a per-command `http.extraHeader` writes nothing at all.
    """
    bootstrap = next(
        _source(c) for _, c in _cells(notebook, "code") if "clone" in _source(c)
    )
    assert "http.extraHeader" in bootstrap, "expected header-based git authentication"
    assert not re.search(r"https://[^\s\"']*\{_?TOKEN", bootstrap), (
        "the token must not be interpolated into a git URL"
    )
    assert not re.search(r"https://x-access-token:\{", bootstrap), (
        "the token must not be interpolated into a git URL"
    )


def test_bootstrap_preflights_the_token(notebook):
    """A bare git failure cannot tell an expired token from a missing repo grant.

    The cell must check the token against the API first so the error names the cause.
    """
    bootstrap = next(
        _source(c) for _, c in _cells(notebook, "code") if "clone" in _source(c)
    )
    assert "api.github.com" in bootstrap, "expected an API preflight before cloning"
    for code in ("401", "403", "404"):
        assert code in bootstrap, f"preflight should explain HTTP {code} distinctly"


def test_outputs_are_not_committed(notebook):
    """Executed outputs bloat diffs and can leak data; the committed notebook is clean."""
    for i, cell in _cells(notebook, "code"):
        assert not cell.get("outputs"), f"code cell {i} has saved outputs; clear them before committing"
        assert cell.get("execution_count") is None, f"code cell {i} has an execution_count"
