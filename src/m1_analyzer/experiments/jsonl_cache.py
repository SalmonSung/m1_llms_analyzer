"""The resumable JSONL cache shared by the phase-A scorers.

One header line, then one line per finished item, each flushed and fsynced so
a Colab pre-emption loses at most one item. On resume the header must agree
with the run (the mismatching field is named), and a truncated last line --
an interrupted write -- is dropped and rescored rather than raising.

`span_costs` (Task 1b) and `splice` (Task 9a) both write caches this way; the
row format is theirs, the file mechanics are here.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from ..utils.logging import get_logger

log = get_logger("jsonl_cache")


def read_jsonl(path: Path) -> tuple[dict | None, list[dict], bool]:
    """``(header, rows, truncated)``; a broken *last* line is dropped with a warning."""
    header, rows, truncated = None, [], False
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if k == len(lines) - 1:
                truncated = True
                log.warning("%s: last line is truncated (interrupted write); it will be rescored.", path)
                break
            raise ValueError(f"{path}: line {k + 1} is not valid JSON and is not the last line.")
        if obj.get("kind") == "header":
            header = obj
        else:
            rows.append(obj)
    return header, rows, truncated


def check_header(existing: dict, wanted: dict, path: Path, unchecked: Iterable[str] = ()) -> None:
    """Refuse to resume a cache written under different settings; name the field."""
    skip = set(unchecked)
    for key, value in wanted.items():
        if key in skip:
            continue
        if key in existing and existing[key] != value:
            raise ValueError(
                f"{path} was written for {key}={existing[key]!r}, but this run has {key}={value!r}. "
                "Use a different cache_path, or delete the file to rescore."
            )


def rewrite(path: Path, header: dict, rows: list[dict]) -> None:
    """Atomically rewrite the file from parsed content (drops a truncated tail)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def write_header(path: Path, header: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def append_row(fh, row: dict[str, Any]) -> None:
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def mirror(path: Path, mirror_path: str | os.PathLike) -> None:
    """Copy the cache somewhere that survives a runtime wipe (e.g. Drive)."""
    target = Path(mirror_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, target)
