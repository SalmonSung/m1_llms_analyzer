"""Command-line entrypoint (`m1-extract`, or `python scripts/run_extraction.py`).

The notebook is the primary interface; this exists so the same pipeline can be
driven from a terminal, a cron job, or CI without importing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config.settings import (
    NPY_MODES,
    POOLING_MODES,
    ExtractionConfig,
    ModelConfig,
    RunConfig,
    StorageConfig,
)
from .container import Analyzer
from .utils.logging import get_logger

log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="m1-extract",
        description="Extract hidden states from a Hugging Face model and save them as JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=ModelConfig().model_id, help="Hub id or local path.")
    parser.add_argument("--revision", default=None, help="Commit SHA / tag to pin.")
    parser.add_argument(
        "--layers", default="-1",
        help=(
            "Comma-separated indices and/or keywords (all, last, middle). "
            "Use '=' for negative values so argparse does not read them as flags: "
            "--layers=-1,middle"
        ),
    )
    parser.add_argument("--pooling", default="last_token", choices=POOLING_MODES)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=ExtractionConfig().batch_size)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16")
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=RunConfig().seed)
    parser.add_argument("--npy-mode", default="auto", choices=NPY_MODES)
    parser.add_argument("--float-precision", type=int, default=StorageConfig().float_precision)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="An input string. Repeatable.")
    source.add_argument(
        "--input-file",
        help="UTF-8 file: one input per line, or a .json array/JSONL of strings.",
    )

    parser.add_argument("--out", required=True, help="Output .json path.")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def parse_layers(raw: str):
    """'-1,middle' -> [-1, 'middle']; a lone token stays scalar."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--layers must not be empty.")
    parsed = []
    for part in parts:
        try:
            parsed.append(int(part))
        except ValueError:
            parsed.append(part)
    return parsed[0] if len(parsed) == 1 else parsed


def read_inputs(path: str) -> list[str]:
    """Accept plain lines, a JSON array of strings, or JSONL objects with a 'text' key."""
    content = Path(path).read_text(encoding="utf-8")
    stripped = content.lstrip()
    if stripped.startswith("["):
        data = json.loads(content)
        return [d if isinstance(d, str) else d["text"] for d in data]
    texts = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            texts.append(json.loads(line)["text"])
        else:
            texts.append(line)
    return texts


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    texts = args.text if args.text else read_inputs(args.input_file)
    texts = [t for t in texts if t.strip()]
    if not texts:
        log.error("No non-empty inputs found.")
        return 2

    out = Path(args.out)
    config = RunConfig(
        model=ModelConfig(
            model_id=args.model,
            revision=args.revision,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        ),
        extraction=ExtractionConfig(
            layers=parse_layers(args.layers),
            pooling=args.pooling,
            max_length=args.max_length,
            batch_size=args.batch_size,
        ),
        storage=StorageConfig(
            output_dir=str(out.parent) if out.parent != Path("") else ".",
            npy_mode=args.npy_mode,
            float_precision=args.float_precision,
        ),
        seed=args.seed,
    )

    analyzer = Analyzer(config)
    result = analyzer.batch(texts, show_progress=not args.no_progress)
    paths = analyzer.save(result, name=out.stem)

    print(f"Wrote {paths}")
    print(f"  records : {len(result.records)}")
    print(f"  failures: {len(result.failures)}")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
