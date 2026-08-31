#!/usr/bin/env python3
"""End-to-end smoke test: load, invoke, batch, save, reload, verify.

Two modes:

    python scripts/smoke_test.py              # downloads sshleifer/tiny-gpt2 (~5 MB)
    python scripts/smoke_test.py --offline    # builds a tiny model locally, no network

`--offline` exists so the pipeline can be verified in sandboxes and CI with no
Hub access. Exits non-zero on any failure, so it is usable as a CI gate.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m1_analyzer import (  # noqa: E402
    Analyzer,
    ExtractionConfig,
    ModelConfig,
    RunConfig,
    StorageConfig,
    load_run,
)

TEXTS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "a b c",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def run(model_id: str, output_dir: str) -> None:
    analyzer = Analyzer(
        RunConfig(
            model=ModelConfig(model_id=model_id),
            # Two layers at once proves middle-layer selection works, not just the last.
            extraction=ExtractionConfig(layers=[-1, "middle"], pooling="last_token", batch_size=2),
            storage=StorageConfig(output_dir=output_dir),
        )
    )
    info = analyzer.describe()
    print(f"\nModel: {info['model_id']}  layers={info['num_hidden_layers']} "
          f"hidden={info['hidden_size']} device={info['device']} dtype={info['dtype']}")
    hidden = info["hidden_size"]

    print("\n[1/5] invoke()")
    record = analyzer.invoke(TEXTS[0])
    check(record.layer(-1).shape == (hidden,), f"last layer is a ({hidden},) vector")
    check(record.layer("middle").shape == (hidden,), "middle layer extracted too")
    check(record.token_count > 0, f"token_count = {record.token_count}")

    print("\n[2/5] batch()")
    result = analyzer.batch(TEXTS)
    check(len(result.records) == len(TEXTS), f"{len(result.records)} records returned")
    check([r.text for r in result.records] == TEXTS, "input order preserved")
    check(result.ok, "no failures")

    print("\n[3/5] invoke/batch agreement")
    import numpy as np

    np.testing.assert_allclose(
        record.layer(-1), result.records[0].layer(-1), rtol=1e-4, atol=1e-5
    )
    check(True, "invoke() and batch() produce the same vector")

    print("\n[4/5] save()")
    paths = analyzer.save(result, name="smoke_test")
    size_kb = Path(paths.json_path).stat().st_size / 1024
    check(Path(paths.json_path).exists(), f"wrote {paths.json_path} ({size_kb:.1f} KB)")

    print("\n[5/5] reload and verify")
    loaded = load_run(paths.json_path)
    check(loaded["counts"]["records"] == len(TEXTS), "record count round-tripped")
    check(
        loaded["records"][0]["layers"]["-1"]["values"].shape == (hidden,),
        "values round-tripped as a numpy array",
    )
    check(loaded["run"]["model_id"] == info["model_id"], "manifest carries the model id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="Hub id or local path.")
    parser.add_argument(
        "--offline", action="store_true",
        help="Build a tiny random model locally instead of downloading one.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        model_id = args.model
        if args.offline:
            from m1_analyzer.testing import build_tiny_local_model

            model_id = build_tiny_local_model(Path(tmp) / "tiny_model")
            print(f"Built offline test model at {model_id}")
        try:
            run(model_id, args.output_dir or str(Path(tmp) / "outputs"))
        except Exception as exc:  # noqa: BLE001
            print(f"\nSMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
