#!/usr/bin/env python3
"""Thin wrapper so the CLI runs from a checkout without installing the package.

    python scripts/run_extraction.py --model sshleifer/tiny-gpt2 --text "hello" --out outputs/x.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m1_analyzer.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
