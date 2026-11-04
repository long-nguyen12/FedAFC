"""Run pooled centralized evidence-conditioned training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmfc.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["train-centralized", *sys.argv[1:]]))
