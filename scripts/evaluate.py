"""Re-score the dev-selected best_model.pt of a finished run on train/dev/test (B2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmfc.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
