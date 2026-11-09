"""Explain a saved CrossTrans prediction by removing supplied evidence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmfc.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["explain", *sys.argv[1:]]))
