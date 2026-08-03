"""Compatibility wrapper for the old command.

Use `python -m scripts.extract_paper --paper-id ...` for new work.
"""
from __future__ import annotations

import sys

from scripts.extract_paper import main


if __name__ == "__main__":
    if "--paper-id" not in sys.argv:
        sys.argv.extend(["--paper-id", "Zhang2019_PtRu"])
    main()
