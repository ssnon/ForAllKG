from __future__ import annotations

import sys

from scripts.discovery.run_research_idea_closed_generation_cycle_v3_0 import main


if __name__ == "__main__":
    if "--adaptive-fertility" not in sys.argv:
        sys.argv.insert(1, "--adaptive-fertility")
    raise SystemExit(main())
