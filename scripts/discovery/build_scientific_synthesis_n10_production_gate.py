from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    promote_scientific_synthesis_n10_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a role-aware N10 v2 candidate gate to scientific "
            "cross-lane synthesis authority without changing gate semantics."
        )
    )
    parser.add_argument("--candidate-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate = json.loads(args.candidate_gate.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict):
        raise ValueError("candidate gate must be a JSON object")
    result = promote_scientific_synthesis_n10_gate(candidate_gate=candidate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Scientific synthesis N10 production gate built")
    print("Authority scope:", result["authority_scope"])
    print("Selections:", result["selection_counts"])
    print("Bounded continuation offered: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
