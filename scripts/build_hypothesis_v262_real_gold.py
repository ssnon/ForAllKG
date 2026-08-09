from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_real_gold import build_real_gold_suite
from dac_her.hypothesis_real_gold_contracts import HypothesisRealGoldSpec


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a provenance-frozen real-output semantic gold suite for "
            "Hypothesis Maker v2.6.2."
        )
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo).resolve()
    spec_path = Path(args.spec).resolve()
    output_path = Path(args.output).resolve()

    spec = HypothesisRealGoldSpec.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )
    suite = build_real_gold_suite(
        spec,
        repo_root=repo_root,
        output_path=output_path,
    )
    _write_json(output_path, suite)

    print("Hypothesis real-output gold suite built")
    print("Suite:", suite.suite_id)
    print("Cases:", len(suite.cases))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
