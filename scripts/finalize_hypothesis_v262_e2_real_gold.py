from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_e2 import worksheet_to_combined_spec
from dac_her.hypothesis_e2_contracts import HypothesisE2HumanReviewWorksheet
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
            "Finalize the human-approved E2 worksheet into a five-case "
            "real-output gold spec and provenance-frozen gold suite."
        )
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--base-spec",
        default="benchmarks/hypothesis_v262/real_gold_spec.k9.example.json",
    )
    parser.add_argument(
        "--worksheet",
        default="benchmarks/hypothesis_v262/e2_review_worksheet.json",
    )
    parser.add_argument(
        "--output-spec",
        default="benchmarks/hypothesis_v262/real_gold_spec.e2_5case.json",
    )
    parser.add_argument(
        "--output-gold",
        default="benchmarks/hypothesis_v262/real/gold_v262_real_e2_5case.json",
    )
    parser.add_argument(
        "--suite-id",
        default="hypothesis-real-output-gold-v262-e2-5case",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    base_spec_path = Path(args.base_spec).resolve()
    worksheet_path = Path(args.worksheet).resolve()
    output_spec_path = Path(args.output_spec).resolve()
    output_gold_path = Path(args.output_gold).resolve()

    base_spec = HypothesisRealGoldSpec.model_validate_json(
        base_spec_path.read_text(encoding="utf-8")
    )
    worksheet = HypothesisE2HumanReviewWorksheet.model_validate_json(
        worksheet_path.read_text(encoding="utf-8")
    )
    combined = worksheet_to_combined_spec(
        base_spec=base_spec,
        worksheet=worksheet,
        output_suite_id=args.suite_id,
    )
    _write_json(output_spec_path, combined)

    frozen = build_real_gold_suite(
        combined,
        repo_root=repo,
        output_path=output_gold_path,
    )
    _write_json(output_gold_path, frozen)

    print("Hypothesis v2.6.2 E2 five-case real-output gold finalized")
    print("Cases:", len(frozen.cases))
    print("Spec:", output_spec_path)
    print("Frozen gold:", output_gold_path)
    print(
        "Note: four E2 cases are fresh live Hypothesis Maker outputs on "
        "controlled evidence contexts, not additional corpus-derived evidence."
    )


if __name__ == "__main__":
    main()
