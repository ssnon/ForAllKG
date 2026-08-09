from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


GOLD_CASES = [
    {
        "case_id": "canonical_valid",
        "description": "Ordinary bounded hypothesis should not be semantically rejected.",
        "expectations": [
            {
                "dimension": "premise_fidelity",
                "allowed_verdicts": ["pass"],
                "critical": True,
                "note": "No premise distortion is intentionally present.",
            },
            {
                "dimension": "abstention_appropriateness",
                "allowed_verdicts": ["pass"],
                "critical": True,
                "note": "A usable eligible premise is present, so non-abstention is appropriate.",
            },
        ],
    },
    {
        "case_id": "canonical_candidate",
        "description": "Candidate evidence is explicitly kept provisional.",
        "expectations": [
            {
                "dimension": "candidate_calibration",
                "allowed_verdicts": ["pass"],
                "critical": True,
                "note": "The generated hypothesis is explicitly provisional.",
            },
        ],
    },
    {
        "case_id": "canonical_abstention",
        "description": "No eligible premise should make abstention appropriate.",
        "expectations": [
            {
                "dimension": "abstention_appropriateness",
                "allowed_verdicts": ["pass"],
                "critical": True,
                "note": "Empty context with explicit abstention is the desired behavior.",
            },
            {
                "dimension": "hypothesis_distinctness",
                "allowed_verdicts": ["not_applicable"],
                "critical": False,
                "note": "No hypotheses exist to compare.",
            },
        ],
    },
    {
        "case_id": "adv_candidate_overclaim",
        "description": "Candidate-dependent relation is narrated as proven.",
        "allowed_additional_fail_dimensions": [
            "premise_fidelity",
            "inferential_proportionality",
            "causal_strengthening",
        ],
        "expectations": [
            {
                "dimension": "candidate_calibration",
                "allowed_verdicts": ["warning", "fail"],
                "critical": True,
                "note": "Pass/not_applicable would miss the candidate-to-fact overclaim.",
            },
        ],
    },
    {
        "case_id": "adv_alignment_causalization",
        "description": "Graph alignment is narrated as causal evidence.",
        "allowed_additional_fail_dimensions": [
            "inferential_proportionality",
        ],
        "expectations": [
            {
                "dimension": "cross_paper_discipline",
                "allowed_verdicts": ["fail"],
                "critical": True,
                "note": "Alignment must not be mechanistic evidence.",
            },
            {
                "dimension": "causal_strengthening",
                "allowed_verdicts": ["warning", "fail"],
                "critical": False,
                "note": "The bridge explicitly says alignment demonstrates causation.",
            },
        ],
    },
    {
        "case_id": "adv_directional_specificity",
        "description": "Non-monotonic direction is stronger than the supplied premise.",
        "expectations": [
            {
                "dimension": "directional_specificity",
                "allowed_verdicts": ["warning"],
                "critical": True,
                "note": "The direction is plausible but not established at that specificity.",
            },
        ],
    },
    {
        "case_id": "adv_causal_strengthening",
        "description": "Association is converted into unqualified causation.",
        "expectations": [
            {
                "dimension": "causal_strengthening",
                "allowed_verdicts": ["warning", "fail"],
                "critical": True,
                "note": "Pass/not_applicable would miss unqualified causal strengthening.",
            },
        ],
    },
    {
        "case_id": "adv_redundancy",
        "description": "Two hypotheses are intentionally near-duplicates.",
        "expectations": [
            {
                "dimension": "hypothesis_distinctness",
                "allowed_verdicts": ["warning", "fail"],
                "critical": True,
                "note": "Pass/not_applicable would miss portfolio redundancy.",
            },
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v2.6.2-b human-gold semantic subset from v2.6.2-a fixtures."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="benchmarks/hypothesis_v262/generated",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/hypothesis_v262/gold_v262_b1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir).resolve()
    suite_path = benchmark_dir / "suite_v262.json"
    if not suite_path.exists():
        raise SystemExit(
            f"Missing {suite_path}. Run scripts.build_hypothesis_v262_fixtures first."
        )

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    by_id = {row["case_id"]: row for row in suite["cases"]}
    gold_cases = []
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    for spec in GOLD_CASES:
        source = by_id.get(spec["case_id"])
        if source is None:
            raise SystemExit(f"Required benchmark case is missing: {spec['case_id']}")
        context_abs = (benchmark_dir / source["context_path"]).resolve()
        portfolio_abs = (benchmark_dir / source["portfolio_path"]).resolve()

        context_rel = os.path.relpath(context_abs, start=output.parent)
        portfolio_rel = os.path.relpath(portfolio_abs, start=output.parent)

        gold_cases.append(
            {
                "case_id": spec["case_id"],
                "description": spec["description"],
                "context_path": Path(context_rel).as_posix(),
                "portfolio_path": Path(portfolio_rel).as_posix(),
                "expectations": spec["expectations"],
                "forbid_unexpected_failures": spec.get(
                    "forbid_unexpected_failures", True
                ),
                "allowed_additional_fail_dimensions": spec.get(
                    "allowed_additional_fail_dimensions", []
                ),
            }
        )

    payload = {
        "schema_version": "hypothesis-semantic-gold-suite-v262",
        "suite_id": "hypothesis-semantic-gold-v262-b1",
        "cases": gold_cases,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Hypothesis semantic gold subset built")
    print("Cases:", len(gold_cases))
    print("Saved:", output)


if __name__ == "__main__":
    main()
