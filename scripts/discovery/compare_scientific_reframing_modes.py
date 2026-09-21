from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.mode_contrast import (
    build_reasoning_mode_contrast,
    load_reasoning_mode_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare scientific reframing candidates across reasoning modes without ranking them."
    )
    parser.add_argument("--reframe-shadow", required=True, type=Path)
    parser.add_argument("--proxy-shadow", type=Path, default=None)
    parser.add_argument("--contradiction-shadow", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    reframe, proxy, contradiction = load_reasoning_mode_inputs(
        reframe_shadow_path=args.reframe_shadow,
        proxy_shadow_path=args.proxy_shadow,
        contradiction_shadow_path=args.contradiction_shadow,
    )
    report = build_reasoning_mode_contrast(
        reframe_shadow=reframe,
        proxy_shadow=proxy,
        contradiction_shadow=contradiction,
    )
    output_base = args.proxy_shadow or args.contradiction_shadow or args.reframe_shadow
    output = args.output or output_base.with_name(
        "scientific_reframing_mode_contrast.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print("Scientific reframing reasoning-mode contrast complete")
    print("LLM calls: 0")
    print(f"Task: {report.source_task_id}")
    print(f"Candidates: {len(report.candidate_profiles)}")
    print(f"Cross-mode pairs: {len(report.pairwise_contrasts)}")
    if report.relation_counts:
        print("Relations:")
        for key, value in report.relation_counts.items():
            print(f"  {key}: {value}")
    for row in report.pairwise_contrasts:
        print(
            f"  {row.operator_a} ↔ {row.operator_b}: {row.relation}; "
            f"premise_jaccard={row.premise_jaccard:.3f}; "
            f"challenge_jaccard={row.challenge_token_jaccard:.3f}; "
            f"prediction_jaccard={row.prediction_observable_token_jaccard:.3f}"
        )
    print("No scientific quality ranking or portfolio selection was performed.")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
