from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.reframing.ablation_evaluation import (
    ablation_condition_counts,
    build_scientific_reasoning_ablation_packet,
)
from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
)
from pipeline_core.discovery.reframing.cross_lane_portfolio import (
    CrossLaneScientificReasoningShadowPortfolio,
)
from pipeline_core.discovery.reframing.proxy_challenge import ProxyChallengeRunReport
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframingShadowReport,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_model(path: Path, model_type):
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic blind SERS scientific-reasoning ablation packet "
            "for RELATIONAL_ONLY, REFRAMING_ONLY, and COMBINED conditions. "
            "This stage performs no judge call, ranking, novelty assessment, or selection."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--cross-lane-portfolio", required=True, type=Path)
    parser.add_argument("--relational-portfolio", required=True, type=Path)
    parser.add_argument("--reframe-shadow", required=True, type=Path)
    parser.add_argument("--proxy-shadow", type=Path, default=None)
    parser.add_argument("--contradiction-shadow", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--key-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_path = args.context.expanduser().resolve()
    cross_lane_path = args.cross_lane_portfolio.expanduser().resolve()
    relational_path = args.relational_portfolio.expanduser().resolve()
    reframe_path = args.reframe_shadow.expanduser().resolve()
    proxy_path = (
        args.proxy_shadow.expanduser().resolve()
        if args.proxy_shadow is not None
        else None
    )
    contradiction_path = (
        args.contradiction_shadow.expanduser().resolve()
        if args.contradiction_shadow is not None
        else None
    )

    context = _read_model(context_path, HypothesisContext)
    cross_lane = _read_model(
        cross_lane_path, CrossLaneScientificReasoningShadowPortfolio
    )
    relational = _read_model(relational_path, HypothesisPortfolio)
    reframe = _read_model(reframe_path, ScientificReframingShadowReport)
    proxy = (
        _read_model(proxy_path, ProxyChallengeRunReport)
        if proxy_path is not None
        else None
    )
    contradiction = (
        _read_model(contradiction_path, ContradictionResolutionRunReport)
        if contradiction_path is not None
        else None
    )

    packet, key = build_scientific_reasoning_ablation_packet(
        context=context,
        cross_lane_portfolio=cross_lane,
        relational_portfolio=relational,
        reframe_shadow=reframe,
        proxy_shadow=proxy,
        contradiction_shadow=contradiction,
        source_cross_lane_portfolio_file_sha256=_sha256_file(cross_lane_path),
        source_context_file_sha256=_sha256_file(context_path),
        source_relational_portfolio_file_sha256=_sha256_file(relational_path),
        source_reframe_shadow_file_sha256=_sha256_file(reframe_path),
        source_proxy_shadow_file_sha256=(
            _sha256_file(proxy_path) if proxy_path is not None else None
        ),
        source_contradiction_shadow_file_sha256=(
            _sha256_file(contradiction_path)
            if contradiction_path is not None
            else None
        ),
    )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else cross_lane_path.parent / "scientific_reasoning_ablation_packet.json"
    )
    key_output = (
        args.key_output.expanduser().resolve()
        if args.key_output is not None
        else cross_lane_path.parent / "scientific_reasoning_ablation_key.json"
    )
    _write_json(output, packet)
    _write_json(key_output, key)

    counts = ablation_condition_counts(key)
    print("Scientific reasoning ablation packet complete")
    print("LLM calls: 0")
    print(f"Task: {key.source_task_id}")
    print(f"Eligible evidence statements: {packet.evidence_statement_count}")
    print(f"Source candidates: {packet.source_candidate_count}")
    print(
        "Conditions: "
        f"RELATIONAL_ONLY={counts['RELATIONAL_ONLY']}, "
        f"REFRAMING_ONLY={counts['REFRAMING_ONLY']}, "
        f"COMBINED={counts['COMBINED']}"
    )
    print(f"Blind pairwise comparisons: {packet.comparison_count}")
    for row in packet.comparisons:
        print(
            f"  {row.comparison_alias}: "
            f"ARM_A={row.arm_a.candidate_count}, "
            f"ARM_B={row.arm_b.candidate_count}, "
            f"count_difference={row.absolute_candidate_count_difference}"
        )
    print("Condition labels hidden from evaluator packet: true")
    print("Candidate source IDs hidden from evaluator packet: true")
    print("Candidate count treated as quality signal: false")
    print("Evaluation judgment performed: false")
    print("Scientific quality ranking performed: false")
    print("Production selection changed: false")
    print(f"Packet: {output}")
    print(f"Blind key (do not supply to evaluator): {key_output}")


if __name__ == "__main__":
    main()
