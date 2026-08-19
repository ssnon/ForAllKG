from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPolicy,
)
from dac_her.hypothesis_trend_compiler import (
    TrendAwareHypothesisCompiler,
)
from dac_her.hypothesis_trend_contracts import (
    TrendAwareFalsificationCriterionDraft,
    TrendAwareHypothesisPortfolioDraft,
    TrendAwareHypothesisProposalDraft,
    TrendAwarePredictedObservationDraft,
    TrendReferenceDraft,
)
from dac_her.hypothesis_trend_grounding import (
    HypothesisTrendGroundingBundle,
)
from dac_her.hypothesis_trend_input import (
    _sha256_json,
    build_trend_aware_hypothesis_input,
)
from dac_her.hypothesis_trend_validator import (
    TrendAwareHypothesisValidator,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grounding",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    grounding = (
        HypothesisTrendGroundingBundle.
        model_validate_json(
            args.grounding.read_text(
                encoding="utf-8"
            )
        )
    )

    trend_summary = next(
        row
        for row in grounding.source_artifacts
        if row.role == "trend_summary"
    )
    summary = json.loads(
        Path(trend_summary.path).read_text(
            encoding="utf-8"
        )
    )
    corpus_id = str(summary["corpus_id"])

    context_payload = {
        "schema_version": "hypothesis-context-v1",
        "context_id":
            "context:alpha4c5c-v2-seen-synthetic",
        "source_packet_id":
            "packet:alpha4c5c-v2-seen-synthetic",
        "source_packet_sha256":
            "synthetic-regression-packet",
        "source_report_id":
            "report:alpha4c5c-v2-seen-synthetic",
        "source_report_sha256":
            "synthetic-regression-report",
        "task_id":
            "alpha4c5c-v2-seen-regression",
        "question": (
            "Synthetic deterministic regression fixture only; "
            "not a scientific report."
        ),
        "corpus_id": corpus_id,
        "domain_profile_id": "sers_au_ag",
        "evidence_statements": [],
        "mechanism_routes": [],
        "mechanistic_motifs": [],
        "reported_design_levers": [],
        "research_gaps": [],
        "partial_absence_blocked_paper_ids": [],
        "policy":
            HypothesisPolicy().model_dump(
                mode="json"
            ),
    }
    context_payload["context_sha256"] = (
        _sha256_json(context_payload)
    )
    context = HypothesisContext(**context_payload)

    source = build_trend_aware_hypothesis_input(
        grounded_context=context,
        trend_grounding=grounding,
        input_semantics_id=(
            "sers_au_ag_hypothesis_trend_input_v1_alpha4c5b"
        ),
    )

    local_views = [
        row for row in source.trend_views
        if row.lane == "local_empirical_support"
    ]
    gap_views = [
        row for row in source.trend_views
        if row.lane == "replication_gap"
    ]
    if len(local_views) != 1 or len(gap_views) != 1:
        raise RuntimeError(
            "v2 seen fixture expected one local support and "
            "one replication gap view."
        )
    local = local_views[0]
    gap = gap_views[0]
    if local.grounding_id != gap.grounding_id:
        raise RuntimeError(
            "v2 local/gap views do not share a grounding."
        )
    if local.cross_context_status != "insufficient":
        raise RuntimeError(
            "v2 local support must remain insufficient "
            "cross-context."
        )

    draft = TrendAwareHypothesisPortfolioDraft(
        hypotheses=[
            TrendAwareHypothesisProposalDraft(
                local_id="h1",
                title=(
                    "Paper-local particle-size/SERS relation "
                    "with unresolved replication"
                ),
                hypothesis_statement=(
                    "Within a comparable local context, the "
                    "reported particle-size-associated SERS "
                    "performance trend should remain observable, "
                    "while its cross-paper generality remains "
                    "unresolved."
                ),
                hypothesis_type="context_dependency",
                premise_statement_ids=[],
                gap_statement_ids=[],
                trend_references=[
                    TrendReferenceDraft(
                        view_id=local.view_id,
                        use_role=(
                            "positive_empirical_support"
                        ),
                    ),
                    TrendReferenceDraft(
                        view_id=gap.view_id,
                        use_role="replication_gap",
                    ),
                ],
                inferential_bridge=(
                    "Treat the paper-local directional association "
                    "as a scoped empirical premise, not as "
                    "cross-paper replication, and test whether it "
                    "persists under comparable context."
                ),
                predicted_observations=[
                    TrendAwarePredictedObservationDraft(
                        local_id="p1",
                        observable="sers_performance",
                        expected_direction="increase",
                        rationale=(
                            "The source Trend result reports a "
                            "positive paper-local direction."
                        ),
                    )
                ],
                falsification_criteria=[
                    TrendAwareFalsificationCriterionDraft(
                        local_id="f1",
                        observable="sers_performance",
                        falsifying_outcome=(
                            "The scoped qualitative increase is "
                            "not observed under the comparable "
                            "context considered by the hypothesis."
                        ),
                    )
                ],
                assumptions=[
                    (
                        "Cross-paper replication is not established "
                        "by the current Trend grounding."
                    )
                ],
            )
        ],
        abstention_reason=None,
    )

    compiler = TrendAwareHypothesisCompiler()
    portfolio = compiler.compile(source, draft)
    validation = TrendAwareHypothesisValidator().validate(
        source,
        portfolio,
    )
    if not validation.passes:
        raise RuntimeError(
            "v2 seen deterministic regression failed: "
            + validation.model_dump_json()
        )

    card = portfolio.hypotheses[0]
    if card.premise_statement_ids != []:
        raise RuntimeError(
            "v2 trend-only regression unexpectedly gained "
            "Explorer premise IDs."
        )
    if len(card.trend_references) != 2:
        raise RuntimeError(
            "v2 trend-only regression did not preserve both "
            "local support and replication gap."
        )
    if card.cross_paper_synthesis:
        raise RuntimeError(
            "v2 insufficient local result was promoted to "
            "cross-paper synthesis."
        )
    if card.evidence_profile.trend_positive_support_count != 1:
        raise RuntimeError(
            "v2 positive Trend support count drifted."
        )
    if card.evidence_profile.trend_gap_count != 1:
        raise RuntimeError(
            "v2 Trend gap count drifted."
        )
    if (
        card.trend_causal_authorization is not False
        or card.trend_universal_authorization is not False
    ):
        raise RuntimeError(
            "v2 Trend provenance authorization escalated."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    (args.output_dir / "trend_input.json").write_text(
        source.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "draft.json").write_text(
        draft.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (
        args.output_dir / "portfolio.json"
    ).write_text(
        portfolio.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (
        args.output_dir / "validation.json"
    ).write_text(
        validation.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "phase": "alpha4c.5c",
        "fixture_kind":
            "synthetic_context_plus_real_v2_seen_trend",
        "scientific_result": False,
        "v3_reserve_used": False,
        "trend_only_positive_support": True,
        "replication_gap_companion_preserved": True,
        "cross_paper_synthesis": False,
        "compiler_semantics_id": compiler.semantics_id,
        "validator_semantics_id":
            validation.semantics_id,
        "validation_passed": validation.passes,
    }
    (
        args.output_dir / "fixture_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("alpha4c.5c v2 seen deterministic regression")
    print("Trend-only positive support: True")
    print("Local support refs: 1")
    print("Replication gap refs: 1")
    print("Explorer premise refs: 0")
    print("Cross-paper synthesis: False")
    print("Trend causal authorization: False")
    print("Trend universal authorization: False")
    print("Validation:", validation.passes)
    print("Errors:", validation.errors)
    print("Warnings:", validation.warnings)
    print("Output:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
