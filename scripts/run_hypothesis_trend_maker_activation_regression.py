from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_trend_contracts import (
    TrendAwareFalsificationCriterionDraft,
    TrendAwareHypothesisPortfolioDraft,
    TrendAwareHypothesisProposalDraft,
    TrendAwarePredictedObservationDraft,
    TrendReferenceDraft,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput
from dac_her.hypothesis_trend_llm import (
    TrendAwareHypothesisDraftGeneration,
)
from dac_her.hypothesis_trend_maker_exposure import (
    build_trend_maker_exposure,
)
from dac_her.hypothesis_trend_prompt import (
    TrendAwareHypothesisPromptAssembler,
)
from dac_her.hypothesis_trend_runtime import (
    TrendAwareHypothesisMakerAgentRuntime,
)


class DeterministicRegressionBackend:
    backend_name = "alpha4c5d_deterministic_regression"
    model_name = "none"
    temperature = 0.0
    instructor_mode = "none"
    base_url = None
    parse_retries = 0

    def __init__(self, draft: TrendAwareHypothesisPortfolioDraft) -> None:
        self.draft = draft
        self.generate_calls = 0
        self.repair_calls = 0

    def generate(self, prompt):
        self.generate_calls += 1
        return TrendAwareHypothesisDraftGeneration(draft=self.draft)

    def repair(self, prompt, previous_draft, feedback):
        self.repair_calls += 1
        raise AssertionError(
            "alpha4c.5d deterministic v2 regression must not need repair"
        )


def _write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[attr-defined]
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = TrendAwareHypothesisInput.model_validate_json(
        args.input.read_text(encoding="utf-8")
    )
    exposure = build_trend_maker_exposure(source)

    if any(row.maker_selectable for row in source.trend_views):
        raise RuntimeError(
            "alpha4c.5d mutated frozen alpha4c.5b maker_selectable."
        )
    if any(row.causal_use_allowed for row in source.trend_views):
        raise RuntimeError("5b causal authorization drifted.")
    if any(row.universal_use_allowed for row in source.trend_views):
        raise RuntimeError("5b universal authorization drifted.")

    local = [
        row for row in exposure.views
        if row.lane == "local_empirical_support"
    ]
    gaps = [
        row for row in exposure.views
        if row.lane == "replication_gap"
    ]
    if len(local) != 1 or len(gaps) != 1:
        raise RuntimeError(
            "v2 seen fixture expected exactly one local support and one "
            "replication gap exposure view."
        )
    local_view = local[0]
    gap_view = gaps[0]
    if local_view.cross_context_status != "insufficient":
        raise RuntimeError("v2 local support must remain insufficient.")
    if local_view.allowed_use_role != "positive_empirical_support":
        raise RuntimeError("v2 local support role drifted.")
    if gap_view.allowed_use_role != "replication_gap":
        raise RuntimeError("v2 replication gap role drifted.")
    if [
        (row.use_role, row.view_id)
        for row in local_view.required_companions
    ] != [("replication_gap", gap_view.view_id)]:
        raise RuntimeError(
            "v2 positive local support did not expose its mandatory gap companion."
        )

    assembler = TrendAwareHypothesisPromptAssembler(max_hypotheses=1)
    prompt = assembler.build(source, exposure=exposure)
    for token in (
        local_view.view_id,
        gap_view.view_id,
        "positive_empirical_support",
        "replication_gap",
        "REQUIRED COMPANIONS IF SELECTED AS POSITIVE SUPPORT",
    ):
        if token not in prompt.user_prompt:
            raise RuntimeError(f"5d prompt omitted required token: {token}")

    draft = TrendAwareHypothesisPortfolioDraft(
        hypotheses=[
            TrendAwareHypothesisProposalDraft(
                local_id="h1",
                title=(
                    "Paper-local particle-size/SERS relation with "
                    "explicit replication gap"
                ),
                hypothesis_statement=(
                    "Within the source-compatible local scope, the "
                    "particle-size-associated SERS performance direction "
                    "should remain qualitatively observable, while its "
                    "cross-paper generality remains unresolved."
                ),
                hypothesis_type="context_dependency",
                premise_statement_ids=[],
                gap_statement_ids=[],
                trend_references=[
                    TrendReferenceDraft(
                        view_id=local_view.view_id,
                        use_role="positive_empirical_support",
                    ),
                    TrendReferenceDraft(
                        view_id=gap_view.view_id,
                        use_role="replication_gap",
                    ),
                ],
                inferential_bridge=(
                    "Use the paper-local directional association only as a "
                    "scoped empirical premise and preserve the unresolved "
                    "replication boundary."
                ),
                predicted_observations=[
                    TrendAwarePredictedObservationDraft(
                        local_id="p1",
                        observable="sers_performance",
                        expected_direction="increase",
                        rationale=(
                            "The selected paper-local Trend support carries a "
                            "positive qualitative direction."
                        ),
                    )
                ],
                falsification_criteria=[
                    TrendAwareFalsificationCriterionDraft(
                        local_id="f1",
                        observable="sers_performance",
                        falsifying_outcome=(
                            "The scoped qualitative increase is not observed "
                            "within the source-compatible local context."
                        ),
                    )
                ],
                assumptions=[
                    "Cross-paper replication is not established by this input."
                ],
            )
        ],
        abstention_reason=None,
    )

    backend = DeterministicRegressionBackend(draft)
    runtime = TrendAwareHypothesisMakerAgentRuntime(
        backend,
        prompt_assembler=assembler,
        max_repairs=0,
    )
    outcome = runtime.run(source)
    if not outcome.accepted:
        raise RuntimeError("alpha4c.5d deterministic runtime regression failed.")
    if backend.generate_calls != 1 or backend.repair_calls != 0:
        raise RuntimeError("alpha4c.5d deterministic backend call count drifted.")
    card = outcome.accepted_portfolio.hypotheses[0]  # type: ignore[union-attr]
    if card.premise_statement_ids:
        raise RuntimeError("Trend-only v2 regression gained Explorer premises.")
    if card.cross_paper_synthesis:
        raise RuntimeError("Insufficient v2 support became cross-paper synthesis.")
    if card.evidence_profile.trend_positive_support_count != 1:
        raise RuntimeError("Trend positive support count drifted.")
    if card.evidence_profile.trend_gap_count != 1:
        raise RuntimeError("Trend gap count drifted.")
    if card.trend_causal_authorization is not False:
        raise RuntimeError("Trend causal authorization escalated.")
    if card.trend_universal_authorization is not False:
        raise RuntimeError("Trend universal authorization escalated.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "exposure.json", exposure)
    (args.output_dir / "prompt.txt").write_text(
        "SYSTEM\n======\n"
        + prompt.system_prompt
        + "\n\nUSER\n====\n"
        + prompt.user_prompt
        + "\n",
        encoding="utf-8",
    )
    _write_json(args.output_dir / "draft.json", draft)
    _write_json(
        args.output_dir / "portfolio.json",
        outcome.accepted_portfolio,
    )
    _write_json(
        args.output_dir / "validation.json",
        outcome.validation,
    )
    _write_json(args.output_dir / "run.json", outcome.run_record)
    _write_json(
        args.output_dir / "fixture_manifest.json",
        {
            "phase": "alpha4c.5d",
            "fixture_kind": "real_v2_seen_trend_plus_synthetic_5c_context",
            "scientific_result": False,
            "llm_calls": 0,
            "v3_reserve_used": False,
            "source_5b_maker_selectable_preserved_false": True,
            "separate_5d_exposure_activation": True,
            "required_replication_gap_companion_exposed": True,
            "trend_only_positive_support": True,
            "cross_paper_synthesis": False,
            "causal_authorization": False,
            "universal_authorization": False,
            "validation_passed": bool(
                outcome.validation and outcome.validation.passes
            ),
        },
    )

    print("alpha4c.5d v2 seen deterministic Maker activation regression")
    print("Source 5b maker_selectable preserved: False")
    print("5d exposure selectable views:", len(exposure.views))
    print("Local support refs: 1")
    print("Replication gap refs: 1")
    print("Mandatory gap companion exposed: True")
    print("Trend-only positive support: True")
    print("Cross-paper synthesis: False")
    print("Trend causal authorization: False")
    print("Trend universal authorization: False")
    print("Validation:", bool(outcome.validation and outcome.validation.passes))
    print("Generation attempts:", outcome.run_record.generation_attempts)
    print("Repair attempts:", outcome.run_record.repair_attempts)
    print("LLM calls: 0")
    print("v3 reserve consumed: False")
    print("Output:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
