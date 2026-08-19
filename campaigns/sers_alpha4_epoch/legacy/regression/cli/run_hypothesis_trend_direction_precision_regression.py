from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_trend_compiler import (
    TrendHypothesisCompileError,
)
from dac_her.hypothesis_trend_contracts import (
    TrendAwareFalsificationCriterionDraft,
    TrendReferenceDraft,
)
from dac_her.hypothesis_trend_directional_compiler import (
    DirectionAwareTrendHypothesisCompiler,
)
from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwarePredictedObservationDraft,
    DirectionAwareTrendHypothesisPortfolioDraft,
    DirectionAwareTrendHypothesisProposalDraft,
    TrendDirectionBindingDraft,
)
from dac_her.hypothesis_trend_directional_exposure import (
    build_directional_trend_maker_exposure,
)
from dac_her.hypothesis_trend_directional_llm import (
    DirectionAwareTrendHypothesisDraftGeneration,
)
from dac_her.hypothesis_trend_directional_prompt import (
    DirectionAwareTrendHypothesisPromptAssembler,
)
from dac_her.hypothesis_trend_directional_runtime import (
    DirectionAwareTrendHypothesisMakerAgentRuntime,
)
from dac_her.hypothesis_trend_directional_validator import (
    DirectionAwareTrendHypothesisValidator,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput


class DeterministicBackend:
    backend_name = "alpha4c5d1_direction_regression"
    model_name = "none"
    temperature = 0.0
    instructor_mode = "none"
    base_url = None
    parse_retries = 0

    def __init__(
        self,
        draft: DirectionAwareTrendHypothesisPortfolioDraft,
    ) -> None:
        self.draft = draft
        self.generate_calls = 0
        self.repair_calls = 0

    def generate(self, prompt):
        self.generate_calls += 1
        return DirectionAwareTrendHypothesisDraftGeneration(
            draft=self.draft
        )

    def repair(self, prompt, previous_draft, feedback):
        self.repair_calls += 1
        raise AssertionError(
            "direction precision regression must not require repair"
        )


def _write_json(path: Path, value: object) -> None:
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _draft(
    local_view_id: str,
    gap_view_id: str,
    *,
    dependent_change: str = "increase",
    decrease_frame: bool = False,
):
    hypothesis_statement = (
        "Within the represented context, decreasing particle size "
        "is hypothesized to improve qualitative SERS performance."
        if decrease_frame
        else (
            "Within the represented context, increasing particle size "
            "is hypothesized to increase qualitative SERS performance."
        )
    )
    inferential_bridge = (
        "If the local association persists, reducing particle size "
        "should improve qualitative SERS performance."
        if decrease_frame
        else (
            "If the local association persists, increasing particle size "
            "should coincide with increased qualitative SERS performance."
        )
    )
    observable = (
        "Qualitative SERS performance as particle size decreases"
        if decrease_frame
        else "Qualitative SERS performance as particle size increases"
    )
    rationale = (
        "Smaller particle size is expected to improve qualitative SERS."
        if decrease_frame
        else (
            "The positive Trend direction is evaluated in the canonical "
            "particle-size increase frame."
        )
    )
    return DirectionAwareTrendHypothesisPortfolioDraft(
        hypotheses=[
            DirectionAwareTrendHypothesisProposalDraft(
                local_id="h1",
                title=(
                    "Canonical particle-size/SERS directional test"
                ),
                hypothesis_statement=hypothesis_statement,
                hypothesis_type="context_dependency",
                premise_statement_ids=[],
                gap_statement_ids=[],
                trend_references=[
                    TrendReferenceDraft(
                        view_id=local_view_id,
                        use_role="positive_empirical_support",
                    ),
                    TrendReferenceDraft(
                        view_id=gap_view_id,
                        use_role="replication_gap",
                    ),
                ],
                inferential_bridge=inferential_bridge,
                predicted_observations=[
                    DirectionAwarePredictedObservationDraft(
                        local_id="p1",
                        observable=observable,
                        expected_direction="increase",
                        rationale=rationale,
                        trend_direction_bindings=[
                            TrendDirectionBindingDraft(
                                view_id=local_view_id,
                                independent_change="increase",
                                dependent_change=dependent_change,
                            )
                        ],
                    )
                ],
                falsification_criteria=[
                    TrendAwareFalsificationCriterionDraft(
                        local_id="f1",
                        observable=(
                            "Qualitative SERS performance as "
                            "particle size increases"
                        ),
                        falsifying_outcome=(
                            "Qualitative SERS performance does not "
                            "increase under the comparable-context "
                            "particle-size increase."
                        ),
                    )
                ],
                assumptions=[
                    "Cross-paper replication is not established."
                ],
            )
        ],
        abstention_reason=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = TrendAwareHypothesisInput.model_validate_json(
        args.input.read_text(encoding="utf-8")
    )
    exposure = build_directional_trend_maker_exposure(source)
    local = [
        row
        for row in exposure.views
        if row.source_view.lane == "local_empirical_support"
    ]
    gaps = [
        row
        for row in exposure.views
        if row.source_view.lane == "replication_gap"
    ]
    if len(local) != 1 or len(gaps) != 1:
        raise RuntimeError(
            "v2 seen fixture expected one local support and one gap"
        )
    local_view = local[0]
    gap_view = gaps[0]
    if local_view.source_view.directions != ["positive"]:
        raise RuntimeError(
            "v2 seen local Trend direction changed from ['positive']; "
            "inspect upstream Trend semantics before continuing."
        )
    if local_view.expected_dependent_change != "increase":
        raise RuntimeError(
            "positive Trend did not map to dependent increase"
        )

    valid_draft = _draft(
        local_view.source_view.view_id,
        gap_view.source_view.view_id,
    )
    assembler = DirectionAwareTrendHypothesisPromptAssembler(
        max_hypotheses=1
    )
    prompt = assembler.build(source, exposure=exposure)
    for token in (
        "canonical_direction",
        "independent_change=increase",
        "expected_dependent_change=increase",
        "MANDATORY DIRECTION BINDING IF SELECTED",
        local_view.source_view.view_id,
    ):
        if token not in prompt.user_prompt:
            raise RuntimeError(
                f"5d.1 prompt omitted direction token: {token}"
            )

    backend = DeterministicBackend(valid_draft)
    runtime = DirectionAwareTrendHypothesisMakerAgentRuntime(
        backend,
        prompt_assembler=assembler,
        max_repairs=0,
    )
    outcome = runtime.run(source)
    if not outcome.accepted:
        raise RuntimeError(
            "valid direction-aware deterministic regression failed"
        )
    card = outcome.accepted_portfolio.hypotheses[0]  # type: ignore[union-attr]
    binding = (
        card.predicted_observations[0].
        trend_direction_bindings[0]
    )
    if binding.independent_change != "increase":
        raise RuntimeError("compiled independent frame drifted")
    if binding.dependent_change != "increase":
        raise RuntimeError("compiled dependent sign drifted")
    if binding.expected_dependent_change != "increase":
        raise RuntimeError("expected dependent sign drifted")

    wrong_binding = _draft(
        local_view.source_view.view_id,
        gap_view.source_view.view_id,
        dependent_change="decrease",
    )
    try:
        DirectionAwareTrendHypothesisCompiler().compile(
            source,
            wrong_binding,
        )
    except TrendHypothesisCompileError as exc:
        if not any(
            row.code == "TREND_DIRECTION_BINDING_MISMATCH"
            for row in exc.issues
        ):
            raise RuntimeError(
                "wrong sign failed for the wrong reason"
            ) from exc
    else:
        raise RuntimeError(
            "positive Trend accepted dependent_change='decrease'"
        )

    inverted_text = _draft(
        local_view.source_view.view_id,
        gap_view.source_view.view_id,
        dependent_change="increase",
        decrease_frame=True,
    )
    inverted_portfolio = (
        DirectionAwareTrendHypothesisCompiler().compile(
            source,
            inverted_text,
        )
    )
    inverted_validation = (
        DirectionAwareTrendHypothesisValidator().validate(
            source,
            inverted_portfolio,
        )
    )
    if inverted_validation.passes:
        raise RuntimeError(
            "lexically inverted decrease-frame hypothesis passed"
        )
    if not any(
        row.code == "NONCANONICAL_TREND_DIRECTION_FRAME"
        for row in inverted_validation.issues
    ):
        raise RuntimeError(
            "decrease-frame inversion was not diagnosed explicitly"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "directional_exposure.json",
        exposure,
    )
    _write_json(args.output_dir / "valid_draft.json", valid_draft)
    _write_json(
        args.output_dir / "valid_portfolio.json",
        outcome.accepted_portfolio,
    )
    _write_json(
        args.output_dir / "valid_validation.json",
        outcome.validation,
    )
    _write_json(
        args.output_dir / "inverted_validation.json",
        inverted_validation,
    )

    print("alpha4c.5c.1 + 5d.1 direction precision regression")
    print("Source Trend direction: positive")
    print("Canonical independent change: increase")
    print("Expected dependent change: increase")
    print("Valid canonical-frame draft: PASS")
    print("Wrong structured sign binding: BLOCKED")
    print("Decrease-frame textual inversion: BLOCKED")
    print("Replication-gap companion: PRESERVED")
    print("Trend causal authorization: False")
    print("Trend universal authorization: False")
    print("LLM calls: 0")
    print("v3 reserve consumed: False")
    print("Output:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
