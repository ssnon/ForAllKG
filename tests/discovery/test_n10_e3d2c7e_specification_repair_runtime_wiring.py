from pathlib import Path

import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.n10_alpha6_resolution_policy import (
    Alpha6N10ResolutionDirective,
    NO_N10_ALPHA6_OVERRIDE,
)
from pipeline_core.discovery.novelty_refinement_runtime import (
    _build_alpha6_specification_repair_context,
)


H = "hypothesis:runtime-repair"


def _plan() -> LiteratureQueryPlan:
    return LiteratureQueryPlan(
        plan_id="plan:runtime-repair",
        plan_sha256="plan-sha",
        source_portfolio_id="portfolio:1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id=H,
                title="Synthetic",
                claims=[
                    NoveltyClaim(
                        claim_id="claim:novel",
                        hypothesis_id=H,
                        claim_rank=1,
                        kind="moderator_interaction",
                        importance="core",
                        novelty_selection_role=(
                            "NOVELTY_BEARING"
                        ),
                        text=(
                            "M moderates relation X to Y."
                        ),
                        rationale="synthetic",
                    ),
                    NoveltyClaim(
                        claim_id="claim:test",
                        hypothesis_id=H,
                        claim_rank=2,
                        kind="distinctive_prediction",
                        importance="core",
                        novelty_selection_role=(
                            "TESTING_PREDICTION"
                        ),
                        text=(
                            "Y differs across M."
                        ),
                        rationale="synthetic",
                    ),
                ],
            )
        ],
    )


def _decision(
    *,
    claim_id: str,
    text: str,
    missing: list[str],
) -> dict:
    return {
        "claim": {
            "hypothesis_id":
                H,
            "claim_id":
                claim_id,
            "claim_text":
                text,
        },
        "specification": {
            "status":
                "NEEDS_REFINEMENT",
            "missing_fields":
                list(missing),
            "reason_codes": [
                "atomic_residue_under_specified",
                *[
                    "missing_" + field
                    for field in missing
                ],
            ],
        },
        "shadow_state":
            "NEEDS_REFINEMENT",
        "next_action":
            "REFINE_HYPOTHESIS_SPECIFICATION",
    }


def _intake() -> dict:
    return {
        "schema_version":
            "nonobviousness-shadow-v1",
        "shadow_only":
            True,
        "scientific_selection_changed":
            False,
        "source_query_plan_id":
            "plan:runtime-repair",
        "source_query_plan_sha256":
            "plan-sha",
        "source_external_report_id":
            "external:1",
        "source_external_report_sha256":
            "external-sha",
        "hypotheses": [
            {
                "hypothesis_id":
                    H,
                "claims": [
                    _decision(
                        claim_id="claim:novel",
                        text=(
                            "M moderates relation X to Y."
                        ),
                        missing=[
                            "required_bridge",
                        ],
                    ),
                    _decision(
                        claim_id="claim:test",
                        text=(
                            "Y differs across M."
                        ),
                        missing=[
                            "falsification_condition",
                        ],
                    ),
                ],
            }
        ],
    }


def _post_generation_gate() -> dict:
    return {
        "schema_version":
            "scientific-novelty-fallback-gate-v2",
        "production_authority":
            True,
        "authority_scope":
            "alpha6_post_generation_candidate",
        "conditional_is_positive":
            False,
        "absence_is_novelty":
            False,
        "candidate_semantics_preserved":
            True,
        "gates": [
            {
                "hypothesis_id":
                    H,
                "selection_class":
                    "CONDITIONAL",
                "base_aggregation_action": (
                    "REFINE_NOVELTY_BEARING_SPECIFICATION"
                ),
                "positive_nonobviousness_authority":
                    False,
                "fallback_allowed":
                    False,
            }
        ],
    }


def _repair_directive():
    return Alpha6N10ResolutionDirective(
        force_bounded_refinement=True,
        use_source_external_without_targeted_search=True,
        bypass_resolved_candidate_external_exit=True,
        reason_code=(
            "n10_refine_novelty_bearing_specification"
        ),
    )


def test_historical_no_diagnostic_inputs_remain_backward_compatible():
    result = (
        _build_alpha6_specification_repair_context(
            source_hypothesis_id=H,
            external_query_plan=_plan(),
            n10_resolution_directive=
                _repair_directive(),
            specification_repair_intake_shadow=None,
            specification_repair_post_generation_gate=None,
        )
    )

    assert result is None


def test_paired_input_requirement_fails_closed():
    with pytest.raises(
        RuntimeError,
        match="must be supplied as a pair",
    ):
        _build_alpha6_specification_repair_context(
            source_hypothesis_id=H,
            external_query_plan=_plan(),
            n10_resolution_directive=
                _repair_directive(),
            specification_repair_intake_shadow=
                _intake(),
            specification_repair_post_generation_gate=
                None,
        )


def test_non_repair_directive_does_not_consume_diagnosis():
    result = (
        _build_alpha6_specification_repair_context(
            source_hypothesis_id=H,
            external_query_plan=_plan(),
            n10_resolution_directive=
                NO_N10_ALPHA6_OVERRIDE,
            specification_repair_intake_shadow=
                _intake(),
            specification_repair_post_generation_gate=
                _post_generation_gate(),
        )
    )

    assert result is None


def test_exact_repair_directive_builds_non_authoritative_context():
    result = (
        _build_alpha6_specification_repair_context(
            source_hypothesis_id=H,
            external_query_plan=_plan(),
            n10_resolution_directive=
                _repair_directive(),
            specification_repair_intake_shadow=
                _intake(),
            specification_repair_post_generation_gate=
                _post_generation_gate(),
        )
    )

    assert result is not None

    assert (
        result.source_hypothesis_id
        == H
    )

    assert result.diagnostic_only is True
    assert result.production_authority is False

    assert (
        result.scientific_evidence_authority
        is False
    )

    assert [
        row.claim_id
        for row in result.claim_diagnostics
    ] == [
        "claim:novel"
    ]

    assert (
        result.claim_diagnostics[0]
        .missing_fields
        == ["required_bridge"]
    )


def test_future_force_directive_cannot_reuse_current_diagnosis_contract():
    future = Alpha6N10ResolutionDirective(
        force_bounded_refinement=True,
        use_source_external_without_targeted_search=False,
        bypass_resolved_candidate_external_exit=True,
        reason_code="some_future_repair",
    )

    with pytest.raises(
        RuntimeError,
        match="exact frozen specification-repair directive",
    ):
        _build_alpha6_specification_repair_context(
            source_hypothesis_id=H,
            external_query_plan=_plan(),
            n10_resolution_directive=future,
            specification_repair_intake_shadow=
                _intake(),
            specification_repair_post_generation_gate=
                _post_generation_gate(),
        )


def test_runtime_passes_context_only_at_same_premise_prompt_seam():
    text = Path(
        "pipeline_core/discovery/"
        "novelty_refinement_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "_build_alpha6_specification_repair_context("
        in text
    )

    assert (
        "specification_repair_context=("
        in text
    )

    assert (
        "NoveltyRefinementPromptAssembler("
        in text
    )


def test_cli_uses_separate_diagnostic_provenance_inputs():
    text = Path(
        "scripts/discovery/"
        "run_novelty_refinement.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "--n10-specification-repair-intake-shadow"
        in text
    )

    assert (
        "--n10-specification-repair-post-generation-gate"
        in text
    )

    assert (
        "specification_repair_intake_shadow=("
        in text
    )

    assert (
        "specification_repair_post_generation_gate=("
        in text
    )


def test_runtime_exposes_used_context_for_observability():
    text = Path(
        "pipeline_core/discovery/"
        "novelty_refinement_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "specification_repair_contexts:"
        in text
    )

    assert (
        "specification_repair_contexts=tuple("
        in text
    )
