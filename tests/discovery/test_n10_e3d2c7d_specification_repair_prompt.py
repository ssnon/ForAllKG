import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyCard,
    HypothesisSearchCoverage,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
)
from pipeline_core.discovery.n10_specification_repair_context import (
    N10SpecificationRepairClaimDiagnostic,
    N10SpecificationRepairContext,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGap,
)
from pipeline_core.discovery.novelty_refinement_prompt import (
    NoveltyRefinementPromptAssembler,
)


H = "hypothesis:synthetic-prompt"


def _original(
    hypothesis_id: str = H,
) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        domain_profile_id="synthetic",
        source_context_id="context:1",
        source_context_sha256="context-sha",
        source_report_id="report:1",
        source_report_sha256="report-sha",
        title="Synthetic hypothesis",
        hypothesis_statement=(
            "Factor M moderates the relation between X and Y."
        ),
        hypothesis_type="descriptor_mediation",
        premise_statement_ids=[
            "stmt:p1",
        ],
        gap_statement_ids=[
            "stmt:g1",
        ],
        inferential_bridge=(
            "The grounded evidence motivates "
            "testing M as a moderator."
        ),
        predicted_observations=[],
        falsification_criteria=[],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=1,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def _gap(
    hypothesis_id: str = H,
) -> NoveltyGap:
    return NoveltyGap(
        gap_id="gap:1",
        hypothesis_id=hypothesis_id,
        source_external_status=(
            "INSUFFICIENT_SEARCH_EVIDENCE"
        ),
        action="keep",
        differentiator=(
            "Whether M moderates X-to-Y."
        ),
        already_known_boundary=[
            "X and Y are individually known."
        ],
        unresolved_boundary=[
            "Moderation remains unresolved."
        ],
    )


def _targeted(
    hypothesis_id: str = H,
) -> ExternalNoveltyCard:
    return ExternalNoveltyCard(
        hypothesis_id=hypothesis_id,
        title="Synthetic hypothesis",
        status="INSUFFICIENT_SEARCH_EVIDENCE",
        claim_reviews=[],
        coverage=HypothesisSearchCoverage(
            hypothesis_id=hypothesis_id,
            query_count=1,
            successful_query_count=1,
            provider_success_count=1,
            unique_work_count=1,
            abstract_work_count=1,
            core_claim_count=1,
            core_claims_with_minimum_abstract_coverage=0,
            sufficient_for_absence_based_novelty=False,
        ),
        interpretation=(
            "Search-bounded synthetic assessment."
        ),
    )


def _repair_context(
    hypothesis_id: str = H,
) -> N10SpecificationRepairContext:
    return N10SpecificationRepairContext(
        context_id="repair-context:1",
        context_sha256="repair-sha",
        source_hypothesis_id=hypothesis_id,
        source_query_plan_id="plan:1",
        source_query_plan_sha256="plan-sha",
        source_external_report_id="external:1",
        source_external_report_sha256="external-sha",
        source_intake_sha256="intake-sha",
        source_n10_gate_sha256="gate-sha",
        claim_diagnostics=[
            N10SpecificationRepairClaimDiagnostic(
                claim_id="claim:novelty",
                claim_text=(
                    "M moderates the relationship between X and Y."
                ),
                missing_fields=[
                    "required_bridge",
                    "falsification_condition",
                ],
                reason_codes=[
                    "atomic_residue_under_specified",
                    "missing_required_bridge",
                    "missing_falsification_condition",
                ],
            )
        ],
    )


def _context() -> HypothesisContext:
    return HypothesisContext.model_construct()


def test_no_diagnostic_context_preserves_legacy_prompt_version():
    prompt = NoveltyRefinementPromptAssembler(
        original=_original(),
        gap=_gap(),
        targeted_card=_targeted(),
    ).build(
        _context()
    )

    assert (
        prompt.prompt_version
        == "novelty-refinement-prompt-v2.8.1-a6-relgap-boundary"
    )

    assert (
        "SPECIFICATION REPAIR DIAGNOSIS"
        not in prompt.user_prompt
    )


def test_diagnostic_context_uses_distinct_prompt_version():
    prompt = NoveltyRefinementPromptAssembler(
        original=_original(),
        gap=_gap(),
        targeted_card=_targeted(),
        specification_repair_context=
            _repair_context(),
    ).build(
        _context()
    )

    assert (
        prompt.prompt_version
        == (
            "novelty-refinement-prompt-"
            "v2.8.2-a6-n10-specification-diagnostic"
        )
    )


def test_diagnostic_prompt_contains_exact_claim_local_targets():
    prompt = NoveltyRefinementPromptAssembler(
        original=_original(),
        gap=_gap(),
        targeted_card=_targeted(),
        specification_repair_context=
            _repair_context(),
    ).build(
        _context()
    )

    text = prompt.user_prompt

    assert (
        "SPECIFICATION REPAIR DIAGNOSIS"
        in text
    )

    assert (
        "claim_id: claim:novelty"
        in text
    )

    assert (
        "M moderates the relationship between X and Y."
        in text
    )

    assert (
        '"required_bridge"'
        in text
    )

    assert (
        '"falsification_condition"'
        in text
    )

    assert (
        '"missing_required_bridge"'
        in text
    )


def test_diagnosis_is_explicitly_non_evidentiary():
    prompt = NoveltyRefinementPromptAssembler(
        original=_original(),
        gap=_gap(),
        targeted_card=_targeted(),
        specification_repair_context=
            _repair_context(),
    ).build(
        _context()
    )

    text = prompt.user_prompt

    assert (
        "It is NOT scientific evidence"
        in text
    )

    assert (
        "NOT a positive premise"
        in text
    )

    assert (
        "NOT evidence of novelty"
        in text
    )

    assert (
        "Do not infer scientific content "
        "from a missing-field label or reason code."
        in text
    )

    assert (
        "If a listed field cannot be repaired "
        "without unsupported scientific content, abstain."
        in text
    )


def test_diagnosis_does_not_replace_existing_epistemic_rules():
    prompt = NoveltyRefinementPromptAssembler(
        original=_original(),
        gap=_gap(),
        targeted_card=_targeted(),
        specification_repair_context=
            _repair_context(),
    ).build(
        _context()
    )

    system = prompt.system_prompt

    assert (
        "Grounded evidence is ONLY the supplied "
        "HypothesisContext premise statements."
        in system
    )

    assert (
        "External prior-art summaries are NOT "
        "positive scientific premises."
        in system
    )

    assert (
        "Never claim novelty, priority, first report, "
        "or literature-wide absence."
        in system
    )


def test_context_identity_mismatch_fails_closed():
    with pytest.raises(
        ValueError,
        match="hypothesis identity mismatch",
    ):
        NoveltyRefinementPromptAssembler(
            original=_original(
                "hypothesis:original"
            ),
            gap=_gap(
                "hypothesis:original"
            ),
            targeted_card=_targeted(
                "hypothesis:original"
            ),
            specification_repair_context=
                _repair_context(
                    "hypothesis:other"
                ),
        )


def test_diagnostic_context_changes_prompt_sha():
    legacy = NoveltyRefinementPromptAssembler(
        original=_original(),
        gap=_gap(),
        targeted_card=_targeted(),
    ).build(
        _context()
    )

    diagnostic = NoveltyRefinementPromptAssembler(
        original=_original(),
        gap=_gap(),
        targeted_card=_targeted(),
        specification_repair_context=
            _repair_context(),
    ).build(
        _context()
    )

    assert (
        diagnostic.prompt_sha256
        != legacy.prompt_sha256
    )


def test_prompt_does_not_describe_diagnosis_as_authority():
    prompt = NoveltyRefinementPromptAssembler(
        original=_original(),
        gap=_gap(),
        targeted_card=_targeted(),
        specification_repair_context=
            _repair_context(),
    ).build(
        _context()
    )

    diagnosis = (
        prompt.user_prompt.split(
            "SPECIFICATION REPAIR DIAGNOSIS",
            1,
        )[1]
    )

    forbidden = [
        "diagnosis proves novelty",
        "diagnosis establishes novelty",
        "diagnosis is evidence",
        "use the diagnosis as evidence",
        "treat the diagnosis as a premise",
    ]

    lowered = diagnosis.lower()

    for phrase in forbidden:
        assert phrase not in lowered
