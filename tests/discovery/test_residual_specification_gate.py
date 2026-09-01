from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDraft,
)
from pipeline_core.discovery.external_novelty_llm import (
    InstructorOpenAICompatibleExternalNoveltyBackend,
    _DECOMPOSE_SYSTEM,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
    assess_residual_specification,
)


def _residue(
    *,
    disposition="RESIDUAL",
    required_bridge="",
    predicted_observation="",
    falsification_condition="",
):
    return NoveltyResidueClaim(
        hypothesis_id="hypothesis:test",
        claim_id="claim:test",
        claim_text=(
            "Laser power moderates the dependence of "
            "SERS enhancement on interparticle spacing."
        ),
        claim_kind="moderator_interaction",
        prior_art_status=(
            "COMPONENTS_ONLY"
            if disposition == "RESIDUAL"
            else "DIRECT_PRIOR_ART"
        ),
        disposition=disposition,
        is_residue=disposition == "RESIDUAL",
        distinguishing_terms=("laser power",),
        prior_art_identity_terms=("laser power",),
        relation_nucleus_terms=(
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ),
        required_bridge=required_bridge,
        predicted_observation=predicted_observation,
        falsification_condition=falsification_condition,
        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=(),
    )


def test_atomic_specification_fields_default_empty():
    row = NoveltyClaimDraft(
        local_id="power",
        kind="moderator_interaction",
        text="Laser power moderates spacing dependence.",
        rationale="Atomic power branch.",
    )

    assert row.required_bridge == ""
    assert row.predicted_observation == ""
    assert row.falsification_condition == ""


def test_under_specified_residue_fails_closed():
    result = assess_residual_specification(
        _residue(
            predicted_observation=(
                "Spacing dependence changes with laser power."
            ),
            falsification_condition=(
                "No change across compared laser powers."
            ),
        )
    )

    assert result.status == "NEEDS_REFINEMENT"
    assert result.missing_fields == (
        "required_bridge",
    )
    assert (
        "atomic_residue_under_specified"
        in result.reason_codes
    )


def test_fully_specified_residue_can_reach_closure():
    result = assess_residual_specification(
        _residue(
            required_bridge=(
                "A branch-specific supplied scientific bridge."
            ),
            predicted_observation=(
                "A branch-specific distinguishing observation."
            ),
            falsification_condition=(
                "A branch-specific falsifying observation."
            ),
        )
    )

    assert result.status == "READY_FOR_CLOSURE"
    assert result.missing_fields == ()


def test_saturated_claim_does_not_enter_gate():
    result = assess_residual_specification(
        _residue(
            disposition="SATURATED",
        )
    )

    assert result.status == "NOT_APPLICABLE"


def test_decomposition_prompt_requires_fail_closed_specification():
    assert (
        "ATOMIC SPECIFICATION PROVENANCE CONTRACT"
        in _DECOMPOSE_SYSTEM
    )
    assert (
        "Empty specification fields are valid"
        in _DECOMPOSE_SYSTEM
    )
    assert (
        "do not invent power-dependent heating"
        in _DECOMPOSE_SYSTEM
    )


def test_decomposition_backend_supplies_falsifiers():
    import inspect

    source = inspect.getsource(
        InstructorOpenAICompatibleExternalNoveltyBackend.decompose
    )

    assert "hypothesis.falsification_criteria" in source
    assert '"falsification_criteria:"' in source
