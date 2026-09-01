from pipeline_core.discovery.nonobviousness_shadow import (
    compile_shadow_claim,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
)


def claim(
    *,
    disposition="RESIDUAL",
    status="COMPONENTS_ONLY",
    bridge="",
    prediction="prediction",
    falsifier="falsifier",
):
    return NoveltyResidueClaim(
        hypothesis_id="hypothesis:test",
        claim_id="claim:test",
        claim_text="Test residual claim.",
        claim_kind="moderator_interaction",
        prior_art_status=status,
        disposition=disposition,
        is_residue=(disposition == "RESIDUAL"),
        distinguishing_terms=("factor",),
        prior_art_identity_terms=("factor",),
        relation_nucleus_terms=(
            "input",
            "outcome",
            "dependence",
        ),
        required_bridge=bridge,
        predicted_observation=prediction,
        falsification_condition=falsifier,
        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=(),
    )


def test_under_specified_residue_stops_before_closure():
    row = compile_shadow_claim(
        claim(
            bridge="",
        )
    )

    assert row["shadow_state"] == "NEEDS_REFINEMENT"
    assert row["closure_status"] == "NOT_RUN"
    assert row["structural_status"] == "NOT_RUN"
    assert row["adjudication_status"] == "NOT_RUN"


def test_well_specified_residue_only_becomes_closure_ready():
    row = compile_shadow_claim(
        claim(
            bridge="Explicit branch bridge.",
        )
    )

    assert row["shadow_state"] == "READY_FOR_CLOSURE"
    assert (
        row["closure_status"]
        == "PENDING_TARGETED_CLOSURE"
    )
    assert row["structural_status"] == "NOT_RUN"
    assert row["adjudication_status"] == "NOT_RUN"


def test_saturated_claim_never_enters_closure():
    row = compile_shadow_claim(
        claim(
            disposition="SATURATED",
            status="DIRECT_PRIOR_ART",
        )
    )

    assert row["shadow_state"] == "SATURATED_PRIOR_ART"
    assert row["next_action"] == "NONE"
    assert row["closure_status"] == "NOT_RUN"


def test_partial_claim_remains_unresolved():
    row = compile_shadow_claim(
        claim(
            disposition="UNRESOLVED_PARTIAL",
            status="PARTIAL_PRIOR_ART",
        )
    )

    assert row["shadow_state"] == "UNRESOLVED_PARTIAL"
    assert (
        row["next_action"]
        == "RESOLVE_PARTIAL_PRIOR_ART"
    )
