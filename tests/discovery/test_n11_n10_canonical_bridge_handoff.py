from __future__ import annotations

import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
)
from pipeline_core.discovery.nonobviousness_shadow import (
    _json_safe,
    reconcile_intake_required_bridge,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
)


BRIDGE = (
    "Interparticle spacing is related to the plasmonic response. "
    "The new inference is that interparticle spacing could alter "
    "the relative mechanistic contribution of electromagnetic "
    "and chemical enhancement."
)


def make_claim(
    *,
    required_bridge: str = "",
) -> NoveltyResidueClaim:
    return NoveltyResidueClaim(
        hypothesis_id="hypothesis:test",
        claim_id="claim:test",
        claim_text=(
            "Interparticle spacing changes the relative "
            "electromagnetic and chemical contribution."
        ),
        claim_kind="moderator_interaction",
        prior_art_status="COMPONENTS_ONLY",
        disposition="RESIDUAL",
        is_residue=True,
        distinguishing_terms=(
            "relative contribution",
        ),
        prior_art_identity_terms=(
            "interparticle spacing",
        ),
        relation_nucleus_terms=(
            "relative electromagnetic chemical contribution",
        ),
        required_bridge=required_bridge,
        predicted_observation=(
            "Relative signatures change with spacing."
        ),
        falsification_condition=(
            "Relative signatures do not change with spacing."
        ),
        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=("work:1",),
        scientific_structure=(
            NoveltyClaimScientificStructure()
        ),
        scientific_structure_reason_codes=(),
    )


def make_hypothesis(
    bridge: str = BRIDGE,
) -> HypothesisCard:
    return HypothesisCard.model_construct(
        hypothesis_id="hypothesis:test",
        inferential_bridge=bridge,
        assumptions=[],
    )


def intake_dict(
    claim: NoveltyResidueClaim,
) -> dict[str, object]:
    value = _json_safe(claim)
    assert isinstance(value, dict)
    return dict(value)


def test_recovers_exact_canonical_bridge() -> None:
    claim = make_claim()
    incoming = intake_dict(claim)
    incoming["required_bridge"] = BRIDGE

    result = reconcile_intake_required_bridge(
        claim,
        intake_claim=incoming,
        specification_provenance={
            "required_bridge":
                "CANONICAL_HYPOTHESIS_INFERENTIAL_BRIDGE",
        },
        hypothesis=make_hypothesis(),
    )

    assert result.required_bridge == BRIDGE
    assert result.claim_text == claim.claim_text
    assert (
        result.predicted_observation
        == claim.predicted_observation
    )


def test_rejects_noncanonical_provenance() -> None:
    claim = make_claim()
    incoming = intake_dict(claim)
    incoming["required_bridge"] = BRIDGE

    with pytest.raises(
        ValueError,
        match="lacks canonical",
    ):
        reconcile_intake_required_bridge(
            claim,
            intake_claim=incoming,
            specification_provenance={
                "required_bridge": "UNRESOLVED",
            },
            hypothesis=make_hypothesis(),
        )


def test_rejects_drift_outside_bridge() -> None:
    claim = make_claim()
    incoming = intake_dict(claim)
    incoming["required_bridge"] = BRIDGE
    incoming["claim_text"] = "mutated claim"

    with pytest.raises(
        ValueError,
        match="drift outside required_bridge",
    ):
        reconcile_intake_required_bridge(
            claim,
            intake_claim=incoming,
            specification_provenance={
                "required_bridge":
                    "CANONICAL_HYPOTHESIS_INFERENTIAL_BRIDGE",
            },
            hypothesis=make_hypothesis(),
        )


def test_rejects_bridge_not_equal_to_canonical_source() -> None:
    claim = make_claim()
    incoming = intake_dict(claim)
    incoming["required_bridge"] = (
        "Interparticle spacing proves another mechanism."
    )

    with pytest.raises(
        ValueError,
        match="does not exactly match",
    ):
        reconcile_intake_required_bridge(
            claim,
            intake_claim=incoming,
            specification_provenance={
                "required_bridge":
                    "CANONICAL_HYPOTHESIS_INFERENTIAL_BRIDGE",
            },
            hypothesis=make_hypothesis(),
        )


def test_existing_query_plan_bridge_cannot_be_replaced() -> None:
    claim = make_claim(
        required_bridge=BRIDGE
    )
    incoming = intake_dict(claim)
    incoming["required_bridge"] = (
        "Interparticle spacing changes something else."
    )

    with pytest.raises(
        ValueError,
        match="attempted to replace",
    ):
        reconcile_intake_required_bridge(
            claim,
            intake_claim=incoming,
            specification_provenance={
                "required_bridge":
                    "CANONICAL_HYPOTHESIS_INFERENTIAL_BRIDGE",
            },
            hypothesis=make_hypothesis(),
        )
