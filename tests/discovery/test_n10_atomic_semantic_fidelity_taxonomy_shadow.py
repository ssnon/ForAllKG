from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    assess_atomic_semantic_fidelity,
    compile_atomic_semantic_fidelity_taxonomy_shadow,
)


def _hypothesis(*, statement: str, bridge: str = ""):
    return SimpleNamespace(
        hypothesis_statement=statement,
        inferential_bridge=bridge,
        assumptions=[],
        predicted_observations=[
            SimpleNamespace(
                observation_id="prediction:p1",
                observable="Outcome for comparable states",
                expected_direction="qualitative_change",
                rationale="A qualitative difference is expected.",
            )
        ],
        falsification_criteria=[
            SimpleNamespace(
                criterion_id="falsifier:f1",
                observable="Outcome for comparable states",
                falsifying_outcome="No qualitative difference is observed.",
            )
        ],
    )


def _claim(
    *,
    text: str,
    basis: str,
    endpoints: list[str],
    bridge: str = "",
    scope: list[str] | None = None,
    direction: list[str] | None = None,
    predicted: str = "",
    falsifier: str = "",
    role: str = "NOVELTY_BEARING",
):
    return NoveltyClaimDraft(
        local_id="c1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role=role,
        text=text,
        rationale="test rationale",
        search_concepts=["factor", "outcome"],
        search_queries=["factor outcome relation"],
        prior_art_identity_terms=["factor"],
        relation_nucleus_terms=["outcome", "relation"],
        required_bridge=bridge,
        predicted_observation=predicted,
        falsification_condition=falsifier,
        semantic_fidelity_binding=NoveltyClaimSemanticFidelityBindingDraft(
            proposition_basis=basis,
            relation_endpoint_anchors=endpoints,
            scope_qualifier_spans=scope or [],
            directional_qualifier_spans=direction or [],
            prediction_observation_id=("prediction:p1" if predicted else None),
            falsification_criterion_id=("falsifier:f1" if falsifier else None),
        ),
    )


def _taxonomy(hypothesis, claim, *, sanitized_bridge=None, sanitized_prediction=None, sanitized_falsifier=None):
    shadow = assess_atomic_semantic_fidelity(hypothesis, claim)
    return compile_atomic_semantic_fidelity_taxonomy_shadow(
        claim,
        semantic_fidelity_shadow=shadow,
        sanitized_required_bridge=(
            claim.required_bridge if sanitized_bridge is None else sanitized_bridge
        ),
        sanitized_predicted_observation=(
            claim.predicted_observation
            if sanitized_prediction is None
            else sanitized_prediction
        ),
        sanitized_falsification_condition=(
            claim.falsification_condition
            if sanitized_falsifier is None
            else sanitized_falsifier
        ),
    )


def test_taxonomy_safe_binding_is_pass_shadow_without_authority():
    source = "Factor M modulates outcome O."
    hypothesis = _hypothesis(statement=source)
    claim = _claim(
        text=source,
        basis=source,
        endpoints=["Factor M", "outcome O"],
        bridge=source,
        predicted="Outcome changes qualitatively.",
        falsifier="No qualitative difference is observed.",
    )

    taxonomy = _taxonomy(hypothesis, claim)

    assert taxonomy["production_authority"] is False
    assert taxonomy["binding_contract"]["status"] == "VALID"
    assert taxonomy["claim_fidelity"]["status"] == "NO_FLAG"
    assert taxonomy["specification_completeness"]["status"] == "NO_FLAG"
    assert taxonomy["specification_self_containment"]["status"] == "NO_FLAG"
    assert taxonomy["overall_review_status"] == "PASS_SHADOW"


def test_taxonomy_binding_invalid_makes_claim_fidelity_not_assessable():
    source = "Factor M modulates outcome O."
    hypothesis = _hypothesis(statement=source)
    claim = _claim(
        text=source,
        basis=source,
        endpoints=["Factor M", "descriptor D"],
        bridge=source,
        predicted="Outcome changes qualitatively.",
        falsifier="No qualitative difference is observed.",
    )

    taxonomy = _taxonomy(hypothesis, claim)

    assert taxonomy["binding_contract"]["status"] == "INVALID"
    assert (
        "atomic_claim_relation_endpoint_missing_from_claim"
        in taxonomy["binding_contract"]["findings"]
    )
    assert taxonomy["claim_fidelity"]["status"] == "NOT_ASSESSABLE_FROM_BINDING"
    assert taxonomy["overall_review_status"] == "NOT_ASSESSABLE_FROM_BINDING"


def test_taxonomy_novelty_bearing_empty_specification_is_review_not_unsafe():
    source = "Factor M modulates outcome O."
    hypothesis = _hypothesis(statement=source)
    claim = _claim(
        text=source,
        basis=source,
        endpoints=["Factor M", "outcome O"],
        bridge="",
        predicted="",
        falsifier="",
    )

    taxonomy = _taxonomy(
        hypothesis,
        claim,
        sanitized_bridge="",
        sanitized_prediction="",
        sanitized_falsifier="",
    )

    assert taxonomy["binding_contract"]["status"] == "VALID"
    assert taxonomy["specification_completeness"]["status"] == "REVIEW_REQUIRED"
    assert taxonomy["specification_completeness"]["findings"] == [
        "NOVELTY_BEARING_REQUIRED_BRIDGE_EMPTY",
        "NOVELTY_BEARING_PREDICTION_EMPTY",
        "NOVELTY_BEARING_FALSIFIER_EMPTY",
    ]
    assert taxonomy["overall_review_status"] == "REVIEW_REQUIRED"


def test_taxonomy_sanitized_anaphoric_bridge_is_direct_self_containment_review():
    source = "Factor M moderates the relationship between descriptor D and outcome O."
    hypothesis = _hypothesis(statement=source)
    claim = _claim(
        text=source,
        basis=source,
        endpoints=["Factor M", "descriptor D", "outcome O"],
        bridge="Factor M moderates this relationship.",
        predicted="Outcome changes qualitatively.",
        falsifier="No qualitative difference is observed.",
    )

    taxonomy = _taxonomy(
        hypothesis,
        claim,
        sanitized_bridge="Factor M moderates this relationship.",
    )

    assert taxonomy["specification_self_containment"]["status"] == "REVIEW_REQUIRED"
    assert taxonomy["specification_self_containment"]["findings"] == [
        "SANITIZED_REQUIRED_BRIDGE_RETAINS_ANAPHORIC_REFERENCE"
    ]
    assert taxonomy["overall_review_status"] == "REVIEW_REQUIRED"


def test_taxonomy_valid_binding_review_signal_never_becomes_fail():
    source = (
        "A more balanced distribution promotes outcome O, whereas a less "
        "balanced distribution is associated with another state."
    )
    hypothesis = _hypothesis(statement=source)
    claim = _claim(
        text="A more balanced distribution promotes outcome O.",
        basis="A more balanced distribution promotes outcome O",
        endpoints=["more balanced distribution", "outcome O"],
        bridge="A more balanced distribution promotes outcome O.",
        predicted="Outcome changes qualitatively.",
        falsifier="No qualitative difference is observed.",
    )

    taxonomy = _taxonomy(hypothesis, claim)

    assert taxonomy["binding_contract"]["status"] == "VALID"
    assert taxonomy["claim_fidelity"]["status"] == "REVIEW_REQUIRED"
    assert "FAIL" not in taxonomy["claim_fidelity"]["status"]
    assert taxonomy["overall_review_status"] == "REVIEW_REQUIRED"
