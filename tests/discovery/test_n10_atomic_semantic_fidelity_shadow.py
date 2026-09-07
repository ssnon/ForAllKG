from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    assess_atomic_semantic_fidelity,
)


def _hypothesis(
    *,
    statement: str,
    bridge: str = "",
    prediction_direction: str = "qualitative_change",
):
    return SimpleNamespace(
        hypothesis_statement=statement,
        inferential_bridge=bridge,
        assumptions=[],
        predicted_observations=[
            SimpleNamespace(
                observation_id="prediction:p1",
                observable=(
                    "Outcome for comparable states with different factor values"
                ),
                expected_direction=prediction_direction,
                rationale="A qualitative difference is expected.",
            )
        ],
        falsification_criteria=[
            SimpleNamespace(
                criterion_id="falsifier:f1",
                observable=(
                    "Outcome for comparable states with different factor values"
                ),
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
):
    return NoveltyClaimDraft(
        local_id="c1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
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
            prediction_observation_id=(
                "prediction:p1" if predicted else None
            ),
            falsification_criterion_id=(
                "falsifier:f1" if falsifier else None
            ),
        ),
    )


def test_shadow_flags_ordered_language_absent_from_source_basis():
    source = (
        "At comparable coupling, a more balanced distribution promotes "
        "adsorption-desorption compatibility."
    )
    hypothesis = _hypothesis(statement=source)

    claim = _claim(
        text=(
            "At comparable coupling, a more balanced distribution produces "
            "greater compatibility and higher activity."
        ),
        basis=source,
        endpoints=["more balanced distribution", "compatibility"],
        scope=["At comparable coupling"],
        direction=["higher activity"],
        predicted=(
            "At comparable coupling, a more balanced distribution produces "
            "higher activity."
        ),
        falsifier="At comparable coupling, no qualitative difference is observed.",
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)

    assert "atomic_claim_direction_qualifier_not_source_backed" in review["reason_codes"]
    assert "atomic_claim_ordered_language_not_in_source_basis" in review["reason_codes"]


def test_shadow_flags_source_condition_and_comparative_scope_loss():
    source = (
        "Under the condition of comparable coupling, a more balanced "
        "distribution promotes adsorption-desorption compatibility."
    )
    hypothesis = _hypothesis(statement=source)

    claim = _claim(
        text="The distribution promotes adsorption-desorption compatibility.",
        basis=source,
        endpoints=["distribution", "adsorption-desorption compatibility"],
        scope=["comparable coupling", "more balanced"],
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)

    assert "atomic_claim_scope_qualifier_not_preserved" in review["reason_codes"]
    assert "atomic_claim_comparative_scope_may_be_broadened" in review["reason_codes"]
    assert "atomic_claim_condition_scope_may_be_dropped" in review["reason_codes"]


def test_shadow_flags_anaphoric_bridge_with_missing_relation_endpoint():
    source = (
        "Factor M moderates the relationship between descriptor D and outcome O."
    )
    hypothesis = _hypothesis(
        statement=source,
        bridge="Factor M moderates this activity relationship.",
    )

    claim = _claim(
        text=source,
        basis=source,
        endpoints=["Factor M", "descriptor D", "outcome O"],
        bridge="Factor M moderates this activity relationship.",
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)

    assert "required_bridge_anaphoric_relation_reference" in review["reason_codes"]
    assert (
        "required_bridge_relation_endpoint_alignment_unresolved"
        in review["reason_codes"]
    )


def test_shadow_accepts_complete_cross_domain_source_binding_without_authority():
    source = (
        "When Reporter R output is comparable across localization states, "
        "subcellular scaffold localization conditions the association between "
        "activation-deactivation compatibility and cell survival."
    )
    hypothesis = _hypothesis(statement=source)

    claim = _claim(
        text=source,
        basis=source,
        endpoints=[
            "subcellular scaffold localization",
            "activation-deactivation compatibility",
            "cell survival",
        ],
        scope=["When Reporter R output is comparable across localization states"],
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)

    assert review["proposition_basis_source_paths"] == ["hypothesis_statement"]
    assert review["production_authority"] is False
    assert not any(
        code.startswith("atomic_claim_")
        for code in review["reason_codes"]
    )


def test_shadow_expands_inner_basis_to_source_sentence_for_scope_loss():
    source = (
        "Under the condition of comparable coupling, a more balanced "
        "distribution promotes adsorption-desorption compatibility."
    )
    hypothesis = _hypothesis(statement=source)

    # Adversarial but exact inner span: the basis itself omits both the
    # leading condition and the comparative qualifier.
    claim = _claim(
        text="The distribution promotes adsorption-desorption compatibility.",
        basis="distribution promotes adsorption-desorption compatibility.",
        endpoints=["distribution", "adsorption-desorption compatibility"],
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)

    assert (
        "atomic_claim_context_condition_scope_may_be_dropped"
        in review["reason_codes"]
    )
    assert (
        "atomic_claim_context_comparative_scope_may_be_broadened"
        in review["reason_codes"]
    )
    assert review["proposition_basis_source_contexts"] == [
        {
            "source_path": "hypothesis_statement",
            "sentence": source,
        }
    ]


def test_shadow_flags_anaphora_even_if_reported_anchors_are_incomplete():
    source = (
        "Factor M moderates the relationship between descriptor D and outcome O."
    )
    hypothesis = _hypothesis(
        statement=source,
        bridge="Factor M moderates this relationship.",
    )

    # Only one endpoint is reported. Anaphora detection must not depend on
    # the model faithfully enumerating every endpoint anchor.
    claim = _claim(
        text=source,
        basis=source,
        endpoints=["Factor M"],
        bridge="Factor M moderates this relationship.",
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)

    assert (
        "required_bridge_anaphoric_relation_reference"
        in review["reason_codes"]
    )


def test_shadow_source_context_does_not_absorb_following_sentence():
    hypothesis = _hypothesis(
        statement="Unrelated hypothesis statement.",
        bridge=(
            "Factor M modulates outcome O. "
            "When condition C holds, a second proposition applies."
        ),
    )
    claim = _claim(
        text="Factor M modulates outcome O.",
        bridge="Factor M modulates outcome O.",
        basis="Factor M modulates outcome O.",
        endpoints=["Factor M", "outcome O"],
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)

    assert review["proposition_basis_source_contexts"] == [
        {
            "source_path": "inferential_bridge",
            "sentence": "Factor M modulates outcome O.",
        }
    ]
    assert "atomic_claim_context_condition_scope_may_be_dropped" not in review["reason_codes"]
    assert "required_bridge_condition_scope_may_be_dropped" not in review["reason_codes"]


def test_shadow_colon_separates_later_conditional_proposition():
    hypothesis = _hypothesis(
        statement=(
            "Factor M moderates the D-O relationship: "
            "under condition C, factor N changes outcome O."
        )
    )
    claim = _claim(
        text="Factor M moderates the D-O relationship",
        bridge="Factor M moderates the D-O relationship",
        basis="Factor M moderates the D-O relationship",
        endpoints=["Factor M", "D-O relationship"],
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)

    assert review["proposition_basis_source_contexts"] == [
        {
            "source_path": "hypothesis_statement",
            "sentence": "Factor M moderates the D-O relationship:",
        }
    ]
    assert "atomic_claim_context_condition_scope_may_be_dropped" not in review["reason_codes"]
    assert "required_bridge_condition_scope_may_be_dropped" not in review["reason_codes"]
