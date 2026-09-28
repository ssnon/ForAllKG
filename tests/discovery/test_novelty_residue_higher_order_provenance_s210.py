from types import SimpleNamespace

from pipeline_core.discovery.novelty_residue import (
    extract_novelty_residue,
)


def test_extract_preserves_higher_order_provenance():
    claim = SimpleNamespace(
        claim_id="claim:composite",
        hypothesis_id="hyp:1",
        kind="composite",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="A moderates B through C.",
        distinguishing_terms=["A", "B", "C"],
        prior_art_identity_terms=["A", "B", "C"],
        relation_nucleus_terms=["moderates"],
        higher_order_relation_basis=[
            "A moderates B through C."
        ],
        higher_order_component_claim_ids=[
            "claim:base",
            "claim:mechanism",
        ],
        required_bridge="A may moderate B through C.",
        predicted_observation="B changes across A.",
        falsification_condition="B is unchanged across A.",
        inference_provenance=None,
        specification_sanitization_reason_codes=[],
        scientific_structure=SimpleNamespace(),
        scientific_structure_reason_codes=[],
    )

    group = SimpleNamespace(claims=[claim])
    plan = SimpleNamespace(claims=[group])

    review = SimpleNamespace(
        claim_id="claim:composite",
        claim_text=claim.text,
        status="COMPONENTS_ONLY",
        importance="core",
        matches=[],
    )

    card = SimpleNamespace(
        hypothesis_id="hyp:1",
        status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        claim_reviews=[review],
    )

    report = SimpleNamespace(cards=[card])

    residues = extract_novelty_residue(
        plan,
        report,
    )

    assert len(residues) == 1
    assert len(residues[0].claims) == 1

    row = residues[0].claims[0]

    assert row.claim_kind == "composite"
    assert row.novelty_selection_role == "NOVELTY_BEARING"
    assert row.higher_order_relation_basis == (
        "A moderates B through C.",
    )
    assert row.higher_order_component_claim_ids == (
        "claim:base",
        "claim:mechanism",
    )
