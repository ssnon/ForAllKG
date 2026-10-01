from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import NoveltyClaim
from pipeline_core.discovery.lower_order_prior_art_saturation import (
    _compact_query,
    _query_texts_for_claim,
)
from pipeline_core.discovery.external_novelty_llm import (
    _EXACT_VERIFICATION_REVIEW_SYSTEM,
)


def _claim() -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Changing the number of Au/Al2O3 bilayers changes Raman signal intensity.",
        rationale="test",
        search_concepts=[
            "Au Al2O3 bilayer number",
            "SERS intensity",
            "multilayer plasmonic structure",
        ],
        search_queries=[
            "number of Au/Al2O3 bilayers Raman signal intensity dependence",
            "Au Al2O3 bilayer number Raman signal intensity multilayer plasmonic structure",
        ],
        prior_art_identity_terms=[
            "Au/Al2O3 bilayer number",
            "multilayer plasmonic structure",
        ],
        relation_nucleus_terms=[
            "Raman signal intensity dependence",
        ],
        diagnostic_structural_terms=[
            "Au/Al2O3 bilayers",
        ],
        diagnostic_relation_terms=[
            "Raman intensity",
        ],
    )


def test_compaction_keeps_material_identity_and_drops_query_wrappers():
    q = _compact_query(
        "number of Au/Al2O3 bilayers Raman signal intensity dependence"
    )
    assert "Au" in q
    assert "Al2O3" in q
    assert "bilayers" in q
    assert "Raman" in q
    assert "intensity" in q
    assert "number" not in q.lower()
    assert "dependence" not in q.lower()


def test_query_ladder_contains_concise_identity_preserving_variant():
    queries = _query_texts_for_claim(_claim())
    assert queries
    assert max(len(q.split()) for q in queries) <= 10
    assert any(
        {"au", "al2o3", "bilayers"}.issubset(
            {tok.lower() for tok in q.split()}
        )
        for q in queries
    )
    assert any("Raman" in q and "intensity" in q for q in queries)


def test_exact_verification_reviewer_requires_structural_identity():
    assert "structural identity is mandatory" in _EXACT_VERIFICATION_REVIEW_SYSTEM
    assert "semiconductor bilayer" in _EXACT_VERIFICATION_REVIEW_SYSTEM
    assert "different material" in _EXACT_VERIFICATION_REVIEW_SYSTEM
