from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import NoveltyClaim
from pipeline_core.discovery.lower_order_prior_art_saturation import (
    _compact_query,
    _query_texts_for_claim,
)
from pipeline_core.discovery.external_novelty_llm import (
    _EXACT_VERIFICATION_REVIEW_SYSTEM,
)


def _h6_claim() -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=(
            "Multidimensional plasmonic coupling among Au nanoparticles, "
            "Ag nanoparticles, and an Au film relates to local-field enhancement."
        ),
        rationale="test",
        search_concepts=[
            "multidimensional plasmonic coupling",
            "Au nanoparticles",
            "Ag nanoparticles",
            "Au film",
            "local field enhancement",
        ],
        search_queries=[
            "multidimensional plasmonic coupling gold nanoparticles silver nanoparticles gold film local field enhancement"
        ],
        prior_art_identity_terms=[
            "Au nanoparticles",
            "Ag nanoparticles",
            "Au film",
        ],
        relation_nucleus_terms=[
            "multidimensional plasmonic coupling",
            "local field enhancement",
        ],
        diagnostic_structural_terms=[
            "Au nanoparticles",
            "Ag nanoparticles",
            "Au film",
        ],
    )


def test_compaction_preserves_repeated_entity_carriers():
    q = _compact_query(
        "Au nanoparticles Ag nanoparticles Au film multidimensional coupling"
    )
    assert q.count("nanoparticles") == 2
    assert q.count("Au") == 2
    assert "Ag nanoparticles" in q
    assert "Au film" in q


def test_query_ladder_preserves_h6_architecture_pairings():
    queries = _query_texts_for_claim(_h6_claim())
    assert any(
        "Au nanoparticles" in q
        and "Ag nanoparticles" in q
        and "Au film" in q
        for q in queries
    )


def test_exact_reviewer_uses_stable_structural_identity_policy():
    p = _EXACT_VERIFICATION_REVIEW_SYSTEM
    assert "DIRECT_PRIOR_ART requires" in p
    assert "PARTIAL_PRIOR_ART requires substantial identity overlap" in p
    assert "structural identity is mandatory" in p
    assert "semiconductor bilayer" in p
    assert "different material" in p
