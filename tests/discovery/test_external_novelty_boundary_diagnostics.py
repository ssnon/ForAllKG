from __future__ import annotations

import inspect

from pipeline_core.discovery.external_novelty import (
    ExternalNoveltyAssessor,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyCard,
    ExternalNoveltyPolicy,
    PriorArtMatchDraft,
)
from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
    _REVIEW_SYSTEM,
)
from pipeline_core.discovery.prior_art_matching import (
    ClaimPriorArtCompiler,
)


def test_diagnostic_relationships_are_contract_valid() -> None:
    lower = PriorArtMatchDraft(
        work_id="prior_art_work:test_lower",
        relationship="LOWER_ORDER_RELATION_PRIOR_ART",
        confidence=0.90,
        rationale="Explicit lower-order relation.",
    )

    boundary = PriorArtMatchDraft(
        work_id="prior_art_work:test_boundary",
        relationship="DIRECTIONAL_COUNTEREVIDENCE",
        confidence=0.90,
        rationale="Neighboring-scope directional boundary.",
    )

    assert (
        lower.relationship
        == "LOWER_ORDER_RELATION_PRIOR_ART"
    )

    assert (
        boundary.relationship
        == "DIRECTIONAL_COUNTEREVIDENCE"
    )


def test_card_exposes_diagnostic_provenance() -> None:
    fields = ExternalNoveltyCard.model_fields

    assert "lower_order_prior_art_work_ids" in fields

    assert (
        "directional_counterevidence_work_ids"
        in fields
    )


def test_query_prompt_requires_bounded_query_diversity() -> None:
    assert "QUERY-DIVERSITY RULE:" in _DECOMPOSE_SYSTEM

    assert (
        "LOWER-ORDER RELATION"
        in _DECOMPOSE_SYSTEM
    )

    assert (
        "BOUNDARY OR COUNTEREVIDENCE"
        in _DECOMPOSE_SYSTEM
    )

    assert (
        "Never insert a known paper title, DOI, author, year"
        in _DECOMPOSE_SYSTEM
    )


def test_review_prompt_distinguishes_diagnostic_evidence() -> None:
    assert (
        "LOWER_ORDER_RELATION_PRIOR_ART"
        in _REVIEW_SYSTEM
    )

    assert (
        "DIRECTIONAL_COUNTEREVIDENCE"
        in _REVIEW_SYSTEM
    )

    assert (
        "They do not by themselves make the full claim "
        "DIRECT_PRIOR_ART or PARTIAL_PRIOR_ART"
        in _REVIEW_SYSTEM
    )


def test_compiler_keeps_diagnostics_in_component_family() -> None:
    source = inspect.getsource(
        ClaimPriorArtCompiler.compile
    )

    start = source.index(
        "components = ["
    )

    end = source.index(
        "title_only = [",
        start,
    )

    block = source[start:end]

    assert (
        "LOWER_ORDER_RELATION_PRIOR_ART"
        in block
    )

    assert (
        "DIRECTIONAL_COUNTEREVIDENCE"
        in block
    )

    assert (
        "lower_order_relation_prior_art_present"
        in source
    )

    assert (
        "directional_counterevidence_present"
        in source
    )


def test_diagnostics_do_not_redefine_hypothesis_status_logic() -> None:
    source = inspect.getsource(
        ExternalNoveltyAssessor._status
    )

    assert (
        "LOWER_ORDER_RELATION_PRIOR_ART"
        not in source
    )

    assert (
        "DIRECTIONAL_COUNTEREVIDENCE"
        not in source
    )


def test_report_aggregates_diagnostic_work_ids() -> None:
    source = inspect.getsource(
        ExternalNoveltyAssessor.compile_report_from_claim_reviews
    )

    assert (
        "lower_order_prior_art_work_ids"
        in source
    )

    assert (
        "directional_counterevidence_work_ids"
        in source
    )


def test_existing_search_and_match_budgets_are_unchanged() -> None:
    policy = ExternalNoveltyPolicy()

    assert policy.max_queries_per_claim == 2
    assert policy.max_ranked_works_per_claim == 8



def test_explicit_diagnostic_query_contract_is_exposed() -> None:
    from pipeline_core.discovery.external_novelty_contracts import (
        LiteratureQuery,
        NoveltyClaim,
        NoveltyClaimDraft,
    )

    assert "diagnostic_query_kind" in NoveltyClaimDraft.model_fields
    assert "diagnostic_search_query" in NoveltyClaimDraft.model_fields

    assert "diagnostic_query_kind" in NoveltyClaim.model_fields
    assert "diagnostic_search_query" in NoveltyClaim.model_fields

    allowed = LiteratureQuery.model_fields[
        "query_kind"
    ].annotation

    assert "claim_diagnostic" in str(allowed)


def test_relation_first_fields_exist_on_draft_and_claim() -> None:
    from pipeline_core.discovery.external_novelty_contracts import (
        NoveltyClaim,
        NoveltyClaimDraft,
    )

    for name in (
        "diagnostic_structural_terms",
        "diagnostic_relation_terms",
    ):
        assert name in NoveltyClaimDraft.model_fields
        assert name in NoveltyClaim.model_fields

    assert (
        "diagnostic_execution_query"
        in NoveltyClaim.model_fields
    )


def test_relation_first_assembler_preserves_structural_carrier() -> None:
    from pipeline_core.discovery.novelty_claim_decomposition import (
        _assemble_diagnostic_relation_query,
    )

    query, structural, relation = (
        _assemble_diagnostic_relation_query(
            ["plasmonic dimer"],
            [
                "SERS",
                "gap size",
                "LSPR",
                "resonance wavelength",
            ],
            fallback="fallback query",
        )
    )

    assert structural == [
        "plasmonic dimer",
    ]

    assert relation == [
        "SERS",
        "gap size",
        "LSPR",
        "resonance wavelength",
    ]

    assert query == (
        "plasmonic dimer SERS gap size "
        "LSPR resonance wavelength"
    )


def test_relation_first_assembler_does_not_invent_terms() -> None:
    from pipeline_core.discovery.novelty_claim_decomposition import (
        _assemble_diagnostic_relation_query,
    )

    query, structural, relation = (
        _assemble_diagnostic_relation_query(
            ["generic structure"],
            [
                "variable X",
                "outcome Y",
            ],
            fallback="fallback query",
        )
    )

    assert query == (
        "generic structure variable X outcome Y"
    )

    assert structural == [
        "generic structure",
    ]

    assert relation == [
        "variable X",
        "outcome Y",
    ]


def test_relation_first_assembler_has_no_sers_case_hardcoding() -> None:
    import inspect

    from pipeline_core.discovery.novelty_claim_decomposition import (
        _assemble_diagnostic_relation_query,
    )

    source = inspect.getsource(
        _assemble_diagnostic_relation_query
    ).lower()

    for forbidden in (
        "au ag",
        "gold silver",
        "nanogap",
        "sers",
        "lspr",
    ):
        assert forbidden not in source



def test_diagnostic_metadata_does_not_replace_ordinary_second_query() -> None:
    from types import SimpleNamespace

    from pipeline_core.discovery.external_novelty_contracts import (
        NoveltyClaimDecompositionDraft,
        NoveltyClaimDraft,
    )
    from pipeline_core.discovery.novelty_claim_decomposition import (
        NoveltyClaimDecomposer,
    )

    class Backend:
        def decompose(
            self,
            hypothesis,
            *,
            max_claims,
        ):
            return NoveltyClaimDecompositionDraft(
                claims=[
                    NoveltyClaimDraft(
                        local_id="c1",
                        kind="moderator_interaction",
                        importance="core",
                        text=(
                            "A moderator changes how X "
                            "relates to Y."
                        ),
                        rationale="test",
                        search_concepts=[],
                        search_queries=[
                            "exact full relation",
                            "ordinary retrieval variant",
                        ],
                        diagnostic_query_kind=(
                            "LOWER_ORDER_RELATION"
                        ),
                        diagnostic_search_query=(
                            "human readable diagnostic"
                        ),
                        diagnostic_structural_terms=[
                            "generic dimer",
                        ],
                        diagnostic_relation_terms=[
                            "variable X",
                            "outcome Y",
                            "dependence",
                        ],
                    )
                ]
            )

    result = NoveltyClaimDecomposer(
        Backend(),
        max_claims_per_hypothesis=1,
        max_queries_per_claim=2,
    ).decompose(
        SimpleNamespace(
            hypothesis_id="hypothesis:test",
            title="test",
        )
    )

    claim = result.claims[0]

    assert claim.search_queries == [
        "exact full relation",
        "ordinary retrieval variant",
    ]

    assert (
        claim.diagnostic_query_kind
        == "LOWER_ORDER_RELATION"
    )

    assert (
        claim.diagnostic_search_query
        == "human readable diagnostic"
    )

    assert (
        claim.diagnostic_execution_query
        == "generic dimer variable X outcome Y dependence"
    )

    assert claim.diagnostic_structural_terms == [
        "generic dimer",
    ]

    assert claim.diagnostic_relation_terms == [
        "variable X",
        "outcome Y",
        "dependence",
    ]


def test_first_pass_planner_labels_second_query_as_variant() -> None:
    from types import SimpleNamespace

    from pipeline_core.discovery.external_novelty_contracts import (
        NoveltyClaimDecompositionDraft,
        NoveltyClaimDraft,
    )
    from pipeline_core.discovery.novelty_claim_decomposition import (
        LiteratureQueryPlanner,
        NoveltyClaimDecomposer,
    )

    hypothesis = SimpleNamespace(
        hypothesis_id="hypothesis:test",
        title="test title",
        hypothesis_statement=(
            "test hypothesis statement"
        ),
    )

    class Backend:
        def decompose(
            self,
            hypothesis,
            *,
            max_claims,
        ):
            return NoveltyClaimDecompositionDraft(
                claims=[
                    NoveltyClaimDraft(
                        local_id="c1",
                        kind="moderator_interaction",
                        importance="core",
                        text=(
                            "A moderator changes how X "
                            "relates to Y."
                        ),
                        rationale="test",
                        search_concepts=[],
                        search_queries=[
                            "exact full relation",
                            "ordinary retrieval variant",
                        ],
                        diagnostic_query_kind=(
                            "LOWER_ORDER_RELATION"
                        ),
                        diagnostic_search_query=(
                            "human readable diagnostic"
                        ),
                        diagnostic_structural_terms=[
                            "generic dimer",
                        ],
                        diagnostic_relation_terms=[
                            "variable X",
                            "outcome Y",
                        ],
                    )
                ]
            )

    decomposition = NoveltyClaimDecomposer(
        Backend(),
        max_claims_per_hypothesis=1,
        max_queries_per_claim=2,
    ).decompose(
        hypothesis
    )

    portfolio = SimpleNamespace(
        portfolio_id="hypothesis_portfolio:test",
        hypotheses=[hypothesis],
    )

    plan = LiteratureQueryPlanner(
        include_hypothesis_composite=False
    ).build(
        portfolio,
        [decomposition],
    )

    claim_queries = [
        row
        for row in plan.queries
        if row.claim_id
        == decomposition.claims[0].claim_id
    ]

    assert [
        row.query_kind
        for row in claim_queries
    ] == [
        "claim_primary",
        "claim_variant",
    ]

    assert [
        row.query_text
        for row in claim_queries
    ] == [
        "exact full relation",
        "ordinary retrieval variant",
    ]

    assert not any(
        row.query_kind == "claim_diagnostic"
        for row in plan.queries
    )

    # Diagnostic candidate is still preserved in plan provenance.
    assert (
        plan.claims[0]
        .claims[0]
        .diagnostic_execution_query
        == "generic dimer variable X outcome Y"
    )


def test_prompt_does_not_reserve_first_pass_diagnostic_slot() -> None:
    from pipeline_core.discovery import (
        external_novelty_llm,
    )

    prompt = (
        external_novelty_llm
        ._DECOMPOSE_SYSTEM
    )

    lower = prompt.lower()

    assert (
        "prefer exact relation plus one diagnostic query"
        not in lower
    )

    assert (
        "may reserve the second query slot"
        not in lower
    )

    assert (
        "does not automatically insert "
        "it into first-pass search_queries"
        in prompt
    )
