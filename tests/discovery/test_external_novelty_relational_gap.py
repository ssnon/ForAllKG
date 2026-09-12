from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.external_novelty import (
    ExternalNoveltyAssessor,
    _lower_order_gap_annotation,
)


CLAIM_ID = "external_novelty_claim:test"
WORK_ID = "prior_art_work:lower"


def _claim(**updates):
    payload = {
        "claim_id": CLAIM_ID,
        "importance": "core",
        "kind": "composite",
        "higher_order_relation_basis": [
            "A jointly organizes B and C."
        ],
        "higher_order_component_claim_ids": [
            "external_novelty_claim:ab",
            "external_novelty_claim:ac",
        ],
        "novelty_selection_role": "NOVELTY_BEARING",
    }
    payload.update(
        updates
    )
    return SimpleNamespace(
        **payload
    )


def _review():
    return SimpleNamespace(
        claim_id=CLAIM_ID,
        importance="core",
        status="COMPONENTS_ONLY",
        matches=[
            SimpleNamespace(
                work_id=WORK_ID,
                relationship=(
                    "LOWER_ORDER_RELATION_PRIOR_ART"
                ),
            )
        ],
    )


def _coverage(
    sufficient: bool = True,
):
    return SimpleNamespace(
        sufficient_for_absence_based_novelty=(
            sufficient
        )
    )


def _annotation(
    claim,
):
    return _lower_order_gap_annotation(
        [_review()],
        _coverage(),
        {
            CLAIM_ID:
                claim,
        },
    )


def test_explicit_novelty_bearing_composite_with_topology_gets_gap():
    (
        kind,
        supported,
        gap_claim_ids,
        work_ids,
    ) = _annotation(
        _claim()
    )

    assert (
        kind
        == "HIGHER_ORDER_RELATIONAL_GAP"
    )
    assert supported == [
        CLAIM_ID
    ]
    assert gap_claim_ids == [
        CLAIM_ID
    ]
    assert work_ids == [
        WORK_ID
    ]


@pytest.mark.parametrize(
    (
        "updates",
        "reason",
    ),
    [
        (
            {
                "kind":
                    "descriptor_interaction",
            },
            "non-composite",
        ),
        (
            {
                "higher_order_relation_basis":
                    [],
            },
            "missing basis",
        ),
        (
            {
                "higher_order_component_claim_ids":
                    [],
            },
            "missing topology",
        ),
        (
            {
                "novelty_selection_role":
                    "REQUIRED_ENABLING_RELATION",
            },
            "enabling role",
        ),
        (
            {
                "novelty_selection_role":
                    None,
            },
            "legacy null role",
        ),
    ],
)
def test_structurally_ineligible_claim_keeps_lower_order_evidence_without_gap(
    updates,
    reason,
):
    del reason

    (
        kind,
        supported,
        gap_claim_ids,
        work_ids,
    ) = _annotation(
        _claim(
            **updates
        )
    )

    assert kind == "NONE"
    assert supported == [
        CLAIM_ID
    ]
    assert gap_claim_ids == []
    assert work_ids == [
        WORK_ID
    ]


def test_missing_planned_claim_fails_closed_for_gap_but_keeps_reviewed_evidence():
    (
        kind,
        supported,
        gap_claim_ids,
        work_ids,
    ) = _lower_order_gap_annotation(
        [_review()],
        _coverage(),
        {},
    )

    assert kind == "NONE"
    assert supported == [
        CLAIM_ID
    ]
    assert gap_claim_ids == []
    assert work_ids == [
        WORK_ID
    ]


def _assessor_for_role_aware_gap_coverage():
    assessor = ExternalNoveltyAssessor.__new__(
        ExternalNoveltyAssessor
    )

    assessor.policy = SimpleNamespace(
        min_successful_queries_for_absence=2,
        min_unique_works_for_absence=10,
        min_abstract_works_for_absence=5,
        min_abstract_works_per_core_claim=3,
    )

    return assessor


def _coverage_row(
    claim_id: str,
    *,
    abstract_work_count: int,
):
    return SimpleNamespace(
        claim_id=claim_id,
        importance="core",
        coverage=SimpleNamespace(
            abstract_work_count=(
                abstract_work_count
            )
        ),
    )


def _hypothesis_coverage(
    *,
    sufficient: bool,
    successful_query_count: int = 3,
    unique_work_count: int = 20,
    abstract_work_count: int = 10,
):
    return SimpleNamespace(
        successful_query_count=(
            successful_query_count
        ),
        unique_work_count=(
            unique_work_count
        ),
        abstract_work_count=(
            abstract_work_count
        ),
        sufficient_for_absence_based_novelty=(
            sufficient
        ),
    )


def test_gap_coverage_ignores_explicit_enabling_core_when_novelty_bearing_core_is_covered():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    target_id = "claim:target"
    enabling_id = "claim:enabling"

    reviews = [
        _coverage_row(
            target_id,
            abstract_work_count=5,
        ),
        _coverage_row(
            enabling_id,
            abstract_work_count=2,
        ),
    ]

    claims_by_id = {
        target_id:
            SimpleNamespace(
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                )
            ),
        enabling_id:
            SimpleNamespace(
                novelty_selection_role=(
                    "REQUIRED_ENABLING_RELATION"
                )
            ),
    }

    coverage = _hypothesis_coverage(
        sufficient=False
    )

    assert (
        assessor
        ._relational_gap_absence_sufficient(
            reviews,
            coverage,
            claims_by_id,
        )
        is True
    )


def test_gap_coverage_keeps_independently_novel_core_in_denominator():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    target_id = "claim:target"
    independent_id = "claim:independent"

    reviews = [
        _coverage_row(
            target_id,
            abstract_work_count=5,
        ),
        _coverage_row(
            independent_id,
            abstract_work_count=2,
        ),
    ]

    claims_by_id = {
        target_id:
            SimpleNamespace(
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                )
            ),
        independent_id:
            SimpleNamespace(
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                )
            ),
    }

    coverage = _hypothesis_coverage(
        sufficient=False
    )

    assert (
        assessor
        ._relational_gap_absence_sufficient(
            reviews,
            coverage,
            claims_by_id,
        )
        is False
    )


def test_gap_coverage_legacy_null_role_falls_back_to_existing_all_core_result():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    target_id = "claim:target"
    legacy_id = "claim:legacy"

    reviews = [
        _coverage_row(
            target_id,
            abstract_work_count=5,
        ),
        _coverage_row(
            legacy_id,
            abstract_work_count=2,
        ),
    ]

    claims_by_id = {
        target_id:
            SimpleNamespace(
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                )
            ),
        legacy_id:
            SimpleNamespace(
                novelty_selection_role=None
            ),
    }

    coverage = _hypothesis_coverage(
        sufficient=False
    )

    assert (
        assessor
        ._relational_gap_absence_sufficient(
            reviews,
            coverage,
            claims_by_id,
        )
        is False
    )


def test_gap_coverage_still_requires_global_search_minima():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    target_id = "claim:target"

    reviews = [
        _coverage_row(
            target_id,
            abstract_work_count=5,
        )
    ]

    claims_by_id = {
        target_id:
            SimpleNamespace(
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                )
            )
    }

    coverage = _hypothesis_coverage(
        sufficient=False,
        successful_query_count=1,
        unique_work_count=20,
        abstract_work_count=10,
    )

    assert (
        assessor
        ._relational_gap_absence_sufficient(
            reviews,
            coverage,
            claims_by_id,
        )
        is False
    )


def test_gap_annotation_accepts_separate_role_aware_absence_authority():
    claim = _claim()

    (
        kind,
        supported,
        gap_claim_ids,
        work_ids,
    ) = _lower_order_gap_annotation(
        [_review()],
        _coverage(
            sufficient=False
        ),
        {
            CLAIM_ID:
                claim,
        },
        gap_absence_sufficient=True,
    )

    assert (
        kind
        == "HIGHER_ORDER_RELATIONAL_GAP"
    )
    assert supported == [
        CLAIM_ID
    ]
    assert gap_claim_ids == [
        CLAIM_ID
    ]
    assert work_ids == [
        WORK_ID
    ]


def test_explicit_empty_diagnostic_authority_does_not_fallback_to_ordinary_lower_order_label():
    (
        kind,
        supported,
        gap_claim_ids,
        work_ids,
    ) = _lower_order_gap_annotation(
        [_review()],
        _coverage(),
        {
            CLAIM_ID:
                _claim(),
        },
        diagnostic_lower_order_signals_by_claim={},
    )

    assert kind == "NONE"
    assert supported == []
    assert gap_claim_ids == []
    assert work_ids == []


def test_diagnostic_lower_order_signal_drives_gap_even_without_ordinary_lower_order_label():
    review = _review()
    review.matches = []

    (
        kind,
        supported,
        gap_claim_ids,
        work_ids,
    ) = _lower_order_gap_annotation(
        [review],
        _coverage(),
        {
            CLAIM_ID:
                _claim(),
        },
        diagnostic_lower_order_signals_by_claim={
            CLAIM_ID: [
                WORK_ID
            ]
        },
    )

    assert (
        kind
        == "HIGHER_ORDER_RELATIONAL_GAP"
    )
    assert supported == [
        CLAIM_ID
    ]
    assert gap_claim_ids == [
        CLAIM_ID
    ]
    assert work_ids == [
        WORK_ID
    ]


def test_diagnostic_lower_order_signal_does_not_override_partial_full_claim_status():
    review = _review()
    review.status = (
        "PARTIAL_PRIOR_ART"
    )

    (
        kind,
        supported,
        gap_claim_ids,
        work_ids,
    ) = _lower_order_gap_annotation(
        [review],
        _coverage(),
        {
            CLAIM_ID:
                _claim(),
        },
        diagnostic_lower_order_signals_by_claim={
            CLAIM_ID: [
                WORK_ID
            ]
        },
    )

    assert kind == "NONE"
    assert supported == [
        CLAIM_ID
    ]
    assert gap_claim_ids == []
    assert work_ids == [
        WORK_ID
    ]


def test_diagnostic_signal_projection_rejects_unplanned_claim():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    diagnostic = SimpleNamespace(
        claim_id="claim:unexpected",
        hypothesis_id="hypothesis:test",
        claim_text="unexpected",
        diagnostic_query_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_execution_query=(
            "A B relation"
        ),
        signal_work_ids=[],
        matches=[],
    )

    with pytest.raises(
        ValueError,
        match="unplanned claim_id",
    ):
        assessor._diagnostic_lower_order_signal_map(
            [diagnostic],
            {},
            diagnostic_plan=SimpleNamespace(
                source_portfolio_id="portfolio:test",
                plan_id="plan:diagnostic",
                queries=[],
            ),
            diagnostic_packet=SimpleNamespace(
                source_portfolio_id="portfolio:test",
                source_query_plan_id="plan:diagnostic",
                works=[],
            ),
        )


def test_diagnostic_signal_projection_requires_abstract_backed_matching_signal():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    planned = SimpleNamespace(
        claim_id=CLAIM_ID,
        hypothesis_id="hypothesis:test",
        text="A jointly organizes B and C.",
        diagnostic_query_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_execution_query=(
            "A B relation"
        ),
    )

    diagnostic = SimpleNamespace(
        claim_id=CLAIM_ID,
        hypothesis_id="hypothesis:test",
        claim_text=planned.text,
        diagnostic_query_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_execution_query=(
            "A B relation"
        ),
        signal_work_ids=[
            WORK_ID
        ],
        matches=[
            SimpleNamespace(
                work_id=WORK_ID,
                relationship=(
                    "LOWER_ORDER_RELATION_PRIOR_ART"
                ),
                abstract_available=False,
            )
        ],
    )

    packet = SimpleNamespace(
        works=[
            SimpleNamespace(
                work_id=WORK_ID,
                abstract=None,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="lacks abstract-backed",
    ):
        assessor._diagnostic_lower_order_signal_map(
            [diagnostic],
            {
                CLAIM_ID:
                    planned,
            },
            diagnostic_plan=SimpleNamespace(
                source_portfolio_id="portfolio:test",
                plan_id="plan:diagnostic",
                queries=[
                    SimpleNamespace(
                        query_kind="claim_diagnostic"
                    )
                ],
            ),
            diagnostic_packet=SimpleNamespace(
                source_portfolio_id="portfolio:test",
                source_query_plan_id="plan:diagnostic",
                works=packet.works,
            ),
        )


def test_nonempty_diagnostic_reviews_require_diagnostic_plan_and_packet():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    planned = SimpleNamespace(
        claim_id=CLAIM_ID,
        hypothesis_id="hypothesis:test",
        text="A jointly organizes B and C.",
        diagnostic_query_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_execution_query=(
            "A B relation"
        ),
    )

    diagnostic = SimpleNamespace(
        claim_id=CLAIM_ID,
        hypothesis_id="hypothesis:test",
        claim_text=planned.text,
        diagnostic_query_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_execution_query=(
            "A B relation"
        ),
        signal_work_ids=[],
        matches=[],
    )

    with pytest.raises(
        ValueError,
        match="require diagnostic plan and packet",
    ):
        assessor._diagnostic_lower_order_signal_map(
            [diagnostic],
            {
                CLAIM_ID:
                    planned,
            },
            diagnostic_plan=None,
            diagnostic_packet=None,
        )


def test_explicit_empty_diagnostics_are_fail_closed_without_packet():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    assert (
        assessor._diagnostic_lower_order_signal_map(
            [],
            {},
            diagnostic_plan=None,
            diagnostic_packet=None,
        )
        == {}
    )


def test_diagnostic_plan_packet_provenance_must_match():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    planned = SimpleNamespace(
        claim_id=CLAIM_ID,
        hypothesis_id="hypothesis:test",
        text="A jointly organizes B and C.",
        diagnostic_query_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_execution_query=(
            "A B relation"
        ),
    )

    diagnostic = SimpleNamespace(
        claim_id=CLAIM_ID,
        hypothesis_id="hypothesis:test",
        claim_text=planned.text,
        diagnostic_query_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_execution_query=(
            "A B relation"
        ),
        signal_work_ids=[],
        matches=[],
    )

    with pytest.raises(
        ValueError,
        match="query-plan provenance mismatch",
    ):
        assessor._diagnostic_lower_order_signal_map(
            [diagnostic],
            {
                CLAIM_ID:
                    planned,
            },
            diagnostic_plan=SimpleNamespace(
                source_portfolio_id="portfolio:test",
                plan_id="plan:diagnostic",
                queries=[
                    SimpleNamespace(
                        query_kind="claim_diagnostic"
                    )
                ],
            ),
            diagnostic_packet=SimpleNamespace(
                source_portfolio_id="portfolio:test",
                source_query_plan_id="plan:wrong",
                works=[],
            ),
        )


def test_assess_resolver_without_diagnostic_inputs_is_fail_closed_empty():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    assessor.review_diagnostic_prior_art = (
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "diagnostic review must not run"
                )
            )
        )
    )

    assert (
        assessor
        ._resolve_diagnostic_reviews_for_assess(
            diagnostic_plan=None,
            diagnostic_packet=None,
            diagnostic_reviews=None,
        )
        == []
    )


def test_assess_resolver_requires_diagnostic_plan_packet_pair():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    with pytest.raises(
        ValueError,
        match="must be supplied together",
    ):
        assessor._resolve_diagnostic_reviews_for_assess(
            diagnostic_plan=SimpleNamespace(),
            diagnostic_packet=None,
            diagnostic_reviews=None,
        )


def test_assess_resolver_runs_diagnostic_review_for_complete_pair():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    expected = [
        SimpleNamespace(
            claim_id="claim:diagnostic"
        )
    ]

    calls = []

    def fake_review(
        plan,
        packet,
    ):
        calls.append(
            (
                plan,
                packet,
            )
        )
        return expected

    assessor.review_diagnostic_prior_art = (
        fake_review
    )

    plan = SimpleNamespace(
        plan_id="plan:diagnostic"
    )
    packet = SimpleNamespace(
        packet_id="packet:diagnostic"
    )

    actual = (
        assessor
        ._resolve_diagnostic_reviews_for_assess(
            diagnostic_plan=plan,
            diagnostic_packet=packet,
            diagnostic_reviews=None,
        )
    )

    assert actual == expected
    assert calls == [
        (
            plan,
            packet,
        )
    ]


def test_assess_resolver_reuses_supplied_diagnostics_without_rerun():
    assessor = (
        _assessor_for_role_aware_gap_coverage()
    )

    supplied = [
        SimpleNamespace(
            claim_id="claim:diagnostic"
        )
    ]

    assessor.review_diagnostic_prior_art = (
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "supplied diagnostics must not rerun"
                )
            )
        )
    )

    assert (
        assessor
        ._resolve_diagnostic_reviews_for_assess(
            diagnostic_plan=SimpleNamespace(),
            diagnostic_packet=SimpleNamespace(),
            diagnostic_reviews=supplied,
        )
        == supplied
    )

