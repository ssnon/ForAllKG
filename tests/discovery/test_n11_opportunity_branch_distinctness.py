from pipeline_core.discovery.nonobviousness_opportunity import (
    _collapse_overlapping_branches,
    _higher_order_operator_rows,
)


def _branch(
    ref,
    statements,
    papers=None,
):
    return {
        "source_refs": [ref],
        "support_statement_ids": statements,
        "paper_ids": papers or [],
    }


def test_route_and_motif_over_same_support_collapse():
    rows = _collapse_overlapping_branches(
        [
            _branch(
                "route:r1",
                ["stmt:a", "stmt:b"],
                ["paper:1", "paper:2"],
            ),
            _branch(
                "motif:m1",
                ["stmt:a", "stmt:b"],
                ["paper:1"],
            ),
        ]
    )

    assert len(rows) == 1

    assert rows[0]["support_statement_ids"] == [
        "stmt:a",
        "stmt:b",
    ]

    assert set(rows[0]["source_refs"]) == {
        "route:r1",
        "motif:m1",
    }


def test_partially_overlapping_wrappers_collapse_conservatively():
    rows = _collapse_overlapping_branches(
        [
            _branch(
                "route:r1",
                ["stmt:a", "stmt:b"],
            ),
            _branch(
                "motif:m1",
                ["stmt:b", "stmt:c"],
            ),
        ]
    )

    assert len(rows) == 1

    assert rows[0]["support_statement_ids"] == [
        "stmt:a",
        "stmt:b",
        "stmt:c",
    ]


def test_disjoint_positive_support_remains_distinct():
    rows = _collapse_overlapping_branches(
        [
            _branch(
                "route:r1",
                ["stmt:a"],
            ),
            _branch(
                "motif:m1",
                ["stmt:b"],
            ),
        ]
    )

    assert len(rows) == 2


def test_one_mechanistic_branch_cannot_enable_switch_or_competition():
    eligible, unsupported = (
        _higher_order_operator_rows(
            mechanistic_branches=[
                {
                    "branch_id": "branch:1",
                    "source_refs": ["route:r1"],
                    "support_statement_ids": [
                        "stmt:a",
                        "stmt:b",
                    ],
                    "paper_ids": ["paper:1"],
                }
            ],
            bound_gap_statement_ids=[
                "stmt:gap"
            ],
            design_lever_statement_ids=[
                "stmt:a"
            ],
        )
    )

    assert eligible == []

    assert "MECHANISM_SWITCH" in unsupported
    assert "PATHWAY_COMPETITION" in unsupported


def test_gap_is_required_even_with_two_distinct_branches():
    eligible, unsupported = (
        _higher_order_operator_rows(
            mechanistic_branches=[
                {
                    "branch_id": "branch:1",
                    "source_refs": ["route:r1"],
                    "support_statement_ids": [
                        "stmt:a"
                    ],
                    "paper_ids": ["paper:1"],
                },
                {
                    "branch_id": "branch:2",
                    "source_refs": ["route:r2"],
                    "support_statement_ids": [
                        "stmt:b"
                    ],
                    "paper_ids": ["paper:2"],
                },
            ],
            bound_gap_statement_ids=[],
            design_lever_statement_ids=[
                "stmt:a"
            ],
        )
    )

    assert eligible == []

    assert "MECHANISM_SWITCH" in unsupported
    assert "PATHWAY_COMPETITION" in unsupported


def test_two_distinct_branches_plus_bound_gap_enable_search_operators():
    eligible, unsupported = (
        _higher_order_operator_rows(
            mechanistic_branches=[
                {
                    "branch_id": "branch:1",
                    "source_refs": ["route:r1"],
                    "support_statement_ids": [
                        "stmt:a"
                    ],
                    "paper_ids": ["paper:1"],
                },
                {
                    "branch_id": "branch:2",
                    "source_refs": ["motif:m2"],
                    "support_statement_ids": [
                        "stmt:b"
                    ],
                    "paper_ids": ["paper:2"],
                },
            ],
            bound_gap_statement_ids=[
                "stmt:gap"
            ],
            design_lever_statement_ids=[
                "stmt:a"
            ],
        )
    )

    by_operator = {
        row["operator"]: row
        for row in eligible
    }

    assert "MECHANISM_SWITCH" in by_operator
    assert "PATHWAY_COMPETITION" in by_operator

    assert (
        by_operator[
            "MECHANISM_SWITCH"
        ]["gap_statement_ids"]
        == ["stmt:gap"]
    )

    assert (
        by_operator[
            "PATHWAY_COMPETITION"
        ]["gap_statement_ids"]
        == ["stmt:gap"]
    )

    # The unresolved gap must never be promoted into positive support.
    assert "stmt:gap" not in (
        by_operator[
            "MECHANISM_SWITCH"
        ]["support_statement_ids"]
    )

    assert "MECHANISM_SWITCH" not in unsupported
    assert "PATHWAY_COMPETITION" not in unsupported
