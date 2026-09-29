from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.novelty_depth_causal_edge_coverage import (
    build_novelty_depth_causal_edge_coverage,
)


def _match(rel, wid):
    return SimpleNamespace(
        relationship=rel,
        work_id=wid,
    )


def _review(
    cid,
    status,
    matches,
    *,
    text="orientation and wavelength jointly alter mode intensity",
    importance="core",
):
    return SimpleNamespace(
        claim_id=cid,
        importance=importance,
        claim_text=text,
        status=status,
        matches=matches,
    )


def _fixture(review, premise_text):
    context = SimpleNamespace(
        context_id="ctx:1",
        evidence_statements=[
            SimpleNamespace(
                statement_id="s1",
                text=premise_text,
            )
        ],
    )
    portfolio = SimpleNamespace(
        portfolio_id="portfolio:1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                premise_statement_ids=["s1"],
            )
        ],
    )
    external = SimpleNamespace(
        report_id="external:1",
        cards=[
            SimpleNamespace(
                hypothesis_id="h1",
                title="fixture",
                status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                claim_reviews=[review],
            )
        ],
    )
    return context, portfolio, external


def test_higher_order_gap_over_multiple_known_lower_order_relations():
    review = _review(
        "c1",
        "COMPONENTS_ONLY",
        [
            _match("LOWER_ORDER_RELATION_PRIOR_ART", "w1"),
            _match("LOWER_ORDER_RELATION_PRIOR_ART", "w2"),
        ],
    )
    context, portfolio, external = _fixture(
        review,
        "orientation changes mode intensity and wavelength changes mode intensity",
    )

    report = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )

    row = report.profiles[0]
    assert row.novelty_depth_class == "HIGHER_ORDER_INTERACTION_GAP"
    assert row.planner_advisory == "KEEP_TESTABLE_GAP"
    assert row.core_lower_order_relation_count == 2


def test_weak_bridge_is_not_rewarded_as_deep_novelty():
    review = _review(
        "c1",
        "COMPONENTS_ONLY",
        [
            _match("COMPONENT_ONLY", "w1"),
            _match("COMPONENT_ONLY", "w2"),
        ],
        text="superlattice order determines adsorption orientation distribution",
    )
    context, portfolio, external = _fixture(
        review,
        "particle uniformity changes lattice order",
    )

    report = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )

    row = report.profiles[0]
    assert row.novelty_depth_class == "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN"
    assert row.planner_advisory == "REAXIS_OR_ABSTAIN"
    assert row.core_unsupported_bridge_count == 1
    assert row.weak_bridge_claim_ids == ["c1"]


def test_single_lower_order_relation_is_local_extension():
    review = _review(
        "c1",
        "COMPONENTS_ONLY",
        [
            _match("LOWER_ORDER_RELATION_PRIOR_ART", "w1"),
            _match("COMPONENT_ONLY", "w2"),
        ],
    )
    context, portfolio, external = _fixture(
        review,
        "orientation and wavelength affect Raman mode intensity",
    )

    report = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )

    row = report.profiles[0]
    assert row.novelty_depth_class == "SHALLOW_LOCAL_EXTENSION"
    assert row.planner_advisory == "SHARPEN_LOCAL_EXTENSION"


def test_direct_prior_art_is_known_relation():
    review = _review(
        "c1",
        "DIRECT_PRIOR_ART",
        [_match("DIRECT_PRIOR_ART", "w1")],
    )
    context, portfolio, external = _fixture(
        review,
        "orientation and wavelength jointly alter mode intensity",
    )

    report = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )

    row = report.profiles[0]
    assert row.novelty_depth_class == "KNOWN_RELATION"
    assert row.planner_advisory == "KNOWN_RELATION"


def test_unresolved_metadata_holds_depth():
    review = _review(
        "c1",
        "INSUFFICIENT_METADATA",
        [_match("INSUFFICIENT_METADATA", "w1")],
    )
    context, portfolio, external = _fixture(
        review,
        "orientation changes Raman intensity",
    )

    report = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )

    row = report.profiles[0]
    assert row.novelty_depth_class == "UNRESOLVED_DEPTH"
    assert row.planner_advisory == "HOLD_UNRESOLVED"
