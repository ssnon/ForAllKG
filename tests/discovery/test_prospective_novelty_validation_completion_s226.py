from pipeline_core.discovery.prospective_novelty_validation_completion_s226 import (
    audit_external_report_payload,
    first_gap_level,
    l3_status_from_external_card,
    research_value_detail_payload,
)


def test_first_gap_stops_on_uncertainty():
    assert first_gap_level(
        [
            ("L1_BROAD", "RELATION_BACKED"),
            ("L2_INTERMEDIATE", "INSUFFICIENT_COVERAGE"),
            ("L3_EXACT", "SEARCH_BOUNDED_GAP"),
        ]
    ) is None


def test_first_gap_returns_earliest_search_bounded_gap():
    assert first_gap_level(
        [
            ("L1_BROAD", "RELATION_BACKED"),
            ("L2_INTERMEDIATE", "SEARCH_BOUNDED_GAP"),
            ("L3_EXACT", "SEARCH_BOUNDED_RELATIONAL_GAP"),
        ]
    ) == "L2_INTERMEDIATE"


def _gap_card_with_title_only():
    return {
        "hypothesis_id": "h1",
        "status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        "relational_gap_kind": "HIGHER_ORDER_RELATIONAL_GAP",
        "coverage": {
            "sufficient_for_absence_based_novelty": True,
        },
        "claim_reviews": [
            {
                "claim_id": "c1",
                "importance": "core",
                "status": "TITLE_ONLY_NEIGHBORS",
                "matches": [
                    {
                        "work_id": "w1",
                        "relationship": "TITLE_ONLY_NEIGHBOR",
                        "confidence": 0.95,
                        "title": "Highly relevant title",
                        "doi": "10.test/example",
                        "relevance_score": 0.8,
                        "abstract_available": False,
                    }
                ],
            }
        ],
    }


def test_l3_gap_is_withheld_when_material_title_only_is_unresolved():
    result = l3_status_from_external_card(
        _gap_card_with_title_only(),
        min_confidence=0.65,
    )
    assert result["status"] == "INSUFFICIENT_COVERAGE"
    assert result["sufficient_coverage"] is False
    assert (
        result["reason_code"]
        == "MATERIAL_CORE_TITLE_ONLY_PRIOR_ART_UNRESOLVED"
    )


def test_external_audit_reports_resolution_risk():
    report = {
        "policy": {"min_match_confidence": 0.65},
        "status_counts": {
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": 1
        },
        "cards": [_gap_card_with_title_only()],
    }
    packet = {
        "works": [
            {"work_id": "w1", "abstract": None},
            {"work_id": "w2", "abstract": "available"},
        ],
        "executions": [
            {"success": True},
            {"success": False},
        ],
    }
    audit = audit_external_report_payload(
        report,
        packet=packet,
    )
    assert audit["disposition"] == "RESOLUTION_RISK"
    assert audit["material_core_title_only_count"] == 1
    assert audit["packet_missing_abstract_count"] == 1
    assert audit["provider_execution_failure_count"] == 1
    assert (
        audit["retrieval_recall_completeness"]
        == "UNOBSERVABLE_WITHOUT_EXTERNAL_REFERENCE_SET"
    )


def test_research_value_detail_collects_dimension_signals():
    report = {
        "hypothesis_count": 1,
        "assessed_count": 1,
        "novelty_signal_consumed": False,
        "research_value_selection_authority": False,
        "production_selection_authority": False,
        "cards": [
            {
                "hypothesis_id": "h1",
                "hypothesis_type": "mechanistic_extension",
                "value_argument_class": "VALUE_ARGUMENT_SUPPORTED",
                "mechanistic_discrimination": {"signal": "STRONG"},
                "two_sided_outcome_informativeness":
                    {"signal": "STRONG"},
                "observable_decisiveness": {"signal": "PARTIAL"},
                "information_gain_proxy": {"signal": "PARTIAL"},
                "experimental_resolvability": {"signal": "STRONG"},
                "experimental_disposition":
                    "experimentally_plausible",
                "relative_cost_burden": "moderate",
                "relative_effort_burden": "moderate",
                "reason_codes": ["x"],
            }
        ],
    }
    result = research_value_detail_payload(report)
    assert result["measurement_status"] == "AVAILABLE"
    card = result["cards"][0]
    assert card["value_argument_class"] == "VALUE_ARGUMENT_SUPPORTED"
    assert card["dimensions"]["mechanistic_discrimination"] == "STRONG"
    assert card["dimensions"]["observable_decisiveness"] == "PARTIAL"
