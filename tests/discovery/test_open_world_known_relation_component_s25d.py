from __future__ import annotations

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxis
from pipeline_core.discovery.open_world_discovery_axis import (
    ExternalAxisDraft,
    _s25d_external_axis_novelty_role,
)


def _draft(*, bounded_synthesis_note: str) -> ExternalAxisDraft:
    return ExternalAxisDraft(
        local_id="a1",
        label="axis",
        proposed_subject="A",
        proposed_relation="affects",
        proposed_object="B",
        source_work_ids=["w1"],
        source_evidence_spans=["A affects B"],
        compatible_grounded_statement_ids=["s1"],
        bounded_synthesis_note=bounded_synthesis_note,
        requires_verification=True,
    )


def test_s25d_source_reported_relation_becomes_known_component():
    role = _s25d_external_axis_novelty_role(
        _draft(bounded_synthesis_note="")
    )

    assert role.inspiration_role == "KNOWN_RELATION_COMPONENT"
    assert role.external_relation_source_mode == "SOURCE_REPORTED"
    assert role.second_order_gap_required is True
    assert "S25D_KNOWN_RELATION_COMPONENT" in role.reason_codes


def test_s25d_bounded_synthesis_remains_exploratory_axis():
    role = _s25d_external_axis_novelty_role(
        _draft(
            bounded_synthesis_note=(
                "bounded synthesis between two supplied source spans"
            )
        )
    )

    assert role.inspiration_role == "EXPLORATORY_AXIS"
    assert role.external_relation_source_mode == "BOUNDED_SYNTHESIS"
    assert role.second_order_gap_required is False
    assert "S25D_BOUNDED_SYNTHESIS_REMAINS_EXPLORATORY" in role.reason_codes


def test_discovery_axis_defaults_preserve_non_open_world_callers():
    axis = DiscoveryAxis(
        axis_id="x",
        axis_rank=1,
        inspiration_id="i",
        source_path_id="p",
        candidate_unit_id="c",
        label="legacy",
        rendered_path="A -> B",
        source_mode="persistent_kg",
        exploration_score=0.5,
        planner_score=0.5,
        mechanistic_continuity_band="medium",
    )

    assert axis.inspiration_role == "EXPLORATORY_AXIS"
    assert axis.external_relation_source_mode == "NOT_APPLICABLE"
    assert axis.second_order_gap_required is False
