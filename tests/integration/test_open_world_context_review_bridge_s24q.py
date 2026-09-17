from __future__ import annotations

import inspect

from domains.sers import context_review_adapter as sers_review
from domains.sers.context_contracts import SERSContextProvenance
from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxis
from pipeline_core.discovery.open_world_discovery_axis import ExternalAxisProvenance
from scripts.discovery import run_dac_discovery_e2e as e2e
from scripts.discovery import run_discovery_axis_hypothesis_maker as maker


def _axis() -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id="axis:external",
        axis_rank=1,
        inspiration_id="external_inspiration:test",
        source_path_id="external_work_set:test",
        candidate_unit_id="external_axis:test",
        label="external axis",
        rendered_path="external evidence -> bounded axis",
        source_mode="external_open_world",
        exploration_score=1.0,
        candidate_unit_score=1.0,
        planner_score=1.0,
        mechanistic_continuity_band="medium",
        requires_verification=True,
    )


def _provenance() -> ExternalAxisProvenance:
    return ExternalAxisProvenance(
        axis_id="axis:external",
        raw_local_id="local:1",
        source_work_ids=["work:1"],
        source_evidence_spans=["Source-backed external evidence span."],
        compatible_grounded_statement_ids=["stmt:1"],
        bounded_synthesis_note="bounded",
        max_control_similarity=0.2,
    )


def test_external_axis_unknown_projection_never_invents_context_values():
    signature = (
        sers_review.SERSDiscoveryAxisContextReviewer
        ._external_axis_unknown_signature(
            axis=_axis(),
            provenance=_provenance(),
        )
    )

    assert signature.scope == "axis_inspiration"
    assert signature.source_ref_id == "external_inspiration:test"
    assert len(signature.facts) > 0
    assert all(row.knowledge_state == "unknown" for row in signature.facts)
    assert all(row.value is None for row in signature.facts)
    assert all(row.normalized_value is None for row in signature.facts)
    assert all(row.binding is None for row in signature.facts)
    assert {
        provenance.kind
        for row in signature.facts
        for provenance in row.provenance
    } == {"external_axis_source_span"}


def test_external_source_span_provenance_is_traceable_but_not_a_premise():
    row = SERSContextProvenance(
        kind="external_axis_source_span",
        paper_ids=["work:1"],
        excerpt="external source span",
    )
    assert row.kind == "external_axis_source_span"
    assert row.statement_ids == []
    assert row.hypothesis_ids == []


def test_alpha4_requires_external_bundle_for_external_open_world_plan():
    src = inspect.getsource(maker.main)
    assert 'axis.source_mode == "external_open_world"' in src
    assert '"--external-axis-bundle"' in inspect.getsource(maker.parse_args)
    assert "external axis plan/bundle axis IDs differ" in src
    assert "bind_external_axis_bundle" in src


def test_realization_search_forwards_external_bundle_to_every_slot():
    candidate = inspect.getsource(e2e._run_realization_candidate_chain)
    production = inspect.getsource(e2e._run_realization_search_production_stage8)
    parent = inspect.getsource(e2e.run_pipeline)

    assert '"--external-axis-bundle"' in candidate
    assert "external_axis_bundle" in production
    assert '"external_axis_bundle"' in parent
    assert "external_axis_bundle=(" in parent
