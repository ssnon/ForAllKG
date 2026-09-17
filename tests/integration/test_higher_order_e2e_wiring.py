from __future__ import annotations

import inspect

from pipeline_core.discovery import higher_order_modifier_eligibility as eligibility
from scripts.discovery import run_dac_discovery_e2e as e2e
from scripts.discovery import run_higher_order_shadow_lane as lane


def test_frozen_modifier_screen_is_authority_neutral():
    src = inspect.getsource(eligibility.screen_confirmed_known_modifiers)
    assert "scientific_quality_ranking_performed" in src
    assert "novelty_authority_created" in src
    assert "domain_synonym_ontology_used" in src


def test_parent_wires_parallel_higher_order_shadow_without_removing_stage8():
    src = inspect.getsource(e2e.run_pipeline)
    assert "run_higher_order_shadow_lane" in src
    assert "run_discovery_axis_hypothesis_maker" in src
    assert "production_selection_changed" in src


def test_lane_reuses_existing_authority_safe_stack():
    src = inspect.getsource(lane.main)
    assert "screen_confirmed_known_modifiers" in src
    assert "compose_higher_order_topologies" in src
    assert "HigherOrderShadowBatchRuntime" in src
    assert "build_higher_order_external_shadow_batch_plan" in src


def test_parent_higher_order_stage_consumes_stage75_endpoint_artifact():
    src = inspect.getsource(e2e.run_pipeline)
    marker = '"[7.55/13] Higher-order composition/generation shadow"'
    start = src.index(marker)
    end = src.index(
        'open_world_outputs: dict[str, Path] | None = None',
        start,
    )
    block = src[start:end]

    assert '"--task-axis-report"' in block
    assert 'str(task_conditioned_axis_report)' in block
    assert '"--requested-source"' not in block
    assert '"--requested-target"' not in block


def test_higher_order_lane_prefers_stage75_resolved_endpoints():
    src = inspect.getsource(lane.main)

    assert '"--task-axis-report"' in src
    assert '"requested_source"' in src
    assert '"requested_target"' in src
    assert 'endpoint_resolution_source = "stage7_5_task_axis_report"' in src
