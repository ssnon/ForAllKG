from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.prospective_routed_execution_plan_v2 import (
    ProspectiveRoutedCaseExecutionPlanV2,
)
from pipeline_core.discovery.prospective_routed_route_compiler_v2 import (
    RegenerationFallbackLineageV2,
    _fallback_lineage,
    _resolve_template,
    _slug,
)


def _case() -> ProspectiveRoutedCaseExecutionPlanV2:
    base = "/tmp/P16/routed_preverifier_v2"
    template = (
        base
        + "/lineage/{final_hypothesis_id_slug}/regeneration_v2"
    )
    return ProspectiveRoutedCaseExecutionPlanV2(
        case_id="P16",
        source_task_id="task:1",
        source_task_sha256="a" * 64,
        run_dir="/tmp/P16",
        generation_model="model",
        critic_model="critic",
        route_model="critic",
        route_audit_model="critic",
        regeneration_model="model",
        initial_main_e2e_argv=[
            "python",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
            "--run-dir",
            "/tmp/P16",
        ],
        hypothesis_context_path="/tmp/P16/hypothesis.context.json",
        full_binding_plan_path="/tmp/P16/relational_atomic_binding_plan.full.json",
        pre_route_gate_path=base + "/pre_route_contract_gate_v2.json",
        routed_dispatch_path=base + "/routed_dispatch.v2.json",
        routed_lineage_dir=base + "/lineage",
        regeneration_lineage_dir_template=template,
        regeneration_unit_result_path_template=(
            template + "/regeneration_unit_v2.result.json"
        ),
        regeneration_portfolio_path_template=(
            template + "/regenerated.portfolio.json"
        ),
        regeneration_downstream_dir_template=(
            template + "/downstream_v2"
        ),
        regeneration_downstream_report_path_template=(
            template + "/downstream_v2/downstream_v2.report.json"
        ),
        regeneration_argv=None,
        post_route_binding_plan_path=base + "/binding.post.json",
        post_route_gate_path=base + "/gate.post.json",
        selection_report_path=base + "/selection.json",
        selected_binding_plan_path=base + "/binding.selected.json",
        endpoint_binding_report_path=base + "/endpoint.json",
        endpoint_binding_prompt_path=base + "/endpoint.prompt.txt",
        endpoint_binding_telemetry_path=base + "/endpoint.telemetry.jsonl",
        projection_preflight_output_dir=base + "/projection",
        relational_verifier_output_dir=base + "/verifier",
    )


def test_slug_is_deterministic_and_removes_colon() -> None:
    assert _slug("hypothesis:abc") == "hypothesis_abc"
    assert _slug("hypothesis:a/b") == "hypothesis_a_b"


def test_template_resolution_eliminates_placeholder() -> None:
    resolved = _resolve_template(
        "/tmp/{final_hypothesis_id_slug}/x",
        "hypothesis:abc",
    )
    assert resolved == "/tmp/hypothesis_abc/x"
    assert "{final_hypothesis_id_slug}" not in resolved


def test_fallback_lineage_uses_frozen_context_and_v2_paths() -> None:
    fallback = _fallback_lineage(
        case=_case(),
        final_hypothesis_id="hypothesis:abc",
    )
    assert fallback.input_hypothesis_context_path == (
        "/tmp/P16/hypothesis.context.json"
    )
    assert fallback.lineage_dir.endswith(
        "/lineage/hypothesis_abc/regeneration_v2"
    )
    assert fallback.unit_result_path.endswith(
        "/regeneration_unit_v2.result.json"
    )
    assert fallback.downstream_report_path.endswith(
        "/downstream_v2/downstream_v2.report.json"
    )
    assert fallback.full_e2e_argv_present is False
    assert fallback.full_e2e_rerun_allowed is False


def test_fallback_budget_is_exactly_one_generate_zero_repair() -> None:
    fallback = _fallback_lineage(
        case=_case(),
        final_hypothesis_id="hypothesis:abc",
    )
    assert fallback.structured_generation_calls_max == 1
    assert fallback.repair_calls_max == 0


def test_lineage_rejects_unresolved_template() -> None:
    with pytest.raises(ValidationError, match="resolve hypothesis slug"):
        RegenerationFallbackLineageV2(
            input_hypothesis_context_path="/tmp/context.json",
            lineage_dir="/tmp/{final_hypothesis_id_slug}",
            unit_result_path="/tmp/result.json",
            regenerated_portfolio_path="/tmp/portfolio.json",
            downstream_dir="/tmp/downstream",
            downstream_report_path="/tmp/downstream/report.json",
        )


def test_template_without_slug_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="lacks final hypothesis slug token"):
        _resolve_template("/tmp/no-token", "hypothesis:abc")


def test_v2_case_has_no_regeneration_argv() -> None:
    case = _case()
    assert case.regeneration_argv is None
    assert case.regeneration_full_e2e_argv_present is False


def test_regeneration_lineage_does_not_contain_e2e_command() -> None:
    fallback = _fallback_lineage(
        case=_case(),
        final_hypothesis_id="hypothesis:abc",
    )
    payload = fallback.model_dump(mode="json")
    assert "argv" not in payload
    assert "run_dac_discovery_e2e" not in str(payload)
