from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.reframing.sparse_replication_materializer import (
    build_sparse_replication_plan,
    discover_reframe_shadow,
    execute_sparse_replication_plan,
)


def _touch(path: Path, text: str = "{}\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _canonical(tmp_path: Path, case: str = "G02_case") -> Path:
    canonical = tmp_path / case / "replicate_01" / "canonical"
    _touch(canonical / "hypothesis.context.json")
    _touch(canonical / "hypothesis_axis_a4.portfolio.json")
    _touch(canonical / "scientific_reframing_shadow_s38.json")
    return canonical


def test_discovers_s38_shadow_without_standard_shadow(tmp_path: Path):
    canonical = _canonical(tmp_path)
    assert discover_reframe_shadow(canonical).name == "scientific_reframing_shadow_s38.json"


def test_prefers_standard_shadow_when_both_exist(tmp_path: Path):
    canonical = _canonical(tmp_path)
    _touch(canonical / "scientific_reframing_shadow.json")
    assert discover_reframe_shadow(canonical).name == "scientific_reframing_shadow.json"


def test_plan_uses_six_zero_llm_materialization_stages(tmp_path: Path):
    plan = build_sparse_replication_plan(canonical_dir=_canonical(tmp_path))
    assert [row.stage_id for row in plan.stages] == [
        "mode_contrast",
        "reframing_portfolio",
        "cross_lane_portfolio",
        "ablation_packet",
        "schema_normalization",
        "generalized_matching",
    ]


def test_absent_optional_operators_are_not_passed_to_downstream_cli(tmp_path: Path):
    plan = build_sparse_replication_plan(canonical_dir=_canonical(tmp_path))
    flattened = " ".join(arg for stage in plan.stages for arg in stage.args)
    assert "--proxy-shadow" not in flattened
    assert "--contradiction-shadow" not in flattened


def test_present_optional_operators_are_reused_not_regenerated(tmp_path: Path):
    canonical = _canonical(tmp_path)
    proxy = _touch(canonical / "scientific_proxy_challenge_shadow.json")
    contradiction = _touch(canonical / "scientific_contradiction_resolution_shadow.json")
    plan = build_sparse_replication_plan(canonical_dir=canonical)
    assert plan.proxy_shadow == proxy
    assert plan.contradiction_shadow == contradiction
    mode = plan.stages[0].args
    assert "--proxy-shadow" in mode
    assert str(proxy) in mode
    assert "--contradiction-shadow" in mode
    assert str(contradiction) in mode


def test_generalized_matching_receives_configured_comparison_cap(tmp_path: Path):
    plan = build_sparse_replication_plan(
        canonical_dir=_canonical(tmp_path),
        max_comparisons=7,
    )
    args = plan.stages[-1].args
    index = args.index("--max-comparisons")
    assert args[index + 1] == "7"


def test_missing_relational_portfolio_fails_before_any_execution(tmp_path: Path):
    canonical = tmp_path / "K06" / "replicate_01" / "canonical"
    _touch(canonical / "hypothesis.context.json")
    _touch(canonical / "scientific_reframing_shadow.json")
    with pytest.raises(FileNotFoundError, match="relational hypothesis portfolio"):
        build_sparse_replication_plan(canonical_dir=canonical)


def test_resume_skips_stage_when_all_expected_outputs_exist(tmp_path: Path):
    plan = build_sparse_replication_plan(canonical_dir=_canonical(tmp_path))
    first = plan.stages[0]
    for output in first.outputs:
        _touch(output)

    calls: list[list[str]] = []

    def runner(command, *, cwd, check):
        calls.append(list(command))
        for stage in plan.stages:
            if stage.module in command:
                for output in stage.outputs:
                    _touch(output)
                break
        return object()

    report = execute_sparse_replication_plan(
        plan=plan,
        repository_root=tmp_path,
        runner=runner,
        python_executable="python",
    )
    assert report.records[0].status == "skipped_existing_outputs"
    assert len(calls) == len(plan.stages) - 1


def test_force_executes_existing_stage(tmp_path: Path):
    plan = build_sparse_replication_plan(canonical_dir=_canonical(tmp_path))
    for output in plan.stages[0].outputs:
        _touch(output)

    calls: list[list[str]] = []

    def runner(command, *, cwd, check):
        calls.append(list(command))
        for stage in plan.stages:
            if stage.module in command:
                for output in stage.outputs:
                    _touch(output)
                break
        return object()

    report = execute_sparse_replication_plan(
        plan=plan,
        repository_root=tmp_path,
        force=True,
        runner=runner,
        python_executable="python",
    )
    assert report.records[0].status == "executed"
    assert len(calls) == len(plan.stages)


def test_execution_fails_closed_if_stage_does_not_write_expected_output(tmp_path: Path):
    plan = build_sparse_replication_plan(canonical_dir=_canonical(tmp_path))

    def runner(command, *, cwd, check):
        return object()

    with pytest.raises(RuntimeError, match="without expected outputs"):
        execute_sparse_replication_plan(
            plan=plan,
            repository_root=tmp_path,
            runner=runner,
            python_executable="python",
        )
