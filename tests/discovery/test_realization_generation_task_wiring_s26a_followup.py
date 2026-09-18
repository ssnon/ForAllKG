from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts/discovery/run_dac_discovery_e2e.py"
MAKER = REPO / "scripts/discovery/run_discovery_axis_hypothesis_maker.py"
RUNTIME = REPO / "pipeline_core/discovery/discovery_axis_runtime.py"


def _function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    raise AssertionError(f"missing function {name} in {path}")


def _class_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    raise AssertionError(f"missing class {name} in {path}")


def test_runtime_contract_already_supports_generation_task_anchors() -> None:
    runtime = _class_source(RUNTIME, "DiscoveryAxisSynthesisRuntime")

    assert "task_source: str | None = None" in runtime
    assert "task_target: str | None = None" in runtime
    assert "task_question: str | None = None" in runtime
    assert "task_source=self.task_source" in runtime
    assert "task_target=self.task_target" in runtime
    assert "task_question=self.task_question" in runtime


def test_maker_exposes_optional_task_anchor_cli_and_wires_runtime() -> None:
    maker = MAKER.read_text(encoding="utf-8")

    assert '"--task-source"' in maker
    assert '"--task-target"' in maker
    assert '"--task-question"' in maker

    assert "task_source=args.task_source" in maker
    assert "task_target=args.task_target" in maker
    assert "task_question=args.task_question" in maker


def test_realization_alpha4_forwards_canonical_task_inputs() -> None:
    candidate_chain = _function_source(
        RUNNER,
        "_run_realization_candidate_chain",
    )

    assert '"--task-question"' in candidate_chain
    assert "str(args.question)" in candidate_chain
    assert '"--task-source"' in candidate_chain
    assert "str(args.source)" in candidate_chain
    assert '"--task-target"' in candidate_chain
    assert "str(args.target)" in candidate_chain


def test_downstream_task_replacement_guard_remains_present() -> None:
    stage8 = _function_source(
        RUNNER,
        "_run_realization_search_production_stage8",
    )
    runner = RUNNER.read_text(encoding="utf-8")

    assert "evaluate_hypothesis_task_preservation" in stage8
    assert "TASK_REPLACING" in runner
