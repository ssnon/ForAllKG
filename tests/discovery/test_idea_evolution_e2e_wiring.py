import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "scripts" / "discovery" / "run_dac_discovery_e2e.py"


def _source() -> str:
    return E2E.read_text(encoding="utf-8")


def _strings() -> set[str]:
    tree = ast.parse(_source())
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_idea_evolution_stage_is_wired_after_frontier_and_before_stage8():
    text = _source()
    strings = _strings()
    assert "[7.72/13] Idea Evolution shadow" in strings
    assert "scripts.discovery.run_idea_evolution_shadow" in strings
    assert "--idea-evolution-shadow" in strings
    assert text.index("[7.71/13] Exploration Frontier factorization shadow") < text.index(
        "[7.72/13] Idea Evolution shadow"
    ) < text.index("stage8_axis_plan_input = _stage8_axis_plan_input(")


def test_idea_evolution_requires_frontier_and_does_not_rebind_stage8_input():
    tree = ast.parse(_source())
    assigned_names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned_names.append(target.id)
    assert "stage8_axis_plan_input" in assigned_names
    # The new shadow block must not assign the canonical task-conditioned plan variable.
    source = _source()
    marker = source.index("if args.idea_evolution_shadow:")
    stage8 = source.index("stage8_axis_plan_input = _stage8_axis_plan_input(")
    block = source[marker:stage8]
    assert "task_conditioned_axis_plan =" not in block
    assert "requires --frontier-idea-population-shadow" in block
