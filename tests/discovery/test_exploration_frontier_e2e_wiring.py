from __future__ import annotations

import ast
from pathlib import Path


def _source() -> str:
    return (
        Path("scripts/discovery/run_dac_discovery_e2e.py")
        .read_text(encoding="utf-8")
    )


def _string_constants(text: str) -> set[str]:
    tree = ast.parse(text)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def _assigns_args_accepted_patterns(text: str) -> bool:
    tree = ast.parse(text)

    def is_target(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "accepted_patterns"
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(is_target(target) for target in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            if is_target(node.target):
                return True
        elif isinstance(node, ast.AugAssign):
            if is_target(node.target):
                return True

    return False


def test_frontier_factorization_runs_before_stage8_input_selection():
    text = _source()
    audit = text.index(
        "[7.71/13] Exploration Frontier factorization shadow"
    )
    stage8 = text.index(
        "stage8_axis_plan_input = _stage8_axis_plan_input("
    )
    assert audit < stage8


def test_higher_order_shadow_can_auto_export_canonical_relationpatterns():
    text = _source()
    strings = _string_constants(text)

    # The production source intentionally uses adjacent string literals for
    # line-length/readability. Python concatenates them at parse time, so the
    # wiring test must inspect parsed string constants rather than raw source
    # substrings.
    assert (
        "[7.545/13] Canonical accepted RelationPattern export shadow"
        in strings
    )
    assert (
        "scripts.discovery.export_accepted_relationpatterns"
        in strings
    )
    assert "higher_order_accepted_patterns" in text
    assert (
        "--higher-order-shadow requires --accepted-patterns"
        not in text
    )


def test_auto_export_does_not_assign_args_accepted_patterns():
    text = _source()
    assert not _assigns_args_accepted_patterns(text)
