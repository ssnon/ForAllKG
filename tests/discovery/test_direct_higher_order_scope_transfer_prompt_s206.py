from __future__ import annotations

import ast
from pathlib import Path


RUNTIME = Path(
    "pipeline_core/discovery/"
    "direct_higher_order_hypothesis_runtime.py"
)


def _runtime_string_constants() -> str:
    tree = ast.parse(
        RUNTIME.read_text(encoding="utf-8")
    )

    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    ]

    return "\n".join(values)


def test_s206_prompt_requires_explicit_cross_system_scope_transfer():
    strings = _runtime_string_constants()

    assert "Preserve SYSTEM AND MATERIAL SCOPE" in strings
    assert (
        "state that difference explicitly before using the premise "
        "as analogical motivation"
        in strings
    )


def test_s206_prompt_blocks_cross_system_fact_attribution_to_modifier():
    strings = _runtime_string_constants()

    assert (
        "attributes a cross-system mechanism to modifier C unless"
        in strings
    )
    assert (
        "for the same modifier identity and system"
        in strings
    )


def test_s206_prompt_requires_reported_fact_and_new_hypothesis_separation():
    strings = _runtime_string_constants()

    assert (
        "first state what the other system reported"
        in strings
    )
    assert (
        "applying an analogous mechanism to C is the NEW HYPOTHESIS"
        in strings
    )
    assert (
        "Do not compress those two epistemic steps into one reported "
        "composite claim"
        in strings
    )


def test_s206_does_not_promote_candidate_mechanism_to_fact():
    strings = _runtime_string_constants()

    assert (
        "does not authorize a candidate-specific mechanism as "
        "reported fact"
        in strings
    )
