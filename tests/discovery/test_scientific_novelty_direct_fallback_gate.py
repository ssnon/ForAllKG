from __future__ import annotations

import ast
from pathlib import Path

from pydantic import TypeAdapter

from pipeline_core.discovery.novelty_refinement_contracts import (
    RefinementDecision,
)


def test_scientific_novelty_rejection_is_explicit_decision():
    adapter = TypeAdapter(
        RefinementDecision
    )

    assert (
        adapter.validate_python(
            "scientific_novelty_rejected"
        )
        == "scientific_novelty_rejected"
    )


def test_all_direct_kept_original_paths_are_gate_aware():
    path = Path(
        "pipeline_core/discovery/"
        "novelty_refinement_runtime.py"
    )

    text = path.read_text(
        encoding="utf-8"
    )

    # Every original-fallback authorization call must be explicitly
    # scientific-gate-aware. Do not freeze the number of call sites:
    # legitimate new fallback paths may be added over time.
    tree = ast.parse(text)

    fallback_calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_original_fallback_allowed"
        )
    ]

    assert fallback_calls

    for call in fallback_calls:
        keyword_names = {
            keyword.arg
            for keyword in call.keywords
            if keyword.arg is not None
        }
        assert "scientific_gate_by_id" in keyword_names

    # This test owns only the two original-fallback bypass closures.
    # Post-generation scientific novelty adds additional legitimate
    # scientific_novelty_rejected paths, so a global decision count is
    # no longer a valid invariant.
    assert (
        text.count(
            "scientific_novelty_gate_blocked_original_fallback"
        )
        == 2
    )

    # Sanity: runtime remains syntactically parseable and the two direct
    # branches still exist rather than being removed.
    assert any(
        isinstance(node, ast.If)
        for node in ast.walk(tree)
    )

    assert (
        'if gap.action == "keep":'
        in text
    )

    assert (
        "in self.RESOLVED_CANDIDATE_EXTERNAL"
        in text
    )
