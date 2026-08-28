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

    # The two previously bypassing direct branches now add two additional
    # authoritative fallback checks: 8 -> 10.
    assert (
        text.count(
            "scientific_gate_by_id=scientific_gate_by_id"
        )
        == 10
    )

    assert (
        text.count(
            'decision="scientific_novelty_rejected"'
        )
        == 2
    )

    tree = ast.parse(text)

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
