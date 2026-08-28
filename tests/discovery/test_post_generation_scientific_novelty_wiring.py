from __future__ import annotations

import ast
from pathlib import Path


RUNTIME = Path(
    "pipeline_core/discovery/"
    "novelty_refinement_runtime.py"
)

CLI = Path(
    "scripts/discovery/"
    "run_novelty_refinement.py"
)


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    return [
        node.value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        )
    ]


def test_post_generation_evaluator_is_wired_to_both_candidate_paths():
    text = RUNTIME.read_text(
        encoding="utf-8"
    )

    assert (
        text.count(
            "evaluate_post_generation_scientific_novelty("
        )
        == 2
    )

    assert (
        "reaxis_post_generation_scientific_assessment"
        in text
    )

    assert (
        "refined_post_generation_scientific_assessment"
        in text
    )


def test_post_generation_rejection_reason_exists_for_both_paths():
    constants = _string_constants(
        RUNTIME
    )

    assert (
        constants.count(
            "post_generation_scientific_novelty_gate_rejected"
        )
        == 2
    )


def test_post_generation_gate_is_optional_and_cli_controlled():
    runtime = RUNTIME.read_text(
        encoding="utf-8"
    )

    cli = CLI.read_text(
        encoding="utf-8"
    )

    assert (
        "post_generation_scientific_novelty_backend"
        in runtime
    )

    assert (
        "--post-generation-scientific-novelty-enforce"
        in cli
    )

    assert (
        "OpenRouterSemanticDistinctivenessBackend"
        in cli
    )


def test_frozen_action_policy_remains_the_authority():
    helper = Path(
        "pipeline_core/discovery/"
        "post_generation_scientific_novelty.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "ScientificNoveltyActionPolicy"
        in helper
    )

    assert (
        "review_pass_index=1"
        in helper
    )

    assert (
        "review_pass_index=2"
        in helper
    )

    assert (
        '== "INELIGIBLE"'
        not in helper
    )
