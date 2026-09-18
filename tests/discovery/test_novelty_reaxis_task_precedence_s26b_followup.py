from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNTIME = (
    REPO
    / "pipeline_core"
    / "discovery"
    / "novelty_refinement_runtime.py"
)

TOK_EXTERNAL = "REAXIS_REJECT_EXTERNAL"
TOK_EVIDENCE = "_reaxis_replacement_evidence_ready"
TOK_HOLD = "held_for_evidence"
TOK_TASK = "reaxis_task_assessment"
TOK_TASK_DECISION = "question_task_rejected"


def _segment(text: str, node: ast.AST) -> str:
    return ast.get_source_segment(text, node) or ""


def _target_run(
    text: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(text)
    lines = text.splitlines()

    matches = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        if node.name != "run":
            continue

        source = "\n".join(
            lines[node.lineno - 1: node.end_lineno]
        )
        if all(
            token in source
            for token in (
                TOK_EXTERNAL,
                TOK_EVIDENCE,
                TOK_HOLD,
                TOK_TASK_DECISION,
            )
        ):
            matches.append(node)

    assert len(matches) == 1
    return matches[0]


def _decision_chain(
    text: str,
) -> tuple[ast.If, ast.If, ast.If]:
    run = _target_run(text)

    candidates = []
    for node in ast.walk(run):
        if not isinstance(node, ast.If):
            continue

        condition = _segment(text, node.test)
        whole_chain = _segment(text, node)

        if (
            TOK_EXTERNAL in condition
            and TOK_EVIDENCE in whole_chain
            and TOK_HOLD in whole_chain
            and TOK_TASK_DECISION in whole_chain
        ):
            candidates.append(node)

    assert len(candidates) == 1
    external = candidates[0]

    assert len(external.orelse) == 1
    first = external.orelse[0]
    assert isinstance(first, ast.If)

    assert len(first.orelse) == 1
    second = first.orelse[0]
    assert isinstance(second, ast.If)

    return external, first, second


def test_task_rejection_precedes_evidence_hold() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    external, first, second = _decision_chain(text)

    first_condition = _segment(text, first.test)
    first_body = "\n".join(
        _segment(text, node) for node in first.body
    )
    second_condition = _segment(text, second.test)
    second_body = "\n".join(
        _segment(text, node) for node in second.body
    )

    # The first sibling after explicit external rejection must be
    # task preservation rejection.
    assert TOK_TASK in first_condition
    assert TOK_TASK_DECISION in first_body

    # Evidence insufficiency is subordinate: it is checked only after
    # task preservation has passed.
    assert TOK_EVIDENCE in second_condition
    assert TOK_HOLD in second_body

    assert external.lineno < first.lineno < second.lineno


def test_external_rejection_still_precedes_task_rejection() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    external, first, _ = _decision_chain(text)

    assert TOK_EXTERNAL in _segment(text, external.test)
    assert TOK_TASK in _segment(text, first.test)
    assert external.lineno < first.lineno
