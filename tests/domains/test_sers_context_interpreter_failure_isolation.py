from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ADAPTER = REPO / "domains" / "sers" / "context_review_adapter.py"
RUNTIME = (
    REPO
    / "pipeline_core"
    / "discovery"
    / "discovery_axis_runtime.py"
)


def _review_function():
    text = ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(text)

    matches = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        if node.name != "review":
            continue

        source = ast.get_source_segment(text, node) or ""
        if (
            "self.interpreter.interpret" in source
            and "source_signatures" in source
        ):
            matches.append(node)

    assert len(matches) == 1
    return text, matches[0]


def test_interpreter_validation_failure_is_axis_local_unavailable():
    text, review = _review_function()

    matching = []
    for handler in ast.walk(review):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        if handler.type is None:
            continue

        type_text = ast.get_source_segment(text, handler.type) or ""
        body_text = "\n".join(
            ast.get_source_segment(text, row) or ""
            for row in handler.body
        )

        if (
            "HypothesisContextInterpreterValidationError" in type_text
            and "AxisContextReviewUnavailableError" in body_text
        ):
            matching.append(handler)

    assert len(matching) == 1


def test_runtime_isolates_unavailable_context_as_context_rejected():
    text = RUNTIME.read_text(encoding="utf-8")

    assert "except AxisContextReviewUnavailableError as exc:" in text
    assert 'decision="context_rejected"' in text


def test_strict_span_and_nonsemantic_canonicalization_contract_remain():
    contracts = (
        REPO
        / "domains"
        / "sers"
        / "hypothesis_context_contracts.py"
    ).read_text(encoding="utf-8")

    interpreter = (
        REPO
        / "domains"
        / "sers"
        / "hypothesis_context_interpreter.py"
    ).read_text(encoding="utf-8")

    assert "mention_text is not an exact " in contracts
    assert "assertion span after whitespace " in contracts
    assert (
        "Canonicalize orchestration/citation noise without semantic repair."
        in interpreter
    )
