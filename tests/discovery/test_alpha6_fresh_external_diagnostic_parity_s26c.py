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
RUNNER = (
    REPO
    / "scripts"
    / "discovery"
    / "run_novelty_refinement.py"
)


def _function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    rows = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.name == name
    ]
    assert len(rows) == 1
    return ast.get_source_segment(text, rows[0]) or ""


def test_alpha6_fresh_external_executes_diagnostic_lane() -> None:
    source = _function_source(RUNTIME, "_fresh_external")

    assert "build_diagnostic_query_plan(plan)" in source
    assert "review_diagnostic_prior_art" in source
    assert "diagnostic_plan=" in source
    assert "diagnostic_packet=" in source
    assert "diagnostic_reviews=" in source


def test_alpha6_diagnostic_lane_does_not_merge_main_packet() -> None:
    source = _function_source(RUNTIME, "_fresh_external")

    assert (
        "packet = self.targeted_retriever.retriever.retrieve(plan).packet"
        in source
    )
    assert "packet = diagnostic_packet" not in source
    assert "merged_packet" not in source


def test_final_artifacts_serialize_diagnostic_sidecars() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert '".diagnostic_queries.json"' in text
    assert '".diagnostic_prior_art.json"' in text
    assert '".diagnostic_review.json"' in text


def test_authority_comment_is_explicit() -> None:
    text = RUNTIME.read_text(encoding="utf-8")

    assert "absence-based coverage sufficient" in text
