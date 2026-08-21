from __future__ import annotations

import ast
from pathlib import Path


FILES = (
    Path(
        "scripts/corpus/"
        "strict_extraction_runtime.py"
    ),
    Path(
        "pipeline_core/corpus/"
        "domain_gate_replay.py"
    ),
    Path(
        "scripts/corpus/"
        "extract_paper.py"
    ),
)

TARGETS = {
    "validate_draft",
    "finalize_draft",
    "_finalize_and_save",
    "load_existing_result",
}


def _name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id

    if isinstance(node.func, ast.Attribute):
        return node.func.attr

    return None


def test_relation_contract_calls_also_receive_semantic_contract():
    missing = []

    for path in FILES:
        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
            filename=str(path),
        )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            if _name(node) not in TARGETS:
                continue

            keywords = {
                kw.arg
                for kw in node.keywords
                if kw.arg is not None
            }

            if (
                "relation_constraints"
                not in keywords
            ):
                continue

            if (
                "semantic_issue_collector"
                not in keywords
            ):
                missing.append(
                    (
                        str(path),
                        node.lineno,
                        _name(node),
                    )
                )

    assert missing == []


def test_run_fingerprint_binds_semantic_contract_and_implementation():
    text = Path(
        "scripts/corpus/extract_paper.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        '"strict_semantic_contract"'
        in text
    )

    assert (
        ".strict_semantic_contract_payload()"
        in text
    )

    assert (
        "semantic_collector_impl_path"
        in text
    )

    assert (
        "inspect.getsourcefile("
        in text
    )
