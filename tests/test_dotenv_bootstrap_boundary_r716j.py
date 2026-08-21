from __future__ import annotations

import ast
from pathlib import Path


CORE = Path("pipeline_core/openrouter_llm.py")
PAPER = Path("scripts/extract_paper.py")
BRIDGE = Path("scripts/extract_bridge_graph.py")


def _tree(path: Path) -> ast.Module:
    return ast.parse(
        path.read_text(encoding="utf-8")
    )


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _module_scope_load_dotenv_calls(
    path: Path,
) -> list[ast.Call]:
    tree = _tree(path)

    return [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and _call_name(node.value) == "load_dotenv"
        )
        for node in [node.value]
    ]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = _tree(path)

    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == name
        ):
            return node

    raise AssertionError(
        f"function not found: {path}:{name}"
    )


def test_generic_openrouter_core_has_no_dotenv_dependency():
    source = CORE.read_text(encoding="utf-8")

    assert "from dotenv import" not in source
    assert "import dotenv" not in source
    assert "load_dotenv" not in source


def test_extraction_scripts_have_no_module_scope_dotenv_call():
    assert _module_scope_load_dotenv_calls(PAPER) == []
    assert _module_scope_load_dotenv_calls(BRIDGE) == []


def test_extract_paper_loads_dotenv_at_parse_bootstrap():
    function = _function(PAPER, "parse_args")

    first = function.body[0]

    assert isinstance(first, ast.Expr)
    assert isinstance(first.value, ast.Call)
    assert _call_name(first.value) == "load_dotenv"


def test_extract_bridge_loads_dotenv_at_parse_bootstrap():
    function = _function(BRIDGE, "parse_args")

    first = function.body[0]

    assert isinstance(first, ast.Expr)
    assert isinstance(first.value, ast.Call)
    assert _call_name(first.value) == "load_dotenv"
