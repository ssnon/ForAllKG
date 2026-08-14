from __future__ import annotations

import ast
import inspect

import dac_her.corpus_acquisition.materialization_package as package_module
import scripts.materialize_corpus_documents as materialize_script


def _extract_paper_subprocess_calls(source: str) -> list[int]:
    tree = ast.parse(source)
    targets = {
        "scripts.extract_paper",
        "scripts/extract_paper.py",
        "extract_paper.py",
    }
    subprocess_calls = {"run", "call", "check_call", "check_output", "Popen"}
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = None
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ):
            call_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id
        if call_name not in subprocess_calls:
            continue
        command_nodes = list(node.args)
        command_nodes.extend(
            keyword.value
            for keyword in node.keywords
            if keyword.arg in {"args", "command", "cmd"}
        )
        if any(
            any(
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and child.value in targets
                for child in ast.walk(command_node)
            )
            for command_node in command_nodes
        ):
            hits.append(node.lineno)
    return hits


def test_materialization_package_serializes_extract_paper_handoff():
    source = inspect.getsource(package_module)
    assert '"scripts.extract_paper"' in source


def test_m4_never_executes_extract_paper():
    assert _extract_paper_subprocess_calls(
        inspect.getsource(package_module)
    ) == []
    assert _extract_paper_subprocess_calls(
        inspect.getsource(materialize_script)
    ) == []
