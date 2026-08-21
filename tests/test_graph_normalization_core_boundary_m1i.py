from __future__ import annotations

import ast
from pathlib import Path

import pipeline_core.corpus.schemas as legacy_schemas

from pipeline_core.corpus.graph.knowledge_graph_schema import (
    KnowledgeGraph,
)
from pipeline_core.corpus.extraction.measurement_schema import (
    Condition,
)


_RUNTIME_FUNCTIONS = {
    "normalize_graph_vocabularies",
    "normalize_networkx_metric_vocabularies",
}


def _runtime_ast():
    source = Path(
        "pipeline_core/corpus/graph/graph_normalization_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    return {
        node.name: node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in _RUNTIME_FUNCTIONS
        )
    }


def test_legacy_schema_surface_reexports_core_condition_and_graph():
    assert legacy_schemas.Condition is Condition
    assert (
        legacy_schemas.KnowledgeGraph
        is KnowledgeGraph
    )


def test_normalization_runtime_registry_dependency_is_minimal():
    functions = _runtime_ast()

    registry_methods = set()

    for function in functions.values():
        for node in ast.walk(function):
            if not (
                isinstance(node, ast.Call)
                and isinstance(
                    node.func,
                    ast.Attribute,
                )
                and isinstance(
                    node.func.value,
                    ast.Name,
                )
                and node.func.value.id
                in {
                    "experiment_registry",
                    "metric_registry",
                }
            ):
                continue

            registry_methods.add(
                node.func.attr
            )

    assert registry_methods == {
        "canonical_or_unregistered",
        "resolve",
        "resolve_parameterized",
    }


def test_normalization_runtime_keeps_metric_refinement_as_external_seam():
    functions = _runtime_ast()

    for function_name in (
        "normalize_graph_vocabularies",
        "normalize_networkx_metric_vocabularies",
    ):
        function = functions[function_name]

        calls = {
            node.func.id
            for node in ast.walk(function)
            if (
                isinstance(node, ast.Call)
                and isinstance(
                    node.func,
                    ast.Name,
                )
            )
        }

        assert (
            "metric_refiner"
            in calls
        )


def test_normalization_runtime_does_not_need_registry_construction_api():
    functions = _runtime_ast()

    rendered = "\n".join(
        ast.unparse(function)
        for function in functions.values()
    )

    forbidden = (
        "from_yaml",
        "load_default_registries",
        "prompt_lines",
        "compiled_patterns",
        "alias_map",
    )

    for token in forbidden:
        assert token not in rendered
