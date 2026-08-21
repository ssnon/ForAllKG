from __future__ import annotations

import ast
from pathlib import Path

from domains.catalysis_mechanism.compact_schema import (
    BROAD_COMPACT_SCHEMA_ID,
)
from domains.extraction_registry import get_extraction_adapter


ROOT = Path(__file__).resolve().parents[2]


def test_broad_compact_schema_is_domain_owned_and_adapter_exposed() -> None:
    old_path = (
        ROOT
        / "pipeline_core"
        / "corpus"
        / "broad_compact_schema.py"
    )

    new_path = (
        ROOT
        / "domains"
        / "catalysis_mechanism"
        / "compact_schema.py"
    )

    assert not old_path.exists()
    assert new_path.exists()

    adapter = get_extraction_adapter(
        "catalysis_mechanism"
    )

    assert (
        adapter.compact_generation_schema_id
        == BROAD_COMPACT_SCHEMA_ID
    )

    assert (
        adapter.compact_domain_gate_recovery_schema_id
        == BROAD_COMPACT_SCHEMA_ID
    )

    assert {
        Path(path).resolve()
        for path
        in adapter.compact_response_model_implementation_paths()
    } == {
        new_path.resolve()
    }

    script = (
        ROOT
        / "scripts"
        / "corpus"
        / "extract_paper.py"
    )

    script_text = script.read_text(
        encoding="utf-8"
    )

    assert (
        "broad_compact_schema_module"
        not in script_text
    )

    tree = ast.parse(script_text)

    direct_domain_imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            direct_domain_imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith(
                    "domains.catalysis_mechanism"
                )
            )

        elif isinstance(node, ast.ImportFrom):
            if (
                node.module
                and node.module.startswith(
                    "domains.catalysis_mechanism"
                )
            ):
                direct_domain_imports.add(
                    node.module
                )

    assert direct_domain_imports == set()
