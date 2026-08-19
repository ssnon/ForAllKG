from __future__ import annotations

import ast
from pathlib import Path

from domains.sers.prompts import (
    SERS_MICRO_REEXTRACT_SYSTEM_PROMPT as CANONICAL_MICRO,
    SERS_PATCH_SYSTEM_PROMPT as CANONICAL_PATCH,
    SERS_PROMPT_VERSION as CANONICAL_VERSION,
    SERS_SYSTEM_PROMPT as CANONICAL_SYSTEM,
)
from dac_her.sers_prompts import (
    SERS_MICRO_REEXTRACT_SYSTEM_PROMPT as LEGACY_MICRO,
    SERS_PATCH_SYSTEM_PROMPT as LEGACY_PATCH,
    SERS_PROMPT_VERSION as LEGACY_VERSION,
    SERS_SYSTEM_PROMPT as LEGACY_SYSTEM,
)


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_version_matches_canonical() -> None:
    assert LEGACY_VERSION == CANONICAL_VERSION


def test_legacy_system_prompt_is_canonical() -> None:
    assert LEGACY_SYSTEM is CANONICAL_SYSTEM


def test_legacy_patch_prompt_is_canonical() -> None:
    assert LEGACY_PATCH is CANONICAL_PATCH


def test_legacy_micro_prompt_is_canonical() -> None:
    assert LEGACY_MICRO is CANONICAL_MICRO


def test_legacy_prompts_module_is_definition_free() -> None:
    path = ROOT / "dac_her" / "sers_prompts.py"

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    definitions = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
    ]

    assert definitions == []


def test_canonical_prompts_do_not_reverse_import_legacy() -> None:
    path = (
        ROOT
        / "domains"
        / "sers"
        / "prompts.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")

        elif isinstance(node, ast.Import):
            modules.extend(
                alias.name
                for alias in node.names
            )

    assert (
        "dac_her.sers_prompts"
        not in modules
    )
