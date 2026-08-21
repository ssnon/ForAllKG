from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _imports(relative_path: str) -> set[str]:
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))

    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)

    return modules


def test_strict_extraction_runtime_has_no_domain_imports() -> None:
    imports = _imports(
        "scripts/corpus/strict_extraction_runtime.py"
    )

    assert not {
        module
        for module in imports
        if module == "domains"
        or module.startswith("domains.")
    }


def test_extract_paper_has_no_direct_scientific_domain_imports() -> None:
    imports = _imports(
        "scripts/corpus/extract_paper.py"
    )

    scientific_prefixes = (
        "domains.dac_her",
        "domains.sers",
        "domains.catalysis_mechanism",
    )

    assert not {
        module
        for module in imports
        if module.startswith(scientific_prefixes)
    }



def test_sers_and_broad_do_not_directly_import_dac_prompt_modules() -> None:
    forbidden_prompt_modules = {
        "domains.dac_her.prompts",
        "domains.dac_her.semantic_patch_prompts",
        "domains.dac_her.micro_reextract_prompts",
    }

    for relative_path in (
        "domains/sers/extraction.py",
        "domains/catalysis_mechanism/extraction.py",
    ):
        imports = _imports(relative_path)

        assert not (
            imports
            & forbidden_prompt_modules
        )
