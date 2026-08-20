from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

ENTRY_MODULE = "scripts.run_dac_discovery_e2e"

EXPECTED_DYNAMIC_STAGES = {
    "scripts.build_demo_viewer",
    "scripts.build_discovery_bundle",
    "scripts.build_dual_hypothesis_context",
    "scripts.build_explorer_packet",
    "scripts.build_hypothesis_context",
    "scripts.run_candidate_unit_traversal",
    "scripts.run_discovery_axis_hypothesis_maker",
    "scripts.run_external_novelty",
    "scripts.run_feasibility_e2e",
    "scripts.run_graph_explorer",
    "scripts.run_graph_traversal",
    "scripts.run_hypothesis_semantic_critic",
    "scripts.run_novelty_refinement",
}

EXPECTED_DOMAIN_SURFACE = {
    "pipeline_core.domain_profile",
    "pipeline_core.feasibility_domain",
    'domains.registry',
    "dac_her.domains.extraction_registry",
    'domains.feasibility_registry',
}

CAMPAIGN_MODULE_PATTERN = re.compile(
    r"(fresh_c|reserve_[abc]|holdout|closeout|freeze)",
    re.IGNORECASE,
)


def _resolve_module(
    module: str,
) -> Path | None:
    candidate = (
        ROOT
        / Path(*module.split("."))
    ).with_suffix(".py")

    if candidate.is_file():
        return candidate.resolve()

    package = (
        ROOT
        / Path(*module.split("."))
        / "__init__.py"
    )

    if package.is_file():
        return package.resolve()

    return None


def _module_for_path(
    path: Path,
) -> str:
    relative = path.resolve().relative_to(ROOT)

    parts = list(
        relative.with_suffix("").parts
    )

    if parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


def _references_from_file(
    path: Path,
) -> set[str]:
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    current_module = _module_for_path(path)

    current_parts = (
        current_module.split(".")
        if current_module
        else []
    )

    references: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                references.add(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""

            if node.level:
                base = current_parts[:-1]
                trim = node.level - 1

                if trim:
                    base = (
                        base[:-trim]
                        if trim <= len(base)
                        else []
                    )

                module = ".".join(
                    base
                    + (
                        module.split(".")
                        if module
                        else []
                    )
                )

            if module:
                references.add(module)

            for alias in node.names:
                if alias.name == "*":
                    continue

                joined = (
                    f"{module}.{alias.name}"
                    if module
                    else alias.name
                )

                if _resolve_module(joined):
                    references.add(joined)

        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith(
                (
                    "scripts.",
                    "dac_her.",
                    "pipeline_core.",
                )
            )
            and _resolve_module(node.value)
        ):
            references.add(node.value)

    return references


def _execution_closure() -> set[str]:
    entry = _resolve_module(
        ENTRY_MODULE
    )

    assert entry is not None

    queue = [entry]
    seen: set[Path] = set()

    while queue:
        path = queue.pop(0)

        if path in seen:
            continue

        seen.add(path)

        for reference in sorted(
            _references_from_file(path)
        ):
            target = _resolve_module(reference)

            if (
                target is not None
                and target not in seen
            ):
                queue.append(target)

    return {
        _module_for_path(path)
        for path in seen
    }


def _resolved_entrypoint_script_literals() -> set[str]:
    entry = _resolve_module(
        ENTRY_MODULE
    )

    assert entry is not None

    tree = ast.parse(
        entry.read_text(
            encoding="utf-8"
        )
    )

    return {
        node.value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("scripts.")
            and _resolve_module(node.value)
        )
    }


def test_e2e_dynamic_stage_capabilities_are_preserved():
    assert (
        _resolved_entrypoint_script_literals()
        == EXPECTED_DYNAMIC_STAGES
    )


def test_e2e_required_domain_surface_is_reachable():
    closure = _execution_closure()

    assert (
        EXPECTED_DOMAIN_SURFACE
        <= closure
    )


def test_e2e_execution_closure_has_no_campaign_modules():
    closure = _execution_closure()

    leaked = sorted(
        module
        for module in closure
        if CAMPAIGN_MODULE_PATTERN.search(module)
    )

    assert leaked == []


def test_e2e_execution_closure_has_no_fresh_c_evaluation_path():
    closure = _execution_closure()

    matches: list[
        tuple[str, int, str]
    ] = []

    for module in sorted(closure):
        path = _resolve_module(module)

        assert path is not None

        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
            filename=str(path),
        )

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
            ):
                continue

            normalized = (
                node.value
                .replace("\\", "/")
                .lower()
            )

            if (
                "evaluation/sers_fresh_c"
                in normalized
            ):
                matches.append(
                    (
                        module,
                        node.lineno,
                        node.value,
                    )
                )

    assert matches == []


def test_pipeline_core_does_not_import_dac_her():
    violations: list[str] = []

    for path in sorted(
        (ROOT / "pipeline_core").rglob("*.py")
    ):
        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
            filename=str(path),
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""

                if (
                    module == "dac_her"
                    or module.startswith(
                        "dac_her."
                    )
                ):
                    violations.append(
                        f"{path.relative_to(ROOT)}:"
                        f"{node.lineno}:"
                        f"{module}"
                    )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.name == "dac_her"
                        or alias.name.startswith(
                            "dac_her."
                        )
                    ):
                        violations.append(
                            f"{path.relative_to(ROOT)}:"
                            f"{node.lineno}:"
                            f"{alias.name}"
                        )

    assert violations == []
