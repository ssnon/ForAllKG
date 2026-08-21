from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_root_parent_depth(
    path: Path,
) -> list[tuple[int, int]]:
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    rows: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if len(node.targets) != 1:
            continue

        target = node.targets[0]

        if (
            not isinstance(target, ast.Name)
            or target.id != "PROJECT_ROOT"
        ):
            continue

        # Match:
        # Path(__file__).resolve().parents[N]
        value = node.value

        if not isinstance(value, ast.Subscript):
            continue

        parents_attr = value.value

        if (
            not isinstance(parents_attr, ast.Attribute)
            or parents_attr.attr != "parents"
        ):
            continue

        slice_node = value.slice

        if not isinstance(slice_node, ast.Constant):
            continue

        if not isinstance(slice_node.value, int):
            continue

        rows.append(
            (
                node.lineno,
                slice_node.value,
            )
        )

    return rows


def test_script_project_roots_resolve_repository_root():
    violations = []

    for path in sorted(
        (REPO_ROOT / "scripts").rglob("*.py")
    ):
        for lineno, depth in _project_root_parent_depth(path):
            resolved = (
                path.resolve()
                .parents[depth]
            )

            if resolved != REPO_ROOT:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:"
                    f"{lineno}: parents[{depth}] "
                    f"resolves to {resolved}, "
                    f"expected {REPO_ROOT}"
                )

    assert violations == []


def test_repository_root_has_expected_runtime_resources():
    assert (REPO_ROOT / "configs").is_dir()
    assert (REPO_ROOT / "domains").is_dir()
    assert (REPO_ROOT / "pipeline_core").is_dir()
