from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _parse(path: str) -> ast.Module:
    return ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)


def _literal_dict_assignment(tree: ast.Module, name: str) -> dict[str, str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError(f"{name} must remain a literal dict for this freeze guard")
        out: dict[str, str] = {}
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    out[key.value] = value.value
        return out
    raise AssertionError(f"assignment {name} not found")


def test_sers_au_ag_profile_id_and_aliases_remain_registered() -> None:
    sers_tree = _parse("dac_her/domains/sers_au_ag.py")
    profile_ids: list[str] = []
    for node in ast.walk(sers_tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "profile_id"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                profile_ids.append(keyword.value.value)
    assert "sers_au_ag" in profile_ids

    registry_tree = _parse("dac_her/domains/registry.py")
    aliases = _literal_dict_assignment(registry_tree, "_ALIASES")
    assert aliases["sers"] == "sers_au_ag"
    assert aliases["au-ag-sers"] == "sers_au_ag"
    assert aliases["sers-au-ag"] == "sers_au_ag"


def test_prior_art_matching_default_is_not_sers_so_runner_must_bind_profile() -> None:
    tree = _parse("dac_her/prior_art_matching.py")
    default_profile_calls: list[str] = []
    fallback_to_default = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_domain_profile"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            default_profile_calls.append(node.args[0].value)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            names = {
                child.id
                for child in node.values
                if isinstance(child, ast.Name)
            }
            if "domain_profile" in names and "_DEFAULT_DOMAIN_PROFILE" in names:
                fallback_to_default = True

    assert "dac_her" in default_profile_calls
    assert fallback_to_default is True


def test_r0_reducer_does_not_import_absence_threshold_router() -> None:
    tree = _parse("dac_her/r0_runtime.py")
    imported_modules: set[str] = set()
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)
            imported_names.update(alias.name for alias in node.names)

    assert "dac_her.external_novelty" not in imported_modules
    assert "ExternalNoveltyPolicy" not in imported_names
    assert "ExternalNoveltyAssessor" not in imported_names


def test_r0_source_contains_no_absence_count_threshold_names() -> None:
    source = (ROOT / "dac_her/r0_runtime.py").read_text(encoding="utf-8")
    forbidden = {
        "min_unique_works_for_absence",
        "min_abstract_works_for_absence",
        "min_abstract_works_per_core_claim",
        "min_successful_queries_for_absence",
        "sufficient_for_absence_based_novelty",
    }
    for name in forbidden:
        assert name not in source
