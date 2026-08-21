from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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
    sers_tree = _parse('domains/sers/profile.py')
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

    registry_tree = _parse('domains/registry.py')
    aliases = _literal_dict_assignment(registry_tree, "_ALIASES")
    assert aliases["sers"] == "sers_au_ag"
    assert aliases["au-ag-sers"] == "sers_au_ag"
    assert aliases["sers-au-ag"] == "sers_au_ag"


def test_prior_art_matching_requires_explicit_domain_profile_binding() -> None:
    tree = _parse("pipeline_core/discovery/prior_art_matching.py")

    # Core scientific matching must not resolve a hidden domain default.
    registry_calls: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_domain_profile"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            registry_calls.append(node.args[0].value)

    assert registry_calls == []

    # Both low-level scientific components require an explicit profile.
    for class_name in ("PriorArtRanker", "ClaimPriorArtCompiler"):
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == class_name
        )
        init = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
        )

        kwonly_names = [
            arg.arg
            for arg in init.args.kwonlyargs
        ]
        assert "domain_profile" in kwonly_names

        index = kwonly_names.index("domain_profile")
        assert init.args.kw_defaults[index] is None

    # Application composition must bind the selected domain explicitly.
    for runner_path in (
        "scripts/discovery/run_external_novelty.py",
        "scripts/discovery/run_novelty_refinement.py",
    ):
        runner = _parse(runner_path)

        seen = {
            "PriorArtRanker": 0,
            "ClaimPriorArtCompiler": 0,
        }

        for node in ast.walk(runner):
            if not isinstance(node, ast.Call):
                continue

            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue

            if name not in seen:
                continue

            seen[name] += 1

            assert any(
                keyword.arg == "domain_profile"
                for keyword in node.keywords
            ), (
                f"{runner_path}: {name} must bind "
                "domain_profile explicitly"
            )

        assert seen == {
            "PriorArtRanker": 1,
            "ClaimPriorArtCompiler": 1,
        }


