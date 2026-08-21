from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path

import domains.catalysis_mechanism.profile as canonical

from domains.registry import get_domain_profile


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PROFILE_SHA256 = (
    "978064836fb2cfe66b72ce72de9c6bdd"
    "5d7fbea8b8e8ac3105879ea2000556dc"
)


def _normalize(value):
    if is_dataclass(value):
        return {
            "__dataclass__": type(value).__qualname__,
            "fields": {
                field.name: _normalize(
                    getattr(value, field.name)
                )
                for field in fields(value)
            },
        }

    if isinstance(value, (frozenset, set)):
        items = [
            _normalize(item)
            for item in value
        ]

        items.sort(
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

        return {
            (
                "__frozenset__"
                if isinstance(value, frozenset)
                else "__set__"
            ): items
        }

    if isinstance(value, tuple):
        return {
            "__tuple__": [
                _normalize(item)
                for item in value
            ]
        }

    if isinstance(value, list):
        return [
            _normalize(item)
            for item in value
        ]

    if isinstance(value, dict):
        rows = [
            (
                _normalize(key),
                _normalize(item),
            )
            for key, item in value.items()
        ]

        rows.sort(
            key=lambda row: json.dumps(
                row[0],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

        return {
            "__dict__": [
                [key, item]
                for key, item in rows
            ]
        }

    if (
        value is None
        or isinstance(
            value,
            (bool, int, float, str),
        )
    ):
        return value

    raise TypeError(
        f"unsupported value: {type(value)!r}"
    )


def _profile_sha(value) -> str:
    payload = json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()




def test_registry_resolves_canonical_profile_object():
    assert (
        get_domain_profile(
            "catalysis_mechanism"
        )
        is canonical.CATALYSIS_MECHANISM_PROFILE
    )


def test_profile_semantics_are_unchanged():
    assert (
        _profile_sha(
            canonical.CATALYSIS_MECHANISM_PROFILE
        )
        == EXPECTED_PROFILE_SHA256
    )


def test_profile_still_uses_shared_contract_type():
    assert (
        type(
            canonical.CATALYSIS_MECHANISM_PROFILE
        ).__module__
        == "pipeline_core.domain.domain_profile"
    )


def test_projection_fallback_boundary_is_preserved():
    assert (
        canonical.CATALYSIS_MECHANISM_PROFILE.projection
        is None
    )


def test_canonical_namespace_does_not_import_legacy_dac():
    violations = []

    domain_root = (
        ROOT
        / "domains"
        / "catalysis_mechanism"
    )

    for path in sorted(
        domain_root.rglob("*.py")
    ):
        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
            filename=str(path),
        )

        for node in ast.walk(tree):
            names = []

            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]

            elif isinstance(node, ast.Import):
                names = [
                    alias.name
                    for alias in node.names
                ]

            for name in names:
                if (
                    name == "dac_her"
                    or name.startswith("dac_her.")
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{name}"
                    )

    assert violations == []
