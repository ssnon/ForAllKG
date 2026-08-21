from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path

import domains.dac_her.profile as canonical

from domains.registry import (
    get_domain_profile,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PROFILE_SHA256 = (
    "8c13fd69baadcd29e07f895624f7bffe"
    "686ef52cf9e215ba078881a0bbc4ddda"
)


def _normalize(value):
    if is_dataclass(value):
        return {
            "__dataclass__":
                type(value).__qualname__,
            "fields": {
                field.name: _normalize(
                    getattr(value, field.name)
                )
                for field in fields(value)
            },
        }

    if isinstance(value, frozenset):
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
            "__frozenset__": items,
        }

    if isinstance(value, set):
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
            "__set__": items,
        }

    if isinstance(value, tuple):
        return {
            "__tuple__": [
                _normalize(item)
                for item in value
            ],
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
            ],
        }

    if value is None or isinstance(
        value,
        (bool, int, float, str),
    ):
        return value

    raise TypeError(
        f"unsupported value: {type(value)!r}"
    )


def _profile_sha256(value) -> str:
    canonical_json = json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()




def test_registry_resolves_canonical_profile_object():
    assert (
        get_domain_profile("dac_her")
        is canonical.DAC_HER_PROFILE
    )


def test_dac_profile_semantics_are_unchanged():
    assert (
        _profile_sha256(
            canonical.DAC_HER_PROFILE
        )
        == EXPECTED_PROFILE_SHA256
    )


def test_canonical_profile_uses_shared_profile_contract():
    assert (
        type(
            canonical.DAC_HER_PROFILE
        ).__module__
        == "pipeline_core.domain.domain_profile"
    )


def test_new_domain_namespace_does_not_import_legacy_dac_package():
    violations: list[str] = []

    domain_root = (
        ROOT
        / "domains"
        / "dac_her"
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
            names: list[str] = []

            if isinstance(
                node,
                ast.ImportFrom,
            ):
                names = [
                    node.module or ""
                ]

            elif isinstance(
                node,
                ast.Import,
            ):
                names = [
                    alias.name
                    for alias in node.names
                ]

            for name in names:
                if (
                    name == "dac_her"
                    or name.startswith(
                        "dac_her."
                    )
                ):
                    violations.append(
                        f"{path.relative_to(ROOT)}:"
                        f"{node.lineno}:"
                        f"{name}"
                    )

    assert violations == []
