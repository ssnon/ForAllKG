from __future__ import annotations

import ast
from pathlib import Path

import domains.dac_her.bridge_audit as canonical
import domains.dac_her.bridge_policy as policy
import domains.dac_her.scientific_signatures as signatures


ROOT = Path(__file__).resolve().parents[1]


AUDIT_OBJECTS = (
    "audit_bridge_graph",
    "write_bridge_audit",
)




def test_audit_objects_are_domain_owned():
    for name in AUDIT_OBJECTS:
        assert (
            getattr(canonical, name).__module__
            == "domains.dac_her.bridge_audit"
        )


def test_canonical_audit_binds_canonical_domain_dependencies():
    assert (
        canonical.BRIDGE_POLICY_VERSION
        == policy.BRIDGE_POLICY_VERSION
    )

    assert (
        canonical.normalize_scientific_text
        is signatures.normalize_scientific_text
    )


def test_canonical_audit_has_no_legacy_reverse_dependency():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "bridge_audit.py"
    )

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    violations = []

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
                or name == "campaigns"
                or name.startswith("campaigns.")
            ):
                violations.append(
                    f"{path}:{node.lineno}:{name}"
                )

    assert violations == []


