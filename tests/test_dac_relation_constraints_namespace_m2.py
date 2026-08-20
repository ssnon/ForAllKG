from __future__ import annotations

import ast
from pathlib import Path

from domains.dac_her.relation_constraints import (
    DAC_HER_STRICT_RELATION_CONSTRAINTS as CANONICAL_DAC,
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS as CANONICAL_LEGACY,
)

from pipeline_core.evidence_relation_constraints import (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
)

from domains.dac_her.relation_constraints import (
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS as CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS,
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS as LEGACY_DAC,
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS as LEGACY_LEGACY,
)
from pipeline_core.evidence_relation_constraints import COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS as SERS_AU_AG_STRICT_RELATION_CONSTRAINTS



ROOT = Path(__file__).resolve().parents[1]


def test_legacy_dac_policy_is_canonical_policy():
    assert LEGACY_DAC is CANONICAL_DAC


def test_legacy_named_policy_identity_is_preserved():
    assert LEGACY_LEGACY is CANONICAL_LEGACY
    assert CANONICAL_DAC is CANONICAL_LEGACY


def test_catalysis_historical_alias_identity_is_preserved():
    assert (
        CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS
        is CANONICAL_LEGACY
    )


def test_sers_still_uses_shared_core_policy():
    assert (
        SERS_AU_AG_STRICT_RELATION_CONSTRAINTS
        is COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
    )




def test_dac_policy_reuses_shared_core_objects():
    embedded = CANONICAL_DAC[5:11]

    assert all(
        left is right
        for left, right in zip(
            embedded,
            COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
        )
    )


def test_canonical_domain_policy_has_no_legacy_import():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "relation_constraints.py"
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
