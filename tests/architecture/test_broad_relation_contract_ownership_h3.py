from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from domains.catalysis_mechanism.relation_constraints import (
    CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS,
)
from domains.dac_her.relation_constraints import (
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS,
)
from domains.extraction_registry import get_extraction_adapter
from pipeline_core.corpus.extraction.evidence_relation_constraints import (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
)


ROOT = Path(__file__).resolve().parents[2]

FROZEN_PRE_H3_PAYLOAD_SHA256 = (
    "1831a7484268151e78a276e01e434467335aa84045d3ae50648c4b73f963a8e8"
)


def _payload_sha(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def test_broad_relation_contract_is_domain_owned_without_cross_domain_imports():
    broad_root = (
        ROOT
        / "domains"
        / "catalysis_mechanism"
    )

    violations = []

    for path in broad_root.rglob("*.py"):
        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            )
        )

        for node in ast.walk(tree):
            modules = []

            if isinstance(node, ast.Import):
                modules = [
                    alias.name
                    for alias in node.names
                ]

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules = [
                        node.module
                    ]

            for module in modules:
                if module.startswith(
                    (
                        "domains.dac_her",
                        "domains.sers",
                    )
                ):
                    violations.append(
                        f"{path.relative_to(ROOT)}:"
                        f"{node.lineno}:{module}"
                    )

    assert violations == []


def test_broad_and_dac_contract_objects_are_independent():
    assert (
        CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS
        is not DAC_LEGACY_STRICT_RELATION_CONSTRAINTS
    )

    broad_adapter = get_extraction_adapter(
        "catalysis_mechanism"
    )

    dac_adapter = get_extraction_adapter(
        "dac_her"
    )

    assert (
        broad_adapter.strict_relation_constraints
        is CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS
    )

    assert (
        dac_adapter.strict_relation_constraints
        is not broad_adapter.strict_relation_constraints
    )


def test_broad_relation_payload_remains_exactly_frozen_pre_h3_semantics():
    broad = get_extraction_adapter(
        "catalysis_mechanism"
    )

    dac = get_extraction_adapter(
        "dac_her"
    )

    broad_payload = (
        broad.strict_relation_contract_payload()
    )

    dac_payload = (
        dac.strict_relation_contract_payload()
    )

    assert broad_payload == dac_payload

    assert (
        _payload_sha(broad_payload)
        == FROZEN_PRE_H3_PAYLOAD_SHA256
    )

    assert (
        _payload_sha(dac_payload)
        == FROZEN_PRE_H3_PAYLOAD_SHA256
    )


def test_broad_contract_reuses_canonical_common_evidence_objects():
    embedded = (
        CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS[
            5:11
        ]
    )

    assert len(embedded) == len(
        COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
    )

    assert all(
        left is right
        for left, right in zip(
            embedded,
            COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
        )
    )
