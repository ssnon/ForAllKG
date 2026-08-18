from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SERS_PROFILE_PATH = (
    "dac_her/domains/sers_au_ag.py"
)

REGISTRY_PATH = (
    "dac_her/domains/registry.py"
)

EXPECTED_SERS_PROFILE_SHA256 = (
    "b705ae934f9cba623651fc2ea7435e998"
    "08a4e85ab5d0be2e7fa4fd017f574cd"
)

HISTORICAL_IDENTITY_MANIFEST = (
    ROOT
    / "evaluation"
    / "sers_fresh_c"
    / "c0_1b_activation_readiness_v1"
    / "historical_identity_sweep_manifest.json"
)

STANDARD2_FILES = (
    ROOT
    / "dac_her"
    / "standard2_ranker_dev_validation.py",

    ROOT
    / "dac_her"
    / "standard2_claim_review_dev_validation.py",

    ROOT
    / "dac_her"
    / "standard2_claim_review_dev_validation_v2.py",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _source_files_to_freeze(
    path: Path,
) -> tuple[str, ...]:
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    for node in tree.body:
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        if not any(
            isinstance(target, ast.Name)
            and target.id
            == "SOURCE_FILES_TO_FREEZE"
            for target in node.targets
        ):
            continue

        if not isinstance(
            node.value,
            ast.Tuple,
        ):
            raise AssertionError(
                "SOURCE_FILES_TO_FREEZE "
                "must remain a tuple"
            )

        values = []

        for item in node.value.elts:
            if not (
                isinstance(item, ast.Call)
                and isinstance(
                    item.func,
                    ast.Name,
                )
                and item.func.id == "Path"
                and len(item.args) == 1
                and isinstance(
                    item.args[0],
                    ast.Constant,
                )
                and isinstance(
                    item.args[0].value,
                    str,
                )
            ):
                raise AssertionError(
                    "SOURCE_FILES_TO_FREEZE "
                    "must use literal Path strings"
                )

            values.append(
                item.args[0].value
            )

        return tuple(values)

    raise AssertionError(
        "SOURCE_FILES_TO_FREEZE "
        f"not found in {path}"
    )


def test_sers_profile_bytes_match_historical_identity():
    profile = (
        ROOT
        / SERS_PROFILE_PATH
    )

    assert (
        _sha256_file(profile)
        == EXPECTED_SERS_PROFILE_SHA256
    )


def test_fresh_c_identity_sweep_records_exact_sers_profile_sha():
    manifest = json.loads(
        HISTORICAL_IDENTITY_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    matches = [
        row
        for row in manifest[
            "scanned_files"
        ]
        if row.get("path")
        == SERS_PROFILE_PATH
    ]

    assert len(matches) == 1

    row = matches[0]

    assert (
        row["sha256"]
        == EXPECTED_SERS_PROFILE_SHA256
    )

    assert (
        row["tracked_in_source_commit"]
        is True
    )


def test_standard2_source_freezes_include_sers_profile_and_registry():
    for path in STANDARD2_FILES:
        frozen = (
            _source_files_to_freeze(
                path
            )
        )

        assert (
            SERS_PROFILE_PATH
            in frozen
        )

        assert (
            REGISTRY_PATH
            in frozen
        )


def test_standard2_validators_fail_closed_on_source_hash_drift():
    ranker = STANDARD2_FILES[0].read_text(
        encoding="utf-8"
    )

    claim_v1 = STANDARD2_FILES[1].read_text(
        encoding="utf-8"
    )

    claim_v2 = STANDARD2_FILES[2].read_text(
        encoding="utf-8"
    )

    assert (
        '_source_hashes(repo_root) '
        '!= spec["source_hashes"]'
        in ranker
    )

    sentinel = (
        'stored.get("source_hashes") '
        '!= source_hashes(repo_root)'
    )

    assert sentinel in claim_v1
    assert sentinel in claim_v2
