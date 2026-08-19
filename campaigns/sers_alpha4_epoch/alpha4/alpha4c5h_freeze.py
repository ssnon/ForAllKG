from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


ALPHA4C5H_FREEZE_SEMANTICS_ID = (
    "sers_trend_v6r2_freeze_v1_alpha4c5h"
)
ALPHA4C5H_CONFIRMATION_PROTOCOL_SEMANTICS_ID = (
    "sers_reserve_b_confirmation_protocol_v1_alpha4c5h"
)

EXPECTED_TREND_SEMANTICS_ID = (
    "sers_au_ag_trend_v6r2_alpha4c5g2r2"
)
EXPECTED_ACTIVE_PRE_FREEZE_TREND_SEMANTICS_ID = (
    "sers_au_ag_trend_v5_alpha4c2121"
)

# IMPORTANT:
# This is the historical semantic SHA stored inside blind_split.json as
# `split_sha256`. It is NOT sha256_file(blind_split.json).
EXPECTED_SPLIT_SEMANTIC_SHA256 = (
    "4b73127ceb27ff0ec7afeb5362485eecc"
    "15fa95fd808377331a57f2b6f497d16"
)
EXPECTED_SPLIT_ID = (
    "sers_alpha4c5f2_blind_split:"
    "bf540bd70cefe49e76ed"
)
EXPECTED_5E_PROTOCOL_ID = (
    "trend_hypothesis_evaluation_protocol:"
    "b97b65fe4bc66c4f5695"
)

EXPECTED_V6R2_FILE_SHA256 = {
    "dac_her/domains/sers_au_ag_trend_alpha4c5g2r2.py":
        "5fff526468d642f82a7037aa2243ebcc82acc28d0a31e6a8997d4e1956f48e59",
    "scripts/run_sers_alpha4c5g2r2_candidate_regression.py":
        "d2522bcff3c774d68af3f0c6e8a8413ad77a85cf07519b96759b602f3dbe0851",
}

KNOWN_FROZEN_BUILDER_SHA256 = {
    "scripts/build_metric_definition_contexts.py":
        "c24c38cc0b25edccc44e7c78b1c9f75eaa7170261a3f0ae440075c65274566c6",
    "scripts/build_comparison_contexts.py":
        "b48e25b2b6cfd0c218d317e20ebe60b5540326b7304f4a6a5b973021049eee8c",
    "scripts/build_trend_evidence.py":
        "9a9270fbcf554ac739135f366e303cf1c9d7ceeb2af27cfa78719ff2f7da6967",
    "scripts/build_trend_precision.py":
        "7e7127b8129a393cc534ec18542fd1d0dc59a59e8f9a90ef4a15d3009678093d",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _norm_key(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value or "").casefold(),
    ).strip("_")


def _paper_id_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    rows = [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]
    if len(rows) != len(value):
        return None
    if len(set(rows)) != len(rows):
        return None
    if not rows:
        return None
    return rows


def find_reserve_b_paper_ids(
    value: Any,
) -> tuple[list[str], str]:
    candidates: list[tuple[list[str], str]] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            for key, child in obj.items():
                normalized = _norm_key(key)
                child_path = (
                    f"{path}.{key}" if path else str(key)
                )

                if (
                    "reserve_b" in normalized
                    or normalized in {
                        "b_reserve",
                        "reserveb",
                    }
                ):
                    direct = _paper_id_list(child)
                    if direct is not None and len(direct) == 25:
                        candidates.append(
                            (direct, child_path)
                        )
                    if isinstance(child, Mapping):
                        for subkey, subvalue in child.items():
                            subnorm = _norm_key(subkey)
                            if (
                                "paper" in subnorm
                                or subnorm in {
                                    "ids",
                                    "members",
                                    "items",
                                }
                            ):
                                rows = _paper_id_list(subvalue)
                                if (
                                    rows is not None
                                    and len(rows) == 25
                                ):
                                    candidates.append(
                                        (
                                            rows,
                                            f"{child_path}.{subkey}",
                                        )
                                    )

                if (
                    "reserve_b" in normalized
                    and (
                        "paper" in normalized
                        or normalized.endswith("_ids")
                    )
                ):
                    rows = _paper_id_list(child)
                    if rows is not None and len(rows) == 25:
                        candidates.append(
                            (rows, child_path)
                        )

                walk(child, child_path)

        elif isinstance(obj, list):
            for index, child in enumerate(obj):
                walk(child, f"{path}[{index}]")

    walk(value, "")

    unique: dict[
        tuple[str, ...],
        list[str],
    ] = {}
    paths: dict[tuple[str, ...], list[str]] = {}
    for rows, path in candidates:
        key = tuple(sorted(rows))
        unique[key] = rows
        paths.setdefault(key, []).append(path)

    if len(unique) != 1:
        raise ValueError(
            "Expected exactly one unique 25-paper Reserve-B "
            f"partition; found {len(unique)}. Candidate paths: "
            + repr(paths)
        )

    key, rows = next(iter(unique.items()))
    return sorted(rows), ",".join(sorted(paths[key]))


def find_scalar_values(
    value: Any,
    *,
    key_predicate,
) -> list[tuple[str, Any]]:
    results: list[tuple[str, Any]] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            for key, child in obj.items():
                child_path = (
                    f"{path}.{key}" if path else str(key)
                )
                if key_predicate(_norm_key(key)):
                    if not isinstance(child, (Mapping, list)):
                        results.append((child_path, child))
                walk(child, child_path)
        elif isinstance(obj, list):
            for index, child in enumerate(obj):
                walk(child, f"{path}[{index}]")

    walk(value, "")
    return results


def find_string_in_json_files(
    root: Path,
    needle: str,
) -> list[Path]:
    matches = []
    if not root.exists():
        return matches
    for path in sorted(root.rglob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if needle in text:
            matches.append(path)
    return matches


def scientific_code_inventory(root: Path) -> list[Path]:
    paths: set[Path] = set()

    dac_root = root / "dac_her"
    if dac_root.exists():
        paths.update(
            path
            for path in dac_root.rglob("*.py")
            if "__pycache__" not in path.parts
        )

    scripts_root = root / "scripts"
    patterns = (
        "build_metric_definition_contexts.py",
        "build_comparison_contexts.py",
        "build_trend_evidence.py",
        "build_trend_precision.py",
        "*cross*context*.py",
        "*hypothesis*.py",
        "*explorer*.py",
        "*alpha4c5*.py",
        "run_dac_discovery_e2e.py",
    )
    if scripts_root.exists():
        for pattern in patterns:
            paths.update(
                path
                for path in scripts_root.glob(pattern)
                if path.is_file()
            )

    return sorted(paths)


def hash_inventory(
    root: Path,
    paths: Iterable[Path],
) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(paths)
    }


def verify_hash_inventory(
    root: Path,
    inventory: Mapping[str, str],
) -> list[dict[str, str]]:
    drift: list[dict[str, str]] = []
    for rel, expected in sorted(inventory.items()):
        path = root / rel
        if not path.exists():
            drift.append(
                {
                    "path": rel,
                    "issue": "missing",
                    "expected": expected,
                    "observed": "",
                }
            )
            continue
        observed = sha256_file(path)
        if observed != expected:
            drift.append(
                {
                    "path": rel,
                    "issue": "sha256_mismatch",
                    "expected": expected,
                    "observed": observed,
                }
            )
    return drift


def make_freeze_id(payload: Mapping[str, Any]) -> str:
    return (
        "sers_alpha4c5h_v6r2_freeze:"
        + semantic_sha256(payload)[:20]
    )


def make_confirmation_protocol_id(
    payload: Mapping[str, Any],
) -> str:
    return (
        "sers_alpha4c5h_reserve_b_confirmation:"
        + semantic_sha256(payload)[:20]
    )
