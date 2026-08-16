from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from dac_her.hypothesis_trend_contract_hardened_compiler import (
    ContractHardenedTrendHypothesisCompiler,
)
from dac_her.hypothesis_trend_contract_hardened_contracts import (
    ContractHardenedTrendHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_contract_hardened_exposure import (
    build_contract_hardened_trend_maker_exposure,
)
from dac_her.hypothesis_trend_contract_hardened_prompt import (
    ContractHardenedTrendHypothesisPromptAssembler,
)
from dac_her.hypothesis_trend_contract_hardened_validator import (
    ContractHardenedTrendHypothesisValidator,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)


ALPHA4C5I_DEV_COMPAT_SEMANTICS_ID = (
    "sers_alpha4c5i_dev_downstream_compatibility_v1"
)

EXPECTED_DEV_COUNT = 53
EXPECTED_TREND_COUNT = 15
EXPECTED_PRECISION_COUNT = 15
EXPECTED_CROSS_CONTEXT_COUNT = 10
EXPECTED_GROUNDING_COUNT = 10

EXPECTED_TREND_SEMANTICS_ID = (
    "sers_au_ag_trend_v6r2_alpha4c5g2r2"
)
EXPECTED_PRECISION_SEMANTICS_ID = (
    "sers_au_ag_trend_precision_v5_alpha4c21211"
)

EXPECTED_POSTMORTEM_ID = (
    "sers_alpha4c5h2_reserve_b_postmortem:46dbf7c9d20b014f3502"
)
EXPECTED_POSTMORTEM_SHA256 = (
    "512447e7cc2790c79b18aee161926587b69e46ad05c04032e5e963b468007e8b"
)
EXPECTED_POSTMORTEM_FILE_SHA256 = (
    "4ca800ab3987b966989938e463b37890eaebe4100cb5818298e9e7eb48455e1e"
)

DEFAULT_5H_FREEZE_MANIFEST = Path(
    "evaluation/sers_alpha4c5h/freeze_v1/freeze_manifest.json"
)
DEFAULT_BLIND_SPLIT = Path(
    "evaluation/sers_alpha4c5f2/pool_v1/blind_split.json"
)
EXPECTED_BLIND_SPLIT_RAW_SHA256 = (
    "16c8fe725468a57a7703e13e843c1486176f51a264bb6b82f593cb7abbd956c5"
)
EXPECTED_BLIND_SPLIT_SEMANTIC_SHA256 = (
    "4b73127ceb27ff0ec7afeb5362485eecc15fa95fd808377331a57f2b6f497d16"
)
EXPECTED_BLIND_SPLIT_ID = (
    "sers_alpha4c5f2_blind_split:bf540bd70cefe49e76ed"
)

DEFAULT_5H2_POSTMORTEM = Path(
    "evaluation/sers_alpha4c5h2/reserve_b_postmortem_v1/"
    "postmortem_manifest.json"
)
DEFAULT_H1_DEV_SUMMARY = Path(
    "evaluation/sers_alpha4c5h1/dev_compat_v1/summary.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "evaluation/sers_alpha4c5i/dev_compat_v1"
)

EXPECTED_5I_COMPONENT_SHA256 = {'dac_her/hypothesis_trend_contract_hardened_compiler.py': '8bd971619e58918e8fa1b90467228dadadd02844f194e033a778f6a5493578f3', 'dac_her/hypothesis_trend_contract_hardened_contracts.py': '9705b19c816145ce8b52846355e5bf285daebbcfcb50d2f802035b66d36eaa8f', 'dac_her/hypothesis_trend_contract_hardened_exposure.py': 'd04b779d1dc01cff37509bdfdab86c753eaabde98373bca6231176773191c30e', 'dac_her/hypothesis_trend_contract_hardened_llm.py': '04f75a8ea68b73f78e97281550f27416595501fdb15d2f55bab25684ca3f166d', 'dac_her/hypothesis_trend_contract_hardened_prompt.py': '1f7ad255a03c8eaca7cf6d558232c841db347a6774c39d0299bdb13d420b6845', 'dac_her/hypothesis_trend_contract_hardened_renderer.py': '4ecfa15eb0db629c692cf2a167279acf9bad01c2c1673dbe69cb2a0252731923', 'dac_her/hypothesis_trend_contract_hardened_run_record.py': '9c839ff04deda8905364180e9abf0973eda5706041ab5df98df5c4a3f0ae5266', 'dac_her/hypothesis_trend_contract_hardened_runtime.py': '1d1a6661575129f93b1c2ae2c423f6275fb88c3f19c321fd9f90fec49f6e197f', 'dac_her/hypothesis_trend_contract_hardened_validator.py': 'f768ec0eafbcee69f2d5b18e3efadd13eec889f4183baa0be4590264fc90879e', 'scripts/run_contract_hardened_trend_hypothesis_maker.py': '3f8a366198804950c7f4f1495b9c5796497248b7f88c15f335b2d7312d2c9e07', 'tests/test_alpha4c5i_contract_hardening.py': '0576f963f16bdfd2a9c366bddb7c6109a218916ba8377d0d0e7a0fe84bc42b37'}

FORBIDDEN_CLOSED_PATH_TOKENS = (
    "reserve_a",
    "reserve_b",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _repo_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def verify_frozen_postmortem(root: Path) -> list[str]:
    issues: list[str] = []
    path = root / DEFAULT_5H2_POSTMORTEM
    if not path.is_file():
        return ["5h.2 postmortem manifest missing"]

    if sha256_file(path) != EXPECTED_POSTMORTEM_FILE_SHA256:
        issues.append("5h.2 postmortem raw file SHA drifted")
        return issues

    payload = read_json(path)
    checks = {
        "postmortem_id": EXPECTED_POSTMORTEM_ID,
        "postmortem_sha256": EXPECTED_POSTMORTEM_SHA256,
        "reserve_consumed": True,
        "campaign_terminal_state": "fail",
        "campaign_closed": True,
        "rerun_allowed": False,
        "reserve_b_failure_authorizes_tuning": False,
    }
    for field, expected in checks.items():
        if payload.get(field) != expected:
            issues.append(
                f"5h.2 postmortem {field} drift: "
                f"expected={expected!r}, observed={payload.get(field)!r}"
            )
    return issues


def verify_5i_component_hashes(root: Path) -> list[str]:
    issues: list[str] = []
    for rel, expected in sorted(
        EXPECTED_5I_COMPONENT_SHA256.items()
    ):
        path = root / rel
        if not path.is_file():
            issues.append(f"5i component missing: {rel}")
            continue
        observed = sha256_file(path)
        if observed != expected:
            issues.append(
                f"5i component SHA drift: {rel}: "
                f"expected={expected}, observed={observed}"
            )
    return issues


def _contains_exact_scalar(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_exact_scalar(child, target)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(
            _contains_exact_scalar(child, target)
            for child in value
        )
    return str(value) == target


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return [str(item) for item in value]


def _collect_dev_candidates(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> list[tuple[str, list[str]]]:
    candidates: list[tuple[str, list[str]]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = path + (key_text,)
            lowered = key_text.lower().replace("-", "_")

            direct = _string_list(child)
            if direct is not None and (
                lowered in {
                    "dev",
                    "development",
                    "dev_paper_ids",
                    "development_paper_ids",
                }
                or (
                    ("dev" in lowered or "development" in lowered)
                    and "reserve" not in lowered
                )
            ):
                candidates.append(
                    (".".join(child_path), direct)
                )

            if isinstance(child, dict) and (
                lowered in {"dev", "development"}
                or "development" in lowered
            ):
                for nested_key in (
                    "paper_ids",
                    "papers",
                    "ids",
                ):
                    nested = _string_list(child.get(nested_key))
                    if nested is not None:
                        candidates.append(
                            (
                                ".".join(
                                    child_path + (nested_key,)
                                ),
                                nested,
                            )
                        )

            candidates.extend(
                _collect_dev_candidates(
                    child,
                    path=child_path,
                )
            )

    elif isinstance(value, list):
        assignments: list[str] = []
        assignment_like = True
        for item in value:
            if not isinstance(item, dict):
                assignment_like = False
                break

            paper_id = None
            for field in ("paper_id", "paper", "id"):
                raw_id = item.get(field)
                if isinstance(raw_id, str):
                    paper_id = raw_id
                    break

            partition = None
            for field in (
                "partition",
                "split",
                "subset",
                "bucket",
                "role",
            ):
                raw_partition = item.get(field)
                if isinstance(raw_partition, str):
                    partition = (
                        raw_partition.lower()
                        .replace("-", "_")
                        .strip()
                    )
                    break

            if paper_id is None or partition is None:
                assignment_like = False
                break

            if partition in {
                "dev",
                "development",
                "development_set",
            }:
                assignments.append(paper_id)

        if assignment_like and assignments:
            candidates.append(
                (".".join(path) + "[assignment-style]", assignments)
            )

        for index, child in enumerate(value):
            candidates.extend(
                _collect_dev_candidates(
                    child,
                    path=path + (f"[{index}]",),
                )
            )

    return candidates


def load_exact_dev_paper_ids(root: Path) -> list[str]:
    path = root / DEFAULT_BLIND_SPLIT
    if not path.is_file():
        raise ValueError(
            f"Canonical blind split missing: {path}"
        )

    observed_raw_sha = sha256_file(path)
    if observed_raw_sha != EXPECTED_BLIND_SPLIT_RAW_SHA256:
        raise ValueError(
            "Canonical blind split raw SHA drift: "
            f"expected={EXPECTED_BLIND_SPLIT_RAW_SHA256}, "
            f"observed={observed_raw_sha}"
        )

    split = read_json(path)

    if not _contains_exact_scalar(
        split,
        EXPECTED_BLIND_SPLIT_ID,
    ):
        raise ValueError(
            "Canonical blind split ID is not present in blind_split.json"
        )

    semantic_sha_present = _contains_exact_scalar(
        split,
        EXPECTED_BLIND_SPLIT_SEMANTIC_SHA256,
    )

    raw_candidates = _collect_dev_candidates(split)
    normalized: dict[tuple[str, ...], list[str]] = {}

    for source_path, candidate in raw_candidates:
        paper_ids = sorted(str(value) for value in candidate)
        if len(paper_ids) != EXPECTED_DEV_COUNT:
            continue
        if len(set(paper_ids)) != EXPECTED_DEV_COUNT:
            continue
        normalized[tuple(paper_ids)] = [
            *normalized.get(tuple(paper_ids), []),
            source_path,
        ]

    if not normalized:
        top_level = ", ".join(sorted(map(str, split.keys())))
        raise ValueError(
            "Could not recover an exact 53-paper DEV partition from "
            "the canonical blind split. "
            f"Top-level keys: {top_level}"
        )

    if len(normalized) != 1:
        sources = [
            ", ".join(paths)
            for paths in normalized.values()
        ]
        raise ValueError(
            "Multiple distinct 53-paper DEV partitions were found; "
            "fail-closed instead of selecting one: "
            + " | ".join(sources)
        )

    paper_ids_tuple, sources = next(iter(normalized.items()))
    paper_ids = list(paper_ids_tuple)

    if paper_ids != sorted(set(paper_ids)):
        raise ValueError("DEV paper IDs are not unique")
    if len(paper_ids) != EXPECTED_DEV_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DEV_COUNT} DEV papers; "
            f"observed {len(paper_ids)}"
        )

    load_exact_dev_paper_ids.last_binding = {
        "blind_split_path": str(DEFAULT_BLIND_SPLIT),
        "blind_split_raw_sha256": observed_raw_sha,
        "blind_split_id": EXPECTED_BLIND_SPLIT_ID,
        "blind_split_semantic_sha256":
            EXPECTED_BLIND_SPLIT_SEMANTIC_SHA256,
        "semantic_sha_serialized_in_split":
            semantic_sha_present,
        "dev_partition_sources": sources,
        "dev_paper_count": len(paper_ids),
    }
    return paper_ids


load_exact_dev_paper_ids.last_binding = {}


def verify_h1_dev_summary(root: Path) -> list[str]:
    issues: list[str] = []
    path = root / DEFAULT_H1_DEV_SUMMARY
    if not path.is_file():
        return ["5h.1 DEV compatibility summary missing"]
    payload = read_json(path)

    if payload.get("passes_downstream_compatibility") is not True:
        issues.append(
            "5h.1 DEV summary is not passes_downstream_compatibility=true"
        )
    if payload.get("scientific_semantics_modified") is not False:
        issues.append(
            "5h.1 DEV summary reports scientific semantics modification"
        )
    if payload.get("precision_algorithm_modified") is not False:
        issues.append(
            "5h.1 DEV summary reports precision algorithm modification"
        )
    if payload.get("trend_semantics_id") != EXPECTED_TREND_SEMANTICS_ID:
        issues.append("5h.1 DEV Trend semantics drifted")
    if (
        payload.get("precision_semantics_id")
        != EXPECTED_PRECISION_SEMANTICS_ID
    ):
        issues.append("5h.1 DEV Precision semantics drifted")
    if payload.get("reserve_a_used") is not False:
        issues.append("5h.1 DEV summary reports Reserve A use")
    if payload.get("reserve_b_used") is not False:
        issues.append("5h.1 DEV summary reports Reserve B use")
    if payload.get("count_thresholds_used_for_acceptance") is not False:
        issues.append(
            "5h.1 DEV summary reports count-threshold acceptance"
        )
    if payload.get("llm_calls") != 0:
        issues.append("5h.1 DEV summary must report llm_calls=0")
    return issues


_COUNT_PATTERNS = {
    "trend_evidence_count": re.compile(
        r"Trend evidence:\s*(\d+)"
    ),
    "precision_count": re.compile(
        r"Precision local results:\s*(\d+)"
    ),
    "cross_context_count": re.compile(
        r"CrossContext assessments:\s*(\d+)"
    ),
    "grounding_count": re.compile(
        r"Grounding relations:\s*(\d+)"
    ),
}


def parse_h1_counts(stdout: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, pattern in _COUNT_PATTERNS.items():
        match = pattern.search(stdout)
        if match:
            counts[key] = int(match.group(1))
    return counts


def run_h1_dev_compatibility(
    root: Path,
) -> tuple[dict[str, int], str]:
    """
    Read and verify the already-materialized alpha4c.5h.1 DEV
    compatibility summary.

    alpha4c.5h.1 is an existing development diagnostic artifact. 5i must
    not try to recreate/overwrite it merely to prove downstream
    compatibility. The caller-facing function name is retained for
    backward compatibility with the 5i orchestration.
    """
    issues = verify_h1_dev_summary(root)
    if issues:
        raise RuntimeError(
            "Existing alpha4c.5h.1 DEV compatibility summary failed "
            "verification:\n- "
            + "\n- ".join(issues)
        )

    summary_path = root / DEFAULT_H1_DEV_SUMMARY
    payload = read_json(summary_path)

    field_map = {
        "trend_evidence_count": "trend_evidence_count",
        "precision_count": "precision_local_result_count",
        "cross_context_count": "cross_context_assessment_count",
        "grounding_count": "grounding_relation_count",
    }
    counts: dict[str, int] = {}
    missing_fields: list[str] = []
    for key, field in field_map.items():
        raw = payload.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int):
            missing_fields.append(field)
            continue
        counts[key] = int(raw)

    if missing_fields:
        raise RuntimeError(
            "Existing alpha4c.5h.1 DEV summary lacks exact integer "
            "invariant fields: "
            + ", ".join(sorted(missing_fields))
        )

    expected = {
        "trend_evidence_count": EXPECTED_TREND_COUNT,
        "precision_count": EXPECTED_PRECISION_COUNT,
        "cross_context_count": EXPECTED_CROSS_CONTEXT_COUNT,
        "grounding_count": EXPECTED_GROUNDING_COUNT,
    }
    drift = [
        f"{key} expected={wanted} observed={counts[key]}"
        for key, wanted in expected.items()
        if counts[key] != wanted
    ]
    if drift:
        raise RuntimeError(
            "DEV upstream invariant drift in existing 5h.1 summary:\n- "
            + "\n- ".join(drift)
        )

    summary_sha = sha256_file(summary_path)
    provenance = (
        f"existing_summary={summary_path};"
        f"sha256={summary_sha};"
        "rerun_performed=false"
    )
    print(
        "5h.1 DEV upstream compatibility source:",
        summary_path,
    )
    print("5h.1 DEV summary SHA256:", summary_sha)
    print("5h.1 DEV rerun performed: False")
    return counts, provenance


def _path_is_closed_reserve(path: Path) -> bool:
    lowered = [
        str(part).lower().replace("-", "_")
        for part in path.parts
    ]
    return any(
        token in part
        for part in lowered
        for token in FORBIDDEN_CLOSED_PATH_TOKENS
    )


def load_and_verify_dev_input(
    *,
    root: Path,
    path: Path,
    dev_paper_ids: list[str],
) -> TrendAwareHypothesisInput:
    path = path if path.is_absolute() else root / path
    if _path_is_closed_reserve(path):
        raise ValueError(
            "Refusing to read Trend input from a closed Reserve A/B path"
        )
    source = TrendAwareHypothesisInput.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    verify_trend_aware_input_sources(source)

    if source.domain_profile_id != "sers_au_ag":
        raise ValueError(
            f"DEV Trend input domain mismatch: {source.domain_profile_id!r}"
        )
    observed = sorted(str(value) for value in source.trend_corpus_binding.paper_ids)
    if observed != dev_paper_ids:
        raise ValueError(
            "Trend-aware hypothesis input is not bound to the exact "
            "53-paper DEV set"
        )
    return source


def discover_exact_dev_inputs(
    *,
    root: Path,
    dev_paper_ids: list[str],
) -> list[Path]:
    evaluation_root = root / "evaluation"
    if not evaluation_root.exists():
        return []

    candidates: list[Path] = []
    for path in sorted(evaluation_root.rglob("*.json")):
        name = path.name.lower()
        if not (
            name == "trend_aware_hypothesis_input.json"
            or (
                "trend" in name
                and "hypothesis" in name
                and "input" in name
            )
        ):
            continue
        if _path_is_closed_reserve(path):
            continue
        try:
            load_and_verify_dev_input(
                root=root,
                path=path,
                dev_paper_ids=dev_paper_ids,
            )
        except Exception:
            continue
        candidates.append(path)
    return candidates


def select_dev_input(
    *,
    root: Path,
    dev_paper_ids: list[str],
    explicit_path: Path | None,
) -> tuple[Path, TrendAwareHypothesisInput]:
    if explicit_path is not None:
        path = (
            explicit_path
            if explicit_path.is_absolute()
            else root / explicit_path
        )
        source = load_and_verify_dev_input(
            root=root,
            path=path,
            dev_paper_ids=dev_paper_ids,
        )
        return path, source

    candidates = discover_exact_dev_inputs(
        root=root,
        dev_paper_ids=dev_paper_ids,
    )
    if not candidates:
        raise ValueError(
            "No exact DEV53 trend-aware hypothesis input was found. "
            "No Reserve A/B file was inspected. Supply --trend-input "
            "with an existing DEV-only 5b input, or build a new DEV-only "
            "Explorer/context + Trend-aware input before running 5i."
        )
    if len(candidates) > 1:
        rendered = "\n- ".join(
            _repo_relative(root, path)
            for path in candidates
        )
        raise ValueError(
            "Multiple exact DEV53 Trend inputs were found; fail-closed "
            "selection requires --trend-input. Candidates:\n- "
            + rendered
        )
    path = candidates[0]
    source = load_and_verify_dev_input(
        root=root,
        path=path,
        dev_paper_ids=dev_paper_ids,
    )
    return path, source


def deterministic_downstream_probe(
    source: TrendAwareHypothesisInput,
) -> dict[str, Any]:
    exposure = build_contract_hardened_trend_maker_exposure(
        source
    )
    assembler = ContractHardenedTrendHypothesisPromptAssembler()
    prompt = assembler.build(source, exposure=exposure)

    draft = ContractHardenedTrendHypothesisPortfolioDraft(
        hypotheses=[],
        abstention_reason=(
            "alpha4c.5i deterministic DEV compatibility probe; "
            "no scientific hypothesis generated"
        ),
    )
    compiler = ContractHardenedTrendHypothesisCompiler()
    portfolio = compiler.compile(source, draft)
    validator = ContractHardenedTrendHypothesisValidator()
    validation = validator.validate(source, portfolio)

    if validation.passes is not True:
        raise RuntimeError(
            "5i empty-abstention downstream compiler/validator probe "
            "did not validate"
        )
    if portfolio.hypotheses:
        raise RuntimeError(
            "Deterministic compatibility probe unexpectedly generated "
            "scientific hypotheses"
        )

    return {
        "hardened_exposure_id": exposure.exposure_id,
        "hardened_exposure_sha256": exposure.exposure_sha256,
        "hardened_view_count": len(exposure.views),
        "prompt_version": prompt.prompt_version,
        "prompt_sha256": prompt.prompt_sha256,
        "empty_abstention_portfolio_id": portfolio.portfolio_id,
        "empty_abstention_validation_passed": True,
        "empty_abstention_validation_errors": validation.errors,
        "empty_abstention_validation_warnings": validation.warnings,
        "scientific_hypotheses_generated": 0,
        "llm_calls": 0,
    }


def write_preview_artifacts(
    *,
    root: Path,
    output_root: Path,
    input_path: Path,
    source: TrendAwareHypothesisInput,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    exposure = build_contract_hardened_trend_maker_exposure(source)
    prompt = ContractHardenedTrendHypothesisPromptAssembler().build(
        source,
        exposure=exposure,
    )

    atomic_json(
        output_root / "hardened_exposure.json",
        exposure.model_dump(mode="json"),
    )
    (output_root / "prompt.txt").write_text(
        "SYSTEM\n======\n"
        + prompt.system_prompt
        + "\n\nUSER\n====\n"
        + prompt.user_prompt
        + "\n",
        encoding="utf-8",
    )
    atomic_json(
        output_root / "prompt.meta.json",
        {
            "prompt_version": prompt.prompt_version,
            "prompt_sha256": prompt.prompt_sha256,
            "hardened_exposure_id": exposure.exposure_id,
            "hardened_exposure_sha256": exposure.exposure_sha256,
            "source_input_path": _repo_relative(root, input_path),
            "source_input_id": source.input_id,
            "source_input_sha256": source.input_sha256,
            "llm_calls": 0,
        },
    )
    return {
        "exposure_path": _repo_relative(
            root, output_root / "hardened_exposure.json"
        ),
        "prompt_path": _repo_relative(
            root, output_root / "prompt.txt"
        ),
        "prompt_meta_path": _repo_relative(
            root, output_root / "prompt.meta.json"
        ),
    }


def make_structural_summary(
    *,
    root: Path,
    input_path: Path,
    source: TrendAwareHypothesisInput,
    h1_counts: Mapping[str, int],
    probe: Mapping[str, Any],
    preview: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version":
            "sers-alpha4c5i-dev-downstream-compatibility-v1",
        "semantics_id": ALPHA4C5I_DEV_COMPAT_SEMANTICS_ID,
        "development_only": True,
        "dev_paper_count": EXPECTED_DEV_COUNT,
        "exact_dev_set_required": True,
        "source_input_path": _repo_relative(root, input_path),
        "source_input_id": source.input_id,
        "source_input_sha256": source.input_sha256,
        "source_input_file_sha256": sha256_file(input_path),
        "upstream_invariants": dict(h1_counts),
        "expected_upstream_invariants": {
            "trend_evidence_count": EXPECTED_TREND_COUNT,
            "precision_count": EXPECTED_PRECISION_COUNT,
                "cross_context_count": EXPECTED_CROSS_CONTEXT_COUNT,
            "grounding_count": EXPECTED_GROUNDING_COUNT,
        },
        "deterministic_probe": dict(probe),
        "preview_artifacts": dict(preview),
        "passes_deterministic_downstream_compatibility": True,
        "scientific_semantics_modified": False,
        "trend_semantics_modified": False,
        "precision_semantics_modified": False,
        "cross_context_semantics_modified": False,
        "closed_reserve_a_used": False,
        "closed_reserve_b_used": False,
        "reserve_b_rerun": False,
        "count_thresholds_used_for_acceptance": False,
        "scientific_hypotheses_generated_by_probe": 0,
        "llm_calls": 0,
    }
