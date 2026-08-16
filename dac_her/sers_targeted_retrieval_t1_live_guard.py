from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.literature_provider_plan import LiteratureProviderPlan
from dac_her.novelty_refinement_contracts import NoveltyGapPlan


EXPECTED_BRANCH = "feat/SERS-targeted-retrieval-live-dev"

RUNTIME_TRACKED_FILES = [
    "dac_her/sers_targeted_retrieval_t1_live_guard.py",
    "dac_her/sers_targeted_retrieval_t1_live_validation.py",
    "scripts/run_sers_targeted_retrieval_t1_live.py",
    "scripts/verify_sers_targeted_retrieval_t0_freeze_v2.py",
    "scripts/verify_sers_targeted_retrieval_t1_input_bundle_v1.py",
    "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/"
    "base_query_plan.json",
    "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/"
    "base_prior_art_packet.json",
    "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/"
    "novelty_gap_plan.json",
    "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/"
    "provider_plan.json",
    "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/"
    "input_bundle_manifest.json",
    "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/"
    "t1_spec.json",
    "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/"
    "PREPARE_PASS.json",
]

FALSE_GUARDS = [
    "runtime_provider_fallback_authorized",
    "query_rewrite_authorized",
    "ranker_authorized",
    "claim_reviewer_authorized",
    "novelty_reassessment_authorized",
    "llm_authorized",
    "hypothesis_rewrite_authorized",
    "fresh_reserve_c_authorized",
    "automatic_next_stage_authorized",
]


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _run_verifier(root: Path, module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"offline verifier failed: {module}\n"
            + result.stdout
            + result.stderr
        )


def validate_t1_pre_network_guard(
    *,
    root: Path,
    spec_root: Path,
    spec: dict[str, Any],
    base_plan: LiteratureQueryPlan,
    base_packet: PriorArtPacket,
    gap_plan: NoveltyGapPlan,
    provider_plan: LiteratureProviderPlan,
) -> dict[str, Any]:
    root = root.resolve()
    spec_root = spec_root.resolve()

    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"unexpected branch: {branch!r}; expected {EXPECTED_BRANCH!r}"
        )

    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "working tree must be clean before one-shot live retrieval:\n"
            + dirty
        )

    for rel in RUNTIME_TRACKED_FILES:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel],
            cwd=root,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "runtime-critical file is not tracked in the live source "
                f"commit: {rel}"
            )

    # Re-run the exact offline handoff verifiers at the live boundary.
    _run_verifier(
        root,
        "scripts.verify_sers_targeted_retrieval_t0_freeze_v2",
    )
    _run_verifier(
        root,
        "scripts.verify_sers_targeted_retrieval_t1_input_bundle_v1",
    )

    # Validate the live spec self-hash and ID.
    body = dict(spec)
    observed_id = body.pop("spec_id", None)
    observed_sha = body.pop("spec_sha256", None)
    expected_sha = _sha256_json(body)
    expected_id = (
        "sers_targeted_retrieval_t1_live_spec:"
        + expected_sha[:20]
    )
    if observed_sha != expected_sha:
        raise RuntimeError("T1 live spec SHA mismatch")
    if observed_id != expected_id:
        raise RuntimeError("T1 live spec ID mismatch")

    # Validate every frozen copy captured by prepare.
    frozen = spec.get("frozen_copy_sha256")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError("T1 spec frozen_copy_sha256 is missing")
    for name, expected in frozen.items():
        path = spec_root / str(name)
        if not path.is_file():
            raise RuntimeError(f"missing frozen spec copy: {name}")
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"frozen spec copy SHA mismatch: {name}"
            )

    prepare_path = spec_root / "PREPARE_PASS.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    if prepare.get("spec_id") != observed_id:
        raise RuntimeError("PREPARE_PASS spec ID mismatch")
    if prepare.get("network_calls") != 0:
        raise RuntimeError("PREPARE_PASS records network calls")
    if prepare.get("llm_calls") != 0:
        raise RuntimeError("PREPARE_PASS records LLM calls")
    if prepare.get("fresh_reserve_c_consumed") is not False:
        raise RuntimeError("PREPARE_PASS records Reserve C consumption")

    # Exact identity linkage between the frozen JSON models and the live spec.
    expected_pairs = {
        "base_query_plan_id": base_plan.plan_id,
        "base_prior_art_packet_id": base_packet.packet_id,
        "novelty_gap_plan_id": gap_plan.plan_id,
        "provider_plan_id": provider_plan.plan_id,
        "provider_mode": provider_plan.mode,
    }
    for key, expected in expected_pairs.items():
        if spec.get(key) != expected:
            raise RuntimeError(
                f"T1 live spec linkage mismatch for {key}: "
                f"{spec.get(key)!r} != {expected!r}"
            )
    if list(spec.get("providers", [])) != list(
        provider_plan.active_providers
    ):
        raise RuntimeError("T1 live spec provider set mismatch")

    if spec.get("provider_set_frozen_for_run") is not True:
        raise RuntimeError("provider set is not frozen for run")
    for key in FALSE_GUARDS:
        if spec.get(key) is not False:
            raise RuntimeError(
                f"forbidden live authorization is not False: {key}"
            )

    # Freeze all provider configuration-presence bits, without persisting secrets.
    observed_env = {
        "openalex_api_key_configured":
            bool(str(os.getenv("OPENALEX_API_KEY", "")).strip()),
        "crossref_mailto_configured":
            bool(str(os.getenv("CROSSREF_MAILTO", "")).strip()),
        "semantic_scholar_api_key_configured":
            bool(
                str(
                    os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
                ).strip()
            ),
    }
    for key, observed in observed_env.items():
        expected = bool(getattr(provider_plan, key))
        if observed != expected:
            raise RuntimeError(
                f"provider environment shape changed after prepare: {key}"
            )

    return {
        "source_git_head": head,
        "source_git_branch": branch,
        "spec_id": observed_id,
        "spec_sha256": observed_sha,
        "provider_plan_id": provider_plan.plan_id,
        "provider_mode": provider_plan.mode,
        "providers": list(provider_plan.active_providers),
        "targeted_query_count": int(spec["targeted_query_count"]),
        "results_per_query": int(spec["results_per_query"]),
        "t0_freeze_verified": True,
        "t1_input_bundle_verified": True,
        "working_tree_clean": True,
        "runtime_files_tracked": True,
        "network_calls_during_guard": 0,
        "llm_calls": 0,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
    }
