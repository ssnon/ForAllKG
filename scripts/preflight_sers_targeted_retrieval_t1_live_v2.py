from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dac_her.sers_targeted_retrieval_t1_live_guard import (
    validate_t1_pre_network_guard,
)
from dac_her.sers_targeted_retrieval_t1_live_recovery_v2 import (
    ROOT,
    SPEC_ROOT,
    V1_RUN_ROOT,
    V2_RUN_ROOT,
    load_frozen_context,
    validate_v1_failure_evidence,
)

V2_RUNTIME_FILES = [
    "dac_her/sers_targeted_retrieval_t1_live_validation_v2.py",
    "dac_her/sers_targeted_retrieval_t1_live_recovery_v2.py",
    "scripts/preflight_sers_targeted_retrieval_t1_live_v2.py",
    "scripts/run_sers_targeted_retrieval_t1_live_v2.py",
    "scripts/verify_sers_targeted_retrieval_t1_live_v2.py",
    "scripts/verify_sers_targeted_retrieval_t1_v1_failure_evidence.py",
    "tests/test_sers_targeted_retrieval_t1_live_validation_v2.py",
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_v1_failure_freeze_v1/"
    "failure_manifest.json",
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v1/"
    "LIVE_ATTEMPT_CONSUMED.json",
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v1/FATAL_ERROR.json",
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v1/gap_01/augmented_plan.json",
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v1/gap_01/delta_plan.json",
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v1/gap_01/delta_packet.json",
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v1/gap_01/merged_packet.json",
]


def _require_v2_runtime_tracked(root: Path) -> None:
    for rel in V2_RUNTIME_FILES:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel],
            cwd=root,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"v2 runtime/evidence file is not tracked: {rel}"
            )


def validate_v2_preflight(
    *,
    root: Path = ROOT,
    spec_root: Path = SPEC_ROOT,
    run_root: Path = V2_RUN_ROOT,
) -> dict[str, object]:
    if run_root.exists():
        raise RuntimeError(
            f"T1 v2 live run root already exists: {run_root}"
        )
    spec, base_plan, base_packet, gap_plan, provider_plan = (
        load_frozen_context(spec_root)
    )
    guard = validate_t1_pre_network_guard(
        root=root,
        spec_root=spec_root,
        spec=spec,
        base_plan=base_plan,
        base_packet=base_packet,
        gap_plan=gap_plan,
        provider_plan=provider_plan,
    )
    _require_v2_runtime_tracked(root)
    recovered = validate_v1_failure_evidence(
        root=root,
        spec_root=spec_root,
        v1_run_root=V1_RUN_ROOT,
    )

    if len(gap_plan.gaps) != 3:
        raise RuntimeError("expected exactly 3 frozen T1 gaps")
    if not gap_plan.gaps[0].targeted_queries:
        raise RuntimeError("gap_01 must be the recovered targeted gap")
    if gap_plan.gaps[1].action != "keep":
        raise RuntimeError("gap_02 must be frozen keep action")
    if gap_plan.gaps[1].targeted_queries:
        raise RuntimeError("gap_02 keep action must have zero queries")
    if not gap_plan.gaps[2].targeted_queries:
        raise RuntimeError("gap_03 must be the remaining targeted gap")

    return {
        "guard": guard,
        "recovered": recovered,
        "spec": spec,
        "base_plan": base_plan,
        "base_packet": base_packet,
        "gap_plan": gap_plan,
        "provider_plan": provider_plan,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=V2_RUN_ROOT,
    )
    args = parser.parse_args()
    try:
        state = validate_v2_preflight(run_root=args.run_root)
    except Exception as exc:
        print("T1 v2 guarded recovery preflight: FAIL")
        print("Exception type:", type(exc).__name__)
        print("Reason:", str(exc))
        print("Network calls:", 0)
        print("LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    guard = state["guard"]
    recovered = state["recovered"]
    gap_plan = state["gap_plan"]
    assert isinstance(guard, dict)
    assert isinstance(recovered, dict)

    print("T1 v2 guarded recovery preflight: PASS")
    print("Source git HEAD:", guard["source_git_head"])
    print("Spec ID:", guard["spec_id"])
    print("Provider mode:", guard["provider_mode"])
    print("Providers:", guard["providers"])
    print("V1 failure evidence verified:", True)
    print("Recovered V1 hypothesis:", recovered["hypothesis_id"])
    print("Recovered V1 queries:", recovered["delta_query_count"])
    print(
        "Recovered V1 provider executions:",
        recovered["successful_execution_count"],
        "success /",
        recovered["failed_execution_count"],
        "failed",
    )
    print(
        "Recovered V1 structural PASS:",
        recovered["structural_pass"],
    )
    print("V1 gap_01 network replay authorized:", False)
    print(
        "Remaining live hypothesis:",
        gap_plan.gaps[2].hypothesis_id,
    )
    print(
        "Remaining live queries:",
        len(gap_plan.gaps[2].targeted_queries),
    )
    print(
        "Expected new provider executions:",
        len(gap_plan.gaps[2].targeted_queries)
        * len(state["provider_plan"].active_providers),
    )
    print("V2 live run root exists:", False)
    print("Network calls during preflight:", 0)
    print("LLM calls:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
