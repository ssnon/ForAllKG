from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from dac_her.alpha4c5f1_sers_readiness import (
    SERS_DOMAIN_PROFILE_ID,
    prepare_sers_canonical_readiness_lock,
)
from dac_her.alpha4c5f_reserve import (
    DEFAULT_5F_PROTOCOL_PATH,
    load_5f_protocol,
    verify_5f_protocol,
)
from dac_her.canonical_readiness import (
    CanonicalReadinessError,
    atomic_json,
    load_and_verify_readiness_lock,
    now_iso,
    sha256_file,
)


ROOT = Path.cwd()
DEFAULT_OUTPUT_ROOT = Path(
    "evaluation/sers_alpha4c5f1/consumed_v3_seen/regression_v1"
)


class SeenRegressionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.5f.1 deterministic seen regression on the already-"
            "consumed v3 reserve. No extraction/Explorer/Maker LLM is called."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_5F_PROTOCOL_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--confirm-refreeze-consumed-seen",
        action="store_true",
        help=(
            "Allow deterministic canonical rebuilds from existing Strict "
            "chunk outputs when readiness requires migration."
        ),
    )
    return parser.parse_args()


def _run(stage: str, args: list[str], *, log_path: Path) -> None:
    print(f"\n[alpha4c.5f.1] {stage}")
    print("[alpha4c.5f.1] command:", " ".join(args))
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": stage,
                    "command": args,
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-8000:],
                    "stderr_tail": result.stderr[-8000:],
                    "finished_at": now_iso(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    if result.returncode != 0:
        raise SeenRegressionError(
            f"{stage} failed with exit code {result.returncode}"
        )


def _python(stage: str, module: str, *args: str, log_path: Path) -> None:
    _run(
        stage,
        [sys.executable, "-m", module, *args],
        log_path=log_path,
    )


def main() -> int:
    args = parse_args()
    if not args.confirm_refreeze_consumed_seen:
        raise SystemExit(
            "--confirm-refreeze-consumed-seen is required. This regression "
            "may deterministically rebuild current canonical graphs from "
            "already-frozen Strict chunk outputs."
        )

    protocol = load_5f_protocol(ROOT / args.protocol)
    frozen_issues = verify_5f_protocol(
        ROOT,
        protocol,
        check_canonical_presence=True,
    )
    if frozen_issues:
        raise SeenRegressionError(
            "Frozen alpha4c.5f implementation/protocol drifted before "
            "5f.1 regression:\n- " + "\n- ".join(frozen_issues)
        )
    historical_root = ROOT / protocol.evaluation_root
    consumed = historical_root / "consumption_started.json"
    failed = historical_root / "CAMPAIGN_FAIL.json"
    passed = historical_root / "CAMPAIGN_PASS.json"
    if not consumed.exists() or not failed.exists() or passed.exists():
        raise SeenRegressionError(
            "Expected an already-consumed failed alpha4c.5f campaign. "
            "This script must never be used on an untouched reserve."
        )

    output_root = ROOT / args.output_root
    if output_root.exists():
        raise SeenRegressionError(
            f"Seen regression output already exists: {output_root}. "
            "Use a new --output-root rather than overwriting provenance."
        )
    output_root.mkdir(parents=True, exist_ok=False)
    lock_path = output_root / "canonical_readiness_lock.json"
    command_log = output_root / "command_log.jsonl"

    print("alpha4c.5f.1 canonical preparation on CONSUMED/SEEN v3 fixture")
    print("Historical reserve rerun: False")
    print("New extraction LLM calls: 0")
    print("Explorer/Maker LLM calls: 0")

    lock = prepare_sers_canonical_readiness_lock(
        paper_ids=protocol.reserve_paper_ids,
        output_path=lock_path,
        allow_refreeze=True,
        source_label=(
            "alpha4c.5f consumed-v3 seen regression; never acceptance"
        ),
    )
    verified = load_and_verify_readiness_lock(
        root=ROOT,
        lock_path=lock_path,
        expected_paper_ids=protocol.reserve_paper_ids,
        expected_domain_profile_id=SERS_DOMAIN_PROFILE_ID,
    )
    if verified["lock_sha256"] != lock["lock_sha256"]:
        raise CanonicalReadinessError("Readiness lock changed after write.")

    work_data = output_root / "work_data_sers"
    for paper_id in protocol.reserve_paper_ids:
        record = lock["paper_records"][paper_id]
        source = ROOT / record["canonical"]["canonical_path"]
        dest = (
            work_data
            / "extracted"
            / paper_id
            / f"{paper_id}.graphml"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        if sha256_file(dest) != record["canonical"]["canonical_sha256"]:
            raise SeenRegressionError(
                f"{paper_id}: seen-regression canonical copy SHA mismatch."
            )

    ids = {
        "corpus": "sers_alpha4c5f1_seen_v3_corpus",
        "measurement_result_identity": (
            "sers_alpha4c5f1_seen_v3_measurement_identity"
        ),
        "metric_definition": "sers_alpha4c5f1_seen_v3_metric_definition",
        "comparison": "sers_alpha4c5f1_seen_v3_comparison",
    }

    for paper_id in protocol.reserve_paper_ids:
        _python(
            f"projection:{paper_id}",
            "scripts.build_graphagents_projection",
            "--paper-id", paper_id,
            "--domain-profile", SERS_DOMAIN_PROFILE_ID,
            "--data-root", str(work_data),
            "--mode", "evidence",
            log_path=command_log,
        )

    _python(
        "corpus",
        "scripts.build_corpus_graph",
        "--corpus-id", ids["corpus"],
        "--domain-profile", SERS_DOMAIN_PROFILE_ID,
        "--data-root", str(work_data),
        "--paper-ids", *protocol.reserve_paper_ids,
        "--mode", "evidence",
        "--allow-critical-partial",
        log_path=command_log,
    )
    _python(
        "measurement_result_identity",
        "scripts.build_measurement_result_identities",
        "--domain-profile", SERS_DOMAIN_PROFILE_ID,
        "--data-root", str(work_data),
        "--corpus-id", ids["corpus"],
        "--mode", "evidence",
        "--measurement-result-identity-id",
        ids["measurement_result_identity"],
        log_path=command_log,
    )
    _python(
        "metric_definition",
        "scripts.build_metric_definition_contexts",
        "--domain-profile", SERS_DOMAIN_PROFILE_ID,
        "--data-root", str(work_data),
        "--corpus-id", ids["corpus"],
        "--mode", "evidence",
        "--metric-definition-id", ids["metric_definition"],
        "--measurement-result-identity-id",
        ids["measurement_result_identity"],
        log_path=command_log,
    )
    _python(
        "comparison",
        "scripts.build_comparison_contexts",
        "--domain-profile", SERS_DOMAIN_PROFILE_ID,
        "--data-root", str(work_data),
        "--corpus-id", ids["corpus"],
        "--mode", "evidence",
        "--comparison-id", ids["comparison"],
        "--metric-definition-id", ids["metric_definition"],
        "--measurement-result-identity-id",
        ids["measurement_result_identity"],
        log_path=command_log,
    )

    refrozen = [
        paper_id
        for paper_id in protocol.reserve_paper_ids
        if lock["paper_records"][paper_id]["refrozen"]
    ]
    summary = {
        "phase": "alpha4c.5f.1",
        "regression_id": "alpha4c5f1_consumed_v3_seen_regression_v1",
        "completed_at": now_iso(),
        "historical_campaign_id": protocol.campaign_id,
        "historical_reserve_consumed": True,
        "historical_reserve_rerun": False,
        "acceptance_evaluation": False,
        "paper_ids": protocol.reserve_paper_ids,
        "readiness_lock_path": str(lock_path),
        "readiness_lock_sha256": lock["lock_sha256"],
        "refrozen_papers": refrozen,
        "refrozen_paper_count": len(refrozen),
        "comparison_completed": True,
        "new_extraction_llm_calls": 0,
        "explorer_llm_calls": 0,
        "maker_llm_calls": 0,
        "count_thresholds_used": False,
        "result": "PASS",
    }
    atomic_json(output_root / "REGRESSION_PASS.json", summary)

    print("\nalpha4c.5f.1 consumed-v3 seen regression: PASS")
    print("Readiness lock:", lock_path)
    print("Refrozen papers:", len(refrozen), refrozen)
    print("Comparison completed: True")
    print("Historical reserve rerun: False")
    print("Acceptance evaluation: False")
    print("New extraction LLM calls: 0")
    print("Explorer/Maker LLM calls: 0")
    print("Count thresholds used: False")
    print("Saved:", output_root / "REGRESSION_PASS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
