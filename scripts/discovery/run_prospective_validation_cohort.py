from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.prospective_validation import (
    ProspectiveCaseAudit,
    build_cohort_audit,
)

DEVELOPMENT_CASES = (
    "sers_raman_orientation_direct_ho_reference",
    "sers_au_ag_structure_broad_control",
    "dac_her_multifactor_optimal_regime",
    "dac_her_charge_transfer_conditional",
    "dac_her_metal_distance_stability",
    "dac_her_low_overpotential_comparison",
)
PAUSED_PROVIDER_BUDGET_EXIT = 75


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cases(
    *,
    cohort_kind: str,
    cohort_root: Path | None,
    manifest: Path | None,
) -> list[tuple[str, Path]]:
    if cohort_kind == "DEVELOPMENT":
        if cohort_root is None:
            raise ValueError("DEVELOPMENT cohort requires --cohort-root")
        return [(name.upper(), (cohort_root / name).resolve()) for name in DEVELOPMENT_CASES]

    if manifest is None:
        raise ValueError("HELD_OUT cohort requires --manifest")
    payload = _load(manifest)
    if payload.get("cohort_kind") != "HELD_OUT":
        raise ValueError('held-out manifest must declare cohort_kind="HELD_OUT"')
    if payload.get("frozen_before_run") is not True:
        raise ValueError("held-out manifest must declare frozen_before_run=true")
    frozen_at = str(payload.get("frozen_at_utc") or "").strip()
    if not frozen_at or "REPLACE" in frozen_at.upper():
        raise ValueError("held-out manifest requires a concrete frozen_at_utc before execution")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("held-out manifest cases must be a non-empty list")

    result: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()
    seen_dirs: set[str] = set()
    dev_names = set(DEVELOPMENT_CASES)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("held-out case entry must be an object")
        case_id = str(row.get("case_id") or "").strip()
        run_dir = str(row.get("run_dir") or "").strip()
        if not case_id or not run_dir:
            raise ValueError("held-out case requires case_id and run_dir")
        resolved = Path(run_dir).expanduser().resolve()
        if case_id in seen_ids or str(resolved) in seen_dirs:
            raise ValueError("held-out manifest contains duplicate case_id or run_dir")
        if case_id.lower() in dev_names or resolved.name.lower() in dev_names:
            raise ValueError(
                f"held-out case overlaps the development cohort: {case_id} -> {resolved}"
            )
        seen_ids.add(case_id)
        seen_dirs.add(str(resolved))
        result.append((case_id, resolved))
    return result


def _heldout_lock_payload(manifest: Path, cases: list[tuple[str, Path]]) -> dict[str, Any]:
    manifest_payload = _load(manifest)
    source_rows = []
    for case_id, run_dir in cases:
        required = {
            "context": run_dir / "hypothesis.context.json",
            "legacy_portfolio": run_dir / "hypothesis_axis_a4.portfolio.json",
            "frontier_population": run_dir / "frontier_augmented" / "frontier_idea_population.full.shadow.json",
            "frontier_audit": run_dir / "frontier_augmented" / "frontier_exploration.audit.json",
            "idea_evolution": run_dir / "frontier_augmented" / "idea_evolution" / "idea_evolution.shadow.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"held-out source freeze missing artifacts for {case_id}: {missing}"
            )
        source_rows.append(
            {
                "case_id": case_id,
                "run_dir": str(run_dir),
                "source_sha256": {
                    name: _sha256_file(path) for name, path in required.items()
                },
            }
        )
    canonical_manifest = json.dumps(
        manifest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": "held-out-prospective-source-lock-v1",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
        "frozen_at_utc": manifest_payload.get("frozen_at_utc"),
        "cases": source_rows,
        "scientific_outcomes_observed_before_lock": False,
        "production_selection_authority": False,
    }


def _enforce_heldout_lock(
    *,
    out: Path,
    manifest: Path,
    cases: list[tuple[str, Path]],
) -> Path:
    lock_path = out / "held_out_manifest.lock.json"
    current = _heldout_lock_payload(manifest, cases)
    if lock_path.is_file():
        locked = _load(lock_path)
        if locked != current:
            raise RuntimeError(
                "held-out manifest/source artifacts changed after the prospective lock was created"
            )
    else:
        _write(lock_path, current)
        print("Held-out prospective source lock created:", lock_path)
    return lock_path


def _case_needs_resume(audit_path: Path) -> bool:
    if not audit_path.is_file():
        return True
    try:
        audit = ProspectiveCaseAudit.model_validate_json(audit_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    if audit.execution_status != "COMPLETE":
        return True
    for row in audit.arms:
        if row.operational_status in {
            "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
            "FAILED_OPERATIONAL",
            "IN_PROGRESS",
        }:
            return True
        # Backward-compatible recovery of v1 audits written before explicit
        # operational status existed.
        if row.operational_status == "UNKNOWN_LEGACY":
            if row.hypothesis_count == 0:
                continue
            if row.semantic_status == "ACCEPTED" and not row.verification_completed:
                return True
            if row.semantic_status == "NOT_RUN":
                return True
    return False


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run/resume the four-arm creativity-stack ablation over a development "
            "or explicitly frozen held-out cohort. Provider quota exhaustion pauses "
            "the cohort without converting it into a scientific failure."
        )
    )
    p.add_argument("--cohort-kind", choices=["DEVELOPMENT", "HELD_OUT"], required=True)
    p.add_argument("--cohort-root", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-hypotheses", type=int, default=8)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--providers", default="auto")
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--force", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.expanduser().resolve() if args.manifest is not None else None
    cases = _cases(
        cohort_kind=args.cohort_kind,
        cohort_root=(args.cohort_root.expanduser().resolve() if args.cohort_root is not None else None),
        manifest=manifest,
    )
    heldout_lock = None
    if args.cohort_kind == "HELD_OUT":
        assert manifest is not None
        heldout_lock = _enforce_heldout_lock(out=out, manifest=manifest, cases=cases)

    status: dict[str, Any] = {
        "schema_version": "prospective-validation-cohort-execution-v2",
        "cohort_kind": args.cohort_kind,
        "planned_case_count": len(cases),
        "held_out_source_lock": str(heldout_lock) if heldout_lock else None,
        "cases": [],
        "paused_provider_budget": False,
        "production_selection_authority": False,
    }
    observed_audits: list[ProspectiveCaseAudit] = []
    paused_at: int | None = None

    for index, (case_id, run_dir) in enumerate(cases):
        case_out = out / case_id
        audit_path = case_out / "case.audit.json"
        print()
        print("#" * 96)
        print("CASE:", case_id)
        print("RUN :", run_dir)
        print("#" * 96)

        needs_resume = args.force or _case_needs_resume(audit_path)
        result_code = 0
        if needs_resume:
            cmd = [
                sys.executable,
                "-m",
                "scripts.discovery.run_prospective_case_ablation",
                "--case-id",
                case_id,
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(case_out),
                "--max-hypotheses",
                str(args.max_hypotheses),
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
                "--providers",
                args.providers,
                "--results-per-query",
                str(args.results_per_query),
            ]
            if args.base_url:
                cmd += ["--base-url", args.base_url]
            if args.force:
                cmd += ["--force"]
            if args.cohort_kind == "HELD_OUT":
                cmd += ["--rebuild-selected-arm"]
            result = subprocess.run(cmd)
            result_code = result.returncode
        else:
            print("Reusing terminal case audit:", audit_path)

        audit = None
        if audit_path.is_file():
            try:
                audit = ProspectiveCaseAudit.model_validate_json(
                    audit_path.read_text(encoding="utf-8")
                )
                observed_audits.append(audit)
            except Exception:
                audit = None

        if result_code == PAUSED_PROVIDER_BUDGET_EXIT or (
            audit is not None
            and audit.execution_status == "PAUSED_PROVIDER_BUDGET_EXHAUSTED"
        ):
            status["paused_provider_budget"] = True
            status["cases"].append(
                {
                    "case_id": case_id,
                    "run_dir": str(run_dir),
                    "status": "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
                    "audit": str(audit_path) if audit_path.is_file() else None,
                    "paused_arm": audit.paused_arm if audit is not None else None,
                }
            )
            paused_at = index
            _write(out / "execution_status.json", status)
            break

        if audit is None:
            status["cases"].append(
                {
                    "case_id": case_id,
                    "run_dir": str(run_dir),
                    "status": "FAILED_OPERATIONAL",
                    "return_code": result_code,
                    "error": "case ablation produced no readable case.audit.json",
                }
            )
        else:
            status["cases"].append(
                {
                    "case_id": case_id,
                    "run_dir": str(run_dir),
                    "status": audit.execution_status,
                    "return_code": result_code,
                    "audit": str(audit_path),
                    "all_four_arms_present": audit.all_four_arms_present,
                    "selected_arm_verification_complete": audit.selected_arm_verification_complete,
                }
            )
        _write(out / "execution_status.json", status)

    if paused_at is not None:
        for case_id, run_dir in cases[paused_at + 1 :]:
            status["cases"].append(
                {
                    "case_id": case_id,
                    "run_dir": str(run_dir),
                    "status": "NOT_STARTED_PROVIDER_BUDGET_PAUSE",
                }
            )
        _write(out / "execution_status.json", status)

    cohort = build_cohort_audit(
        cohort_kind=args.cohort_kind,
        cases=observed_audits,
        planned_case_count=len(cases),
    )
    _write(out / "cohort.audit.json", cohort)

    # Diagnostic-only materialization survival sidecar. This reader never
    # changes selection, materialization, verification, provider usage, or
    # cohort exit semantics. A sidecar failure must not invalidate or resume
    # the primary prospective experiment.
    materialization_survival_path = (
        out / "materialization_survival.cohort.audit.json"
    )
    materialization_survival_cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_materialization_survival_audit",
        "--prospective-output-root",
        str(out),
        "--output",
        str(materialization_survival_path),
    ]
    materialization_survival_run = subprocess.run(
        materialization_survival_cmd
    )
    if materialization_survival_run.returncode != 0:
        print(
            "WARNING: Materialization survival audit failed; continuing primary cohort semantics",
            file=sys.stderr,
        )
    else:
        print(
            "materialization survival audit:",
            materialization_survival_path,
        )

    print()
    print("===== PROSPECTIVE VALIDATION COHORT =====")
    print("kind:", args.cohort_kind)
    print("observed cases:", cohort.case_count, "/", cohort.planned_case_count)
    print("complete cases:", cohort.complete_case_count)
    print("partial cases:", cohort.partial_case_count)
    print("provider-budget paused cases:", cohort.paused_provider_budget_case_count)
    print("four-arm records present:", cohort.four_arm_record_present_case_count)
    print("four-arm terminal complete:", cohort.complete_four_arm_case_count)

    def _fmt_optional_metric(mapping: dict[str, float | None], arm: str):
        value = mapping.get(arm)
        return "NA" if value is None else round(value, 3)

    for arm in (
        "LEGACY",
        "FRONTIER_BALANCED",
        "EVOLUTION_BALANCED",
        "PORTFOLIO_SELECTED",
    ):
        print(
            arm,
            "| cases=", cohort.arm_case_counts.get(arm, 0),
            "| semantic accepted=", cohort.arm_semantic_accepted_counts.get(arm, 0),
            "| external complete=", cohort.arm_external_complete_counts.get(arm, 0),
            "| N9 complete=", cohort.arm_n9_complete_counts.get(arm, 0),
            "| median hypotheses=", cohort.arm_hypothesis_count_medians.get(arm, 0.0),
            "| median materialization yield=", _fmt_optional_metric(cohort.arm_materialization_yield_medians, arm),
            "| median conceptual family fraction=", _fmt_optional_metric(cohort.arm_conceptual_family_fraction_medians, arm),
            "| median premise distinct fraction=", round(cohort.arm_premise_distinct_set_fraction_medians.get(arm, 0.0), 3),
            "| median premise Jaccard=", round(cohort.arm_mean_pairwise_premise_jaccard_medians.get(arm, 0.0), 3),
        )
        print("   operational:", cohort.arm_operational_status_counts.get(arm, {}))
        print("   external statuses:", cohort.arm_external_status_counts.get(arm, {}))

    print(
        "selected Frontier retention:",
        cohort.portfolio_selected_frontier_retention_case_count,
        "/",
        cohort.case_count,
    )
    print(
        "selected Evolution retention:",
        cohort.portfolio_selected_evolution_retention_case_count,
        "/",
        cohort.case_count,
    )
    print("authority invariants:", cohort.all_authority_invariants_hold)
    print("SCIENTIFIC_SUPERIORITY_ESTABLISHED=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("audit:", out / "cohort.audit.json")
    if status["paused_provider_budget"]:
        print("COHORT_PAUSED_PROVIDER_BUDGET_EXHAUSTED")
        return PAUSED_PROVIDER_BUDGET_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
