from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from domains.feasibility_registry import resolve_optional_feasibility_adapter
from domains.registry import get_domain_profile
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.scientific_portfolio_materialization import (
    ScientificPortfolioMaterializationReport,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(label: str, cmd: list[str], out_dir: Path) -> subprocess.CompletedProcess[str]:
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    safe = "_".join(label.lower().split())
    (out_dir / f"{safe}.stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (out_dir / f"{safe}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run existing semantic, external-prior-art, N9 non-obviousness, and "
            "feasibility machinery over a grounded shadow portfolio materialized "
            "from Scientific Portfolio Selection. No N10 or production authority is run."
        )
    )
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--materialization-report", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--domain-profile", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model", default=(os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL") or os.getenv("OPENROUTER_AGENT_MODEL") or ""))
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--providers", default="auto")
    p.add_argument("--provider-plan", type=Path, default=None)
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--skip-n9-full", action="store_true")
    p.add_argument(
        "--production-enforce",
        action="store_true",
        help=(
            "After complete N9, promote the same role-aware N10-v2 decision "
            "to an authoritative Scientific Portfolio production subset."
        ),
    )
    return p


def main() -> int:
    args = parser().parse_args()
    if not args.model:
        raise SystemExit("--model is required unless a configured model environment variable is set")
    if args.results_per_query < 1:
        raise ValueError("--results-per-query must be >= 1")

    context = args.context.expanduser().resolve()
    materialization_path = args.materialization_report.expanduser().resolve()
    portfolio_path = args.portfolio.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    materialization = ScientificPortfolioMaterializationReport.model_validate_json(
        materialization_path.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    if materialization.output_portfolio_id != portfolio.portfolio_id:
        raise ValueError("materialization report / portfolio lineage mismatch")

    summary: dict[str, Any] = {
        "schema_version": "scientific-portfolio-verification-shadow-v1",
        "source_materialization_report_id": materialization.report_id,
        "source_portfolio_id": portfolio.portfolio_id,
        "hypothesis_count": len(portfolio.hypotheses),
        "semantic": {"status": "NOT_RUN"},
        "external_novelty": {"status": "NOT_RUN"},
        "n9": {"status": "NOT_RUN"},
        "feasibility": {"status": "NOT_RUN"},
        "n10_run": False,
        "novelty_certification_authority": False,
        "production_selection_authority": False,
        "production_binding": {"status": "NOT_RUN"},
        "stage8_input_changed": False,
        "canonical_graph_mutated": False,
    }

    if not portfolio.hypotheses:
        summary["status"] = "SKIPPED_EMPTY_MATERIALIZED_PORTFOLIO"
        _write(out / "verification.summary.json", summary)
        print("Scientific Portfolio verification skipped: materialized portfolio is empty")
        return 0

    semantic_prefix = out / "semantic"
    semantic_cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_hypothesis_semantic_critic",
        "--context",
        str(context),
        "--portfolio",
        str(portfolio_path),
        "--model",
        args.model,
        "--api-key-env",
        args.api_key_env,
        "--output-prefix",
        str(semantic_prefix),
        "--save-prompt",
    ]
    if args.base_url:
        semantic_cmd += ["--base-url", args.base_url]
    semantic_run = _run("semantic verification", semantic_cmd, out)
    semantic_run_path = Path(str(semantic_prefix) + ".run.json")
    semantic_review_path = Path(str(semantic_prefix) + ".review.json")
    semantic_record = _load(semantic_run_path) if semantic_run_path.is_file() else {}
    semantic_accepted = bool(semantic_record.get("accepted")) and semantic_review_path.is_file()
    summary["semantic"] = {
        "status": "ACCEPTED" if semantic_accepted else "REJECTED_OR_FAILED",
        "return_code": semantic_run.returncode,
        "run": str(semantic_run_path),
        "review": str(semantic_review_path) if semantic_review_path.is_file() else None,
        "failure_stage": semantic_record.get("failure_stage"),
    }

    # Feasibility consumes the same standard portfolio but requires an accepted
    # semantic review. Capability absence is a valid multidomain state and must
    # not be reported as a scientific/runtime failure.
    if semantic_accepted:
        try:
            profile = get_domain_profile(args.domain_profile)
            feasibility_adapter = resolve_optional_feasibility_adapter(profile)
            feasibility_capability_error = None
        except Exception as exc:
            feasibility_adapter = None
            feasibility_capability_error = f"{type(exc).__name__}: {exc}"

        if feasibility_capability_error is not None:
            summary["feasibility"] = {
                "status": "FAILED_CAPABILITY_RESOLUTION",
                "error": feasibility_capability_error,
            }
        elif feasibility_adapter is None:
            summary["feasibility"] = {
                "status": "SKIPPED_UNSUPPORTED_DOMAIN",
                "domain_profile_id": args.domain_profile,
                "reason": "domain profile declares no feasibility adapter",
            }
        else:
            feasibility_dir = out / "feasibility"
            feas_cmd = [
                sys.executable,
                "-m",
                "scripts.discovery.run_feasibility_e2e",
                "--context",
                str(context),
                "--domain-profile",
                args.domain_profile,
                "--portfolio",
                str(portfolio_path),
                "--semantic-review",
                str(semantic_review_path),
                "--output-dir",
                str(feasibility_dir),
            ]
            feas_run = _run("feasibility verification", feas_cmd, out)
            decision_path = feasibility_dir / "decision" / "portfolio.json"
            decision_counts: dict[str, int] = {}
            if decision_path.is_file():
                payload = _load(decision_path)
                decision_counts = dict(
                    sorted(
                        Counter(
                            str(row.get("final_disposition"))
                            for row in payload.get("cards", [])
                        ).items()
                    )
                )
            summary["feasibility"] = {
                "status": (
                    "COMPLETE"
                    if feas_run.returncode == 0 and decision_path.is_file()
                    else "FAILED"
                ),
                "return_code": feas_run.returncode,
                "adapter_id": feasibility_adapter.adapter_id,
                "decision_portfolio": (
                    str(decision_path) if decision_path.is_file() else None
                ),
                "final_disposition_counts": decision_counts,
            }
    else:
        summary["feasibility"] = {
            "status": "SKIPPED_SEMANTIC_NOT_ACCEPTED"
        }

    if semantic_accepted:
        external_prefix = out / "external_novelty"
        ext_cmd = [
            sys.executable,
            "-m",
            "scripts.discovery.run_external_novelty",
            "--portfolio",
            str(portfolio_path),
            "--domain-profile",
            args.domain_profile,
            "--model",
            args.model,
            "--api-key-env",
            args.api_key_env,
            "--providers",
            args.providers,
            "--results-per-query",
            str(args.results_per_query),
            "--output-prefix",
            str(external_prefix),
            "--pre-review-metadata-resolution",
            "--pre-review-coverage-shadow",
            "--downstream-gate-shadow",
            "--source-bound-topology-shadow",
            "--save-prompts",
        ]
        if args.base_url:
            ext_cmd += ["--base-url", args.base_url]
        if args.provider_plan is not None:
            ext_cmd += ["--provider-plan", str(args.provider_plan.expanduser().resolve())]
        ext_run = _run("external novelty verification", ext_cmd, out)
        ext_report = Path(str(external_prefix) + ".report.json")
        ext_plan = Path(str(external_prefix) + ".claims_queries.json")
        ext_prior = Path(str(external_prefix) + ".prior_art.json")
        ext_provider_plan = Path(str(external_prefix) + ".provider_plan.json")
        ext_payload = _load(ext_report) if ext_report.is_file() else {}
        ext_complete = (
            ext_run.returncode == 0
            and ext_report.is_file()
            and ext_plan.is_file()
            and ext_prior.is_file()
            and ext_provider_plan.is_file()
        )
        summary["external_novelty"] = {
            "status": "COMPLETE" if ext_complete else "FAILED",
            "return_code": ext_run.returncode,
            "report": str(ext_report) if ext_report.is_file() else None,
            "query_plan": str(ext_plan) if ext_plan.is_file() else None,
            "prior_art": str(ext_prior) if ext_prior.is_file() else None,
            "provider_plan": str(ext_provider_plan) if ext_provider_plan.is_file() else None,
            "status_counts": ext_payload.get("status_counts", {}),
            "epistemic_usage": ext_payload.get("epistemic_usage"),
        }

        if ext_complete:
            n9_intake = out / "n9.intake.shadow.json"
            intake_cmd = [
                sys.executable,
                "-m",
                "scripts.discovery.build_nonobviousness_shadow",
                "--query-plan",
                str(ext_plan),
                "--external-report",
                str(ext_report),
                "--portfolio",
                str(portfolio_path),
                "--output",
                str(n9_intake),
            ]
            intake_run = _run("N9 intake shadow", intake_cmd, out)
            n9_complete = intake_run.returncode == 0 and n9_intake.is_file()
            n9_full = out / "n9.full_shadow.json"
            full_run_code = None
            if n9_complete and not args.skip_n9_full:
                full_cmd = [
                    sys.executable,
                    "-m",
                    "scripts.discovery.run_nonobviousness_full_shadow",
                    "--query-plan",
                    str(ext_plan),
                    "--external-report",
                    str(ext_report),
                    "--external-prior-art",
                    str(ext_prior),
                    "--portfolio",
                    str(portfolio_path),
                    "--hypothesis-context",
                    str(context),
                    "--intake-shadow",
                    str(n9_intake),
                    "--provider-plan",
                    str(ext_provider_plan),
                    "--domain-profile",
                    args.domain_profile,
                    "--model",
                    args.model,
                    "--api-key-env",
                    args.api_key_env,
                    "--results-per-query",
                    str(args.results_per_query),
                    "--output",
                    str(n9_full),
                ]
                if args.base_url:
                    full_cmd += ["--base-url", args.base_url]
                full_run = _run("N9 full non-obviousness shadow", full_cmd, out)
                full_run_code = full_run.returncode
                n9_complete = n9_complete and full_run.returncode == 0 and n9_full.is_file()
            summary["n9"] = {
                "status": (
                    "INTAKE_ONLY"
                    if args.skip_n9_full and n9_complete
                    else ("COMPLETE" if n9_complete else "FAILED")
                ),
                "intake": str(n9_intake) if n9_intake.is_file() else None,
                "full_shadow": str(n9_full) if n9_full.is_file() else None,
                "intake_return_code": intake_run.returncode,
                "full_return_code": full_run_code,
            }
        else:
            summary["n9"] = {"status": "SKIPPED_EXTERNAL_NOVELTY_FAILED"}
    else:
        summary["external_novelty"] = {"status": "SKIPPED_SEMANTIC_NOT_ACCEPTED"}
        summary["n9"] = {"status": "SKIPPED_SEMANTIC_NOT_ACCEPTED"}

    if args.production_enforce:
        if args.skip_n9_full:
            raise ValueError(
                "--production-enforce is incompatible with --skip-n9-full"
            )

        if summary["n9"].get("status") != "COMPLETE":
            summary["production_binding"] = {
                "status": "BLOCKED_N9_INCOMPLETE",
            }
        else:
            n10_candidate_gate = out / "n10.production_candidate_gate.json"
            n10_production_gate = out / "n10.production_gate.json"
            production_candidate_portfolio = (
                out / "production.candidate.portfolio.json"
            )
            production_certification_report = (
                out / "n10.certification.json"
            )
            production_certified_portfolio = (
                out / "n10.certified.portfolio.json"
            )

            candidate_run = _run(
                "N10 role-aware candidate gate",
                [
                    sys.executable,
                    "-m",
                    "scripts.discovery.build_nonobviousness_production_gate_v2_candidate",
                    "--query-plan",
                    str(Path(summary["external_novelty"]["query_plan"])),
                    "--intake-shadow",
                    str(Path(summary["n9"]["intake"])),
                    "--full-shadow",
                    str(Path(summary["n9"]["full_shadow"])),
                    "--output",
                    str(n10_candidate_gate),
                ],
                out,
            )

            production_gate_run = _run(
                "N10 role-aware production gate",
                [
                    sys.executable,
                    "-m",
                    "scripts.discovery.build_nonobviousness_production_gate_v2",
                    "--candidate-gate",
                    str(n10_candidate_gate),
                    "--output",
                    str(n10_production_gate),
                ],
                out,
            )

            binding_run = _run(
                "Scientific Portfolio N10 certification",
                [
                    sys.executable,
                    "-m",
                    "scripts.discovery.bind_scientific_portfolio_production",
                    "--authority-mode",
                    "certification_only",
                    "--portfolio",
                    str(portfolio_path),
                    "--n10-production-gate",
                    str(n10_production_gate),
                    "--output-candidate-portfolio",
                    str(production_candidate_portfolio),
                    "--output-certification-report",
                    str(production_certification_report),
                    "--output-certified-portfolio",
                    str(production_certified_portfolio),
                ],
                out,
            )

            production_complete = (
                candidate_run.returncode == 0
                and production_gate_run.returncode == 0
                and binding_run.returncode == 0
                and n10_candidate_gate.is_file()
                and n10_production_gate.is_file()
                and production_candidate_portfolio.is_file()
                and production_certification_report.is_file()
                and production_certified_portfolio.is_file()
            )

            certification_payload = (
                _load(production_certification_report)
                if production_certification_report.is_file()
                else {}
            )

            summary["production_binding"] = {
                "status": "COMPLETE" if production_complete else "FAILED",
                "mode": "certification_only",
                "candidate_gate": str(n10_candidate_gate) if n10_candidate_gate.is_file() else None,
                "production_gate": str(n10_production_gate) if n10_production_gate.is_file() else None,
                "scientific_candidate_portfolio": str(production_candidate_portfolio) if production_candidate_portfolio.is_file() else None,
                "certification_report": str(production_certification_report) if production_certification_report.is_file() else None,
                "certified_novelty_portfolio": str(production_certified_portfolio) if production_certified_portfolio.is_file() else None,
                "scientific_candidate_count": certification_payload.get("hypothesis_count", 0),
                "novelty_certified_count": certification_payload.get("certified_count", 0),
                "conditional_count": certification_payload.get("conditional_count", 0),
                "ineligible_count": certification_payload.get("ineligible_count", 0),
                "scientific_candidate_authority": certification_payload.get("scientific_candidate_authority", False),
                "n10_candidate_survival_authority": certification_payload.get("n10_candidate_survival_authority", True),
                "conditional_candidates_retained": certification_payload.get("conditional_candidates_retained", False),
            }

            if production_complete:
                summary["n10_run"] = True
                summary["novelty_certification_authority"] = True
                summary["production_selection_authority"] = True

    terminal_statuses = {
        "semantic": summary["semantic"].get("status"),
        "external_novelty": summary["external_novelty"].get("status"),
        "n9": summary["n9"].get("status"),
        "feasibility": summary["feasibility"].get("status"),
    }
    expected_ok = {
        "semantic": {"ACCEPTED"},
        "external_novelty": {"COMPLETE"},
        "n9": {"COMPLETE", "INTAKE_ONLY"},
        "feasibility": {"COMPLETE", "SKIPPED_UNSUPPORTED_DOMAIN"},
    }
    base_complete = all(
        terminal_statuses[k] in expected_ok[k]
        for k in terminal_statuses
    )
    production_complete = (
        summary.get("production_binding", {}).get("status") == "COMPLETE"
    )
    summary["status"] = (
        "COMPLETE_PRODUCTION_VERIFICATION"
        if args.production_enforce and base_complete and production_complete
        else (
            "COMPLETE_SHADOW_VERIFICATION"
            if (not args.production_enforce) and base_complete
            else "PARTIAL_SHADOW_VERIFICATION"
        )
    )
    _write(out / "verification.summary.json", summary)
    print()
    print("Scientific Portfolio downstream verification shadow complete")
    print("semantic:", summary["semantic"]["status"])
    print("external novelty:", summary["external_novelty"]["status"])
    print("N9:", summary["n9"]["status"])
    print("feasibility:", summary["feasibility"]["status"])
    print("N10_RUN=", summary["n10_run"])
    print(
        "PRODUCTION_SELECTION_AUTHORITY=",
        summary["production_selection_authority"],
    )
    print("summary:", out / "verification.summary.json")
    if args.production_enforce and (
        summary.get("production_binding", {}).get("status") != "COMPLETE"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
