from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from domains.feasibility_registry import resolve_optional_feasibility_adapter
from domains.registry import get_domain_profile
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.prospective_validation import (
    provider_budget_exhausted_from_text,
)

PAUSED_PROVIDER_BUDGET_EXIT = 75


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(label: str, cmd: list[str], out: Path) -> subprocess.CompletedProcess[str]:
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    safe = "_".join(label.lower().split())
    (out / f"{safe}.stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (out / f"{safe}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result


def _provider_plan_uses_openalex(provider_plan: Path | None, providers: str) -> bool:
    if provider_plan is not None and provider_plan.expanduser().is_file():
        try:
            payload = _load(provider_plan.expanduser().resolve())
            return "openalex" in json.dumps(payload, ensure_ascii=False).lower()
        except Exception:
            pass
    values = {x.strip().lower() for x in str(providers or "").split(",") if x.strip()}
    return "auto" in values or "openalex" in values


def _probe_openalex_budget() -> dict[str, Any] | None:
    key = str(os.getenv("OPENALEX_API_KEY") or "").strip()
    if not key:
        return None
    url = (
        "https://api.openalex.org/rate-limit?api_key="
        + urllib.parse.quote(key, safe="")
    )
    try:
        with urllib.request.urlopen(url, timeout=12.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {
            "checked": True,
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    rate = payload.get("rate_limit") if isinstance(payload, dict) else None
    if not isinstance(rate, dict):
        return {"checked": True, "available": False, "error": "missing rate_limit"}
    remaining_usd = rate.get("daily_remaining_usd")
    remaining_credits = rate.get("credits_remaining")
    exhausted = False
    try:
        if remaining_usd is not None and float(remaining_usd) <= 0:
            exhausted = True
    except (TypeError, ValueError):
        pass
    try:
        if remaining_credits is not None and int(remaining_credits) <= 0:
            exhausted = True
    except (TypeError, ValueError):
        pass
    return {
        "checked": True,
        "available": True,
        "exhausted": exhausted,
        "daily_remaining_usd": remaining_usd,
        "credits_remaining": remaining_credits,
        "resets_at": rate.get("resets_at"),
        "resets_in_seconds": rate.get("resets_in_seconds"),
    }


def _stage_budget_probe(
    *,
    providers: str,
    provider_plan: Path | None,
    disabled: bool,
) -> dict[str, Any] | None:
    if disabled or not _provider_plan_uses_openalex(provider_plan, providers):
        return None
    return _probe_openalex_budget()


def _external_artifacts(prefix: Path) -> tuple[Path, Path, Path, Path]:
    return (
        Path(str(prefix) + ".report.json"),
        Path(str(prefix) + ".claims_queries.json"),
        Path(str(prefix) + ".prior_art.json"),
        Path(str(prefix) + ".provider_plan.json"),
    )


def _external_complete(prefix: Path, portfolio_id: str) -> bool:
    report, plan, prior, provider_plan = _external_artifacts(prefix)
    if not all(path.is_file() for path in (report, plan, prior, provider_plan)):
        return False
    try:
        return str(_load(report).get("source_portfolio_id") or "") == portfolio_id
    except Exception:
        return False


def _decision_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    payload = _load(path)
    return dict(
        sorted(
            Counter(
                str(row.get("final_disposition"))
                for row in payload.get("cards", [])
                if isinstance(row, dict)
            ).items()
        )
    )


def _checkpoint(out: Path, summary: dict[str, Any]) -> None:
    _write(out / "verification.checkpoint.json", summary)
    _write(out / "verification.summary.json", summary)


def _pause(
    *,
    out: Path,
    summary: dict[str, Any],
    stage: str,
    budget: dict[str, Any] | None,
    stage_key: str,
    return_code: int | None = None,
) -> int:
    summary[stage_key] = {
        **dict(summary.get(stage_key) or {}),
        "status": "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
        "return_code": return_code,
    }
    summary["operational_status"] = "PAUSED_PROVIDER_BUDGET_EXHAUSTED"
    summary["resume_stage"] = stage
    summary["provider_budget"] = budget or {"exhausted": True, "source": "subprocess"}
    summary["status"] = "PAUSED_PROVIDER_BUDGET_EXHAUSTED"
    _checkpoint(out, summary)
    print()
    print("Verification paused: provider budget exhausted")
    print("resume stage:", stage)
    if budget:
        print("provider budget:", budget)
    return PAUSED_PROVIDER_BUDGET_EXIT


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run semantic, external-prior-art, N9, and capability-aware feasibility "
            "over a standard HypothesisPortfolio with stage checkpoint/resume. "
            "Provider-budget exhaustion is PAUSED, never a scientific failure."
        )
    )
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--portfolio", type=Path, required=True)
    p.add_argument("--domain-profile", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--providers", default="auto")
    p.add_argument("--provider-plan", type=Path, default=None)
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--skip-n9-full", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-provider-budget-preflight", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    if not args.model:
        raise SystemExit("--model is required unless a configured model environment variable is set")
    if args.results_per_query < 1:
        raise ValueError("--results-per-query must be >= 1")

    context = args.context.expanduser().resolve()
    portfolio_path = args.portfolio.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    provider_plan_arg = (
        args.provider_plan.expanduser().resolve() if args.provider_plan is not None else None
    )

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    summary: dict[str, Any] = {
        "schema_version": "standard-portfolio-verification-shadow-v2",
        "source_portfolio_id": portfolio.portfolio_id,
        "hypothesis_count": len(portfolio.hypotheses),
        "semantic": {"status": "NOT_RUN"},
        "external_novelty": {"status": "NOT_RUN"},
        "n9": {"status": "NOT_RUN"},
        "feasibility": {"status": "NOT_RUN"},
        "operational_status": "IN_PROGRESS",
        "resume_stage": None,
        "provider_budget": None,
        "n10_run": False,
        "novelty_certification_authority": False,
        "production_selection_authority": False,
        "canonical_graph_mutated": False,
    }
    _checkpoint(out, summary)

    if not portfolio.hypotheses:
        summary["operational_status"] = "SCIENTIFIC_TERMINAL"
        summary["status"] = "SKIPPED_EMPTY_PORTFOLIO"
        _checkpoint(out, summary)
        print("Standard portfolio verification skipped: portfolio is empty")
        return 0

    # ------------------------------------------------------------------
    # Semantic critic: reuse an accepted/rejected completed run record.
    # ------------------------------------------------------------------
    semantic_prefix = out / "semantic"
    semantic_run_path = Path(str(semantic_prefix) + ".run.json")
    semantic_review_path = Path(str(semantic_prefix) + ".review.json")
    semantic_record: dict[str, Any] = {}
    semantic_reused = False
    if not args.force and semantic_run_path.is_file():
        try:
            semantic_record = _load(semantic_run_path)
            semantic_reused = True
        except Exception:
            semantic_record = {}
            semantic_reused = False

    if not semantic_reused:
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
        semantic_record = _load(semantic_run_path) if semantic_run_path.is_file() else {}
        semantic_return = semantic_run.returncode
    else:
        semantic_return = int(semantic_record.get("return_code") or 0)
        print("Reusing completed semantic stage:", semantic_run_path)

    semantic_accepted = bool(semantic_record.get("accepted")) and semantic_review_path.is_file()
    semantic_generated = bool(semantic_record)
    summary["semantic"] = {
        "status": "ACCEPTED" if semantic_accepted else "REJECTED_OR_FAILED",
        "return_code": semantic_return,
        "reused": semantic_reused,
        "run": str(semantic_run_path) if semantic_run_path.is_file() else None,
        "review": str(semantic_review_path) if semantic_review_path.is_file() else None,
        "failure_stage": semantic_record.get("failure_stage"),
    }
    _checkpoint(out, summary)

    if not semantic_accepted:
        if semantic_generated:
            summary["feasibility"] = {"status": "SKIPPED_SEMANTIC_NOT_ACCEPTED"}
            summary["external_novelty"] = {"status": "SKIPPED_SEMANTIC_NOT_ACCEPTED"}
            summary["n9"] = {"status": "SKIPPED_SEMANTIC_NOT_ACCEPTED"}
            summary["operational_status"] = "SCIENTIFIC_TERMINAL"
            summary["status"] = "SCIENTIFIC_TERMINAL_SEMANTIC_REJECTED"
            _checkpoint(out, summary)
            return 0
        summary["operational_status"] = "FAILED_OPERATIONAL"
        summary["status"] = "FAILED_OPERATIONAL"
        _checkpoint(out, summary)
        return 1

    # ------------------------------------------------------------------
    # Feasibility: independent of external prior-art; reuse when complete.
    # ------------------------------------------------------------------
    profile = get_domain_profile(args.domain_profile)
    adapter = resolve_optional_feasibility_adapter(profile)
    if adapter is None:
        summary["feasibility"] = {
            "status": "SKIPPED_UNSUPPORTED_DOMAIN",
            "domain_profile_id": profile.profile_id,
            "reused": True,
        }
    else:
        feasibility_dir = out / "feasibility"
        decision_path = feasibility_dir / "decision" / "portfolio.json"
        if not args.force and decision_path.is_file():
            summary["feasibility"] = {
                "status": "COMPLETE",
                "return_code": 0,
                "reused": True,
                "decision_portfolio": str(decision_path),
                "final_disposition_counts": _decision_counts(decision_path),
            }
        else:
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
            complete = feas_run.returncode == 0 and decision_path.is_file()
            summary["feasibility"] = {
                "status": "COMPLETE" if complete else "FAILED",
                "return_code": feas_run.returncode,
                "reused": False,
                "decision_portfolio": str(decision_path) if decision_path.is_file() else None,
                "final_disposition_counts": _decision_counts(decision_path),
            }
    _checkpoint(out, summary)

    # ------------------------------------------------------------------
    # External novelty: provider-budget aware and resumable.
    # ------------------------------------------------------------------
    external_prefix = out / "external_novelty"
    ext_report, ext_plan, ext_prior, ext_provider_plan = _external_artifacts(external_prefix)
    ext_complete = (not args.force) and _external_complete(external_prefix, portfolio.portfolio_id)
    ext_run_code: int | None = 0 if ext_complete else None
    ext_stdout = ""
    ext_stderr = ""

    if not ext_complete:
        budget = _stage_budget_probe(
            providers=args.providers,
            provider_plan=provider_plan_arg,
            disabled=args.skip_provider_budget_preflight,
        )
        if budget and budget.get("available") and budget.get("exhausted"):
            return _pause(
                out=out,
                summary=summary,
                stage="EXTERNAL_NOVELTY",
                budget=budget,
                stage_key="external_novelty",
            )

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
            "--pre-review-coverage-shadow",
            "--downstream-gate-shadow",
            "--save-prompts",
        ]
        if args.base_url:
            ext_cmd += ["--base-url", args.base_url]
        if provider_plan_arg is not None:
            ext_cmd += ["--provider-plan", str(provider_plan_arg)]
        ext_run = _run("external novelty verification", ext_cmd, out)
        ext_run_code = ext_run.returncode
        ext_stdout, ext_stderr = ext_run.stdout or "", ext_run.stderr or ""
        ext_complete = _external_complete(external_prefix, portfolio.portfolio_id) and ext_run.returncode == 0
        if not ext_complete and provider_budget_exhausted_from_text(ext_stdout, ext_stderr):
            return _pause(
                out=out,
                summary=summary,
                stage="EXTERNAL_NOVELTY",
                budget={"exhausted": True, "source": "external_novelty_subprocess"},
                stage_key="external_novelty",
                return_code=ext_run.returncode,
            )

    ext_payload = _load(ext_report) if ext_report.is_file() else {}
    summary["external_novelty"] = {
        "status": "COMPLETE" if ext_complete else "FAILED",
        "return_code": ext_run_code,
        "reused": bool(ext_complete and ext_run_code == 0 and not ext_stdout and not args.force),
        "report": str(ext_report) if ext_report.is_file() else None,
        "query_plan": str(ext_plan) if ext_plan.is_file() else None,
        "prior_art": str(ext_prior) if ext_prior.is_file() else None,
        "provider_plan": str(ext_provider_plan) if ext_provider_plan.is_file() else None,
        "status_counts": dict(ext_payload.get("status_counts", {})),
        "epistemic_usage": ext_payload.get("epistemic_usage"),
    }
    _checkpoint(out, summary)

    if not ext_complete:
        summary["n9"] = {"status": "SKIPPED_EXTERNAL_NOVELTY_FAILED"}
        summary["operational_status"] = "FAILED_OPERATIONAL"
        summary["status"] = "FAILED_OPERATIONAL"
        _checkpoint(out, summary)
        return 1

    # ------------------------------------------------------------------
    # N9: intake is deterministic; full stage may retrieve and can pause.
    # ------------------------------------------------------------------
    n9_intake = out / "n9.intake.shadow.json"
    if args.force or not n9_intake.is_file():
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
        intake_code = intake_run.returncode
    else:
        print("Reusing completed N9 intake:", n9_intake)
        intake_code = 0
    n9_intake_complete = intake_code == 0 and n9_intake.is_file()

    n9_full = out / "n9.full_shadow.json"
    full_code: int | None = None
    n9_complete = n9_intake_complete
    if n9_intake_complete and not args.skip_n9_full:
        if not args.force and n9_full.is_file():
            print("Reusing completed N9 full shadow:", n9_full)
            full_code = 0
            n9_complete = True
        else:
            budget = _stage_budget_probe(
                providers=args.providers,
                provider_plan=ext_provider_plan,
                disabled=args.skip_provider_budget_preflight,
            )
            if budget and budget.get("available") and budget.get("exhausted"):
                summary["n9"] = {
                    "status": "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
                    "intake": str(n9_intake),
                    "full_shadow": None,
                    "intake_return_code": intake_code,
                    "full_return_code": None,
                }
                return _pause(
                    out=out,
                    summary=summary,
                    stage="N9_FULL",
                    budget=budget,
                    stage_key="n9",
                )
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
            full_code = full_run.returncode
            n9_complete = full_run.returncode == 0 and n9_full.is_file()
            if not n9_complete and provider_budget_exhausted_from_text(
                full_run.stdout or "", full_run.stderr or ""
            ):
                summary["n9"] = {
                    "status": "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
                    "intake": str(n9_intake),
                    "full_shadow": str(n9_full) if n9_full.is_file() else None,
                    "intake_return_code": intake_code,
                    "full_return_code": full_code,
                }
                return _pause(
                    out=out,
                    summary=summary,
                    stage="N9_FULL",
                    budget={"exhausted": True, "source": "n9_subprocess"},
                    stage_key="n9",
                    return_code=full_code,
                )

    summary["n9"] = {
        "status": (
            "INTAKE_ONLY"
            if args.skip_n9_full and n9_complete
            else "COMPLETE" if n9_complete else "FAILED"
        ),
        "intake": str(n9_intake) if n9_intake.is_file() else None,
        "full_shadow": str(n9_full) if n9_full.is_file() else None,
        "intake_return_code": intake_code,
        "full_return_code": full_code,
    }
    _checkpoint(out, summary)

    expected_ok = {
        "semantic": {"ACCEPTED"},
        "external_novelty": {"COMPLETE"},
        "n9": {"COMPLETE", "INTAKE_ONLY"},
        "feasibility": {"COMPLETE", "SKIPPED_UNSUPPORTED_DOMAIN"},
    }
    terminal = {key: (summary[key].get("status") if isinstance(summary.get(key), dict) else None) for key in expected_ok}
    complete = all(terminal[key] in expected_ok[key] for key in expected_ok)
    summary["operational_status"] = "COMPLETE" if complete else "FAILED_OPERATIONAL"
    summary["resume_stage"] = None if complete else "UNKNOWN"
    summary["status"] = "COMPLETE_SHADOW_VERIFICATION" if complete else "FAILED_OPERATIONAL"
    _checkpoint(out, summary)

    print()
    print("Standard portfolio downstream verification shadow complete")
    print("semantic:", summary["semantic"]["status"])
    print("external novelty:", summary["external_novelty"]["status"])
    print("N9:", summary["n9"]["status"])
    print("feasibility:", summary["feasibility"]["status"])
    print("operational status:", summary["operational_status"])
    print("N10_RUN=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("summary:", out / "verification.summary.json")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
