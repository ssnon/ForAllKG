from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    run_prospective_identification_shadow,
)
from pipeline_core.discovery.prospective_validation import provider_budget_exhausted_from_text
from pipeline_core.discovery.research_idea_active_realization import (
    ExactFeedbackAttachmentReport,
    run_active_realization_search,
)
from pipeline_core.discovery.research_idea_closed_loop import RealizationLifecycleReport
from pipeline_core.discovery.research_idea_closed_loop_v2_6 import (
    CreditContinuityReportV2,
    build_credit_continuity_v2,
)
from pipeline_core.discovery.research_idea_multigeneration import ConceptualDeltaAuditReport
from pipeline_core.discovery.research_idea_offspring_execution import OffspringExecutionReport
from pipeline_core.discovery.research_idea_program_families import (
    FrontierProgramReferenceCalibration,
    ScientificProgramFamilyReport,
    build_scientific_program_families,
    calibrate_frontier_program_reference,
)
from pipeline_core.discovery.research_idea_search_contracts import GenerationalIdeaSearchShadowReport
from pipeline_core.discovery.research_idea_verified_active_search import (
    FreshResidualVerificationReport,
    ResidualActionOutcome,
    VerifiedLocalRoundRecord,
    attach_feedback_with_fresh_residual,
    build_case_summary,
    build_cost_audit,
    build_idea_evolution_requests,
    combine_active_reports,
    merge_active_into_lifecycle,
    normalize_lifecycle_strict,
    prior_action_map,
)
from pipeline_core.discovery.scientific_portfolio_closed_loop import compile_residual_epistemic_state
from scripts.discovery.run_scientific_portfolio_closed_loop_shadow import (
    load as load_json,
    lower_order,
    run_aggregation,
    run_fulltext,
    run_topology,
)


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    label, raw = value.split("=", 1)
    if not label.strip() or not raw.strip():
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    return label.strip(), Path(raw).expanduser().resolve()


def _reference(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--family-reference-population must use LABEL=/path/to/frontier.json")
    label, raw = value.split("=", 1)
    return label.strip(), Path(raw).expanduser().resolve()


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--header must use KEY=VALUE")
    key, raw = value.split("=", 1)
    return key.strip(), raw


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run SIS-v2.7 Scientific Program Families & Verified Active Search: "
            "coarse scientific-program family projection, fresh residual verification barriers, "
            "one failure-specific local action per verified round, typed local-to-global evolution "
            "requests, and cost-aware closed-loop audit. G4 generation is never executed."
        )
    )
    p.add_argument("--case", action="append", type=_case, required=True)
    p.add_argument("--family-reference-population", action="append", type=_reference, default=[])
    p.add_argument("--auto-family-reference-root", action="append", type=Path, default=[])
    p.add_argument("--auto-family-reference-min-topology", type=int, default=100)
    p.add_argument("--max-auto-family-references", type=int, default=3)
    p.add_argument("--disable-auto-family-reference", action="store_true")
    p.add_argument(
        "--model",
        default=(os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL") or os.getenv("OPENROUTER_AGENT_MODEL") or ""),
    )
    p.add_argument(
        "--critic-model",
        default=(os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL") or os.getenv("OPENROUTER_CRITIC_MODEL") or os.getenv("OPENROUTER_AGENT_MODEL") or ""),
    )
    p.add_argument("--prospective-model", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", type=_header, default=[])
    p.add_argument("--providers", default="auto")
    p.add_argument("--provider-plan", type=Path, default=None)
    p.add_argument("--domain-profile", default=None)
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--max-ranked-works", type=int, default=12)
    p.add_argument("--max-fulltext-candidates", type=int, default=4)
    p.add_argument("--max-excerpt-chars", type=int, default=24000)
    p.add_argument("--max-active-g2-ideas", type=int, default=8)
    p.add_argument("--max-active-g3-ideas", type=int, default=8)
    p.add_argument("--max-verified-local-rounds", type=int, default=2)
    p.add_argument("--max-repair-attempts", type=int, default=1)
    p.add_argument("--max-prospective-audits-per-round", type=int, default=12)
    p.add_argument("--skip-prospective", action="store_true")
    p.add_argument("--program-family-similarity-floor", type=float, default=0.46)
    p.add_argument("--program-family-endpoint-floor", type=float, default=0.55)
    p.add_argument("--force-verification", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected artifact paths exist: " + ", ".join(map(str, paths)))


def _paths(root: Path) -> dict[str, Path]:
    out = root / "scientific_portfolio_shadow"
    return {
        "context": _first_existing(root / "hypothesis.context.json", out / "hypothesis.context.json"),
        "source_generational": _first_existing(
            out / "sis_v2_2.source_v2_1_1.generational_shadow.json",
            out / "sis_v2_1_1.feedback_aware.generational_shadow.json",
            out / "sis_v2_1.generational_shadow.json",
        ),
        "g2_execution": out / "sis_v2_3.offspring_execution.json",
        "g3_execution": out / "sis_v2_4.g3_offspring_execution.json",
        "g2_audit": out / "sis_v2_4.g2_conceptual_delta.json",
        "g3_audit": out / "sis_v2_4.g3_conceptual_delta.json",
        "g2_lifecycle": out / "sis_v2_5.g2_realization_lifecycle.json",
        "g2_portfolio": out / "sis_v2_5.g2_lifecycle.portfolio.json",
        "g3_lifecycle": out / "sis_v2_5.g3_realization_lifecycle.json",
        "g3_portfolio": out / "sis_v2_5.g3_lifecycle.portfolio.json",
        "v2_6_credit": out / "sis_v2_6.credit_continuity_v2.json",
        "root": out / "sis_v2_7_verified_active_search",
        "program_families": out / "sis_v2_7.program_families.json",
        "g2_final_lifecycle": out / "sis_v2_7.g2_final_lifecycle.json",
        "g2_final_portfolio": out / "sis_v2_7.g2_final.portfolio.json",
        "g3_final_lifecycle": out / "sis_v2_7.g3_final_lifecycle.json",
        "g3_final_portfolio": out / "sis_v2_7.g3_final.portfolio.json",
        "evolution_requests": out / "sis_v2_7.idea_evolution_requests.json",
        "cost_audit": out / "sis_v2_7.cost_audit.json",
        "summary": out / "sis_v2_7.case_summary.json",
        "telemetry": out / "sis_v2_7.active_realization.telemetry.jsonl",
    }


def _require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")


def _run_cmd(label: str, cmd: list[str], out: Path) -> tuple[int, str, str]:
    out.mkdir(parents=True, exist_ok=True)
    print()
    print("-" * 88)
    print(label)
    print("-" * 88)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    safe = "_".join(label.lower().split())
    (out / f"{safe}.stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (out / f"{safe}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.returncode, result.stdout or "", result.stderr or ""


def _verification_report(
    *,
    generation_index: int,
    round_index: int,
    portfolio: HypothesisPortfolio,
    status: str,
    out: Path,
    state_counts: dict[str, int] | None = None,
    stage_return_codes: dict[str, int] | None = None,
    subprocess_calls: int = 0,
) -> FreshResidualVerificationReport:
    epistemic = out / "epistemic_state.json"
    external = out / "external.report.json"
    query = out / "external.claims_queries.json"
    aggregation = out / "residual_aggregation.report.json"
    provider = out / "external.provider_plan.json"
    provisional = FreshResidualVerificationReport(
        report_id="pending",
        report_sha256="pending",
        generation_index=generation_index,
        round_index=round_index,
        source_portfolio_id=portfolio.portfolio_id,
        hypothesis_count=len(portfolio.hypotheses),
        status=status,
        residual_state_counts=dict(state_counts or {}),
        epistemic_state_path=str(epistemic) if epistemic.is_file() else None,
        external_report_path=str(external) if external.is_file() else None,
        query_plan_path=str(query) if query.is_file() else None,
        aggregation_path=str(aggregation) if aggregation.is_file() else None,
        provider_plan_path=str(provider) if provider.is_file() else None,
        stage_return_codes=dict(stage_return_codes or {}),
        verification_subprocess_call_count=subprocess_calls,
        fresh_exact_lineage_residual_created=(status == "COMPLETE" and epistemic.is_file()),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    import hashlib
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return provisional.model_copy(
        update={
            "report_id": f"g{generation_index}_fresh_residual_r{round_index}:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def _fresh_residual_verification(
    *,
    args: argparse.Namespace,
    generation_index: int,
    round_index: int,
    context_path: Path,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    out: Path,
) -> tuple[FreshResidualVerificationReport, dict[str, Any] | None]:
    out.mkdir(parents=True, exist_ok=True)
    portfolio_path = out / "input.portfolio.json"
    _write(portfolio_path, portfolio)
    epistemic = out / "epistemic_state.json"
    if not args.force_verification and epistemic.is_file():
        try:
            payload = load_json(epistemic)
            if str(payload.get("source_portfolio_id") or "") == str(portfolio.portfolio_id):
                report = _verification_report(
                    generation_index=generation_index,
                    round_index=round_index,
                    portfolio=portfolio,
                    status="COMPLETE",
                    out=out,
                    state_counts=dict(payload.get("state_counts") or {}),
                    stage_return_codes={"REUSED": 0},
                    subprocess_calls=0,
                )
                _write(out / "barrier.report.json", report)
                return report, payload
        except Exception:
            pass

    if not portfolio.hypotheses:
        state = {
            "schema_version": "scientific-portfolio-residual-epistemic-state-v1",
            "source_portfolio_id": portfolio.portfolio_id,
            "state_counts": {},
            "hypotheses": [],
            "shadow_only": True,
            "novelty_authority_created": False,
            "production_selection_changed": False,
        }
        _write(epistemic, state)
        report = _verification_report(
            generation_index=generation_index,
            round_index=round_index,
            portfolio=portfolio,
            status="SKIPPED_EMPTY_PORTFOLIO",
            out=out,
            state_counts={},
        )
        _write(out / "barrier.report.json", report)
        return report, state

    domain_profile = args.domain_profile or context.domain_profile_id
    critic_model = args.critic_model or args.model
    external_prefix = out / "external"
    cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_external_novelty",
        "--portfolio", str(portfolio_path),
        "--domain-profile", str(domain_profile),
        "--model", str(critic_model),
        "--api-key-env", str(args.api_key_env),
        "--providers", str(args.providers),
        "--results-per-query", str(args.results_per_query),
        "--max-ranked-works", str(args.max_ranked_works),
        "--parse-retries", str(args.parse_retries),
        "--pre-review-metadata-resolution",
        "--pre-review-coverage-shadow",
        "--downstream-gate-shadow",
        "--source-bound-topology-shadow",
        "--output-prefix", str(external_prefix),
        "--save-prompts",
    ]
    if args.base_url:
        cmd += ["--base-url", str(args.base_url)]
    if args.provider_plan is not None:
        cmd += ["--provider-plan", str(args.provider_plan.expanduser().resolve())]
    code, stdout, stderr = _run_cmd(
        f"SIS-v2.7 G{generation_index} round {round_index} fresh external novelty",
        cmd,
        out,
    )
    stage_codes = {"EXTERNAL_NOVELTY": code}
    calls = 1
    if code != 0:
        status = (
            "PAUSED_PROVIDER_BUDGET_EXHAUSTED"
            if provider_budget_exhausted_from_text(stdout, stderr)
            else "FAILED_OPERATIONAL"
        )
        report = _verification_report(
            generation_index=generation_index,
            round_index=round_index,
            portfolio=portfolio,
            status=status,
            out=out,
            stage_return_codes=stage_codes,
            subprocess_calls=calls,
        )
        _write(out / "barrier.report.json", report)
        return report, None

    query = out / "external.claims_queries.json"
    external = out / "external.report.json"
    binding = out / "external.atomic_source_binding.json"
    generated_provider = out / "external.provider_plan.json"
    required = [query, external, binding, generated_provider]
    if any(not path.is_file() for path in required):
        report = _verification_report(
            generation_index=generation_index,
            round_index=round_index,
            portfolio=portfolio,
            status="FAILED_OPERATIONAL",
            out=out,
            stage_return_codes=stage_codes,
            subprocess_calls=calls,
        )
        _write(out / "barrier.report.json", report)
        return report, None

    helper_args = SimpleNamespace(
        provider_plan=generated_provider,
        domain_profile=domain_profile,
        critic_model=critic_model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        results_per_query=args.results_per_query,
        max_ranked_works=args.max_ranked_works,
        parse_retries=args.parse_retries,
        max_fulltext_candidates=args.max_fulltext_candidates,
        max_excerpt_chars=args.max_excerpt_chars,
        output_dir=out,
    )
    try:
        lower_report, lower_reviews, lower_packet = lower_order(
            args=helper_args,
            portfolio=portfolio_path,
            query_plan=query,
            external_report=external,
            prefix=out / "lower",
        )
        calls += 1
        stage_codes["LOWER_ORDER"] = 0
        topology = out / "topology.report.json"
        run_topology(
            args=helper_args,
            query_plan=query,
            saturation_report=lower_report,
            external_report=external,
            source_binding=binding,
            output=topology,
        )
        calls += 1
        stage_codes["TOPOLOGY"] = 0
        fulltext = out / "fulltext.report.json"
        run_fulltext(
            args=helper_args,
            query_plan=query,
            reviews=lower_reviews,
            packet=lower_packet,
            topology_report=topology,
            output=fulltext,
        )
        calls += 1
        stage_codes["FULLTEXT"] = 0
        aggregation = out / "residual_aggregation.report.json"
        run_aggregation(
            args=helper_args,
            topology_report=topology,
            reviews=lower_reviews,
            fulltext_report=fulltext,
            output=aggregation,
        )
        calls += 1
        stage_codes["AGGREGATION"] = 0
        state = compile_residual_epistemic_state(
            portfolio=portfolio.model_dump(mode="json"),
            query_plan=load_json(query),
            external_report=load_json(external),
            aggregation=load_json(aggregation),
        )
        _write(epistemic, state)
    except Exception as exc:
        stage_codes["PIPELINE_EXCEPTION"] = 1
        (out / "verification_exception.txt").write_text(f"{type(exc).__name__}:{exc}\n", encoding="utf-8")
        report = _verification_report(
            generation_index=generation_index,
            round_index=round_index,
            portfolio=portfolio,
            status="FAILED_OPERATIONAL",
            out=out,
            stage_return_codes=stage_codes,
            subprocess_calls=calls,
        )
        _write(out / "barrier.report.json", report)
        return report, None

    report = _verification_report(
        generation_index=generation_index,
        round_index=round_index,
        portfolio=portfolio,
        status="COMPLETE",
        out=out,
        state_counts=dict(state.get("state_counts") or {}),
        stage_return_codes=stage_codes,
        subprocess_calls=calls,
    )
    _write(out / "barrier.report.json", report)
    return report, state


def _auto_reference_candidates(roots: list[Path], minimum_topology: int, limit: int) -> list[tuple[str, Path]]:
    rows: list[tuple[int, str, Path]] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for path in root.rglob("frontier_idea_population.shadow.json"):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            try:
                population = FrontierIdeaPopulation.model_validate_json(path.read_text(encoding="utf-8"))
                topology = sum(getattr(idea, "topology_signature", None) is not None for idea in population.ideas)
            except Exception:
                continue
            if topology >= minimum_topology:
                rows.append((topology, path.parent.name or "reference", path))
    rows.sort(key=lambda item: (-item[0], str(item[2])))
    return [(f"auto_{count}_{index+1}", path) for index, (count, _name, path) in enumerate(rows[:limit])]


def _feedback_by_idea(lifecycle: RealizationLifecycleReport, feedback: ExactFeedbackAttachmentReport) -> dict[str, Any]:
    by_h = {row.hypothesis_id: row for row in feedback.records}
    links_by_idea: dict[str, list[Any]] = defaultdict(list)
    for link in lifecycle.links:
        links_by_idea[link.idea_id].append(link)
    result: dict[str, Any] = {}
    for idea_id, links in links_by_idea.items():
        links.sort(key=lambda row: (row.attempt_index, row.realization_id))
        for link in reversed(links):
            if link.hypothesis_id is None:
                continue
            row = by_h.get(str(link.hypothesis_id))
            if row is not None:
                result[idea_id] = row
                break
    return result


def _usable_ideas(lifecycle: RealizationLifecycleReport) -> set[str]:
    return {row.idea_id for row in lifecycle.local_states if row.usable_grounded_realization_count > 0}


def _run_generation_verified_search(
    *,
    args: argparse.Namespace,
    label: str,
    generation_index: int,
    execution: OffspringExecutionReport,
    lifecycle: RealizationLifecycleReport,
    portfolio: HypothesisPortfolio,
    context: HypothesisContext,
    context_path: Path,
    backend: InstructorOpenAICompatibleHypothesisBackend,
    prospective_runner,
    case_root: Path,
    work_root: Path,
    max_active_ideas: int,
) -> tuple[
    RealizationLifecycleReport,
    HypothesisPortfolio,
    list[Any],
    list[FreshResidualVerificationReport],
    ExactFeedbackAttachmentReport,
    ExactFeedbackAttachmentReport,
    list[ResidualActionOutcome],
]:
    lifecycle = normalize_lifecycle_strict(lifecycle, extra_local_budget=args.max_verified_local_rounds)
    active_reports: list[Any] = []
    verification_reports: list[FreshResidualVerificationReport] = []
    residual_action_outcomes: list[ResidualActionOutcome] = []
    prior_actions: dict[str, list[Any]] = {}
    prospective_roots = [case_root, work_root]

    initial_barrier, residual_state = _fresh_residual_verification(
        args=args,
        generation_index=generation_index,
        round_index=0,
        context_path=context_path,
        context=context,
        portfolio=portfolio,
        out=work_root / "verification_round_0",
    )
    verification_reports.append(initial_barrier)
    initial_feedback = attach_feedback_with_fresh_residual(
        lifecycle=lifecycle,
        fresh_residual_state=residual_state,
        prospective_search_roots=prospective_roots,
    )
    current_feedback = initial_feedback
    current_residual_state = residual_state

    for round_index in range(1, args.max_verified_local_rounds + 1):
        pre_feedback_by_idea = _feedback_by_idea(lifecycle, current_feedback)
        before_usable = _usable_ideas(lifecycle)
        active_result = run_active_realization_search(
            execution=execution,
            lifecycle=lifecycle,
            portfolio=portfolio,
            context=context,
            feedback=current_feedback,
            backend=backend,
            output_dir=work_root / f"local_round_{round_index}",
            prospective_runner=prospective_runner,
            max_active_ideas=max_active_ideas,
            max_actions_per_idea=1,
            max_repair_attempts=args.max_repair_attempts,
            max_prospective_audits=args.max_prospective_audits_per_round,
            prior_actions_by_idea=prior_actions,
        )
        active_reports.append(active_result.report)
        prior_actions = prior_action_map(active_reports)

        scientific_content_changed = any(
            link.hypothesis_id is not None and link.materialization_status == "MATERIALIZED"
            for link in active_result.report.new_links
        )
        next_portfolio = active_result.portfolio if scientific_content_changed else portfolio
        lifecycle = merge_active_into_lifecycle(
            lifecycle=lifecycle,
            active=active_result.report,
            output_portfolio=next_portfolio,
            remaining_round_budget=args.max_verified_local_rounds - round_index,
        )
        portfolio = next_portfolio

        terminal_only = bool(active_result.report.action_records) and all(
            row.action in {
                "KEEP", "KEEP_UNTIL_EVALUATED", "RETRIEVE_MORE",
                "AXIS_MUTATION", "REQUEST_GRAPH_RETRAVERSAL", "STOP",
            }
            for row in active_result.report.action_records
        )

        if scientific_content_changed:
            barrier, current_residual_state = _fresh_residual_verification(
                args=args,
                generation_index=generation_index,
                round_index=round_index,
                context_path=context_path,
                context=context,
                portfolio=portfolio,
                out=work_root / f"verification_round_{round_index}",
            )
            verification_reports.append(barrier)
        else:
            barrier = verification_reports[-1]

        current_feedback = attach_feedback_with_fresh_residual(
            lifecycle=lifecycle,
            fresh_residual_state=current_residual_state,
            prospective_search_roots=prospective_roots,
        )
        post_feedback_by_idea = _feedback_by_idea(lifecycle, current_feedback)
        after_usable = _usable_ideas(lifecycle)
        for record in active_result.report.action_records:
            pre = pre_feedback_by_idea.get(record.idea_id)
            post = post_feedback_by_idea.get(record.idea_id)
            pre_state = getattr(pre, "residual_epistemic_state", None) if pre is not None else None
            post_state = getattr(post, "residual_epistemic_state", None) if post is not None else None
            residual_action_outcomes.append(
                ResidualActionOutcome(
                    generation_index=generation_index,
                    idea_id=record.idea_id,
                    pre_residual_state=pre_state,
                    action=record.action,
                    post_residual_state=post_state,
                    usable_after_action=record.idea_id in after_usable,
                    changed_residual_state=pre_state != post_state,
                )
            )

        round_record = VerifiedLocalRoundRecord(
            generation_index=generation_index,
            round_index=round_index,
            source_lifecycle_report_id=active_result.report.source_lifecycle_report_id,
            source_portfolio_id=(verification_reports[-2].source_portfolio_id if scientific_content_changed and len(verification_reports) >= 2 else portfolio.portfolio_id),
            feedback_report_id=active_result.report.source_feedback_attachment_report_id,
            active_report_id=active_result.report.report_id,
            output_lifecycle_report_id=lifecycle.report_id,
            output_portfolio_id=portfolio.portfolio_id,
            action_counts=active_result.report.action_counts,
            generated_realization_count=len(active_result.report.new_links),
            same_idea_rescue_count=len(active_result.report.rescued_within_same_idea_ids),
            local_llm_call_count=active_result.report.local_llm_call_count,
            prospective_audit_llm_call_count=active_result.report.prospective_audit_llm_call_count,
            fresh_verification_report_id=(barrier.report_id if scientific_content_changed else None),
        )
        _write(work_root / f"round_{round_index}.record.json", round_record)
        _write(work_root / f"round_{round_index}.active_report.json", active_result.report)
        _write(work_root / f"round_{round_index}.portfolio.json", portfolio)
        _write(work_root / f"round_{round_index}.lifecycle.json", lifecycle)
        _write(work_root / f"round_{round_index}.feedback.json", current_feedback)

        if not active_result.report.action_records or terminal_only:
            break
        if before_usable == after_usable and not scientific_content_changed and round_index >= args.max_verified_local_rounds:
            break

    # Final barrier must correspond to the final scientific portfolio. If the last local action
    # produced a materialized candidate, it was already verified above; otherwise reuse is exact.
    final_feedback = current_feedback
    return (
        lifecycle,
        portfolio,
        active_reports,
        verification_reports,
        initial_feedback,
        final_feedback,
        residual_action_outcomes,
    )


def main() -> int:
    args = parser().parse_args()
    if not args.model:
        raise SystemExit("No hypothesis model configured; pass --model or set GRAPHAGENTS_HYPOTHESIS_MODEL.")
    if not args.critic_model:
        args.critic_model = args.model
    if args.max_verified_local_rounds < 1:
        raise SystemExit("--max-verified-local-rounds must be >= 1")
    if not 0.0 <= args.program_family_similarity_floor <= 1.0:
        raise SystemExit("--program-family-similarity-floor must be in [0,1]")
    headers = dict(args.header)
    output = args.output.expanduser().resolve()
    cohort: dict[str, Any] = {}
    aggregate = Counter()

    explicit_references = list(args.family_reference_population)
    auto_roots = [path.expanduser().resolve() for path in args.auto_family_reference_root]
    if not auto_roots:
        auto_roots = sorted({root.parent for _label, root in args.case})
    auto_references = [] if args.disable_auto_family_reference else _auto_reference_candidates(
        auto_roots,
        args.auto_family_reference_min_topology,
        args.max_auto_family_references,
    )
    reference_calibrations: dict[str, Any] = {}
    for ref_label, ref_path in [*explicit_references, *auto_references]:
        try:
            population = _load(ref_path, FrontierIdeaPopulation)
            calibration = calibrate_frontier_program_reference(
                population,
                similarity_floor=args.program_family_similarity_floor,
                endpoint_floor=args.program_family_endpoint_floor,
            )
            reference_calibrations[ref_label] = {
                "path": str(ref_path),
                **calibration.model_dump(mode="json"),
            }
        except Exception as exc:
            reference_calibrations[ref_label] = {
                "path": str(ref_path),
                "error": f"{type(exc).__name__}:{exc}",
            }

    for label, root in args.case:
        paths = _paths(root)
        for key in (
            "context", "source_generational", "g2_execution", "g3_execution",
            "g2_audit", "g3_audit", "g2_lifecycle", "g2_portfolio",
            "g3_lifecycle", "g3_portfolio", "v2_6_credit",
        ):
            _require(paths[key], key)
        work_root = paths["root"]
        work_root.mkdir(parents=True, exist_ok=True)

        context = _load(paths["context"], HypothesisContext)
        source_generational = _load(paths["source_generational"], GenerationalIdeaSearchShadowReport)
        g2_execution = _load(paths["g2_execution"], OffspringExecutionReport)
        g3_execution = _load(paths["g3_execution"], OffspringExecutionReport)
        g2_audit = _load(paths["g2_audit"], ConceptualDeltaAuditReport)
        g3_audit = _load(paths["g3_audit"], ConceptualDeltaAuditReport)
        g2_initial_lifecycle = normalize_lifecycle_strict(
            _load(paths["g2_lifecycle"], RealizationLifecycleReport),
            extra_local_budget=args.max_verified_local_rounds,
        )
        g2_initial_portfolio = _load(paths["g2_portfolio"], HypothesisPortfolio)
        g3_initial_lifecycle = normalize_lifecycle_strict(
            _load(paths["g3_lifecycle"], RealizationLifecycleReport),
            extra_local_budget=args.max_verified_local_rounds,
        )
        g3_initial_portfolio = _load(paths["g3_portfolio"], HypothesisPortfolio)
        initial_credit = _load(paths["v2_6_credit"], CreditContinuityReportV2)

        nodes = [
            *source_generational.research_ideas,
            *g2_execution.offspring_nodes,
            *g3_execution.offspring_nodes,
        ]
        node_by_id = {row.idea_id: row for row in nodes}
        nodes = list(node_by_id.values())
        families = build_scientific_program_families(
            nodes,
            conceptual_audits=[g2_audit, g3_audit],
            similarity_floor=args.program_family_similarity_floor,
            endpoint_floor=args.program_family_endpoint_floor,
        )
        _write(paths["program_families"], families)

        backend = InstructorOpenAICompatibleHypothesisBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=headers,
            telemetry_path=paths["telemetry"],
            telemetry_context={"pipeline": "sis_v2_7_verified_active_search", "case": label},
        )
        prospective_model = args.prospective_model or args.critic_model
        prospective_runner = None
        if not args.skip_prospective:
            def prospective_runner(*, context, candidate, source_stage, output_prefix, _m=prospective_model):
                return run_prospective_identification_shadow(
                    context=context,
                    candidate=candidate,
                    source_stage=source_stage,
                    model=_m,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                    parse_retries=args.parse_retries,
                    output_prefix=output_prefix,
                )

        g2 = _run_generation_verified_search(
            args=args,
            label=label,
            generation_index=2,
            execution=g2_execution,
            lifecycle=g2_initial_lifecycle,
            portfolio=g2_initial_portfolio,
            context=context,
            context_path=paths["context"],
            backend=backend,
            prospective_runner=prospective_runner,
            case_root=root,
            work_root=work_root / "g2",
            max_active_ideas=args.max_active_g2_ideas,
        )
        g3 = _run_generation_verified_search(
            args=args,
            label=label,
            generation_index=3,
            execution=g3_execution,
            lifecycle=g3_initial_lifecycle,
            portfolio=g3_initial_portfolio,
            context=context,
            context_path=paths["context"],
            backend=backend,
            prospective_runner=prospective_runner,
            case_root=root,
            work_root=work_root / "g3",
            max_active_ideas=args.max_active_g3_ideas,
        )
        (
            g2_final_lifecycle, g2_final_portfolio, g2_active_reports, g2_verification,
            g2_initial_feedback, g2_final_feedback, g2_outcomes,
        ) = g2
        (
            g3_final_lifecycle, g3_final_portfolio, g3_active_reports, g3_verification,
            g3_initial_feedback, g3_final_feedback, g3_outcomes,
        ) = g3

        _write(paths["g2_final_lifecycle"], g2_final_lifecycle)
        _write(paths["g2_final_portfolio"], g2_final_portfolio)
        _write(paths["g3_final_lifecycle"], g3_final_lifecycle)
        _write(paths["g3_final_portfolio"], g3_final_portfolio)

        combined_g2 = combine_active_reports(g2_active_reports, generation_index=2)
        combined_g3 = combine_active_reports(g3_active_reports, generation_index=3)
        final_credit = build_credit_continuity_v2(
            generation2_execution=g2_execution,
            generation2_lifecycle=g2_initial_lifecycle,
            generation2_active=combined_g2,
            generation3_execution=g3_execution,
            generation3_lifecycle=g3_initial_lifecycle,
            generation3_active=combined_g3,
        )
        _write(work_root / "final_credit_continuity.json", final_credit)

        g2_requests = build_idea_evolution_requests(
            generation_index=2,
            lifecycle=g2_final_lifecycle,
            final_feedback=g2_final_feedback,
            action_reports=g2_active_reports,
            families=families,
        )
        g3_requests = build_idea_evolution_requests(
            generation_index=3,
            lifecycle=g3_final_lifecycle,
            final_feedback=g3_final_feedback,
            action_reports=g3_active_reports,
            families=families,
        )
        all_requests = [*g2_requests.requests, *g3_requests.requests]
        scope_counts = Counter(row.requested_mutation_scope for row in all_requests)
        from pipeline_core.discovery.research_idea_verified_active_search import IdeaEvolutionRequestReport
        import hashlib
        provisional_requests = IdeaEvolutionRequestReport(
            report_id="pending",
            report_sha256="pending",
            requests=all_requests,
            request_count=len(all_requests),
            scope_counts=dict(sorted(scope_counts.items())),
            source_generation_counts=dict(sorted(Counter(f"G{row.source_generation_index}" for row in all_requests).items())),
        )
        req_payload = provisional_requests.model_dump(mode="json")
        req_payload.pop("report_id", None)
        req_payload.pop("report_sha256", None)
        digest = hashlib.sha256(json.dumps(req_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        evolution_requests = provisional_requests.model_copy(update={
            "report_id": f"idea_evolution_requests:{digest[:20]}",
            "report_sha256": digest,
        })
        _write(paths["evolution_requests"], evolution_requests)

        cost = build_cost_audit(
            family_report=families,
            active_reports=[*g2_active_reports, *g3_active_reports],
            fresh_verification_reports=[*g2_verification, *g3_verification],
            initial_credit=initial_credit,
            final_credit=final_credit,
            generation3_execution=g3_execution,
            residual_action_outcomes=[*g2_outcomes, *g3_outcomes],
        )
        _write(paths["cost_audit"], cost)

        successful_reference_errors = [
            int(value["absolute_family_count_error"])
            for value in reference_calibrations.values()
            if isinstance(value, dict) and value.get("absolute_family_count_error") is not None
        ]
        reference_error = min(successful_reference_errors) if successful_reference_errors else None
        g2_round_records = [
            _load(path, VerifiedLocalRoundRecord)
            for path in sorted((work_root / "g2").glob("round_*.record.json"))
        ]
        g3_round_records = [
            _load(path, VerifiedLocalRoundRecord)
            for path in sorted((work_root / "g3").glob("round_*.record.json"))
        ]
        summary = build_case_summary(
            idea_count=len(nodes),
            family_report=families,
            family_reference_absolute_error=reference_error,
            generation2_rounds=g2_round_records,
            generation3_rounds=g3_round_records,
            initial_feedback_reports=[g2_initial_feedback, g3_initial_feedback],
            final_feedback_reports=[g2_final_feedback, g3_final_feedback],
            evolution_requests=evolution_requests,
            cost_audit=cost,
        )
        _write(paths["summary"], summary)

        cohort[label] = summary.model_dump(mode="json")
        aggregate["case_count"] += 1
        aggregate["idea_count"] += summary.research_idea_count
        aggregate["program_family_count"] += summary.scientific_program_family_count
        aggregate["same_idea_rescue_count"] += summary.total_same_idea_rescue_count
        aggregate["idea_evolution_request_count"] += summary.idea_evolution_request_count
        aggregate["initial_residual_evaluated"] += summary.initial_residual_evaluated_count
        aggregate["final_residual_evaluated"] += summary.final_residual_evaluated_count
        aggregate["local_llm_calls"] += cost.local_llm_calls
        aggregate["fresh_verification_barrier_runs"] += cost.fresh_verification_barrier_runs
        aggregate["fresh_verification_subprocess_calls"] += cost.fresh_verification_subprocess_calls
        aggregate["unresolved_reduction"] += cost.unresolved_reduction

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print("ideas / scientific program families:", summary.research_idea_count, "/", summary.scientific_program_family_count)
        print("family fragmentation ratio:", round(summary.family_fragmentation_ratio, 4))
        print("family reference absolute error:", summary.family_reference_absolute_error)
        print("verified local rounds G2/G3:", summary.generation2_verified_round_count, "/", summary.generation3_verified_round_count)
        print("same-idea rescues G2/G3:", summary.generation2_same_idea_rescue_count, "/", summary.generation3_same_idea_rescue_count)
        print("initial/final residual evaluated:", summary.initial_residual_evaluated_count, "/", summary.final_residual_evaluated_count)
        print("IdeaEvolutionRequests:", summary.idea_evolution_request_count, summary.idea_evolution_request_scope_counts)
        print("LOCAL_LLM_CALLS:", cost.local_llm_calls)
        print("SAME_IDEA_RESCUE_PER_LOCAL_LLM_CALL:", round(cost.same_idea_rescue_per_local_llm_call, 4))
        print("UNRESOLVED_REDUCTION:", cost.unresolved_reduction)
        print("FRESH_VERIFICATION_BARRIERS:", cost.fresh_verification_barrier_runs)
        print("G4_GENERATION=False")
        print("FAMILY_IS_HARD_GATE=False")
        print("PRODUCTION_GENERATION_AUTHORITY=False")
        print("PRODUCTION_SELECTION_AUTHORITY=False")

    payload = {
        "schema_version": "sis-v2-7-scientific-program-verified-active-search-cohort-v1",
        "cases": cohort,
        "aggregate": dict(aggregate),
        "family_reference_calibrations": reference_calibrations,
        "scientific_program_family_is_soft_and_recomputable": True,
        "failure_specific_local_action_is_reverified_before_next_action": True,
        "fresh_residual_feedback_uses_exact_portfolio_lineage": True,
        "idea_evolution_request_executes_g4_generation": False,
        "g4_generation_executed": False,
        "single_scalar_fitness_used": False,
        "production_generation_authority": False,
        "production_selection_authority": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
    }
    _write(output, payload)
    print()
    print("SIS-v2.7 Scientific Program Families & Verified Active Search complete")
    print("cases:", len(cohort))
    print("aggregate:", dict(aggregate))
    print("family references:", list(reference_calibrations))
    print("G4_GENERATION=False")
    print("PRODUCTION_GENERATION_AUTHORITY=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
