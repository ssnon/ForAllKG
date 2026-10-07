from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    run_prospective_identification_shadow,
)
from pipeline_core.discovery.research_idea_closed_loop import (
    build_population_closed_loop_case,
    run_realization_lifecycle,
)
from pipeline_core.discovery.research_idea_multigeneration import ConceptualDeltaAuditReport
from pipeline_core.discovery.research_idea_offspring_execution import (
    OffspringExecutionReport,
    OffspringRealizationReport,
)
from pipeline_core.discovery.research_idea_search_contracts import GenerationalIdeaSearchShadowReport


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    label, raw = value.split("=", 1)
    label = label.strip()
    raw = raw.strip()
    if not label or not raw:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    return label, Path(raw).expanduser().resolve()


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--header values must use KEY=VALUE")
    key, item = value.split("=", 1)
    if not key.strip():
        raise argparse.ArgumentTypeError("header key may not be empty")
    return key.strip(), item


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run SIS-v2.5 scientific-population semantics and closed-loop validation: "
            "soft/versioned conceptual families, explicit 1:N ResearchIdea realizations, "
            "bounded same-idea local search before idea escalation, prospective-identification "
            "feedback, adaptive-controller trace adapters, family-aware search pressure, and "
            "G2->G3 credit-continuity audits."
        )
    )
    p.add_argument("--case", action="append", type=_case, required=True)
    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument("--prospective-model", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header)
    p.add_argument("--max-g2-ideas", type=int, default=8)
    p.add_argument("--max-g3-ideas", type=int, default=8)
    p.add_argument("--max-per-parent", type=int, default=2)
    p.add_argument("--max-realizations-per-idea", type=int, default=2)
    p.add_argument("--max-repair-attempts", type=int, default=1)
    p.add_argument("--max-prospective-audits-per-generation", type=int, default=12)
    p.add_argument("--skip-prospective", action="store_true")
    p.add_argument("--family-overconcentration-threshold", type=float, default=0.35)
    p.add_argument("--semantic-sample-size", type=int, default=24)
    p.add_argument("--max-adaptive-plan-files", type=int, default=64)
    p.add_argument("--output", required=True, type=Path)
    return p


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected artifact paths exist: " + ", ".join(map(str, paths)))


def _case_paths(root: Path) -> dict[str, Path]:
    out = root / "scientific_portfolio_shadow"
    return {
        "context": _first_existing(root / "hypothesis.context.json", out / "hypothesis.context.json"),
        "source_generational": _first_existing(
            out / "sis_v2_2.source_v2_1_1.generational_shadow.json",
            out / "sis_v2_1_1.feedback_aware.generational_shadow.json",
            out / "sis_v2_1.generational_shadow.json",
        ),
        "g2_execution": out / "sis_v2_3.offspring_execution.json",
        "g2_seed_realization": out / "sis_v2_3.realization.report.json",
        "g2_seed_portfolio": out / "sis_v2_3.materialized.shadow.portfolio.json",
        "g2_audit": out / "sis_v2_4.g2_conceptual_delta.json",
        "g3_execution": out / "sis_v2_4.g3_offspring_execution.json",
        "g3_audit": out / "sis_v2_4.g3_conceptual_delta.json",
        "family": out / "sis_v2_5.family_population.json",
        "g2_lifecycle": out / "sis_v2_5.g2_realization_lifecycle.json",
        "g2_portfolio": out / "sis_v2_5.g2_lifecycle.portfolio.json",
        "g3_lifecycle": out / "sis_v2_5.g3_realization_lifecycle.json",
        "g3_portfolio": out / "sis_v2_5.g3_lifecycle.portfolio.json",
        "policy": out / "sis_v2_5.closed_loop_policy.json",
        "credit": out / "sis_v2_5.credit_continuity.json",
        "semantic_sample": out / "sis_v2_5.semantic_calibration_sample.json",
        "verification_facets": out / "sis_v2_5.verification_facet_integration.json",
        "summary": out / "sis_v2_5.case_summary.json",
        "telemetry": out / "sis_v2_5.realization.telemetry.jsonl",
        "prospective_root": out / "sis_v2_5_prospective",
    }


def _adaptive_plans(root: Path, *, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    candidates = []
    for path in root.rglob("*.json"):
        name = path.name.casefold()
        full = str(path).casefold()
        if "adaptive" not in full and "controller" not in full:
            continue
        if "plan" not in name and "controller" not in name:
            continue
        candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    plans: list[dict[str, Any]] = []
    artifacts: list[str] = []
    for path in candidates[: max(0, limit)]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != "adaptive-discovery-controller-plan-v1":
            continue
        plans.append(payload)
        artifacts.append(str(path))
    return plans, artifacts


def _residual_reports(root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in root.rglob("*.json"):
        if "epistemic_state" not in str(path).casefold() and "residual" not in str(path).casefold():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == "scientific-portfolio-residual-epistemic-state-v1"
        ):
            reports.append(payload)
    return reports


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")


def main() -> int:
    args = parser().parse_args()
    if not args.model:
        raise SystemExit(
            "No model configured. Set GRAPHAGENTS_HYPOTHESIS_MODEL / OPENROUTER_AGENT_MODEL or pass --model."
        )
    if args.max_realizations_per_idea < 1:
        raise SystemExit("--max-realizations-per-idea must be >= 1")
    if args.max_g2_ideas < 1 or args.max_g3_ideas < 1:
        raise SystemExit("--max-g2-ideas and --max-g3-ideas must be >= 1")

    headers = dict(args.header)
    cohort: dict[str, Any] = {}
    aggregate = {
        "case_count": 0,
        "conceptual_family_count_sum": 0,
        "same_idea_rescue_count": 0,
        "child_rescue_count": 0,
        "productive_lineage_count": 0,
        "unresolved_credit_count": 0,
        "g2_usable_grounded_realizations": 0,
        "g3_usable_grounded_realizations": 0,
        "adaptive_local_events": 0,
        "adaptive_boundary_sensitive_events": 0,
        "local_realization_llm_calls": 0,
        "prospective_audit_llm_calls": 0,
        "prospective_observations": 0,
        "residual_matched_hypotheses": 0,
    }

    for label, root in args.case:
        paths = _case_paths(root)
        for key in ("g2_execution", "g2_audit", "g3_execution", "g3_audit"):
            _require(paths[key], key)

        context = _load(paths["context"], HypothesisContext)
        source_generational = _load(paths["source_generational"], GenerationalIdeaSearchShadowReport)
        g2_execution = _load(paths["g2_execution"], OffspringExecutionReport)
        g2_audit = _load(paths["g2_audit"], ConceptualDeltaAuditReport)
        g3_execution = _load(paths["g3_execution"], OffspringExecutionReport)
        g3_audit = _load(paths["g3_audit"], ConceptualDeltaAuditReport)

        seed_realization = (
            _load(paths["g2_seed_realization"], OffspringRealizationReport)
            if paths["g2_seed_realization"].exists()
            else None
        )
        seed_portfolio = (
            _load(paths["g2_seed_portfolio"], HypothesisPortfolio)
            if paths["g2_seed_portfolio"].exists()
            else None
        )

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
            telemetry_context={"pipeline": "sis_v2_5_population_closed_loop", "case": label},
        )

        prospective_model = args.prospective_model or args.model
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

        g2_lifecycle, g2_portfolio = run_realization_lifecycle(
            execution=g2_execution,
            context=context,
            backend=backend,
            output_dir=paths["prospective_root"] / "g2",
            max_ideas=args.max_g2_ideas,
            max_realizations_per_idea=args.max_realizations_per_idea,
            max_per_parent=args.max_per_parent,
            max_repair_attempts=args.max_repair_attempts,
            prospective_runner=prospective_runner,
            max_prospective_audits=args.max_prospective_audits_per_generation,
            seed_realization_report=seed_realization,
            seed_portfolio=seed_portfolio,
        )
        g3_lifecycle, g3_portfolio = run_realization_lifecycle(
            execution=g3_execution,
            context=context,
            backend=backend,
            output_dir=paths["prospective_root"] / "g3",
            max_ideas=args.max_g3_ideas,
            max_realizations_per_idea=args.max_realizations_per_idea,
            max_per_parent=args.max_per_parent,
            max_repair_attempts=args.max_repair_attempts,
            prospective_runner=prospective_runner,
            max_prospective_audits=args.max_prospective_audits_per_generation,
        )

        adaptive_raw, adaptive_paths = _adaptive_plans(
            root,
            limit=args.max_adaptive_plan_files,
        )
        artifacts = build_population_closed_loop_case(
            source_generational=source_generational,
            generation2_execution=g2_execution,
            generation2_lifecycle=g2_lifecycle,
            generation2_audit=g2_audit,
            generation3_execution=g3_execution,
            generation3_lifecycle=g3_lifecycle,
            generation3_audit=g3_audit,
            adaptive_raw_plans=adaptive_raw,
            adaptive_source_artifacts=adaptive_paths,
            residual_reports=_residual_reports(root),
            additional_residual_source_portfolio_ids=(
                [seed_portfolio.portfolio_id] if seed_portfolio is not None else []
            ),
            family_overconcentration_threshold=args.family_overconcentration_threshold,
            semantic_sample_size=args.semantic_sample_size,
        )

        _write(paths["g2_lifecycle"], artifacts.generation2_lifecycle)
        _write(paths["g2_portfolio"], g2_portfolio)
        _write(paths["g3_lifecycle"], artifacts.generation3_lifecycle)
        _write(paths["g3_portfolio"], g3_portfolio)
        _write(paths["family"], artifacts.family_report)
        _write(paths["policy"], artifacts.policy_report)
        _write(paths["credit"], artifacts.credit_report)
        _write(paths["semantic_sample"], artifacts.semantic_sample)
        _write(paths["verification_facets"], artifacts.verification_facets)
        _write(paths["summary"], artifacts.summary)

        s = artifacts.summary
        cohort[label] = s.model_dump(mode="json")
        aggregate["case_count"] += 1
        aggregate["conceptual_family_count_sum"] += s.conceptual_family_count
        aggregate["same_idea_rescue_count"] += (
            artifacts.credit_report.same_idea_rescue_count
        )
        aggregate["child_rescue_count"] += artifacts.credit_report.child_rescue_count
        aggregate["productive_lineage_count"] += artifacts.credit_report.productive_lineage_count
        aggregate["unresolved_credit_count"] += artifacts.credit_report.unresolved_count
        aggregate["g2_usable_grounded_realizations"] += s.generation2_usable_grounded_realization_count
        aggregate["g3_usable_grounded_realizations"] += s.generation3_usable_grounded_realization_count
        aggregate["adaptive_local_events"] += s.adaptive_local_event_count
        aggregate["adaptive_boundary_sensitive_events"] += s.adaptive_boundary_sensitive_event_count
        aggregate["local_realization_llm_calls"] += s.local_realization_llm_calls
        aggregate["prospective_audit_llm_calls"] += s.prospective_audit_llm_calls
        aggregate["prospective_observations"] += s.prospective_observation_count
        aggregate["residual_matched_hypotheses"] += s.residual_matched_hypothesis_count

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print("ideas / soft families:", s.total_research_idea_nodes, "/", s.conceptual_family_count)
        print("families by generation:", s.generation_family_counts)
        print("family births:", s.family_birth_count_by_generation)
        print(
            "family compute HHI / largest share:",
            round(s.family_compute_hhi, 4),
            "/",
            round(s.largest_family_compute_share, 4),
        )
        print(
            "G2 realizations / usable / same-idea rescues:",
            s.generation2_realization_count,
            "/",
            s.generation2_usable_grounded_realization_count,
            "/",
            s.generation2_same_idea_rescue_count,
        )
        print(
            "G3 realizations / usable / same-idea rescues:",
            s.generation3_realization_count,
            "/",
            s.generation3_usable_grounded_realization_count,
            "/",
            s.generation3_same_idea_rescue_count,
        )
        print("credit continuity:", s.credit_continuity_outcome_counts)
        print("policy decisions:", s.policy_decision_counts)
        print(
            "adaptive local / boundary-sensitive events:",
            s.adaptive_local_event_count,
            "/",
            s.adaptive_boundary_sensitive_event_count,
        )
        print("semantic human-calibration sample:", s.semantic_calibration_sample_count)
        print(
            "prospective observations / residual matches:",
            s.prospective_observation_count,
            "/",
            s.residual_matched_hypothesis_count,
        )
        print("LOCAL_REALIZATION_LLM_CALLS:", s.local_realization_llm_calls)
        print("PROSPECTIVE_AUDIT_LLM_CALLS:", s.prospective_audit_llm_calls)
        print("FAMILY_IS_HARD_GATE=False")
        print("ONE_REALIZATION_FAILURE_IS_IDEA_FAILURE=False")
        print("PRODUCTION_GENERATION_AUTHORITY=False")
        print("PRODUCTION_SELECTION_AUTHORITY=False")

    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": "sis-v2-5-population-closed-loop-cohort-v1",
        "cases": cohort,
        "aggregate": aggregate,
        "conceptual_family_is_soft_and_recomputable": True,
        "observation_policy_separated": True,
        "one_realization_failure_is_not_idea_failure": True,
        "local_search_precedes_idea_escalation": True,
        "single_scalar_fitness_used": False,
        "production_generation_authority": False,
        "production_selection_authority": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
    }
    _write(output, payload)
    print()
    print("SIS-v2.5 scientific-population closed-loop cohort complete")
    print("cases:", len(cohort))
    print("aggregate:", aggregate)
    print("FAMILY_IS_HARD_GATE=False")
    print("ONE_REALIZATION_FAILURE_IS_IDEA_FAILURE=False")
    print("PRODUCTION_GENERATION_AUTHORITY=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
