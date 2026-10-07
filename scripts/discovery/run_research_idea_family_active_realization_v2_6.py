from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    run_prospective_identification_shadow,
)
from pipeline_core.discovery.research_idea_active_realization import (
    attach_exact_feedback,
    build_boundary_escalation_traces,
    run_active_realization_search,
)
from pipeline_core.discovery.research_idea_closed_loop import RealizationLifecycleReport
from pipeline_core.discovery.research_idea_closed_loop_v2_6 import (
    build_active_closed_loop_case_report,
    build_credit_continuity_v2,
)
from pipeline_core.discovery.research_idea_family_calibration import (
    build_family_calibration,
    build_program_family_compute_report,
    calibrate_frontier_population,
)
from pipeline_core.discovery.research_idea_offspring_execution import OffspringExecutionReport
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
    if not label.strip() or not raw.strip():
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    return label.strip(), Path(raw).expanduser().resolve()


def _reference_population(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--family-reference-population must use LABEL=/path/to/frontier.json"
        )
    label, raw = value.split("=", 1)
    if not label.strip() or not raw.strip():
        raise argparse.ArgumentTypeError(
            "--family-reference-population must use LABEL=/path/to/frontier.json"
        )
    return label.strip(), Path(raw).expanduser().resolve()


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
            "Run SIS-v2.6 Family-Calibrated Active Realization Search over the frozen "
            "G2/G3 population: graph-based family calibration, exact-lineage feedback "
            "resolution, active same-idea repair/re-axis before global mutation, local/global "
            "AXIS_MUTATION boundary audit, and generation-independent credit continuity."
        )
    )
    p.add_argument("--case", action="append", type=_case, required=True)
    p.add_argument("--family-reference-population", action="append", type=_reference_population, default=[])
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
    p.add_argument("--max-active-g2-ideas", type=int, default=8)
    p.add_argument("--max-active-g3-ideas", type=int, default=8)
    p.add_argument("--max-actions-per-idea", type=int, default=2)
    p.add_argument("--max-repair-attempts", type=int, default=1)
    p.add_argument("--max-prospective-audits-per-generation", type=int, default=12)
    p.add_argument("--skip-prospective", action="store_true")
    p.add_argument("--feedback-search-root", action="append", type=Path, default=[])
    p.add_argument("--program-family-similarity-floor", type=float, default=0.58)
    p.add_argument("--family-curve-threshold", action="append", type=float, default=[])
    p.add_argument("--family-overconcentration-threshold", type=float, default=0.35)
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
        "g3_execution": out / "sis_v2_4.g3_offspring_execution.json",
        "g2_lifecycle": out / "sis_v2_5.g2_realization_lifecycle.json",
        "g2_portfolio": out / "sis_v2_5.g2_lifecycle.portfolio.json",
        "g3_lifecycle": out / "sis_v2_5.g3_realization_lifecycle.json",
        "g3_portfolio": out / "sis_v2_5.g3_lifecycle.portfolio.json",
        "family_calibration": out / "sis_v2_6.family_calibration.json",
        "family_compute": out / "sis_v2_6.family_compute.json",
        "g2_feedback": out / "sis_v2_6.g2_exact_feedback.json",
        "g3_feedback": out / "sis_v2_6.g3_exact_feedback.json",
        "g2_active": out / "sis_v2_6.g2_active_realization.json",
        "g2_active_portfolio": out / "sis_v2_6.g2_active.portfolio.json",
        "g3_active": out / "sis_v2_6.g3_active_realization.json",
        "g3_active_portfolio": out / "sis_v2_6.g3_active.portfolio.json",
        "boundary": out / "sis_v2_6.local_global_boundary.json",
        "credit": out / "sis_v2_6.credit_continuity_v2.json",
        "summary": out / "sis_v2_6.case_summary.json",
        "telemetry": out / "sis_v2_6.active_realization.telemetry.jsonl",
        "active_root": out / "sis_v2_6_active_realization",
    }


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")


def _search_roots(case_root: Path, explicit: list[Path]) -> list[Path]:
    values = [case_root, case_root.parent, *explicit]
    seen: set[Path] = set()
    out: list[Path] = []
    for value in values:
        path = value.expanduser().resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        out.append(path)
    return out


def main() -> int:
    args = parser().parse_args()
    if not args.model:
        raise SystemExit(
            "No model configured. Set GRAPHAGENTS_HYPOTHESIS_MODEL / OPENROUTER_AGENT_MODEL or pass --model."
        )
    if args.max_actions_per_idea < 1:
        raise SystemExit("--max-actions-per-idea must be >= 1")
    if not 0.0 <= args.program_family_similarity_floor <= 1.0:
        raise SystemExit("--program-family-similarity-floor must be in [0,1]")
    thresholds = (
        args.family_curve_threshold
        if args.family_curve_threshold
        else [0.42, 0.50, 0.58, 0.66, 0.78]
    )
    headers = dict(args.header)
    cohort: dict[str, Any] = {}
    aggregate = Counter()

    reference_reports: dict[str, Any] = {}
    for label, path in args.family_reference_population:
        _require(path, f"family reference population {label}")
        population = _load(path, FrontierIdeaPopulation)
        report = calibrate_frontier_population(
            population,
            program_similarity_floor=args.program_family_similarity_floor,
            curve_thresholds=thresholds,
        )
        reference_reports[label] = report.model_dump(mode="json")

    for label, root in args.case:
        paths = _case_paths(root)
        for key in (
            "g2_execution", "g3_execution", "g2_lifecycle", "g2_portfolio",
            "g3_lifecycle", "g3_portfolio",
        ):
            _require(paths[key], key)

        context = _load(paths["context"], HypothesisContext)
        source_generational = _load(paths["source_generational"], GenerationalIdeaSearchShadowReport)
        g2_execution = _load(paths["g2_execution"], OffspringExecutionReport)
        g3_execution = _load(paths["g3_execution"], OffspringExecutionReport)
        g2_lifecycle = _load(paths["g2_lifecycle"], RealizationLifecycleReport)
        g2_portfolio = _load(paths["g2_portfolio"], HypothesisPortfolio)
        g3_lifecycle = _load(paths["g3_lifecycle"], RealizationLifecycleReport)
        g3_portfolio = _load(paths["g3_portfolio"], HypothesisPortfolio)

        nodes = [
            *source_generational.research_ideas,
            *g2_execution.offspring_nodes,
            *g3_execution.offspring_nodes,
        ]
        node_by_id = {row.idea_id: row for row in nodes}
        nodes = list(node_by_id.values())
        family = build_family_calibration(
            nodes,
            program_similarity_floor=args.program_family_similarity_floor,
            curve_thresholds=thresholds,
        )

        roots = _search_roots(root, args.feedback_search_root)
        g2_feedback = attach_exact_feedback(lifecycle=g2_lifecycle, search_roots=roots)
        g3_feedback = attach_exact_feedback(lifecycle=g3_lifecycle, search_roots=roots)

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
            telemetry_context={"pipeline": "sis_v2_6_active_realization", "case": label},
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

        g2_result = run_active_realization_search(
            execution=g2_execution,
            lifecycle=g2_lifecycle,
            portfolio=g2_portfolio,
            context=context,
            feedback=g2_feedback,
            backend=backend,
            output_dir=paths["active_root"] / "g2",
            prospective_runner=prospective_runner,
            max_active_ideas=args.max_active_g2_ideas,
            max_actions_per_idea=args.max_actions_per_idea,
            max_repair_attempts=args.max_repair_attempts,
            max_prospective_audits=args.max_prospective_audits_per_generation,
        )
        g3_result = run_active_realization_search(
            execution=g3_execution,
            lifecycle=g3_lifecycle,
            portfolio=g3_portfolio,
            context=context,
            feedback=g3_feedback,
            backend=backend,
            output_dir=paths["active_root"] / "g3",
            prospective_runner=prospective_runner,
            max_active_ideas=args.max_active_g3_ideas,
            max_actions_per_idea=args.max_actions_per_idea,
            max_repair_attempts=args.max_repair_attempts,
            max_prospective_audits=args.max_prospective_audits_per_generation,
        )

        family_compute = build_program_family_compute_report(
            calibration=family,
            nodes=nodes,
            generation2_execution=g2_execution,
            generation3_execution=g3_execution,
            generation2_lifecycle=g2_lifecycle,
            generation3_lifecycle=g3_lifecycle,
            active_reports=[g2_result.report, g3_result.report],
            overconcentration_threshold=args.family_overconcentration_threshold,
        )
        boundary = build_boundary_escalation_traces(
            generation2_active=g2_result.report,
            generation3_active=g3_result.report,
            generation3_execution=g3_execution,
        )
        credit = build_credit_continuity_v2(
            generation2_execution=g2_execution,
            generation2_lifecycle=g2_lifecycle,
            generation2_active=g2_result.report,
            generation3_execution=g3_execution,
            generation3_lifecycle=g3_lifecycle,
            generation3_active=g3_result.report,
        )
        case_report = build_active_closed_loop_case_report(
            family_calibration=family,
            family_compute=family_compute,
            generation2_feedback=g2_feedback,
            generation3_feedback=g3_feedback,
            generation2_active=g2_result.report,
            generation3_active=g3_result.report,
            boundary_traces=boundary,
            credit_continuity=credit,
        )

        _write(paths["family_calibration"], family)
        _write(paths["family_compute"], family_compute)
        _write(paths["g2_feedback"], g2_feedback)
        _write(paths["g3_feedback"], g3_feedback)
        _write(paths["g2_active"], g2_result.report)
        _write(paths["g2_active_portfolio"], g2_result.portfolio)
        _write(paths["g3_active"], g3_result.report)
        _write(paths["g3_active_portfolio"], g3_result.portfolio)
        _write(paths["boundary"], {"schema_version": "sis-v2-6-local-global-boundary-v1", "traces": [row.model_dump(mode="json") for row in boundary]})
        _write(paths["credit"], credit)
        _write(paths["summary"], case_report.summary)

        s = case_report.summary
        cohort[label] = s.model_dump(mode="json")
        aggregate["case_count"] += 1
        aggregate["idea_count"] += s.total_research_idea_nodes
        aggregate["program_family_count"] += s.program_family_count
        aggregate["g2_active_same_idea_rescue_count"] += s.generation2_active_same_idea_rescue_count
        aggregate["g3_active_same_idea_rescue_count"] += s.generation3_active_same_idea_rescue_count
        aggregate["total_active_same_idea_rescue_count"] += s.total_active_same_idea_rescue_count
        aggregate["child_rescue_count"] += s.child_rescue_count
        aggregate["productive_lineage_count"] += s.productive_lineage_count
        aggregate["unresolved_credit_count"] += s.unresolved_credit_count
        aggregate["prospective_feedback_evaluated"] += s.prospective_feedback_evaluated_count
        aggregate["prospective_feedback_not_evaluated"] += s.prospective_feedback_not_evaluated_count
        aggregate["residual_feedback_evaluated"] += s.residual_feedback_evaluated_count
        aggregate["residual_feedback_not_evaluated"] += s.residual_feedback_not_evaluated_count
        aggregate["local_llm_calls"] += s.local_llm_call_count
        aggregate["prospective_audit_llm_calls"] += s.prospective_audit_llm_call_count

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print(
            "ideas / tight neighborhoods / program families:",
            s.total_research_idea_nodes,
            "/",
            s.tight_neighborhood_count,
            "/",
            s.program_family_count,
        )
        print("family fragmentation ratio:", round(s.family_fragmentation_ratio, 4))
        print(
            "program-family compute HHI / largest share:",
            round(s.program_family_compute_hhi, 4),
            "/",
            round(s.largest_program_family_compute_share, 4),
        )
        print("overconcentrated program families:", s.overconcentrated_program_family_count)
        print(
            "G2/G3 active targets:",
            s.generation2_active_target_count,
            "/",
            s.generation3_active_target_count,
        )
        print(
            "G2/G3 same-idea rescues:",
            s.generation2_active_same_idea_rescue_count,
            "/",
            s.generation3_active_same_idea_rescue_count,
        )
        print("active local actions:", s.active_local_action_counts)
        print("boundary resolutions:", s.boundary_escalation_resolution_counts)
        print(
            "prospective evaluated/not-evaluated:",
            s.prospective_feedback_evaluated_count,
            "/",
            s.prospective_feedback_not_evaluated_count,
        )
        print(
            "residual evaluated/not-evaluated:",
            s.residual_feedback_evaluated_count,
            "/",
            s.residual_feedback_not_evaluated_count,
        )
        print("credit continuity:", s.credit_continuity_outcome_counts)
        print("LOCAL_LLM_CALLS:", s.local_llm_call_count)
        print("PROSPECTIVE_AUDIT_LLM_CALLS:", s.prospective_audit_llm_call_count)
        print("FAMILY_IS_HARD_GATE=False")
        print("NOT_EVALUATED_IS_NEGATIVE_FEEDBACK=False")
        print("G4_GENERATION=False")
        print("PRODUCTION_GENERATION_AUTHORITY=False")
        print("PRODUCTION_SELECTION_AUTHORITY=False")

    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": "sis-v2-6-family-active-realization-cohort-v1",
        "cases": cohort,
        "aggregate": dict(aggregate),
        "family_reference_calibrations": reference_reports,
        "program_family_similarity_floor": args.program_family_similarity_floor,
        "family_curve_thresholds": thresholds,
        "family_is_soft_and_recomputable": True,
        "family_is_not_hard_gate": True,
        "not_evaluated_is_not_negative_feedback": True,
        "active_local_search_precedes_global_axis_mutation": True,
        "axis_mutation_requires_semantic_boundary_resolution": True,
        "g4_generation_executed": False,
        "production_generation_authority": False,
        "production_selection_authority": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
    }
    _write(output, payload)
    print()
    print("SIS-v2.6 Family-Calibrated Active Realization Search complete")
    print("cases:", len(cohort))
    print("aggregate:", dict(aggregate))
    print("family references:", list(reference_reports))
    print("FAMILY_IS_HARD_GATE=False")
    print("NOT_EVALUATED_IS_NEGATIVE_FEEDBACK=False")
    print("G4_GENERATION=False")
    print("PRODUCTION_GENERATION_AUTHORITY=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
