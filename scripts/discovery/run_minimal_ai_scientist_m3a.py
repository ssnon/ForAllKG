"""M3-A opt-in orchestration: Stage7 idea sources -> direct P0 -> SIS-v3.4 -> verifier.

No new scientific semantics, and no mutation of the legacy Full Current runner.
The default existing-case seed mode is offline. Anything paid needs --allow-paid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PREFIX = "scripts.discovery."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-mode", choices=("existing", "generate"), default="existing")
    p.add_argument("--case-dir", type=Path, help="Existing Stage7 output; required in source-mode existing")
    p.add_argument("--output-dir", type=Path, required=True, help="New output directory, never overwritten")
    p.add_argument("--through", choices=("seed", "sis", "verified"), default="seed")
    p.add_argument("--allow-paid", action="store_true", help="Explicit permission for Stage7/LLM/SIS/external/verifier calls")
    p.add_argument("--plan-only", action="store_true", help="Print full command plan; no commands or writes")

    # Frozen-case parity pins apply to existing-case stage only.
    p.add_argument("--expected-selection", type=Path)
    p.add_argument("--expected-p0-execution", type=Path)
    p.add_argument("--m1-freeze-manifest", type=Path)

    # The fresh source preset deliberately matches the Q-A Full Current source lane;
    # no extra KG/HO/Stage7 science is added here.
    p.add_argument("--source", help="Scientific relation source for a new Stage7 run")
    p.add_argument("--target", help="Scientific relation target for a new Stage7 run")
    p.add_argument("--question", help="Scientific question for a new Stage7 run")
    p.add_argument("--data-root", type=Path, help="Corpus runtime path (fresh runs only)")
    p.add_argument("--corpus-id", default="sers500_final_v2")
    p.add_argument("--domain-profile", default="sers_au_ag")
    p.add_argument("--model", default=os.getenv("OPENROUTER_AGENT_MODEL") or "")
    p.add_argument("--critic-model", default=os.getenv("OPENROUTER_CRITIC_MODEL") or None)
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or None)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--max-selected-parents", type=int, default=3)
    p.add_argument("--novelty-results-per-query", type=int, default=6)
    p.add_argument("--providers", default="auto")
    p.add_argument("--provider-plan", type=Path)
    p.add_argument("--skip-provider-budget-preflight", action="store_true", help="Explicit legacy OpenAlex workaround; not used by default")
    return p


def preflight(a: argparse.Namespace) -> tuple[Path, Path]:
    out = a.output_dir.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"M3-A refuses to overwrite output: {out}")
    if a.cycles < 1 or a.max_selected_parents < 1 or a.novelty_results_per_query < 1:
        raise ValueError("search budgets must be >= 1")
    if a.source_mode == "existing":
        if a.case_dir is None:
            raise ValueError("--case-dir required for existing source mode")
        case = a.case_dir.expanduser().resolve()
        if not case.is_dir():
            raise FileNotFoundError(case)
        if case == out or case in out.parents or out in case.parents:
            raise ValueError("output must be outside the existing case directory and its ancestors")
        for name in ("hypothesis.context.json", "frontier_idea_population.shadow.json", "idea_evolution.shadow.json"):
            if not (case / name).is_file():
                raise FileNotFoundError(case / name)
    else:
        if a.case_dir is not None or a.expected_selection or a.expected_p0_execution or a.m1_freeze_manifest:
            raise ValueError("fresh Stage7 generation cannot use frozen source-case arguments")
        if not all((a.source, a.target, a.question, a.data_root)):
            raise ValueError("fresh generation requires --source, --target, --question, --data-root")
        if not a.data_root.expanduser().resolve().is_dir():
            raise FileNotFoundError(a.data_root)
        case = out / "case"
    for pin in (a.expected_selection, a.expected_p0_execution, a.m1_freeze_manifest, a.provider_plan):
        if pin is not None and not pin.expanduser().resolve().is_file():
            raise FileNotFoundError(pin)
    if (a.source_mode == "generate" or a.through != "seed") and not a.allow_paid and not a.plan_only:
        raise PermissionError("Paid/network stages require explicit --allow-paid")
    if a.source_mode == "generate" or a.through != "seed":
        if not a.model:
            raise ValueError("--model or OPENROUTER_AGENT_MODEL is required")
    return case, out


def _prefix(module: str) -> list[str]:
    return [sys.executable, "-m", PREFIX + module]


def stage7_cmd(a: argparse.Namespace, case: Path) -> list[str]:
    # Keep the frozen Full Current scientific idea-source lane and budgets; omit
    # Stage7 scientific-portfolio-selection-shadow / early materialization.
    return _prefix("run_dac_discovery_e2e") + [
        "--corpus-id", a.corpus_id,
        "--data-root", str(a.data_root.expanduser().resolve()),
        "--domain-profile", a.domain_profile,
        "--run-dir", str(case),
        "--source", a.source, "--target", a.target, "--question", a.question,
        "--objective", "explain_connection", "--title", "MINIMAL_AI_SCIENTIST_M3A",
        "--grounding-policy", "semantic_stop_fallback_top_n", "--context-review-mode", "auto",
        "--question-task-preservation-shadow", "--open-world-discovery",
        "--direct-relationpattern-task-shadow", "--direct-relationpattern-top-k", "20",
        "--direct-higher-order-shadow", "--direct-higher-order-max-contexts", "4",
        "--higher-order-shadow", "--higher-order-max-contexts", "12",
        "--frontier-idea-population-shadow", "--idea-evolution-shadow",
        "--idea-evolution-max-cross-source-outputs", "6",
        "--idea-evolution-max-backbone-mutation-outputs", "6",
        "--idea-evolution-max-candidate-interpretation-outputs", "4",
        "--idea-evolution-max-candidate-parent-pool", "8",
        "--post-generation-n10-authority-mode", "certification_only",
        "--results-per-query", "12",
        "--model", a.model, "--critic-model", a.critic_model or a.model,
        "--api-key-env", a.api_key_env, "--stop-after-initial-semantic",
    ] + (["--base-url", a.base_url] if a.base_url else [])


def seed_cmd(a: argparse.Namespace, case: Path, out: Path) -> list[str]:
    cmd = _prefix("run_direct_research_idea_seed_m2c") + [
        "--context", str(case / "hypothesis.context.json"),
        "--population", str(case / "frontier_idea_population.shadow.json"),
        "--evolution-report", str(case / "idea_evolution.shadow.json"),
        "--output-dir", str(out / "direct_seed"),
    ]
    if a.source_mode == "existing":
        # Frozen replay consumes original 35-candidate evaluation, never re-pays.
        cmd += [
            "--candidate-pool", str(case / "scientific_portfolio_shadow" / "candidate_pool.json"),
            "--evaluation", str(case / "scientific_portfolio_shadow" / "evaluation.json"),
        ]
        for flag, value in (("--expected-selection", a.expected_selection),
                            ("--expected-p0-execution", a.expected_p0_execution),
                            ("--m1-freeze-manifest", a.m1_freeze_manifest)):
            if value is not None:
                cmd += [flag, str(value.expanduser().resolve())]
    else:
        cmd += [
            "--frontier-audit", str(case / "frontier_exploration.audit.json"),
            "--task-source", a.source, "--task-target", a.target,
            "--model", a.model, "--api-key-env", a.api_key_env,
            "--allow-evaluation-llm",
            "--max-evaluation-candidates", "48", "--max-retained", "8",
            "--max-retained-per-profile", "2",
        ]
        if a.base_url:
            cmd += ["--base-url", a.base_url]
    return cmd


def sis_cmd(a: argparse.Namespace, case: Path, out: Path) -> list[str]:
    cmd = _prefix("run_research_idea_e2e_search_v3_4") + [
        "--run-dir", str(case),
        "--p0-execution", str(out / "direct_seed" / "p0.execution.json"),
        "--output-dir", str(out / "sis_v3_4"),
        "--cycles", str(a.cycles),
        "--model", a.model, "--api-key-env", a.api_key_env,
        "--max-selected-parents", str(a.max_selected_parents),
        "--novelty-results-per-query", str(a.novelty_results_per_query),
        "--providers", a.providers,
    ]
    if a.base_url:
        cmd += ["--base-url", a.base_url]
    if a.provider_plan:
        cmd += ["--provider-plan", str(a.provider_plan.expanduser().resolve())]
    # Deliberately no M2-D cache flags until matched scientific-quality gate.
    return cmd


def archive_cmd(out: Path, final_sha: str) -> list[str]:
    return _prefix("build_scientific_program_preservation_m2f") + [
        "--v34-dir", str(out / "sis_v3_4"),
        "--p0-execution", str(out / "direct_seed" / "p0.execution.json"),
        "--expected-final-sha256", final_sha,
        "--output-dir", str(out / "program_archive"),
    ]


def verify_cmd(a: argparse.Namespace, case: Path, out: Path, final: Path) -> list[str]:
    cmd = _prefix("run_standard_portfolio_verification_shadow") + [
        "--context", str(case / "hypothesis.context.json"),
        "--portfolio", str(final), "--domain-profile", a.domain_profile,
        "--output-dir", str(out / "verification"),
        "--model", a.critic_model or a.model, "--api-key-env", a.api_key_env,
    ]
    if a.base_url:
        cmd += ["--base-url", a.base_url]
    if a.skip_provider_budget_preflight:
        cmd.append("--skip-provider-budget-preflight")
    return cmd


def run_stage(name: str, cmd: list[str], logdir: Path) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / (name + ".stdout.txt")).open("w", encoding="utf-8") as stdout, \
         (logdir / (name + ".stderr.txt")).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(cmd, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode:
        raise RuntimeError(f"{name} returned {completed.returncode}; logs: {logdir}")


def validate_seed(out: Path) -> dict[str, Any]:
    p = out / "direct_seed"
    summary = read_json(p / "direct_seed.summary.json")
    exe = read_json(p / "p0.execution.json")
    if summary.get("status") != "DIRECT_P0_READY":
        raise ValueError("direct-seed summary not ready")
    if summary.get("p0_execution_report_id") != exe.get("report_id"):
        raise ValueError("P0 execution report ID mismatch")
    if summary.get("selected_research_idea_count") != exe.get("g4_population_count"):
        raise ValueError("P0 count mismatch")
    if summary.get("early_materialization_llm_calls") != 0 or summary.get("early_hypothesis_materialization_executed") is not False:
        raise ValueError("M3-A early materialization was unexpectedly executed")
    if exe.get("generation_index") != 2 or not exe.get("grounding_required_before_scientific_claim"):
        raise ValueError("P0 grounding boundary invalid")
    return summary


def validate_sis(out: Path, seed: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    sp = out / "sis_v3_4"
    summary = read_json(sp / "arm.summary.json")
    if summary.get("source_p0_report_id") != seed["p0_execution_report_id"]:
        raise ValueError("SIS consumed a different P0")
    if summary.get("child_generation_steps", 0) < 1 or summary.get("grounding_required_before_scientific_claim") is not True:
        raise ValueError("SIS generation/grounding policy mismatch")
    if summary.get("canonical_graph_mutated") is not False or summary.get("external_prior_art_promoted_to_positive_premise") is not False:
        raise ValueError("SIS violated exploration/claim authority")
    final = (sp / "final.materialized.portfolio.json").resolve()
    reported_final = Path(summary["final_portfolio"]).resolve()
    if final != reported_final or not final.is_file():
        raise ValueError("SIS final path mismatch or missing output")
    portfolio = read_json(final)
    rows = portfolio.get("hypotheses")
    if not isinstance(rows, list) or summary.get("final_hypothesis_count") != len(rows):
        raise ValueError("SIS final hypothesis count mismatch")
    ids = [row.get("hypothesis_id") for row in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("invalid hypothesis identities")
    return summary, final, sha256(final)


def plan(a: argparse.Namespace, case: Path, out: Path) -> list[tuple[str, list[str]]]:
    cmds: list[tuple[str, list[str]]] = []
    if a.source_mode == "generate":
        cmds.append(("stage7_sources", stage7_cmd(a, case)))
    cmds.append(("direct_seed", seed_cmd(a, case, out)))
    if a.through != "seed":
        cmds.append(("sis_v3_4", sis_cmd(a, case, out)))
        cmds.append(("program_archive", archive_cmd(out, "<SHA256 computed from SIS final>")))
        if a.through == "verified":
            cmds.append(("common_verifier", verify_cmd(a, case, out, out / "sis_v3_4" / "final.materialized.portfolio.json")))
    return cmds


def run(a: argparse.Namespace) -> dict[str, Any]:
    case, out = preflight(a)
    commands = plan(a, case, out)
    if a.plan_only:
        for name, cmd in commands:
            print(f"[{name}] {shlex.join(cmd)}")
        return {"status": "PLAN_ONLY", "command_count": len(commands)}

    out.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": "minimal-ai-scientist-m3a-orchestration-v1",
        "status": "STARTED", "source_mode": a.source_mode, "through": a.through,
        "case_dir": str(case), "stage_status": {},
        "old_full_current_modified": False, "strict_grounding_relaxed": False,
        "external_prior_art_positive_premise_authority": False,
        "novelty_or_scientific_quality_certified": False,
    }
    record = out / "m3a.run.json"
    write_json(record, state)

    def step(label: str, cmd: list[str]) -> None:
        state["stage_status"][label] = "RUNNING"
        write_json(record, state)
        try:
            run_stage(label, cmd, out / "logs")
        except Exception:
            state["status"] = f"FAILED_AT_{label.upper()}"
            state["stage_status"][label] = "FAILED"
            write_json(record, state)
            raise
        state["stage_status"][label] = "COMPLETE"
        write_json(record, state)

    def checked(label: str, fn):
        try:
            return fn()
        except Exception:
            state["status"] = "FAILED_VALIDATION_AT_" + label.upper()
            write_json(record, state)
            raise

    if a.source_mode == "generate":
        step("stage7_sources", stage7_cmd(a, case))
        for filename in ("hypothesis.context.json", "frontier_idea_population.shadow.json", "idea_evolution.shadow.json", "frontier_exploration.audit.json"):
            if not (case / filename).is_file():
                state["status"] = "FAILED_STAGE7_BOUNDARY"
                write_json(record, state)
                raise FileNotFoundError(case / filename)

    step("direct_seed", seed_cmd(a, case, out))
    seed = checked("seed", lambda: validate_seed(out))
    state["p0_report_id"] = seed["p0_execution_report_id"]
    state["p0_count"] = seed["selected_research_idea_count"]
    state["p0_sha256"] = sha256(out / "direct_seed" / "p0.execution.json")
    state["early_materialization_llm_calls"] = 0
    state["portfolio_evaluation_llm_calls"] = seed["portfolio_evaluation_llm_calls"]
    state["p0_parity_checked"] = bool(seed["old_p0_exact_parity_checked"])
    if a.through != "seed":
        step("sis_v3_4", sis_cmd(a, case, out))
        sis, final, digest = checked("sis", lambda: validate_sis(out, seed))
        state["final_portfolio"] = str(final)
        state["final_portfolio_sha256"] = digest
        state["final_hypotheses"] = sis["final_hypothesis_count"]
        state["sis_generation_calls_partial"] = sis["continuation_generation_llm_calls"]
        state["sis_realization_calls_partial"] = sis["continuation_realization_llm_calls"]
        step("program_archive", archive_cmd(out, digest))
        def validate_archive():
            archival = read_json(out / "program_archive" / "M2F_SUMMARY.json")
            if archival.get("status") != "M2F_ARCHIVE_AND_TERMINAL_VIEW_READY":
                raise ValueError("archive not ready")
            if archival.get("archived_research_ideas", 0) < archival.get("terminal_active_research_ideas", 0):
                raise ValueError("archive has fewer unique ideas than active population")
            if archival.get("terminal_hypotheses") != sis["final_hypothesis_count"]:
                raise ValueError("archive vs final portfolio hypothesis count mismatch")
            return archival
        archival = checked("archive", validate_archive)
        state["archived_research_ideas"] = archival["archived_research_ideas"]
        state["historical_only_research_ideas"] = archival["replaced_parent_ideas_preserved_in_archive"]
        if a.through == "verified":
            step("common_verifier", verify_cmd(a, case, out, final))
            verified = out / "verification" / "verification.summary.json"
            verify_payload = checked("verifier", lambda: read_json(verified))
            state["verification_summary"] = str(verified)
            state["verification_summary_sha256"] = sha256(verified)
            state["common_verifier_reported_status"] = verify_payload.get("status", "NOT_REPORTED")
            state["common_verifier_executed"] = True
            # Standard verifier is itself a shadow diagnostic; never call it production-certified.
            state["novelty_or_scientific_quality_certified"] = False
    state["status"] = {"seed": "SEED_READY", "sis": "SIS_FINISHED_UNVERIFIED", "verified": "COMMON_VERIFIER_COMPLETED_NONAUTHORITATIVE"}[a.through]
    state["known_llm_counts_exclude_prior_art_family_verifier"] = True
    write_json(record, state)
    (out / "M3A_REPORT.md").write_text(
        "# M3-A — Minimal E2E Consolidation\n\n"
        f"- Status: `{state['status']}`\n"
        f"- Source mode: `{a.source_mode}`\n"
        f"- P0 count: `{state['p0_count']}`\n"
        f"- P0 exact parity checked: `{state['p0_parity_checked']}`\n"
        f"- Early Stage7 materialization LLM calls: **0**\n"
        f"- Final hypothesis count: `{state.get('final_hypotheses','not run')}`\n"
        f"- Common verifier executed: `{state.get('common_verifier_executed',False)}`\n"
        "- Search feedback, strict realization, and verifier remain distinct; this is not a novelty certificate.\n"
        "- M2-D caches, rescue automation, source-provider removal, and scheduler-policy changes are deliberately not enabled.\n",
        encoding="utf-8",
    )
    return state


def main() -> int:
    state = run(parser().parse_args())
    print("M3-A:", state["status"])
    if state["status"] != "PLAN_ONLY":
        print("Record:", state["case_dir"] if "case_dir" in state else "", "(see output m3a.run.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
