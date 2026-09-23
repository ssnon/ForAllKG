from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.reframing.e2e_shadow_companion import (
    require_completed_e2e_run,
)
from pipeline_core.discovery.reframing.live_reframing_companion import (
    common_scope_args,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run(name: str, argv: list[str], expected: list[Path]) -> None:
    print()
    print(f"== {name} ==")
    subprocess.run([sys.executable, *argv], check=True)
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"{name} completed without expected outputs: {missing}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build live scientific reframing and cross-lane synthesis from the "
            "pre-N10 Alpha6 portfolio, even when the legacy N10 final portfolio is empty. "
            "The output is a no-authority HypothesisPortfolio for fresh N10 evaluation."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument(
        "--duplicate-paper-policy",
        choices=("error", "latest_attempt"),
        default="latest_attempt",
    )
    parser.add_argument(
        "--cross-root-duplicate-policy",
        choices=("error", "prefer_last_root"),
        default="prefer_last_root",
    )
    parser.add_argument("--max-syntheses", type=int, default=3)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--telemetry", default=None)
    parser.add_argument(
        "--candidate-contract-only",
        action="store_true",
        help=(
            "Stop after materializing the production-facing candidate contract. "
            "This is intended for prospective atomic synthesis wrappers and does "
            "not run the older cross-lane synthesis/projection path."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_dir, legacy_manifest = require_completed_e2e_run(args.run_dir)

    packet = run_dir / "explorer.packet.json"
    explorer_report = run_dir / "explorer.report.json"
    context = run_dir / "hypothesis.context.json"
    alpha6 = run_dir / "novelty_refinement_a6.portfolio.json"
    for path, label in [
        (packet, "explorer packet"),
        (explorer_report, "explorer report"),
        (context, "hypothesis context"),
        (alpha6, "pre-N10 Alpha6 portfolio"),
    ]:
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    alpha6_portfolio = HypothesisPortfolio.model_validate_json(
        alpha6.read_text(encoding="utf-8")
    )
    manifest_path = run_dir / "scientific_pre_n10_synthesis_companion_manifest.json"
    manifest = {
        "schema_version": "scientific-pre-n10-synthesis-companion-manifest-v1",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "status": "planned" if args.dry_run else "running",
        "legacy_e2e_status": legacy_manifest.get("status"),
        "source_alpha6_portfolio": str(alpha6),
        "source_alpha6_hypothesis_count": len(alpha6_portfolio.hypotheses),
        "legacy_final_survivor_count_is_not_a_precondition": True,
        "candidate_contract_only": bool(args.candidate_contract_only),
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "n10_authority_created": False,
        "failure": None,
    }
    _write(manifest_path, manifest)

    if not alpha6_portfolio.hypotheses:
        manifest["status"] = "abstained_no_alpha6_hypotheses"
        manifest["finished_at_utc"] = _now()
        _write(manifest_path, manifest)
        print("Pre-N10 scientific synthesis abstained: Alpha6 portfolio is empty.")
        return 0

    scope = common_scope_args(
        roots=args.root,
        duplicate_paper_policy=args.duplicate_paper_policy,
        cross_root_duplicate_policy=args.cross_root_duplicate_policy,
    )

    readiness = run_dir / "scientific_pre_n10_reframing_readiness.json"
    evidence = run_dir / "scientific_pre_n10_reframing_evidence.json"
    tensions = run_dir / "scientific_pre_n10_reframing_tensions.json"
    triggers = run_dir / "scientific_pre_n10_reframing_triggers.json"
    reframe = run_dir / "scientific_pre_n10_reframing_shadow.json"
    contrast = run_dir / "scientific_pre_n10_mode_contrast.json"
    reasoning = run_dir / "scientific_pre_n10_reasoning_portfolio.json"
    cross_lane = run_dir / "scientific_pre_n10_cross_lane_portfolio.json"
    candidates = run_dir / "scientific_pre_n10_candidate_portfolio.json"
    synthesis = run_dir / "scientific_pre_n10_cross_lane_synthesis.json"
    synthesis_prompt = run_dir / "scientific_pre_n10_cross_lane_synthesis_prompt.txt"
    projected = run_dir / "scientific_synthesis_pre_n10.portfolio.json"
    projected_lineage = run_dir / "scientific_synthesis_pre_n10.lineage.json"

    readiness_cmd = [
        "-m", "scripts.discovery.inspect_reframing_operator_readiness",
        *scope,
        "--packet", str(packet),
        "--context", str(context),
        "--output", str(readiness),
    ]
    evidence_cmd = [
        "-m", "scripts.discovery.run_scientific_reframing_shadow",
        *scope,
        "--packet", str(packet),
        "--context", str(context),
        "--readiness", str(readiness),
        "--dry-run",
        "--evidence-output", str(evidence),
        "--output", str(reframe),
    ]
    trigger_cmd = [
        "-m", "scripts.discovery.inspect_scientific_reframe_triggers",
        "--report", str(explorer_report),
        "--evidence", str(evidence),
        "--packet", str(packet),
        "--output", str(triggers),
        "--tension-output", str(tensions),
    ]
    generation_cmd = [
        "-m", "scripts.discovery.run_scientific_reframing_shadow",
        *scope,
        "--packet", str(packet),
        "--context", str(context),
        "--readiness", str(readiness),
        "--trigger", str(triggers),
        "--evidence-output", str(evidence),
        "--output", str(reframe),
    ]
    if args.model:
        generation_cmd += ["--model", args.model]
    if args.base_url:
        generation_cmd += ["--base-url", args.base_url]
    generation_cmd += ["--api-key-env", args.api_key_env]
    if args.save_prompts:
        generation_cmd += ["--save-prompts"]
    if args.telemetry:
        generation_cmd += ["--telemetry", args.telemetry]

    mode_cmd = [
        "-m", "scripts.discovery.compare_scientific_reframing_modes",
        "--reframe-shadow", str(reframe),
        "--output", str(contrast),
    ]
    reasoning_cmd = [
        "-m", "scripts.discovery.build_scientific_reasoning_portfolio",
        "--mode-contrast", str(contrast),
        "--output", str(reasoning),
    ]
    cross_lane_cmd = [
        "-m", "scripts.discovery.build_cross_lane_scientific_reasoning_portfolio",
        "--context", str(context),
        "--relational-portfolio", str(alpha6),
        "--reframing-portfolio", str(reasoning),
        "--reframe-shadow", str(reframe),
        "--output", str(cross_lane),
    ]
    candidate_cmd = [
        "-m", "scripts.discovery.build_production_facing_scientific_candidate_portfolio",
        "--cross-lane-portfolio", str(cross_lane),
        "--relational-portfolio", str(alpha6),
        "--reframe-shadow", str(reframe),
        "--output", str(candidates),
    ]
    synthesis_cmd = [
        "-m", "scripts.discovery.run_cross_lane_scientific_synthesis_shadow",
        "--portfolio", str(candidates),
        "--max-syntheses", str(args.max_syntheses),
        "--prompt-output", str(synthesis_prompt),
        "--output", str(synthesis),
    ]
    if args.model:
        synthesis_cmd += ["--model", args.model]
    if args.base_url:
        synthesis_cmd += ["--base-url", args.base_url]
    synthesis_cmd += ["--api-key-env", args.api_key_env]
    if args.telemetry:
        synthesis_cmd += ["--telemetry", args.telemetry]

    projection_cmd = [
        "-m", "scripts.discovery.build_scientific_synthesis_pre_n10_portfolio",
        "--context", str(context),
        "--alpha6-portfolio", str(alpha6),
        "--synthesis-report", str(synthesis),
        "--output", str(projected),
        "--lineage-output", str(projected_lineage),
    ]

    commands = [
        ("operator_readiness", readiness_cmd, [readiness]),
        ("evidence_materialization", evidence_cmd, [evidence]),
        ("trigger_detection", trigger_cmd, [triggers, tensions]),
        ("reframing_generation", generation_cmd, [reframe, evidence]),
        ("mode_contrast", mode_cmd, [contrast]),
        ("reasoning_portfolio", reasoning_cmd, [reasoning]),
        ("pre_n10_cross_lane", cross_lane_cmd, [cross_lane]),
        ("pre_n10_candidate_contract", candidate_cmd, [candidates]),
    ]
    if not args.candidate_contract_only:
        commands.extend(
            [
                ("pre_n10_cross_lane_synthesis", synthesis_cmd, [synthesis]),
                (
                    "synthesis_n10_projection",
                    projection_cmd,
                    [projected, projected_lineage],
                ),
            ]
        )
    manifest["planned_stages"] = [
        {"name": name, "argv": argv}
        for name, argv, _ in commands
    ]
    _write(manifest_path, manifest)

    print("Pre-N10 scientific synthesis companion")
    print("Alpha6 hypotheses:", len(alpha6_portfolio.hypotheses))
    print("Legacy final survivor count is a precondition: false")
    print("Max possible LLM calls before abstention: 3")
    print("Candidate-contract-only:", args.candidate_contract_only)
    print("N10 authority created: false")
    print("Production selection changed: false")

    if args.dry_run:
        for name, argv, _ in commands:
            print(f"  {name}:")
            print("    $", sys.executable, *argv)
        print("Execution performed: false")
        print("Manifest:", manifest_path)
        return 0

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    try:
        _run("operator_readiness", readiness_cmd, [readiness])
        _run("evidence_materialization", evidence_cmd, [evidence])
        _run("trigger_detection", trigger_cmd, [triggers, tensions])
        _run("reframing_generation", generation_cmd, [reframe, evidence])

        reframe_payload = _load(reframe)
        rows = reframe_payload.get("candidates", [])
        reframe_count = len(rows) if isinstance(rows, list) else 0
        manifest["reframing_candidate_count"] = reframe_count
        if reframe_count == 0:
            manifest["status"] = "abstained_no_reframing_candidates"
            manifest["finished_at_utc"] = _now()
            _write(manifest_path, manifest)
            print("Disposition: abstained_no_reframing_candidates")
            return 0

        _run("mode_contrast", mode_cmd, [contrast])
        _run("reasoning_portfolio", reasoning_cmd, [reasoning])
        _run("pre_n10_cross_lane", cross_lane_cmd, [cross_lane])
        _run("pre_n10_candidate_contract", candidate_cmd, [candidates])
        if args.candidate_contract_only:
            candidate_payload = _load(candidates)
            candidate_rows = candidate_payload.get("candidates", [])
            manifest["candidate_count"] = (
                len(candidate_rows) if isinstance(candidate_rows, list) else 0
            )
            manifest["candidate_portfolio"] = str(candidates)
            manifest["status"] = "complete_candidate_contract_only"
            manifest["finished_at_utc"] = _now()
            manifest["execution_performed"] = True
            _write(manifest_path, manifest)
            print()
            print("Pre-N10 candidate contract complete")
            print("Candidates:", manifest["candidate_count"])
            print("N10 authority created: false")
            print("Production selection changed: false")
            print("Portfolio:", candidates)
            print("Manifest:", manifest_path)
            return 0
        _run("pre_n10_cross_lane_synthesis", synthesis_cmd, [synthesis])
        _run(
            "synthesis_n10_projection",
            projection_cmd,
            [projected, projected_lineage],
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _now()
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write(manifest_path, manifest)
        raise

    projected_portfolio = HypothesisPortfolio.model_validate_json(
        projected.read_text(encoding="utf-8")
    )
    manifest["projected_synthesis_hypothesis_count"] = len(
        projected_portfolio.hypotheses
    )
    manifest["status"] = (
        "complete"
        if projected_portfolio.hypotheses
        else "abstained_no_cross_lane_syntheses"
    )
    manifest["finished_at_utc"] = _now()
    manifest["execution_performed"] = True
    manifest["projected_synthesis_portfolio"] = str(projected)
    manifest["projected_synthesis_lineage"] = str(projected_lineage)
    _write(manifest_path, manifest)

    print()
    print("Pre-N10 scientific synthesis companion complete")
    print("Projected synthesis hypotheses:", len(projected_portfolio.hypotheses))
    print("N10 authority created: false")
    print("Production selection changed: false")
    print("Portfolio:", projected)
    print("Lineage:", projected_lineage)
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
