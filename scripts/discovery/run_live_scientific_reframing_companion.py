from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_core.discovery.reframing.authoritative_final_binding import (
    resolve_authoritative_final_portfolio,
)
from pipeline_core.discovery.reframing.e2e_shadow_companion import (
    require_completed_e2e_run,
)
from pipeline_core.discovery.reframing.live_reframing_companion import (
    common_scope_args,
    resolve_live_reframing_paths,
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


def _run(name: str, argv: list[str], *, expected: list[Path] = ()) -> None:
    print()
    print(f"== {name} ==")
    subprocess.run([sys.executable, *argv], check=True)
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"{name} completed without expected outputs: {missing}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize live LATENT_VARIABLE/REGIME_BOUNDARY scientific reframing "
            "from a completed legacy E2E run, then attach an authoritative-final "
            "integrated shadow. Sparse operator occupancy and explicit abstention are valid."
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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_dir, e2e_manifest = require_completed_e2e_run(args.run_dir)
    paths = resolve_live_reframing_paths(run_dir)

    required_base = [paths.packet, paths.explorer_report, paths.context]
    missing_base = [str(path) for path in required_base if not path.is_file()]
    if missing_base:
        raise SystemExit(
            "completed E2E run is missing reframing source artifacts: "
            + ", ".join(missing_base)
        )

    scope = common_scope_args(
        roots=args.root,
        duplicate_paper_policy=args.duplicate_paper_policy,
        cross_root_duplicate_policy=args.cross_root_duplicate_policy,
    )

    manifest = {
        "schema_version": "scientific-live-reframing-companion-manifest-v1",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "status": "planned" if args.dry_run else "running",
        "run_dir": str(run_dir),
        "legacy_e2e_status": e2e_manifest.get("status"),
        "semantic_roots": [
            str(Path(root).expanduser().resolve())
            for root in args.root
        ],
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "legacy_e2e_manifest_mutated": False,
        "execution_performed": False,
        "failure": None,
    }
    _write(paths.live_manifest, manifest)

    if e2e_manifest.get("status") == "complete_no_hypotheses_after_alpha4":
        manifest["status"] = "abstained_no_authoritative_hypotheses"
        manifest["finished_at_utc"] = _now()
        manifest["authoritative_hypothesis_count"] = 0
        _write(paths.live_manifest, manifest)
        print("Live scientific reframing companion abstained: no authoritative hypotheses.")
        print("Planned/actual companion LLM calls: 0")
        print("Production selection changed: false")
        return 0

    authority = resolve_authoritative_final_portfolio(run_dir=run_dir)
    manifest["authoritative_final"] = authority.model_dump(mode="json")
    manifest["authoritative_hypothesis_count"] = authority.hypothesis_count
    if authority.hypothesis_count == 0:
        manifest["status"] = "abstained_no_authoritative_hypotheses"
        manifest["finished_at_utc"] = _now()
        manifest["execution_performed"] = False
        _write(paths.live_manifest, manifest)
        print("Live scientific reframing companion")
        print("Authority kind:", authority.authority_kind)
        print("Authoritative hypotheses: 0")
        print("Disposition: abstained_no_authoritative_hypotheses")
        print("Planned/actual companion LLM calls: 0")
        print("Production selection changed: false")
        print("Canonical graph mutated: false")
        print("Legacy E2E manifest mutated: false")
        print("Manifest:", paths.live_manifest)
        return 0

    readiness_cmd = [
        "-m",
        "scripts.discovery.inspect_reframing_operator_readiness",
        *scope,
        "--packet",
        str(paths.packet),
        "--context",
        str(paths.context),
        "--output",
        str(paths.readiness),
    ]
    evidence_cmd = [
        "-m",
        "scripts.discovery.run_scientific_reframing_shadow",
        *scope,
        "--packet",
        str(paths.packet),
        "--context",
        str(paths.context),
        "--readiness",
        str(paths.readiness),
        "--dry-run",
        "--evidence-output",
        str(paths.evidence),
        "--output",
        str(paths.reframe_shadow),
    ]
    trigger_cmd = [
        "-m",
        "scripts.discovery.inspect_scientific_reframe_triggers",
        "--report",
        str(paths.explorer_report),
        "--evidence",
        str(paths.evidence),
        "--packet",
        str(paths.packet),
        "--output",
        str(paths.triggers),
        "--tension-output",
        str(paths.tensions),
    ]
    generation_cmd = [
        "-m",
        "scripts.discovery.run_scientific_reframing_shadow",
        *scope,
        "--packet",
        str(paths.packet),
        "--context",
        str(paths.context),
        "--readiness",
        str(paths.readiness),
        "--trigger",
        str(paths.triggers),
        "--evidence-output",
        str(paths.evidence),
        "--output",
        str(paths.reframe_shadow),
    ]
    if args.model:
        generation_cmd += ["--model", args.model]
    if args.base_url:
        generation_cmd += ["--base-url", args.base_url]
    if args.api_key_env:
        generation_cmd += ["--api-key-env", args.api_key_env]
    if args.save_prompts:
        generation_cmd += ["--save-prompts"]
    if args.telemetry:
        generation_cmd += ["--telemetry", args.telemetry]

    mode_cmd = [
        "-m",
        "scripts.discovery.compare_scientific_reframing_modes",
        "--reframe-shadow",
        str(paths.reframe_shadow),
        "--output",
        str(paths.mode_contrast),
    ]
    portfolio_cmd = [
        "-m",
        "scripts.discovery.build_scientific_reasoning_portfolio",
        "--mode-contrast",
        str(paths.mode_contrast),
        "--output",
        str(paths.reasoning_portfolio),
    ]
    attach_cmd = [
        "-m",
        "scripts.discovery.attach_scientific_shadow_companion",
        "--run-dir",
        str(run_dir),
        "--reframing-portfolio",
        str(paths.reasoning_portfolio),
        "--reframe-shadow",
        str(paths.reframe_shadow),
        "--max-syntheses",
        str(args.max_syntheses),
    ]
    if args.model:
        attach_cmd += ["--model", args.model]
    if args.base_url:
        attach_cmd += ["--base-url", args.base_url]
    if args.api_key_env:
        attach_cmd += ["--api-key-env", args.api_key_env]

    commands = [
        ("operator_readiness", readiness_cmd),
        ("evidence_materialization", evidence_cmd),
        ("trigger_detection", trigger_cmd),
        ("reframing_generation", generation_cmd),
        ("mode_contrast", mode_cmd),
        ("reasoning_portfolio", portfolio_cmd),
        ("authoritative_shadow_attach", attach_cmd),
    ]
    manifest["planned_stages"] = [
        {"name": name, "argv": argv}
        for name, argv in commands
    ]
    _write(paths.live_manifest, manifest)

    print("Live scientific reframing companion")
    print("Legacy E2E status:", e2e_manifest.get("status"))
    print("Semantic roots:", len(args.root))
    print("Max possible LLM calls before abstention: 3")
    print("Production selection changed: false")
    print("Canonical graph mutated: false")
    print("Legacy E2E manifest mutated: false")

    if args.dry_run:
        for name, argv in commands:
            print(f"  {name}:")
            print("    $", sys.executable, *argv)
        print("Execution performed: false")
        print("Manifest:", paths.live_manifest)
        return 0

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    try:
        _run("operator_readiness", readiness_cmd, expected=[paths.readiness])
        _run("evidence_materialization", evidence_cmd, expected=[paths.evidence])
        _run(
            "trigger_detection",
            trigger_cmd,
            expected=[paths.triggers, paths.tensions],
        )
        _run(
            "reframing_generation",
            generation_cmd,
            expected=[paths.reframe_shadow, paths.evidence],
        )

        shadow_payload = _load(paths.reframe_shadow)
        candidates = shadow_payload.get("candidates", [])
        candidate_count = len(candidates) if isinstance(candidates, list) else 0
        manifest["reframing_candidate_count"] = candidate_count
        if candidate_count == 0:
            manifest["status"] = "abstained_no_reframing_candidates"
            manifest["finished_at_utc"] = _now()
            manifest["execution_performed"] = True
            _write(paths.live_manifest, manifest)
            print()
            print("No live reframing candidates were justified.")
            print("Disposition: abstained_no_reframing_candidates")
            print("Production selection changed: false")
            print("Manifest:", paths.live_manifest)
            return 0

        _run("mode_contrast", mode_cmd, expected=[paths.mode_contrast])
        _run(
            "reasoning_portfolio",
            portfolio_cmd,
            expected=[paths.reasoning_portfolio],
        )
        integrated_output = (
            run_dir / "scientific_authoritative_integrated_hypothesis_shadow.json"
        )
        _run(
            "authoritative_shadow_attach",
            attach_cmd,
            expected=[integrated_output],
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _now()
        manifest["execution_performed"] = True
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write(paths.live_manifest, manifest)
        raise

    manifest["status"] = "complete"
    manifest["finished_at_utc"] = _now()
    manifest["execution_performed"] = True
    manifest["integrated_shadow_output"] = str(
        run_dir / "scientific_authoritative_integrated_hypothesis_shadow.json"
    )
    _write(paths.live_manifest, manifest)

    print()
    print("Live scientific reframing companion complete")
    print("Reframing candidates:", manifest["reframing_candidate_count"])
    print("Integrated shadow:", manifest["integrated_shadow_output"])
    print("Production selection changed: false")
    print("Canonical graph mutated: false")
    print("Legacy E2E manifest mutated: false")
    print("Manifest:", paths.live_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
