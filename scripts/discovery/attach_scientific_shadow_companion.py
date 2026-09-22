from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_core.discovery.reframing.e2e_shadow_companion import (
    build_attach_command,
    build_scientific_shadow_companion_plan,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attach an opt-in scientific-reframing integrated shadow to a completed "
            "legacy E2E run. The legacy E2E manifest, production selection, and canonical "
            "graph are never modified."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--reframing-portfolio", required=True, type=Path)
    parser.add_argument("--reframe-shadow", required=True, type=Path)
    parser.add_argument("--proxy-shadow", type=Path, default=None)
    parser.add_argument("--contradiction-shadow", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = build_scientific_shadow_companion_plan(
        run_dir=args.run_dir,
        reframing_portfolio_path=args.reframing_portfolio,
        reframe_shadow_path=args.reframe_shadow,
        proxy_shadow_path=args.proxy_shadow,
        contradiction_shadow_path=args.contradiction_shadow,
        output_dir=args.output_dir,
        max_syntheses=args.max_syntheses,
    )

    manifest_path = Path(plan.companion_manifest_output)
    manifest = {
        "schema_version": "scientific-e2e-shadow-companion-manifest-v1",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "status": "planned" if args.dry_run else "running",
        "plan": plan.model_dump(mode="json"),
        "execution_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "legacy_e2e_manifest_mutated": False,
        "failure": None,
    }
    _write(manifest_path, manifest)

    print("Scientific E2E shadow companion")
    print("Legacy E2E status:", plan.e2e_status)
    print("Disposition:", plan.disposition)
    if plan.authoritative_final is not None:
        print("Authority kind:", plan.authoritative_final.authority_kind)
        print("Authoritative hypotheses:", plan.authoritative_final.hypothesis_count)
    print("Legacy E2E manifest mutated: false")
    print("Production selection changed: false")
    print("Canonical graph mutated: false")

    if plan.disposition != "ready":
        manifest["status"] = plan.disposition
        manifest["finished_at_utc"] = _now()
        _write(manifest_path, manifest)
        print("Execution performed: false")
        print("Companion manifest:", manifest_path)
        return 0

    command = build_attach_command(
        plan,
        model=args.model or None,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    if args.dry_run:
        print("$", sys.executable, *command)
        print("Execution performed: false")
        print("Companion manifest:", manifest_path)
        return 0

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    try:
        subprocess.run([sys.executable, *command], check=True)
        integrated = Path(plan.integrated_shadow_output)
        if not integrated.is_file():
            raise RuntimeError(
                "authoritative scientific integration completed without expected output: "
                f"{integrated}"
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

    manifest["status"] = "complete"
    manifest["finished_at_utc"] = _now()
    manifest["execution_performed"] = True
    manifest["integrated_shadow_output"] = plan.integrated_shadow_output
    _write(manifest_path, manifest)

    print("Scientific E2E shadow companion complete")
    print("Integrated shadow:", plan.integrated_shadow_output)
    print("Legacy E2E manifest mutated: false")
    print("Production selection changed: false")
    print("Canonical graph mutated: false")
    print("Companion manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
