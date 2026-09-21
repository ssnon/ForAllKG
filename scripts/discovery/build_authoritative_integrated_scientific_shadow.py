from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.reframing.authoritative_final_binding import (
    build_authoritative_shadow_commands,
    resolve_authoritative_final_portfolio,
    validate_authoritative_final_context,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebind the scientific-reframing integration lane to the actual authoritative "
            "Alpha6/post-N10 E2E hypothesis portfolio, then materialize a shadow-only "
            "cross-lane synthesis and integrated hypothesis view. The pre-refinement "
            "hypothesis_axis_a4 portfolio is never accepted as an automatic final fallback."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--authoritative-portfolio", type=Path, default=None)
    parser.add_argument("--context", type=Path, default=None)
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
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir
    )
    context_path = (
        args.context.expanduser().resolve()
        if args.context is not None
        else run_dir / "hypothesis.context.json"
    )
    if not context_path.is_file():
        raise SystemExit(f"missing hypothesis context: {context_path}")

    resolution = resolve_authoritative_final_portfolio(
        run_dir=run_dir,
        explicit_portfolio=(
            args.authoritative_portfolio.expanduser().resolve()
            if args.authoritative_portfolio is not None
            else None
        ),
    )
    context = HypothesisContext.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )
    authoritative = validate_authoritative_final_context(
        resolution=resolution,
        context=context,
    )
    if not authoritative.hypotheses:
        raise SystemExit(
            "authoritative final portfolio is empty; no integrated hypothesis shadow can be built"
        )

    commands = build_authoritative_shadow_commands(
        context_path=context_path,
        authoritative_portfolio_path=Path(resolution.portfolio_path),
        reframing_portfolio_path=args.reframing_portfolio.expanduser().resolve(),
        reframe_shadow_path=args.reframe_shadow.expanduser().resolve(),
        proxy_shadow_path=(
            args.proxy_shadow.expanduser().resolve()
            if args.proxy_shadow is not None
            else None
        ),
        contradiction_shadow_path=(
            args.contradiction_shadow.expanduser().resolve()
            if args.contradiction_shadow is not None
            else None
        ),
        output_dir=output_dir,
        max_syntheses=args.max_syntheses,
        model=args.model or None,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )

    manifest_path = output_dir / "scientific_authoritative_integrated_shadow_manifest.json"
    manifest = {
        "schema_version": "scientific-authoritative-integrated-shadow-manifest-v1",
        "run_dir": str(run_dir),
        "context_path": str(context_path),
        "authoritative_final_portfolio": resolution.model_dump(mode="json"),
        "reframing_portfolio_path": str(args.reframing_portfolio.expanduser().resolve()),
        "reframe_shadow_path": str(args.reframe_shadow.expanduser().resolve()),
        "proxy_shadow_path": (
            str(args.proxy_shadow.expanduser().resolve())
            if args.proxy_shadow is not None
            else None
        ),
        "contradiction_shadow_path": (
            str(args.contradiction_shadow.expanduser().resolve())
            if args.contradiction_shadow is not None
            else None
        ),
        "max_syntheses": args.max_syntheses,
        "planned_stages": [
            {"stage": name, "expected_output": str(expected)}
            for name, _, expected in commands
        ],
        "execution_performed": False,
        "llm_calls_planned": 1,
        "pre_refinement_alpha4_used_as_authoritative_final": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    _write(manifest_path, manifest)

    print("Authoritative final scientific integration shadow")
    print("Run:", run_dir)
    print("Authority kind:", resolution.authority_kind)
    print("Authoritative portfolio:", resolution.portfolio_path)
    print("Authoritative hypotheses:", resolution.hypothesis_count)
    print("Pre-refinement alpha4 used as final: false")
    print("Planned LLM calls:", 1)
    for name, command, expected in commands:
        print(f"  {name}: {expected.name}")
        if args.dry_run:
            print("    $", sys.executable, *command)

    if args.dry_run:
        print("Execution performed: false")
        print("Production selection changed: false")
        print("Manifest:", manifest_path)
        return 0

    for name, command, expected in commands:
        print()
        print(f"== {name} ==")
        subprocess.run([sys.executable, *command], check=True)
        if not expected.is_file():
            raise RuntimeError(
                f"stage {name} completed without expected output: {expected}"
            )

    manifest["execution_performed"] = True
    manifest["completed_outputs"] = [
        str(expected) for _, _, expected in commands
    ]
    _write(manifest_path, manifest)

    print()
    print("Authoritative final scientific integration shadow complete")
    print("Pre-refinement alpha4 used as final: false")
    print("Production selection changed: false")
    print("Canonical graph mutated: false")
    print("Integrated shadow:", commands[-1][2])
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
