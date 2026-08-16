from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.alpha4c5i_dev_input_builder import (
    DEFAULT_GROUNDING,
    DEFAULT_OUTPUT_ROOT,
    build_dev_trend_input,
    build_manifest,
    path_is_closed_reserve,
    sha256_file,
)


ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the missing alpha4c.5i DEV-only 5b "
            "TrendAwareHypothesisInput from frozen DEV grounding and an "
            "existing valid DEV HypothesisContext, or deterministically "
            "derive the context from an existing valid DEV "
            "GraphExplorerPacket + ExplorationReport. No LLM calls."
        )
    )
    parser.add_argument(
        "--grounding",
        type=Path,
        default=DEFAULT_GROUNDING,
    )
    parser.add_argument("--context", type=Path, default=None)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--build", action="store_true")

    parser.add_argument(
        "--confirm-development-only",
        action="store_true",
    )
    return parser.parse_args()


def _resolve(value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else ROOT / value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()

    grounding = _resolve(args.grounding)
    context = _resolve(args.context)
    packet = _resolve(args.packet)
    report = _resolve(args.report)
    output_root = _resolve(args.output_root)
    assert grounding is not None
    assert output_root is not None

    for name, path in (
        ("grounding", grounding),
        ("context", context),
        ("packet", packet),
        ("report", report),
        ("output-root", output_root),
    ):
        if path is not None and path_is_closed_reserve(path):
            raise SystemExit(
                f"Refusing {name} under a closed Reserve A/B path: {path}"
            )

    try:
        value, context_candidate, _ = build_dev_trend_input(
            root=ROOT,
            grounding_path=grounding,
            explicit_context=context,
            explicit_packet=packet,
            explicit_report=report,
        )
    except Exception as exc:
        print("alpha4c.5i DEV 5b input readiness: FAIL")
        print("Reason:", exc)
        print("Reserve A used: False")
        print("Reserve B used: False")
        print("Reserve B rerun: False")
        print("LLM calls: 0")
        return 2

    print("alpha4c.5i DEV 5b input readiness")
    print("Input ID:", value.input_id)
    print("Input SHA256:", value.input_sha256)
    print("DEV papers:", len(value.trend_corpus_binding.paper_ids))
    print("Trend views:", len(value.trend_views))
    print("Lane counts:", json.dumps(value.lane_counts, sort_keys=True))
    print("Context ID:", value.grounded_context.context_id)
    print("Context source:", context_candidate.source_kind)
    print("Reserve A used: False")
    print("Reserve B used: False")
    print("Reserve B rerun: False")
    print("LLM calls: 0")

    if args.preflight:
        print("Preflight: PASS")
        print("Write performed: False")
        return 0

    if not args.confirm_development_only:
        raise SystemExit(
            "--confirm-development-only is required for --build."
        )

    input_path = output_root / "trend_aware_hypothesis_input.json"
    context_path = output_root / "hypothesis_context.json"
    manifest_path = output_root / "build_manifest.json"

    existing = [
        path
        for path in (input_path, context_path, manifest_path)
        if path.exists()
    ]
    if existing:
        raise SystemExit(
            "Refusing overwrite of existing DEV 5b outputs:\n- "
            + "\n- ".join(str(path) for path in existing)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        context_path,
        value.grounded_context.model_dump_json(indent=2) + "\n",
    )
    _atomic_text(
        input_path,
        value.model_dump_json(indent=2) + "\n",
    )
    manifest = build_manifest(
        root=ROOT,
        value=value,
        context_candidate=context_candidate,
        grounding_path=grounding,
    )
    manifest["output_context_path"] = str(
        context_path.relative_to(ROOT)
    )
    manifest["output_context_file_sha256"] = sha256_file(context_path)
    manifest["output_input_path"] = str(input_path.relative_to(ROOT))
    manifest["output_input_file_sha256"] = sha256_file(input_path)
    _atomic_text(
        manifest_path,
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    print("Build: PASS")
    print("Saved context:", context_path)
    print("Saved input:", input_path)
    print("Saved manifest:", manifest_path)
    print("Scientific semantics modified: False")
    print("New scientific extraction: False")
    print("LLM calls: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
