from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisPlan,
)
from pipeline_core.discovery.dual_hypothesis_context import (
    DualHypothesisContext,
)
from pipeline_core.discovery.node_mapping import NodeMapper
from pipeline_core.discovery.open_world_discovery_axis import (
    build_external_axis_prompt_payload,
)
from pipeline_core.discovery.open_world_discovery_axis_runtime import (
    InstructorOpenAICompatibleExternalAxisBackend,
    OpenWorldDiscoveryAxisOutcome,
    OpenWorldDiscoveryAxisRuntime,
    build_external_axis_messages,
)
from pipeline_core.discovery.prior_art_provider_plan import (
    LiteratureProviderPlan,
    build_literature_providers,
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Header must be KEY=VALUE"
        )
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            "Header key may not be empty"
        )
    return key, item


def _provider_request(
    raw: str,
) -> list[str] | None:
    text = str(raw or "").strip()
    if not text or text.lower() == "auto":
        return None
    values = [
        item.strip().lower()
        for item in text.split(",")
        if item.strip()
    ]
    if not values:
        raise argparse.ArgumentTypeError(
            "--providers must be 'auto' or a comma-separated provider list"
        )
    return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if hasattr(value, "model_dump"):
        value = value.model_dump(
            mode="json"
        )
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_control_plan(
    path: Path,
    dual: DualHypothesisContext,
) -> DiscoveryAxisPlan:
    plan = DiscoveryAxisPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if (
        plan.source_dual_context_id
        != dual.dual_context_id
    ):
        raise ValueError(
            "control axis plan dual_context_id mismatch"
        )
    if (
        plan.source_dual_context_sha256
        != dual.dual_context_sha256
    ):
        raise ValueError(
            "control axis plan dual_context_sha256 mismatch"
        )
    return plan


def _generation_payload(
    outcome: OpenWorldDiscoveryAxisOutcome,
) -> dict[str, Any]:
    generation = outcome.generation
    return {
        "draft":
            generation.draft.model_dump(mode="json"),
        "input_tokens":
            generation.input_tokens,
        "output_tokens":
            generation.output_tokens,
        "response_id":
            generation.response_id,
        "elapsed_seconds":
            generation.elapsed_seconds,
    }


def _stage_manifest(
    *,
    dual: DualHypothesisContext,
    control_plan: DiscoveryAxisPlan,
    provider_plan: LiteratureProviderPlan,
    outcome: OpenWorldDiscoveryAxisOutcome,
    index_dir: Path,
) -> dict[str, Any]:
    external_plan = outcome.axis_plan.plan
    return {
        "schema_version":
            "open-world-discovery-axis-stage-manifest-v1",
        "source_dual_context_id":
            dual.dual_context_id,
        "source_dual_context_sha256":
            dual.dual_context_sha256,
        "control_plan_id":
            control_plan.plan_id,
        "control_plan_sha256":
            control_plan.plan_sha256,
        "provider_plan_id":
            provider_plan.plan_id,
        "provider_plan_sha256":
            provider_plan.plan_sha256,
        "provider_mode":
            provider_plan.mode,
        "active_providers":
            list(provider_plan.active_providers),
        "index_dir":
            str(index_dir),
        "retrieval_complete":
            outcome.retrieval.complete,
        "retrieval_seed_count":
            len(outcome.retrieval.seeds),
        "provider_execution_count":
            len(outcome.retrieval.executions),
        "selected_abstract_count":
            len(outcome.selected_works),
        "synthesized_axis_draft_count":
            len(outcome.generation.draft.axes),
        "source_validated_axis_count":
            len(outcome.validation.accepted_axes),
        "source_rejected_axis_count":
            len(outcome.validation.rejected_axes),
        "external_axis_count":
            len(external_plan.axes),
        "control_duplicate_rejection_count":
            len(outcome.axis_plan.rejected_axes),
        "external_plan_id":
            external_plan.plan_id,
        "external_plan_sha256":
            external_plan.plan_sha256,
        "external_bundle_id":
            outcome.axis_plan.bundle.bundle_id,
        "external_bundle_sha256":
            outcome.axis_plan.bundle.bundle_sha256,
        "external_literature_authority":
            "INSPIRATION_ONLY",
        "positive_premise_authority_changed":
            False,
        "novelty_authority_created":
            False,
        "hypothesis_generation_performed":
            False,
        "canonical_fallback_authorized":
            False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded S17 open-world discovery-axis stage. "
            "This stage retrieves external literature, synthesizes "
            "source-bounded inspiration-only axes, and writes a "
            "DiscoveryAxisPlan for the ordinary hypothesis runtime. "
            "It does not generate hypotheses or judge novelty."
        )
    )
    parser.add_argument(
        "--dual-context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--control-axis-plan",
        required=True,
        type=Path,
        help=(
            "Existing persistent-KG/control DiscoveryAxisPlan "
            "for duplicate filtering and planner-policy reuse."
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=(
            "Mechanism node index used by NodeMapper for "
            "control-axis semantic duplicate checks. "
            "Defaults to data_dac/corpus/<corpus>/mechanism/"
            "navigation/node_index."
        ),
    )
    parser.add_argument(
        "--providers",
        default="auto",
        help=(
            "'auto' or comma-separated explicit providers. "
            "Uses the existing frozen literature-provider plan."
        ),
    )
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
        help=(
            "Structured LLM model for the single external-axis "
            "synthesis call."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--instructor-mode",
        default="JSON",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        type=_header,
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-selected-works",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-control-similarity",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--max-source-span-words",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help=(
            "Persist the exact external-axis synthesis prompt. "
            "The prompt contains supplied abstracts but no secret values."
        ),
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    dual = DualHypothesisContext.model_validate_json(
        args.dual_context.read_text(
            encoding="utf-8"
        )
    )
    control_plan = _load_control_plan(
        args.control_axis_plan,
        dual,
    )

    requested = _provider_request(
        args.providers
    )
    provider_plan = resolve_literature_provider_plan(
        requested=requested,
    )
    if requested is None:
        require_standard_or_full_auto_plan(
            provider_plan
        )
    providers = build_literature_providers(
        provider_plan
    )

    if not args.model:
        raise SystemExit(
            "--model is required unless "
            "GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set."
        )

    index_dir = args.index_dir or (
        PROJECT_ROOT
        / "data_dac"
        / "corpus"
        / dual.grounded_context.corpus_id
        / "mechanism"
        / "navigation"
        / "node_index"
    )
    mapper = NodeMapper.from_directory(
        index_dir,
        device=args.device,
    )

    backend = (
        InstructorOpenAICompatibleExternalAxisBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=0.0,
            parse_retries=0,
            timeout=args.timeout,
            extra_headers=dict(args.header),
        )
    )
    runtime = OpenWorldDiscoveryAxisRuntime(
        providers=providers,
        backend=backend,
        mapper=mapper,
        results_per_query=args.results_per_query,
        max_selected_works=
            args.max_selected_works,
        max_control_similarity=
            args.max_control_similarity,
        max_source_span_words=
            args.max_source_span_words,
    )

    outcome = runtime.run(
        dual=dual,
        control_plan=control_plan,
    )

    prefix = args.output_prefix
    provider_plan_path = Path(
        str(prefix)
        + ".provider_plan.json"
    )
    retrieval_path = Path(
        str(prefix)
        + ".retrieval.json"
    )
    selected_path = Path(
        str(prefix)
        + ".selected_works.json"
    )
    generation_path = Path(
        str(prefix)
        + ".axis_synthesis.json"
    )
    validation_path = Path(
        str(prefix)
        + ".axis_validation.json"
    )
    plan_path = Path(
        str(prefix)
        + ".external_axis_plan.json"
    )
    bundle_path = Path(
        str(prefix)
        + ".external_axis_bundle.json"
    )
    rejection_path = Path(
        str(prefix)
        + ".external_axis_rejections.json"
    )
    manifest_path = Path(
        str(prefix)
        + ".manifest.json"
    )

    _write_json(
        provider_plan_path,
        provider_plan,
    )
    _write_json(
        retrieval_path,
        outcome.retrieval,
    )
    _write_json(
        selected_path,
        [
            row.model_dump(mode="json")
            for row in outcome.selected_works
        ],
    )
    _write_json(
        generation_path,
        _generation_payload(outcome),
    )
    _write_json(
        validation_path,
        outcome.validation,
    )
    _write_json(
        plan_path,
        outcome.axis_plan.plan,
    )
    _write_json(
        bundle_path,
        outcome.axis_plan.bundle,
    )
    _write_json(
        rejection_path,
        [
            row.model_dump(mode="json")
            for row in outcome.axis_plan.rejected_axes
        ],
    )

    if args.save_prompt:
        prompt_payload = (
            build_external_axis_prompt_payload(
                dual=dual,
                control_plan=control_plan,
                selected_works=list(
                    outcome.selected_works
                ),
            )
        )
        _write_json(
            Path(
                str(prefix)
                + ".axis_synthesis_prompt.json"
            ),
            {
                "messages":
                    build_external_axis_messages(
                        prompt_payload
                    ),
            },
        )

    manifest = _stage_manifest(
        dual=dual,
        control_plan=control_plan,
        provider_plan=provider_plan,
        outcome=outcome,
        index_dir=index_dir,
    )
    _write_json(
        manifest_path,
        manifest,
    )

    print("Open-world discovery-axis stage complete")
    print(
        "Provider plan:",
        provider_plan.plan_id,
        provider_plan.mode,
    )
    print(
        "Retrieval executions:",
        len(outcome.retrieval.executions),
        "complete=",
        outcome.retrieval.complete,
    )
    print(
        "Selected abstract-backed works:",
        len(outcome.selected_works),
    )
    print(
        "Source-validated draft axes:",
        len(outcome.validation.accepted_axes),
    )
    print(
        "External axes after control filtering:",
        len(outcome.axis_plan.plan.axes),
    )
    print(
        "External plan:",
        plan_path,
    )
    print(
        "Manifest:",
        manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
