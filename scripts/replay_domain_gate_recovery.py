from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from dac_her.domain_gate_replay import (
    DomainGateReplayFixture,
    build_zero_loss_summary,
    evaluate_replay_draft,
    verify_fixture_contract,
)
from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.llm_openrouter import OpenRouterLLM
from dac_her.vocab_registry import load_default_registries


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen domain-gate recovery inputs against full and compact "
            "response schemas. Evaluation-only; production policy is untouched."
        )
    )
    parser.add_argument(
        "--fixture",
        action="append",
        required=True,
        help="Captured *.fixture.json path. Repeat for multiple fixtures.",
    )
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument(
        "--output-dir",
        default="data_broad/replay/domain_gate_recovery",
    )
    parser.add_argument(
        "--allow-contract-drift",
        action="store_true",
        help=(
            "Allow adapter/prompt/schema fingerprints to differ from capture. "
            "Not recommended for causal A/B."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing replay output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Replay output already exists: {output_dir}. "
                "Use a new --output-dir or --force."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    telemetry_path = output_dir / "telemetry.jsonl"
    results_path = output_dir / "results.jsonl"
    experiment_registry, metric_registry = load_default_registries(
        PROJECT_ROOT
    )

    fixtures: list[DomainGateReplayFixture] = []
    for raw_path in args.fixture:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        fixture = DomainGateReplayFixture.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        adapter = get_extraction_adapter(fixture.domain_profile_id)
        problems = verify_fixture_contract(fixture, adapter)
        if problems and not args.allow_contract_drift:
            raise RuntimeError(
                f"Fixture contract drift for {path}:\n- "
                + "\n- ".join(problems)
            )
        fixtures.append(fixture)

    all_rows: list[dict[str, Any]] = []

    for fixture in fixtures:
        adapter = get_extraction_adapter(fixture.domain_profile_id)
        model = args.model or fixture.captured_model
        provider = (
            args.provider
            if args.provider is not None
            else fixture.captured_provider
        )

        for replicate in range(1, args.replicates + 1):
            conditions = (
                ("full", "compact")
                if replicate % 2 == 1
                else ("compact", "full")
            )
            for order_index, condition in enumerate(conditions, 1):
                compact = condition == "compact"
                response_model = (
                    adapter.domain_gate_recovery_response_model(
                        compact=compact
                    )
                )
                run_id = (
                    f"{fixture.fixture_id}:{condition}:r{replicate}"
                )
                debug_path = (
                    output_dir
                    / "raw_invalid"
                    / fixture.fixture_id.replace(":", "__")
                    / condition
                    / f"r{replicate}.json"
                )
                llm = OpenRouterLLM(
                    model=model,
                    provider=provider,
                    reproducible=False,
                    zdr=True,
                    telemetry_path=telemetry_path,
                    telemetry_context={
                        "run_id": run_id,
                        "pipeline": "domain_gate_replay",
                        "stage": "domain_gate_recovery_replay",
                        "call_kind": "domain_gate_replay",
                        "paper_id": fixture.paper_id,
                        "chunk_id": fixture.chunk_id,
                        "attempt": replicate,
                    },
                )

                row: dict[str, Any] = {
                    "fixture_id": fixture.fixture_id,
                    "paper_id": fixture.paper_id,
                    "chunk_id": fixture.chunk_id,
                    "condition": condition,
                    "replicate": replicate,
                    "order_index": order_index,
                    "response_model": response_model.__name__,
                    "schema_estimated_tokens": (
                        fixture.compact_schema_estimated_tokens
                        if compact
                        else fixture.full_schema_estimated_tokens
                    ),
                    "llm_success": False,
                    "domain_gate_pass": False,
                    "strict_valid": False,
                    "finalization_success": False,
                    "mechanism_connected": False,
                    "issue_counts": {},
                    "finalization_issue_counts": {},
                    "measurement_issue_count": 0,
                    "mechanism_claim_count": 0,
                    "mechanism_incident_edge_count": 0,
                    "node_count": 0,
                    "edge_count": 0,
                    "finalized_node_count": 0,
                    "finalized_edge_count": 0,
                }

                try:
                    generated = llm.generate_structured(
                        system_prompt=fixture.system_prompt,
                        prompt=fixture.user_prompt,
                        response_model=response_model,
                        temperature=fixture.temperature,
                        max_tokens=fixture.max_completion_tokens,
                        debug_path=debug_path,
                    )
                    row["llm_success"] = True
                    evaluated = evaluate_replay_draft(
                        generated=generated,
                        fixture=fixture,
                        extraction_adapter=adapter,
                        experiment_registry=experiment_registry,
                        metric_registry=metric_registry,
                    )
                    canonical = evaluated.pop("canonical_draft")
                    row.update(evaluated)

                    canonical_path = (
                        output_dir
                        / "canonical_outputs"
                        / fixture.fixture_id.replace(":", "__")
                        / condition
                        / f"r{replicate}.json"
                    )
                    _write_json(
                        canonical_path,
                        canonical.model_dump(mode="json"),
                    )
                    row["canonical_output_path"] = str(canonical_path)
                except Exception as error:
                    row["error_type"] = type(error).__name__
                    row["error_message"] = str(error)

                usage = llm.last_usage
                if usage is not None:
                    row["provider_input_tokens"] = usage.input_tokens
                    row["provider_output_tokens"] = usage.output_tokens
                    row["provider_total_tokens"] = usage.total_tokens

                _append_jsonl(results_path, row)
                all_rows.append(row)

                print(
                    fixture.paper_id,
                    condition,
                    f"r{replicate}",
                    f"input={row.get('provider_input_tokens')}",
                    f"final={row.get('finalization_success')}",
                    f"mechanism={row.get('mechanism_connected')}",
                    flush=True,
                )

    summary = build_zero_loss_summary(all_rows)
    _write_json(output_dir / "summary.json", summary)

    print()
    print("Replay complete")
    print("Rows:", len(all_rows))
    print("Summary:", output_dir / "summary.json")
    print("Verdict:", summary["verdict"])
    if summary["hard_failures"]:
        print("Hard quality-loss signals:")
        for item in summary["hard_failures"]:
            print(" -", item)
    if summary["manual_holds"]:
        print("Manual-review holds:")
        for item in summary["manual_holds"]:
            print(" -", item)


if __name__ == "__main__":
    main()
