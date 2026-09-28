from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.direct_higher_order_hypothesis_runtime import (
    DirectHigherOrderShadowHypothesisRuntime,
    authorize_direct_higher_order_shadow_generation,
    materialize_direct_higher_order_hypothesis_context,
)
from pipeline_core.discovery.direct_higher_order_synthesis_context import (
    DirectHigherOrderSynthesisContext,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)


def _json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _portfolio_payload(outcome):
    portfolio = outcome.canonical_outcome.accepted_portfolio
    if portfolio is None:
        return None
    return portfolio.model_dump(mode="json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--synthesis-context-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "OPENAI_BASE_URL"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    source_context = HypothesisContext.model_validate(
        _json(args.source_context)
    )

    context_payload = _json(
        args.synthesis_context_report
    )

    contexts = [
        DirectHigherOrderSynthesisContext.model_validate(
            row
        )
        for row in context_payload.get(
            "contexts",
            [],
        )
    ][: args.max_contexts]

    if not contexts:
        raise RuntimeError(
            "no direct higher-order synthesis contexts"
        )

    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=0.0,
        parse_retries=args.parse_retries,
        telemetry_path=args.output.with_suffix(
            ".telemetry.jsonl"
        ),
        telemetry_context={
            "pipeline":
                "direct_higher_order_shadow",
            "stage":
                "generation",
        },
    )

    runtime = DirectHigherOrderShadowHypothesisRuntime(
        backend,
        max_repairs=args.max_repairs,
    )

    arms = []

    for index, direct_context in enumerate(
        contexts,
        start=1,
    ):
        projection = (
            materialize_direct_higher_order_hypothesis_context(
                source_context=source_context,
                direct_context=direct_context,
            )
        )

        authorization = (
            authorize_direct_higher_order_shadow_generation(
                projection=projection,
                direct_context=direct_context,
            )
        )

        outcome = runtime.run(
            projection=projection,
            direct_context=direct_context,
            authorization=authorization,
        )

        canonical = outcome.canonical_outcome

        arms.append(
            {
                "arm_index":
                    index,
                "direct_context_id":
                    direct_context.context_id,
                "direct_topology_id":
                    direct_context.direct_higher_order_topology_id,
                "modifier_component_id":
                    direct_context.modifier_component_id,
                "modifier_text":
                    direct_context.structural_opportunity.modifier_text,
                "status":
                    outcome.status,
                "shadow_contract_passed":
                    outcome.shadow_contract_passed,
                "shadow_failure_code":
                    outcome.shadow_failure_code,
                "shadow_failure_message":
                    outcome.shadow_failure_message,
                "materialization":
                    projection.materialization.model_dump(
                        mode="json"
                    ),
                "authorization":
                    authorization.model_dump(
                        mode="json"
                    ),
                "canonical_run_record":
                    canonical.run_record.model_dump(
                        mode="json"
                    ),
                "final_draft":
                    (
                        canonical.final_draft.model_dump(
                            mode="json"
                        )
                        if canonical.final_draft
                        is not None
                        else None
                    ),
                "accepted_portfolio":
                    _portfolio_payload(
                        outcome
                    ),
            }
        )

    report = {
        "schema_version":
            "direct-higher-order-shadow-generation-report-v1",
        "source_context":
            str(args.source_context),
        "synthesis_context_report":
            str(
                args.synthesis_context_report
            ),
        "model":
            args.model,
        "input_context_count":
            len(contexts),
        "proposed_count":
            sum(
                row["status"] == "proposed"
                for row in arms
            ),
        "abstained_count":
            sum(
                row["status"] == "abstained"
                for row in arms
            ),
        "canonical_rejected_count":
            sum(
                row["status"]
                == "canonical_rejected"
                for row in arms
            ),
        "shadow_contract_rejected_count":
            sum(
                row["status"]
                == "shadow_contract_rejected"
                for row in arms
            ),
        "arms":
            arms,
        "authority": {
            "shadow_only":
                True,
            "canonical_hypothesis_runtime_reused":
                True,
            "restricted_composition_relations_as_positive_premise":
                False,
            "novelty_authority_created":
                False,
            "external_novelty_review_performed":
                False,
            "n10_review_performed":
                False,
            "production_selection_changed":
                False,
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=== S204 DIRECT HIGHER-ORDER HYPOTHESIS SHADOW ==="
    )
    print(
        "contexts:",
        len(contexts),
    )
    print(
        "proposed:",
        report[
            "proposed_count"
        ],
    )
    print(
        "abstained:",
        report[
            "abstained_count"
        ],
    )
    print(
        "canonical rejected:",
        report[
            "canonical_rejected_count"
        ],
    )
    print(
        "shadow rejected:",
        report[
            "shadow_contract_rejected_count"
        ],
    )

    for arm in arms:
        print(
            "\narm",
            arm[
                "arm_index"
            ],
            "modifier=",
            arm[
                "modifier_text"
            ],
            "status=",
            arm[
                "status"
            ],
        )

        portfolio = arm[
            "accepted_portfolio"
        ]

        if portfolio is not None:
            for card in portfolio.get(
                "hypotheses",
                [],
            ):
                print(
                    " hypothesis:",
                    card[
                        "hypothesis_statement"
                    ],
                )
                print(
                    " bridge:",
                    card[
                        "inferential_bridge"
                    ],
                )

    print(
        "\nNo novelty or production authority created."
    )
    print(
        "artifact:",
        args.output,
    )
    print(
        "S204_SHADOW_COMPLETE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
