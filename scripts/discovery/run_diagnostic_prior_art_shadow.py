from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.diagnostic_prior_art_retrieval import (
    build_diagnostic_query_plan,
)
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    load_literature_provider_plan,
    require_standard_or_full_auto_plan,
)
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureRetriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded shadow-only diagnostic "
            "prior-art retrieval pass from canonical "
            "claim diagnostic metadata. This command "
            "does not modify the ordinary query plan, "
            "prior-art packet, external novelty report, "
            "N9 state, or N10 selection."
        )
    )

    parser.add_argument(
        "--query-plan",
        required=True,
    )

    parser.add_argument(
        "--provider-plan",
        required=True,
    )

    parser.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
    )

    return parser.parse_args()


def _write(
    path: Path,
    value: object,
) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(
            mode="json"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
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


def main() -> None:
    args = parse_args()

    base_plan = (
        LiteratureQueryPlan
        .model_validate_json(
            Path(
                args.query_plan
            ).read_text(
                encoding="utf-8"
            )
        )
    )

    diagnostic_plan = (
        build_diagnostic_query_plan(
            base_plan
        )
    )

    provider_plan = (
        load_literature_provider_plan(
            args.provider_plan
        )
    )

    require_standard_or_full_auto_plan(
        provider_plan
    )

    providers = (
        build_literature_providers(
            provider_plan
        )
    )

    packet = LiteratureRetriever(
        providers,
        results_per_query=(
            args.results_per_query
        ),
    ).retrieve(
        diagnostic_plan
    ).packet

    prefix = Path(
        args.output_prefix
    )

    query_path = prefix.with_suffix(
        ".queries.json"
    )

    packet_path = prefix.with_suffix(
        ".prior_art.json"
    )

    provider_path = prefix.with_suffix(
        ".provider_plan.json"
    )

    summary_path = prefix.with_suffix(
        ".summary.json"
    )

    _write(
        query_path,
        diagnostic_plan,
    )

    _write(
        packet_path,
        packet,
    )

    _write(
        provider_path,
        provider_plan,
    )

    successful = sum(
        row.success
        for row in packet.executions
    )

    summary = {
        "schema_version":
            "diagnostic-prior-art-shadow-v1",
        "shadow_only": True,
        "scientific_selection_changed":
            False,
        "source_query_plan_id":
            base_plan.plan_id,
        "diagnostic_query_plan_id":
            diagnostic_plan.plan_id,
        "diagnostic_query_count":
            len(
                diagnostic_plan.queries
            ),
        "provider_count":
            len(providers),
        "provider_query_execution_count":
            len(packet.executions),
        "successful_provider_query_executions":
            successful,
        "unique_work_count":
            len(packet.works),
        "epistemic_usage": (
            "diagnostic_prior_art_only_"
            "not_positive_full_claim_evidence"
        ),
    }

    _write(
        summary_path,
        summary,
    )

    print(
        "Diagnostic prior-art shadow "
        "retrieval complete"
    )

    print(
        "Source query plan:",
        base_plan.plan_id,
    )

    print(
        "Diagnostic query plan:",
        diagnostic_plan.plan_id,
    )

    print(
        "Diagnostic queries:",
        len(
            diagnostic_plan.queries
        ),
    )

    for index, query in enumerate(
        diagnostic_plan.queries,
        start=1,
    ):
        claim = next(
            (
                claim
                for group
                in base_plan.claims
                for claim
                in group.claims
                if claim.claim_id
                == query.claim_id
            ),
            None,
        )

        print()
        print(
            f"[{index}]",
            query.query_kind,
        )

        print(
            "    claim:",
            query.claim_id,
        )

        if claim is not None:
            print(
                "    diagnostic kind:",
                claim.diagnostic_query_kind,
            )

        print(
            "    query:",
            query.query_text,
        )

    print()
    print(
        "Search providers:",
        ", ".join(
            packet.providers_requested
        ),
    )

    print(
        "Successful provider-query "
        "executions:",
        successful,
        "/",
        len(packet.executions),
    )

    print(
        "Unique works:",
        len(packet.works),
    )

    print(
        "Saved queries:",
        query_path,
    )

    print(
        "Saved prior art:",
        packet_path,
    )

    print(
        "Saved summary:",
        summary_path,
    )

    print()
    print(
        "SHADOW ONLY: ordinary external "
        "novelty / N9 / N10 artifacts "
        "were not modified."
    )


if __name__ == "__main__":
    main()
