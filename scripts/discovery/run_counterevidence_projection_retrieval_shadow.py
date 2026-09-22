from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.counterevidence_projection_retrieval import (
    build_counterevidence_projection_query_plan,
    build_counterevidence_retrieval_report,
    build_counterevidence_transport_plan,
)
from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    load_literature_provider_plan,
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
)
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureRetriever,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded counterevidence retrieval shadow over grounded "
            "relation projections. Search intent targets null, independence, "
            "decoupling, opposite-direction, and distributional-divergence "
            "literature where structurally applicable. Retrieved works are "
            "not automatically counterevidence."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument(
        "--providers",
        default="auto",
    )
    parser.add_argument(
        "--provider-plan",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-lower-order-factor-order",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-source-alias-variants-per-projection",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    projection = GroundedFactorProjectionReport.model_validate_json(
        args.projection.read_text(encoding="utf-8")
    )

    counter_plan = build_counterevidence_projection_query_plan(
        portfolio=portfolio,
        projection_report=projection,
        max_lower_order_factor_order=(
            args.max_lower_order_factor_order
        ),
        max_source_alias_variants_per_projection=(
            args.max_source_alias_variants_per_projection
        ),
    )
    transport_plan = build_counterevidence_transport_plan(
        counter_plan
    )

    prefix = args.output_prefix
    counter_plan_path = prefix.with_suffix(
        ".counterevidence_query_plan.json"
    )
    transport_path = prefix.with_suffix(
        ".transport_query_plan.json"
    )
    provider_path = prefix.with_suffix(
        ".provider_plan.json"
    )
    packet_path = prefix.with_suffix(
        ".prior_art.json"
    )
    report_path = prefix.with_suffix(
        ".retrieval_report.json"
    )

    _write(counter_plan_path, counter_plan)
    _write(transport_path, transport_plan)

    if args.provider_plan is not None:
        provider_plan = load_literature_provider_plan(
            args.provider_plan
        )
    else:
        raw = str(args.providers or "").strip()
        requested = (
            None
            if raw.lower() == "auto"
            else [
                value.strip()
                for value in raw.split(",")
                if value.strip()
            ]
        )
        provider_plan = resolve_literature_provider_plan(
            requested=requested
        )

    require_standard_or_full_auto_plan(provider_plan)
    _write(provider_path, provider_plan)

    providers = build_literature_providers(
        provider_plan
    )

    packet = LiteratureRetriever(
        providers,
        results_per_query=args.results_per_query,
    ).retrieve(
        transport_plan
    ).packet
    _write(packet_path, packet)

    report = build_counterevidence_retrieval_report(
        counterevidence_plan=counter_plan,
        transport_plan=transport_plan,
        packet=packet,
    )
    _write(report_path, report)

    print("Counterevidence projection retrieval shadow complete")
    print(
        "Selected projections:",
        counter_plan.selected_projection_count,
    )
    print(
        "Counterevidence bindings:",
        counter_plan.counterevidence_binding_count,
    )
    print(
        "Transport queries:",
        counter_plan.transport_query_count,
    )
    print(
        "Deduplicated network queries:",
        counter_plan.deduplicated_network_query_count,
    )
    print("Modes:", counter_plan.modes_present)
    print("Providers:", provider_plan.active_providers)
    print(
        "Provider-query successes:",
        report.successful_provider_query_count,
    )
    print(
        "Provider-query failures:",
        report.failed_provider_query_count,
    )
    print("Unique works:", report.unique_work_count)
    print(
        "Abstract-bearing works:",
        report.abstract_work_count,
    )

    by_claim: dict[str, list] = {}
    for row in report.projection_mode_coverage:
        by_claim.setdefault(
            row.claim_id,
            [],
        ).append(row)

    for claim_id, rows in by_claim.items():
        print()
        print("Claim:", claim_id)
        for row in rows:
            print(
                " ",
                row.projection_kind,
                "order=",
                row.factor_order,
                "mode=",
                row.counterevidence_mode,
                "queries=",
                row.query_count,
                "works=",
                row.unique_work_count,
                "abstracts=",
                row.abstract_work_count,
                "provider_ok=",
                row.successful_provider_query_count,
                "provider_fail=",
                row.failed_provider_query_count,
            )

    print()
    print("Retrieval lane: COUNTEREVIDENCE_PRIOR_ART")
    print("Relation adjudication performed: false")
    print("Counterevidence relationship assigned: false")
    print("Absence-based novelty authorized: false")
    print("Positive non-obviousness authority created: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Counterevidence plan:", counter_plan_path)
    print("Transport plan:", transport_path)
    print("Provider plan:", provider_path)
    print("Prior-art packet:", packet_path)
    print("Retrieval report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
