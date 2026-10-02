
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.sers_closed_loop_decision_consolidation_v2 import (
    build_effective_gen1_portfolio,
    build_final_freeze,
    build_provenance_pin,
    build_replacement_map,
    consolidate_post_verification_decisions,
    render_final_report,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gen0-portfolio", required=True, type=Path)
    p.add_argument("--gen0-closeout", required=True, type=Path)
    p.add_argument("--gen0-external", required=True, type=Path)
    p.add_argument("--gen0-query-plan", required=True, type=Path)
    p.add_argument("--gen0-prior-art", required=True, type=Path)
    p.add_argument("--gen0-materialization", required=True, type=Path)
    p.add_argument("--gen0-aggregation-a", required=True, type=Path)
    p.add_argument("--gen0-aggregation-b", required=True, type=Path)
    p.add_argument("--gen0-readiness", required=True, type=Path)
    p.add_argument("--gen0-selection-audit", required=True, type=Path)
    p.add_argument("--gen0-stability", required=True, type=Path)
    p.add_argument("--generation-report", required=True, type=Path)
    p.add_argument("--gen1-portfolio", required=True, type=Path)
    p.add_argument("--gen1-query-plan", required=True, type=Path)
    p.add_argument("--gen1-external", required=True, type=Path)
    p.add_argument("--gen1-aggregation", required=True, type=Path)
    p.add_argument("--gen1-cohort-audit", required=True, type=Path)
    p.add_argument("--old-closed-loop-audit", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    gen1_portfolio = load(args.gen1_portfolio)
    generation = load(args.generation_report)
    query = load(args.gen1_query_plan)
    external = load(args.gen1_external)
    aggregation = load(args.gen1_aggregation)
    cohort = load(args.gen1_cohort_audit)
    old_audit = load(args.old_closed_loop_audit)

    decisions = consolidate_post_verification_decisions(
        gen1_portfolio=gen1_portfolio,
        generation_report=generation,
        query_plan=query,
        external_report=external,
        aggregation=aggregation,
        cohort_audit=cohort,
    )

    effective = build_effective_gen1_portfolio(
        gen1_portfolio=gen1_portfolio,
        decisions=decisions,
    )

    replacement = build_replacement_map(
        generation_report=generation,
        decisions=decisions,
    )

    pin = build_provenance_pin(
        artifacts={
            "gen0_portfolio": args.gen0_portfolio,
            "gen0_closeout": args.gen0_closeout,
            "gen0_external": args.gen0_external,
            "gen0_query_plan": args.gen0_query_plan,
            "gen0_prior_art": args.gen0_prior_art,
            "gen0_materialization": args.gen0_materialization,
            "gen0_aggregation_a": args.gen0_aggregation_a,
            "gen0_aggregation_b": args.gen0_aggregation_b,
            "gen0_readiness": args.gen0_readiness,
            "gen0_selection_audit": args.gen0_selection_audit,
            "gen0_stability": args.gen0_stability,
            "generation_report": args.generation_report,
            "gen1_portfolio": args.gen1_portfolio,
            "gen1_query_plan": args.gen1_query_plan,
            "gen1_external": args.gen1_external,
            "gen1_aggregation": args.gen1_aggregation,
            "gen1_cohort_audit": args.gen1_cohort_audit,
        },
        identifiers={
            "gen0_portfolio_id": load(
                args.gen0_portfolio
            ).get("portfolio_id"),
            "gen0_closeout_schema": load(
                args.gen0_closeout
            ).get("schema_version"),
            "gen0_external_report_id": load(
                args.gen0_external
            ).get("report_id"),
            "gen0_query_plan_id": load(
                args.gen0_query_plan
            ).get("plan_id"),
            "gen0_prior_art_packet_id": load(
                args.gen0_prior_art
            ).get("packet_id"),
            "gen1_portfolio_id": gen1_portfolio.get("portfolio_id"),
            "gen1_external_report_id": external.get("report_id"),
            "gen1_query_plan_id": query.get("plan_id"),
        },
    )

    freeze = build_final_freeze(
        provenance_pin=pin,
        decisions=decisions,
        effective_portfolio=effective,
        replacement_map=replacement,
        old_audit=old_audit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "gen0_epistemic_provenance.pin.json", pin)
    write(args.output_dir / "post_verification_decisions.json", decisions)
    write(args.output_dir / "effective_gen1.portfolio.json", effective)
    write(args.output_dir / "gen0_to_gen1.replacement_map.json", replacement)
    write(args.output_dir / "closed_loop_decision.freeze.json", freeze)
    (args.output_dir / "SERS_CLOSED_LOOP_FINAL_REPORT.md").write_text(
        render_final_report(
            freeze=freeze,
            decisions=decisions,
            replacement_map=replacement,
        ),
        encoding="utf-8",
    )

    print("===== CLOSED-LOOP DECISION CONSOLIDATION V2 =====")
    print("decision counts:", decisions["decision_counts"])
    print("effective Gen1:", freeze["effective_gen1_count"])
    print("effective IDs:", freeze["effective_gen1_hypothesis_ids"])
    print(
        "replacement transitions:",
        replacement["transition_counts"],
    )
    print(
        "superseded v1 mismatches:",
        len(freeze["superseded_v1_audit_mismatches"]),
    )

    for row in decisions["rows"]:
        print(
            row["hypothesis_id"],
            "| source=", row["source_hypothesis_id"],
            "| route=", row["generation_route"],
            "| external=", row["fresh_external_status"],
            "| composites=", row["composite_count"],
            "| decision=", row["post_verification_decision"],
        )
        for comp in row["composites"]:
            print(
                "   ",
                comp["claim_id"],
                "| role=", comp["role_class"],
                "| full=", comp["full_relation_status"],
                "| topology=", comp["topology_state"],
                "| disposition=", comp["aggregation_disposition"],
            )
        for warning in row["warnings"]:
            print("   WARN:", warning)

    print("SHADOW_ONLY=True")
    print("CARRIED_FORWARD_EPISTEMIC_IMMUNITY=False")
    print("CLAIM_ROLE_AWARE_AGGREGATION=True")
    print("NOVELTY_AUTHORITY_CREATED=False")
    print("GENERATION_AUTHORITY_CREATED=False")
    print("N9_AUTHORITY_CREATED=False")
    print("N10_AUTHORITY_CREATED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
