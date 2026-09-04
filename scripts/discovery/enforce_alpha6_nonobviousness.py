from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.nonobviousness_post_generation import (
    assert_candidate_final_authority_equivalent,
    filter_alpha6_portfolio_by_nonobviousness,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
)


_GENERATED = {
    "accepted_refinement",
    "accepted_reaxis",
}


def _write_json(
    path: Path,
    value: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(
        value,
        "model_dump",
    ):
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


def _run(
    module: str,
    args: list[str],
) -> None:
    cmd = [
        sys.executable,
        "-m",
        module,
        *args,
    ]

    subprocess.run(
        cmd,
        check=True,
    )


def generated_candidate_ids(
    report: NoveltyRefinementReport,
) -> tuple[str, ...]:
    ids = []

    for attempt in report.attempts:
        if (
            attempt.decision
            not in _GENERATED
        ):
            continue

        candidate_id = str(
            attempt.candidate_hypothesis_id
            or ""
        ).strip()

        if not candidate_id:
            raise ValueError(
                "accepted generated Alpha6 attempt "
                "missing candidate_hypothesis_id"
            )

        ids.append(
            candidate_id
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            "duplicate accepted Alpha6 candidate ID"
        )

    return tuple(ids)


def find_final_external_bundle(
    *,
    external_dir: Path,
    candidate_id: str,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
]:
    suffix = (
        candidate_id
        .split(":")[-1]
    )

    plans = sorted(
        external_dir.glob(
            "final_*_"
            + suffix
            + ".claims_queries.json"
        )
    )

    if len(plans) != 1:
        raise ValueError(
            "expected exactly one fresh final external "
            "query plan for candidate "
            + candidate_id
            + f"; found={len(plans)}"
        )

    plan = plans[0]

    base = Path(
        str(plan)[
            :-len(
                ".claims_queries.json"
            )
        ]
    )

    source_portfolio = Path(
        str(base)
        + ".portfolio.json"
    )

    prior = Path(
        str(base)
        + ".prior_art.json"
    )

    report = Path(
        str(base)
        + ".report.json"
    )

    if not source_portfolio.is_file():
        raise ValueError(
            "missing exact fresh candidate source-portfolio artifact: "
            + str(source_portfolio)
        )

    if not prior.is_file():
        raise ValueError(
            "missing fresh final prior-art artifact: "
            + str(prior)
        )

    if not report.is_file():
        raise ValueError(
            "missing fresh final report artifact: "
            + str(report)
        )

    parsed_portfolio = (
        HypothesisPortfolio
        .model_validate_json(
            source_portfolio.read_text(
                encoding="utf-8"
            )
        )
    )

    candidate_portfolio_ids = {
        card.hypothesis_id
        for card
        in parsed_portfolio.hypotheses
    }

    if candidate_portfolio_ids != {candidate_id}:
        raise ValueError(
            "fresh source portfolio does not contain exactly "
            "the expected Alpha6 candidate: "
            + candidate_id
        )

    parsed_plan = (
        LiteratureQueryPlan
        .model_validate_json(
            plan.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        parsed_plan.source_portfolio_id
        != parsed_portfolio.portfolio_id
    ):
        raise ValueError(
            "fresh query-plan/source-portfolio provenance mismatch"
        )

    planned_ids = {
        group.hypothesis_id
        for group
        in parsed_plan.claims
    }

    if candidate_id not in planned_ids:
        raise ValueError(
            "fresh final query plan does not contain "
            "expected candidate "
            + candidate_id
        )

    parsed_prior = (
        PriorArtPacket
        .model_validate_json(
            prior.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        parsed_prior.source_portfolio_id
        != parsed_portfolio.portfolio_id
    ):
        raise ValueError(
            "fresh prior-art/source-portfolio provenance mismatch"
        )

    if (
        parsed_prior.source_query_plan_id
        != parsed_plan.plan_id
    ):
        raise ValueError(
            "fresh prior-art/query-plan provenance mismatch"
        )

    parsed_report = (
        ExternalNoveltyReport
        .model_validate_json(
            report.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        parsed_report.source_portfolio_id
        != parsed_portfolio.portfolio_id
    ):
        raise ValueError(
            "fresh external-report/source-portfolio provenance mismatch"
        )

    report_ids = {
        card.hypothesis_id
        for card
        in parsed_report.cards
    }

    if candidate_id not in report_ids:
        raise ValueError(
            "fresh final external report does not contain "
            "expected candidate "
            + candidate_id
        )

    return (
        source_portfolio,
        plan,
        prior,
        report,
    )


def find_final_external_triplet(
    *,
    external_dir: Path,
    candidate_id: str,
) -> tuple[
    Path,
    Path,
    Path,
]:
    """Backward-compatible non-authoritative artifact locator."""

    (
        _source_portfolio,
        plan,
        prior,
        report,
    ) = find_final_external_bundle(
        external_dir=external_dir,
        candidate_id=candidate_id,
    )

    return (
        plan,
        prior,
        report,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run fresh N10 non-obviousness adjudication over "
            "Alpha6 accepted refinement/re-axis candidates and "
            "filter the Alpha6 portfolio accordingly."
        )
    )

    p.add_argument(
        "--portfolio",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--hypothesis-context",
        required=True,
        type=Path,
        help=(
            "Canonical grounded HypothesisContext inherited by "
            "all Alpha6 candidate portfolios."
        ),
    )

    p.add_argument(
        "--refinement-report",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--external-dir",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--provider-plan",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--domain-profile",
        required=True,
    )

    p.add_argument(
        "--model",
        required=True,
    )

    p.add_argument(
        "--base-url",
        default=None,
    )

    p.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )

    p.add_argument(
        "--device",
        default=None,
    )

    p.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )

    p.add_argument(
        "--max-ranked-works",
        type=int,
        default=8,
    )

    p.add_argument(
        "--work-dir",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--output-portfolio",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--output-report",
        required=True,
        type=Path,
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    portfolio = (
        HypothesisPortfolio
        .model_validate_json(
            args.portfolio.read_text(
                encoding="utf-8"
            )
        )
    )

    refinement = (
        NoveltyRefinementReport
        .model_validate_json(
            args.refinement_report
            .read_text(
                encoding="utf-8"
            )
        )
    )

    candidate_ids = (
        generated_candidate_ids(
            refinement
        )
    )

    gates = {}
    artifact_audit = []

    for index, candidate_id in enumerate(
        candidate_ids,
        1,
    ):
        (
            source_portfolio,
            query_plan,
            prior_art,
            external_report,
        ) = find_final_external_bundle(
            external_dir=args.external_dir,
            candidate_id=candidate_id,
        )

        generated_attempts = [
            attempt
            for attempt
            in refinement.attempts
            if (
                attempt.decision
                in _GENERATED
                and str(
                    attempt.candidate_hypothesis_id
                    or ""
                ).strip()
                == candidate_id
            )
        ]

        if len(generated_attempts) != 1:
            raise ValueError(
                "expected exactly one generated Alpha6 "
                "attempt for candidate "
                + candidate_id
            )

        generated_attempt = (
            generated_attempts[0]
        )

        final_id = str(
            generated_attempt.final_hypothesis_id
            or ""
        ).strip()

        if not final_id:
            raise ValueError(
                "generated Alpha6 candidate lacks "
                "final_hypothesis_id before N10 authority transfer: "
                + candidate_id
            )

        candidate_source_portfolio = (
            HypothesisPortfolio
            .model_validate_json(
                source_portfolio.read_text(
                    encoding="utf-8"
                )
            )
        )

        candidate_cards = [
            card
            for card
            in candidate_source_portfolio.hypotheses
            if card.hypothesis_id
            == candidate_id
        ]

        if len(candidate_cards) != 1:
            raise ValueError(
                "fresh candidate source portfolio "
                "does not resolve exactly one candidate card: "
                + candidate_id
            )

        final_cards = [
            card
            for card
            in portfolio.hypotheses
            if card.hypothesis_id
            == final_id
        ]

        if len(final_cards) != 1:
            raise ValueError(
                "Alpha6 final portfolio does not resolve "
                "exactly one final card for candidate "
                + candidate_id
                + "; final="
                + final_id
            )

        assert_candidate_final_authority_equivalent(
            candidate=candidate_cards[0],
            final=final_cards[0],
        )

        candidate_dir = (
            args.work_dir
            / (
                f"{index:02d}_"
                + candidate_id
                .replace(
                    ":",
                    "_",
                )
            )
        )

        candidate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        intake = (
            candidate_dir
            / "intake_shadow.json"
        )

        full = (
            candidate_dir
            / "full_shadow.json"
        )

        gate_path = (
            candidate_dir
            / "production_gate.json"
        )

        _run(
            "scripts.discovery."
            "build_nonobviousness_shadow",
            [
                "--query-plan",
                str(query_plan),
                "--external-report",
                str(external_report),
                "--output",
                str(intake),
            ],
        )

        intake_payload = json.loads(
            intake.read_text(
                encoding="utf-8"
            )
        )

        ready_count = sum(
            len(
                row.get(
                    "ready_for_closure_claim_ids",
                    [],
                )
            )
            for row
            in intake_payload.get(
                "hypotheses",
                [],
            )
        )

        full_args = [
            "--query-plan",
            str(query_plan),
            "--external-report",
            str(external_report),
            "--external-prior-art",
            str(prior_art),
            "--portfolio",
            str(source_portfolio),
            "--hypothesis-context",
            str(args.hypothesis_context),
            "--intake-shadow",
            str(intake),
            "--provider-plan",
            str(args.provider_plan),
            "--domain-profile",
            args.domain_profile,
            "--model",
            args.model,
            "--api-key-env",
            args.api_key_env,
            "--results-per-query",
            str(
                args.results_per_query
            ),
            "--max-ranked-works",
            str(
                args.max_ranked_works
            ),
            "--max-ready-claims",
            str(
                max(
                    1,
                    ready_count,
                )
            ),
            "--output",
            str(full),
        ]

        if args.base_url:
            full_args.extend(
                [
                    "--base-url",
                    args.base_url,
                ]
            )

        if args.device:
            full_args.extend(
                [
                    "--device",
                    args.device,
                ]
            )

        _run(
            "scripts.discovery."
            "run_nonobviousness_full_shadow",
            full_args,
        )

        candidate_gate_path = (
            gate_path.with_name(
                "production_gate_v2.candidate.json"
            )
        )

        _run(
            "scripts.discovery."
            "build_nonobviousness_production_gate_v2_candidate",
            [
                "--query-plan",
                str(query_plan),
                "--intake-shadow",
                str(intake),
                "--full-shadow",
                str(full),
                "--output",
                str(candidate_gate_path),
            ],
        )

        _run(
            "scripts.discovery."
            "build_nonobviousness_post_generation_production_gate_v2",
            [
                "--candidate-gate",
                str(candidate_gate_path),
                "--output",
                str(gate_path),
            ],
        )

        gate_payload = json.loads(
            gate_path.read_text(
                encoding="utf-8"
            )
        )

        gates[
            candidate_id
        ] = gate_payload

        matching = [
            row
            for row
            in gate_payload.get(
                "gates",
                [],
            )
            if (
                row.get(
                    "hypothesis_id"
                )
                == candidate_id
            )
        ]

        if len(matching) != 1:
            raise ValueError(
                "fresh N10 gate does not contain exactly "
                "one candidate row for "
                + candidate_id
            )

        artifact_audit.append(
            {
                "candidate_id":
                    candidate_id,
                "final_hypothesis_id":
                    final_id,
                "candidate_final_authority_equivalent":
                    True,
                "source_portfolio":
                    str(source_portfolio),
                "hypothesis_context":
                    str(args.hypothesis_context),
                "query_plan":
                    str(query_plan),
                "prior_art":
                    str(prior_art),
                "external_report":
                    str(external_report),
                "intake_shadow":
                    str(intake),
                "full_shadow":
                    str(full),
                "production_gate":
                    str(gate_path),
                "ready_claim_count":
                    ready_count,
                "selection_class":
                    matching[0].get(
                        "selection_class"
                    ),
                "fallback_allowed":
                    matching[0].get(
                        "fallback_allowed"
                    ),
            }
        )

    filtered, audit = (
        filter_alpha6_portfolio_by_nonobviousness(
            portfolio=portfolio,
            refinement_report=refinement,
            gates_by_candidate_id=gates,
        )
    )

    audit[
        "candidate_artifacts"
    ] = artifact_audit

    _write_json(
        args.output_portfolio,
        filtered,
    )

    _write_json(
        args.output_report,
        audit,
    )

    print(
        "Alpha6 fresh-candidate N10 enforcement complete"
    )
    print(
        "Generated candidates:",
        len(candidate_ids),
    )
    print(
        "Alpha6 survivors:",
        len(portfolio.hypotheses),
    )
    print(
        "Final N10 survivors:",
        len(filtered.hypotheses),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
