from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.novelty_selection_role_annotation import (
    InstructorOpenAICompatibleSelectionRoleBackend,
    compile_role_annotation,
)


def _load(path: str) -> dict[str, Any]:
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def _sha256_file(path: str) -> str:
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--portfolio",
        required=True,
    )
    parser.add_argument(
        "--query-plan",
        required=True,
    )
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--base-url",
        default=None,
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
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--telemetry",
        default=None,
    )

    args = parser.parse_args()

    portfolio = HypothesisPortfolio.model_validate(
        _load(args.portfolio)
    )

    query_plan = LiteratureQueryPlan.model_validate(
        _load(args.query_plan)
    )

    if (
        portfolio.portfolio_id
        != query_plan.source_portfolio_id
    ):
        raise ValueError(
            "portfolio/query-plan source mismatch"
        )

    by_hypothesis = {
        row.hypothesis_id: row
        for row in query_plan.claims
    }

    if len(by_hypothesis) != len(
        query_plan.claims
    ):
        raise ValueError(
            "duplicate hypothesis claim-group "
            "in query plan"
        )

    backend = (
        InstructorOpenAICompatibleSelectionRoleBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.temperature,
            parse_retries=args.parse_retries,
            telemetry_path=args.telemetry,
            telemetry_context={
                "source_portfolio_id":
                    portfolio.portfolio_id,
                "source_query_plan_id":
                    query_plan.plan_id,
            },
        )
    )

    hypothesis_rows: list[
        dict[str, Any]
    ] = []

    role_counter: Counter[str] = Counter()
    null_count = 0
    total_claim_count = 0

    for hypothesis in portfolio.hypotheses:
        canonical = by_hypothesis.get(
            hypothesis.hypothesis_id
        )

        if canonical is None:
            raise ValueError(
                "missing query-plan claim group for "
                + hypothesis.hypothesis_id
            )

        if not canonical.claims:
            raise ValueError(
                "empty canonical claim group for "
                + hypothesis.hypothesis_id
            )

        draft = backend.annotate(
            hypothesis,
            canonical.claims,
        )

        compiled = compile_role_annotation(
            hypothesis=hypothesis,
            claims=canonical.claims,
            draft=draft,
        )

        for row in compiled:
            total_claim_count += 1

            role = row.get(
                "novelty_selection_role"
            )

            if role is None:
                null_count += 1
            else:
                role_counter[str(role)] += 1

        prompt_record = (
            backend.prompt_records[-1]
        )

        hypothesis_rows.append(
            {
                "hypothesis_id":
                    hypothesis.hypothesis_id,
                "claim_count":
                    len(compiled),
                "assignments":
                    compiled,
                "prompt_sha256":
                    prompt_record.prompt_sha256,
            }
        )

    artifact = {
        "schema_version":
            "novelty-selection-role-shadow-v1",
        "shadow_only": True,
        "scientific_selection_changed": False,
        "production_authority": False,
        "outcome_blind": True,
        "role_assignment_inputs": [
            "hypothesis_structure",
            "canonical_atomic_claim_identity",
            "canonical_atomic_claim_kind",
            "canonical_atomic_claim_text",
            "canonical_atomic_claim_rationale",
        ],
        "explicitly_excluded_inputs": [
            "claim_importance",
            "prior_art_report",
            "prior_art_packet",
            "external_novelty_status",
            "n9_intake_state",
            "n9_full_verdict",
            "search_coverage",
        ],
        "source_portfolio_id":
            portfolio.portfolio_id,
        "source_portfolio_sha256":
            _sha256_file(args.portfolio),
        "source_query_plan_id":
            query_plan.plan_id,
        "source_query_plan_sha256":
            query_plan.plan_sha256,
        "source_query_plan_file_sha256":
            _sha256_file(args.query_plan),
        "model": args.model,
        "hypothesis_count":
            len(hypothesis_rows),
        "claim_count":
            total_claim_count,
        "null_role_count":
            null_count,
        "role_counts":
            dict(
                sorted(role_counter.items())
            ),
        "hypotheses":
            hypothesis_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "N10-B0 selection-role shadow built"
    )
    print(
        "Hypotheses:",
        artifact["hypothesis_count"],
    )
    print(
        "Claims:",
        artifact["claim_count"],
    )
    print(
        "Role counts:",
        artifact["role_counts"],
    )
    print(
        "Null roles:",
        artifact["null_role_count"],
    )


if __name__ == "__main__":
    main()
