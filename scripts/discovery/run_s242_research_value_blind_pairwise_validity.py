from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scripts.discovery.run_s226_validation_completion import get_client
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairwiseValueReview(StrictModel):
    disposition: Literal[
        "MEANINGFUL_VALUE_STRUCTURE_DIFFERENCE",
        "SUBSTANTIALLY_SIMILAR_VALUE_STRUCTURE",
        "INDETERMINATE",
    ]
    differing_dimensions: list[
        Literal[
            "SCIENTIFIC_DISCRIMINATION",
            "TWO_SIDED_OUTCOME_INFORMATION",
            "OBSERVABLE_DECISIVENESS",
            "EXPERIMENTAL_RESOLVABILITY",
            "RESOURCE_BURDEN",
        ]
    ] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


SYSTEM = """You are a blinded scientific research-value structure reviewer.

You receive TWO hypotheses from the SAME scientific task, together with each
hypothesis's validation specification and experimental realizability report.

You are NOT judging novelty, importance, prestige, publication potential, or
which hypothesis is "better". Do not rank the hypotheses.

Your only task is to decide whether the two candidates have a meaningful
difference in RESEARCH-VALUE STRUCTURE along these dimensions:

1. SCIENTIFIC_DISCRIMINATION
Does the proposed validation distinguish among mechanisms, explanations,
conditions, or competing scientific possibilities?

2. TWO_SIDED_OUTCOME_INFORMATION
Would both supportive and falsifying/negative outcomes teach something
scientifically interpretable, rather than only one outcome being informative?

3. OBSERVABLE_DECISIVENESS
Are the predicted and falsifying outcomes tied to sufficiently explicit,
comparable observables to make the experiment/computation diagnostic?

4. EXPERIMENTAL_RESOLVABILITY
Does the existing feasibility report indicate a meaningful difference in how
well the candidates can actually be resolved/tested?

5. RESOURCE_BURDEN
Does the existing feasibility report indicate a meaningful cost/effort
difference? High burden is descriptive, not automatically worse.

Use only the supplied artifacts. Do not use outside scientific knowledge.
Do not infer novelty. Do not select a winner.

Return:
- MEANINGFUL_VALUE_STRUCTURE_DIFFERENCE when at least one listed dimension
  differs materially between A and B.
- SUBSTANTIALLY_SIMILAR_VALUE_STRUCTURE when the artifacts support essentially
  the same structure on the listed dimensions.
- INDETERMINATE when the supplied artifacts are insufficient to decide.

List only dimensions that materially differ."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_hypothesis_cards(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("hypotheses", []) if isinstance(payload, dict) else []
    return {
        str(row["hypothesis_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("hypothesis_id")
    }


def load_hypothesis_index(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return result

    for path in sorted(directory.glob("*.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        hid = payload.get("hypothesis_id")
        if not hid:
            continue
        key = str(hid)
        if key in result and result[key] != payload:
            raise RuntimeError(
                f"conflicting artifacts for hypothesis {key} in {directory}"
            )
        result[key] = payload
    return result


def profile_key(profile: dict[str, Any]) -> str:
    return json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "hypothesis_id": card.get("hypothesis_id"),
        "hypothesis_type": card.get("hypothesis_type"),
        "statement": card.get("statement"),
        "mechanism": card.get("mechanism"),
        "conditions": card.get("conditions"),
        "predicted_observations": card.get("predicted_observations", []),
        "falsification_criteria": card.get("falsification_criteria", []),
    }


def compact_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "validation_strategy": spec.get("validation_strategy"),
        "requires_candidate_concretization":
            spec.get("requires_candidate_concretization"),
        "primary_observables": spec.get("primary_observables", []),
        "required_comparisons": spec.get("required_comparisons", []),
        "success_patterns": spec.get("success_patterns", []),
        "falsification_patterns": spec.get("falsification_patterns", []),
    }


def compact_experimental(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "disposition": report.get("disposition"),
        "relative_cost_burden": report.get("relative_cost_burden"),
        "relative_effort_burden": report.get("relative_effort_burden"),
        "reason_codes": report.get("reason_codes", []),
        "notes": report.get("notes"),
    }


def build_user(
    *,
    card_a: dict[str, Any],
    spec_a: dict[str, Any],
    exp_a: dict[str, Any],
    card_b: dict[str, Any],
    spec_b: dict[str, Any],
    exp_b: dict[str, Any],
) -> str:
    payload = {
        "candidate_A": {
            "hypothesis": compact_card(card_a),
            "validation_specification": compact_spec(spec_a),
            "experimental_realizability": compact_experimental(exp_a),
        },
        "candidate_B": {
            "hypothesis": compact_card(card_b),
            "validation_specification": compact_spec(spec_b),
            "experimental_realizability": compact_experimental(exp_b),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S242 blind pairwise validity test for Research Value profile-v2. "
            "The reviewer never sees S240 profiles and never ranks candidates."
        )
    )
    p.add_argument("--collector", required=True, type=Path)
    p.add_argument("--s240-summary", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_AGENT_MODEL") or "openai/gpt-5.6-luna",
    )
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
    )
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--max-retries", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    collector_path = args.collector.expanduser().resolve()
    s240_path = args.s240_summary.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    for path in (collector_path, s240_path):
        if not path.is_file():
            raise RuntimeError("missing input: " + str(path))
    if output_path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: " + str(output_path)
        )

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"missing API key environment variable: {args.api_key_env}"
        )

    collector = load_json(collector_path)
    s240 = load_json(s240_path)

    profiles = {
        str(row["hypothesis_id"]): row
        for row in s240.get("cards", [])
        if isinstance(row, dict) and row.get("hypothesis_id")
    }

    case_material: dict[str, dict[str, Any]] = {}

    for case in collector.get("cases", []):
        if not isinstance(case, dict):
            continue
        rv = case.get("research_value") or {}
        if rv.get("measurement_status") != "AVAILABLE":
            continue
        artifact_raw = rv.get("artifact")
        if not artifact_raw:
            continue

        run_dir = Path(artifact_raw).expanduser().resolve().parent
        portfolio_path = run_dir / "novelty_refinement_a6.portfolio.json"
        validation_dir = run_dir / "feasibility_final" / "validation"
        experimental_dir = run_dir / "feasibility_final" / "experimental"

        cards = load_hypothesis_cards(portfolio_path)
        specs = load_hypothesis_index(validation_dir)
        exps = load_hypothesis_index(experimental_dir)

        common = sorted(set(cards) & set(specs) & set(exps) & set(profiles))
        if len(common) < 2:
            continue

        case_material[str(case.get("source_case_id"))] = {
            "ids": common,
            "cards": cards,
            "specs": specs,
            "exps": exps,
        }

    client = get_client(
        api_key=api_key,
        base_url=args.base_url,
        instructor_mode="tools",
        timeout=120.0,
    )

    rows: list[dict[str, Any]] = []

    print("=== S242 BLIND PAIRWISE RV PROFILE VALIDITY ===")
    print("profiles shown to reviewer: false")
    print("candidate ranking requested: false")
    print("literature retrieval: false")
    print("novelty consumed: false")

    total_pairs = sum(
        len(list(combinations(material["ids"], 2)))
        for material in case_material.values()
    )
    pair_index = 0

    for case_id, material in sorted(case_material.items()):
        for a, b in combinations(material["ids"], 2):
            pair_index += 1
            print(f"[{pair_index}/{total_pairs}] {case_id} | {a} vs {b}")

            user = build_user(
                card_a=material["cards"][a],
                spec_a=material["specs"][a],
                exp_a=material["exps"][a],
                card_b=material["cards"][b],
                spec_b=material["specs"][b],
                exp_b=material["exps"][b],
            )

            result, _event = run_instructor_structured_call(
                client.chat.completions,
                model=args.model,
                response_model=PairwiseValueReview,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_retries=args.max_retries,
                telemetry_context={
                    "pipeline": "research_value_profile_validation",
                    "stage": "blind_pairwise_value_structure_review",
                    "call_kind": "structured",
                    "source_case_id": case_id,
                    "hypothesis_a": a,
                    "hypothesis_b": b,
                },
            )

            if not isinstance(result, PairwiseValueReview):
                result = PairwiseValueReview.model_validate(result)

            profile_a = profiles[a].get("profile") or {}
            profile_b = profiles[b].get("profile") or {}
            same_profile = profile_key(profile_a) == profile_key(profile_b)

            if result.disposition == "INDETERMINATE":
                agreement = "INDETERMINATE"
            elif same_profile:
                agreement = (
                    "AGREES"
                    if result.disposition
                    == "SUBSTANTIALLY_SIMILAR_VALUE_STRUCTURE"
                    else "DISAGREES"
                )
            else:
                agreement = (
                    "AGREES"
                    if result.disposition
                    == "MEANINGFUL_VALUE_STRUCTURE_DIFFERENCE"
                    else "DISAGREES"
                )

            rows.append(
                {
                    "source_case_id": case_id,
                    "hypothesis_a": a,
                    "hypothesis_b": b,
                    "same_s240_profile": same_profile,
                    "review": result.model_dump(mode="json"),
                    "profile_validity_agreement": agreement,
                }
            )

    determinate = [
        row for row in rows
        if row["profile_validity_agreement"] != "INDETERMINATE"
    ]
    agreement_count = sum(
        row["profile_validity_agreement"] == "AGREES"
        for row in determinate
    )

    same_profile_rows = [row for row in rows if row["same_s240_profile"]]
    different_profile_rows = [row for row in rows if not row["same_s240_profile"]]

    disposition_counts = Counter(
        row["review"]["disposition"] for row in rows
    )
    dimension_counts = Counter(
        dim
        for row in rows
        for dim in row["review"]["differing_dimensions"]
    )

    payload = {
        "schema_version":
            "research-value-profile-blind-validity-s242-v1",
        "source_collector": str(collector_path),
        "source_s240_summary": str(s240_path),
        "pair_count": len(rows),
        "same_profile_pair_count": len(same_profile_rows),
        "different_profile_pair_count": len(different_profile_rows),
        "determinate_pair_count": len(determinate),
        "agreement_count": agreement_count,
        "agreement_fraction": (
            agreement_count / len(determinate)
            if determinate
            else None
        ),
        "review_disposition_counts":
            dict(sorted(disposition_counts.items())),
        "review_differing_dimension_counts":
            dict(sorted(dimension_counts.items())),
        "rows": rows,
        "interpretation_policy": {
            "blind_reviewer_saw_s240_profile": False,
            "candidate_winner_selected": False,
            "overall_value_score_computed": False,
            "ranking_computed": False,
            "novelty_signal_consumed": False,
            "literature_retrieval_performed": False,
            "research_value_selection_authority": False,
            "production_selection_changed": False,
        },
    }

    write_json(output_path, payload)

    print("\nS242 complete")
    print("pairs:", len(rows))
    print("same-profile pairs:", len(same_profile_rows))
    print("different-profile pairs:", len(different_profile_rows))
    print("determinate pairs:", len(determinate))
    print(
        "profile/reviewer agreement:",
        f"{agreement_count}/{len(determinate)}"
        if determinate
        else "n/a",
        (
            f"({payload['agreement_fraction']:.3f})"
            if payload["agreement_fraction"] is not None
            else ""
        ),
    )
    print("review dispositions:", dict(sorted(disposition_counts.items())))
    print("differing dimensions:", dict(sorted(dimension_counts.items())))
    print("ranking computed: false")
    print("production selection changed: false")
    print("artifact:", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
