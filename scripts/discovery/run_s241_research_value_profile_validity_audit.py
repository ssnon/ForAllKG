from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def norm(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", norm(value))
        if len(token) >= 2
    }


def lexical_similarity(a: str, b: str) -> float:
    ta = tokens(a)
    tb = tokens(b)
    if not ta or not tb:
        return 0.0
    if ta <= tb or tb <= ta:
        return min(len(ta), len(tb)) / max(len(ta), len(tb))
    return len(ta & tb) / len(ta | tb)


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


def factor_purity(
    rows: list[dict[str, Any]],
    factor: str,
) -> tuple[float, dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        value = str(row.get(factor) or "MISSING")
        grouped[value][row["profile_key"]] += 1

    total = len(rows)
    if total <= 0:
        return 1.0, {}

    correct = sum(max(counts.values()) for counts in grouped.values())
    detail = {
        value: dict(sorted(counts.items()))
        for value, counts in sorted(grouped.items())
    }
    return correct / total, detail


def observable_names(card: dict[str, Any], key: str) -> list[str]:
    return [
        str(row.get("observable") or "")
        for row in card.get(key, [])
        if isinstance(row, dict) and norm(row.get("observable"))
    ]


def primary_alignment_diagnostic(
    *,
    card: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    predicted = observable_names(card, "predicted_observations")
    falsifiers = observable_names(card, "falsification_criteria")
    primary = [
        str(value)
        for value in spec.get("primary_observables", [])
        if norm(value)
    ]

    shared_exact = sorted(
        set(map(norm, predicted))
        & set(map(norm, falsifiers))
    )
    exact_primary = sorted(
        set(shared_exact)
        & set(map(norm, primary))
    )

    near_pairs = []
    for shared in shared_exact:
        for target in primary:
            score = lexical_similarity(shared, target)
            if score >= 0.5:
                near_pairs.append(
                    {
                        "shared_observable": shared,
                        "primary_observable": target,
                        "lexical_similarity": round(score, 6),
                    }
                )

    if exact_primary:
        disposition = "EXACT_PRIMARY_ALIGNMENT_PRESENT"
    elif near_pairs:
        disposition = "LEXICALLY_NEAR_PRIMARY_ALIGNMENT_ONLY"
    else:
        disposition = "NO_LEXICAL_PRIMARY_ALIGNMENT"

    return {
        "predicted_observables": predicted,
        "falsifier_observables": falsifiers,
        "primary_observables": primary,
        "shared_prediction_falsifier_observables": shared_exact,
        "exact_primary_alignment": exact_primary,
        "lexically_near_primary_pairs": near_pairs,
        "disposition": disposition,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S241 artifact-only Research Value profile validity audit: "
            "check whether S240 profile separation is merely a proxy for "
            "hypothesis/validation templates, and diagnose the universal "
            "NO_PRIMARY_ALIGNMENT signal."
        )
    )
    p.add_argument("--collector", required=True, type=Path)
    p.add_argument("--s240-summary", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
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

    collector = load_json(collector_path)
    s240 = load_json(s240_path)

    profile_rows = {
        str(row["hypothesis_id"]): row
        for row in s240.get("cards", [])
        if isinstance(row, dict) and row.get("hypothesis_id")
    }

    rows: list[dict[str, Any]] = []

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

        if not portfolio_path.is_file():
            continue

        cards = load_hypothesis_cards(portfolio_path)
        specs = load_hypothesis_index(validation_dir)

        for hid in sorted(set(cards) & set(specs) & set(profile_rows)):
            p_row = profile_rows[hid]
            profile = p_row.get("profile") or {}

            rows.append(
                {
                    "source_case_id": str(case.get("source_case_id") or "UNKNOWN"),
                    "case_role": str(case.get("case_role") or "MISSING"),
                    "hypothesis_id": hid,
                    "hypothesis_type": str(
                        cards[hid].get("hypothesis_type") or "MISSING"
                    ),
                    "validation_strategy": str(
                        specs[hid].get("validation_strategy") or "MISSING"
                    ),
                    "profile": profile,
                    "profile_key": profile_key(profile),
                    "primary_alignment_diagnostic":
                        primary_alignment_diagnostic(
                            card=cards[hid],
                            spec=specs[hid],
                        ),
                }
            )

    type_purity, type_detail = factor_purity(rows, "hypothesis_type")
    strategy_purity, strategy_detail = factor_purity(
        rows, "validation_strategy"
    )
    case_purity, case_detail = factor_purity(rows, "source_case_id")

    alignment_counts = Counter(
        row["primary_alignment_diagnostic"]["disposition"]
        for row in rows
    )

    unique_profiles = len({row["profile_key"] for row in rows})

    if len(rows) < 2:
        profile_validity = "INSUFFICIENT_SAMPLE"
    elif (
        unique_profiles > 1
        and type_purity < 1.0
        and strategy_purity < 1.0
    ):
        profile_validity = "PROFILE_VARIATION_NOT_FULLY_TEMPLATE_EXPLAINED"
    elif unique_profiles > 1:
        profile_validity = "PROFILE_VARIATION_TEMPLATE_CONFOUNDED"
    else:
        profile_validity = "PROFILE_NON_DISCRIMINATIVE"

    near_only = alignment_counts.get(
        "LEXICALLY_NEAR_PRIMARY_ALIGNMENT_ONLY", 0
    )
    exact = alignment_counts.get(
        "EXACT_PRIMARY_ALIGNMENT_PRESENT", 0
    )

    if exact > 0:
        primary_diagnosis = "SOME_TRUE_PRIMARY_ALIGNMENT_PRESENT"
    elif near_only > 0:
        primary_diagnosis = "EXACT_STRING_MATCHING_LIKELY_TOO_STRICT"
    else:
        primary_diagnosis = "PRIMARY_ALIGNMENT_ABSENCE_NOT_EXPLAINED_LEXICALLY"

    payload = {
        "schema_version":
            "research-value-profile-validity-audit-s241-v1",
        "source_collector": str(collector_path),
        "source_s240_summary": str(s240_path),
        "card_count": len(rows),
        "unique_profile_count": unique_profiles,
        "profile_validity": profile_validity,
        "template_confounding": {
            "hypothesis_type_profile_purity": round(type_purity, 6),
            "validation_strategy_profile_purity":
                round(strategy_purity, 6),
            "source_case_profile_purity": round(case_purity, 6),
            "hypothesis_type_crosstab": type_detail,
            "validation_strategy_crosstab": strategy_detail,
            "source_case_crosstab": case_detail,
        },
        "primary_alignment": {
            "disposition_counts":
                dict(sorted(alignment_counts.items())),
            "diagnosis": primary_diagnosis,
        },
        "cards": rows,
        "interpretation_policy": {
            "profile_variation_is_value_validity_proof": False,
            "profile_variation_is_ranking_authority": False,
            "template_purity_is_causal_proof": False,
            "lexical_near_alignment_is_semantic_equivalence": False,
            "novelty_signal_consumed": False,
            "ranking_computed": False,
            "research_value_selection_authority": False,
            "production_selection_changed": False,
        },
    }

    write_json(output_path, payload)

    print("=== S241 RESEARCH VALUE PROFILE VALIDITY AUDIT ===")
    print("artifact only: true")
    print("cards:", len(rows))
    print("unique profiles:", unique_profiles)
    print(
        "hypothesis-type -> profile purity:",
        f"{type_purity:.3f}",
    )
    print(
        "validation-strategy -> profile purity:",
        f"{strategy_purity:.3f}",
    )
    print(
        "source-case -> profile purity:",
        f"{case_purity:.3f}",
    )
    print("profile validity:", profile_validity)
    print(
        "primary alignment dispositions:",
        dict(sorted(alignment_counts.items())),
    )
    print("primary alignment diagnosis:", primary_diagnosis)
    print("ranking computed: false")
    print("production selection changed: false")
    print("artifact:", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
