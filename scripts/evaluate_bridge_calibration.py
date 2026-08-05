from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_CALIBRATION_DIR = Path("calibration/bridge_semantic")

RELATION_TRUE_VALUES = frozenset({"1", "true", "yes"})
GOLD_ACCEPT_DECISIONS = frozenset({
    "ACCEPT_AS_IS",
    "RELABEL_AND_ACCEPT",
})
GOLD_DECISIONS = frozenset({
    "ACCEPT_AS_IS",
    "RELABEL_AND_ACCEPT",
    "KEEP_REJECTED",
})

CONFIRMED_STATUSES = frozenset({"ACCEPTED_PATTERN"})
CANDIDATE_STATUSES = frozenset({
    "SEMANTIC_CANDIDATE",
    "CANDIDATE",
})
FATAL_REJECTION_STATUSES = frozenset({
    "REJECTED",
    "FATAL_REJECTION",
    "FATAL_REJECTED",
})


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm(value: Any) -> str:
    return " ".join(_text(value).lower().split())


def _is_true(value: Any) -> bool:
    return _norm(value) in RELATION_TRUE_VALUES


def _ratio(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {path}"
            )

        rows = [
            {
                str(key): (
                    "" if value is None else str(value)
                )
                for key, value in row.items()
            }
            for row in reader
        ]

    return rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8-sig",
        )
        return

    fields: list[str] = []

    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: _csv_value(
                    row.get(field, "")
                )
                for field in fields
            })


def _csv_value(value: Any) -> Any:
    if isinstance(
        value,
        (dict, list, tuple, set),
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return value


def _json_list(value: Any) -> list[Any]:
    text = _text(value)

    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        return parsed

    return []


def _int_value(value: Any) -> int:
    text = _text(value)

    if not text:
        return 0

    try:
        return int(float(text))
    except ValueError:
        return 0


def _relation_rows(
    rows: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if _is_true(
            row.get(
                "in_relation_calibration",
                "",
            )
        )
    ]


def _index_by_candidate(
    rows: Iterable[dict[str, str]],
    *,
    source_name: str,
) -> dict[str, dict[str, str]]:
    result: dict[
        str,
        dict[str, str],
    ] = {}

    for row in rows:
        key = _text(
            row.get(
                "candidate_key",
                "",
            )
        )

        if not key:
            raise ValueError(
                f"{source_name} contains a row "
                "without candidate_key."
            )

        if key in result:
            raise ValueError(
                f"{source_name} contains duplicate "
                f"candidate_key={key!r}."
            )

        result[key] = row

    return result


def _require_columns(
    rows: Sequence[dict[str, str]],
    columns: Iterable[str],
    *,
    source_name: str,
) -> None:
    if not rows:
        raise ValueError(
            f"{source_name} has no rows."
        )

    available = set(rows[0])

    missing = [
        column
        for column in columns
        if column not in available
    ]

    if missing:
        raise ValueError(
            f"{source_name} is missing columns: "
            f"{missing!r}"
        )


def _validate_gold(
    gold_rows: Sequence[dict[str, str]],
) -> None:
    unreviewed: list[str] = []
    invalid_decisions: list[
        tuple[str, str]
    ] = []

    for row in gold_rows:
        key = _text(
            row.get(
                "candidate_key",
                "",
            )
        )
        decision = _text(
            row.get(
                "manual_decision",
                "",
            )
        )

        if not decision:
            unreviewed.append(key)
            continue

        if decision not in GOLD_DECISIONS:
            invalid_decisions.append(
                (key, decision)
            )
            continue

        if decision == "RELABEL_AND_ACCEPT":
            relation = _text(
                row.get(
                    "manual_pattern_relation",
                    "",
                )
            )

            if not relation:
                raise ValueError(
                    "RELABEL_AND_ACCEPT requires "
                    "manual_pattern_relation for "
                    f"candidate {key!r}."
                )

    if unreviewed:
        raise ValueError(
            "Gold adjudication is incomplete. "
            f"Unreviewed relation candidates: "
            f"{len(unreviewed)}; examples: "
            f"{unreviewed[:10]!r}"
        )

    if invalid_decisions:
        raise ValueError(
            "Gold contains invalid manual_decision "
            f"values: {invalid_decisions[:10]!r}"
        )


def _gold_accepts(
    row: Mapping[str, str],
) -> bool:
    return _text(
        row.get(
            "manual_decision",
            "",
        )
    ) in GOLD_ACCEPT_DECISIONS


def _gold_tuple(
    row: Mapping[str, str],
) -> tuple[str, str, str] | None:
    decision = _text(
        row.get(
            "manual_decision",
            "",
        )
    )

    if decision == "KEEP_REJECTED":
        return None

    raw_subject = _text(
        row.get(
            "raw_pattern_subject",
            "",
        )
    )
    raw_relation = _text(
        row.get(
            "raw_pattern_relation",
            "",
        )
    )
    raw_object = _text(
        row.get(
            "raw_pattern_object",
            "",
        )
    )

    if decision == "ACCEPT_AS_IS":
        return (
            raw_subject,
            raw_relation,
            raw_object,
        )

    return (
        _text(
            row.get(
                "manual_pattern_subject",
                "",
            )
        )
        or raw_subject,
        _text(
            row.get(
                "manual_pattern_relation",
                "",
            )
        )
        or raw_relation,
        _text(
            row.get(
                "manual_pattern_object",
                "",
            )
        )
        or raw_object,
    )


def _predicted_tuple(
    row: Mapping[str, str],
) -> tuple[str, str, str]:
    return (
        _text(
            row.get(
                "effective_pattern_subject",
                "",
            )
        )
        or _text(
            row.get(
                "raw_pattern_subject",
                "",
            )
        ),
        _text(
            row.get(
                "effective_pattern_relation",
                "",
            )
        )
        or _text(
            row.get(
                "raw_pattern_relation",
                "",
            )
        ),
        _text(
            row.get(
                "effective_pattern_object",
                "",
            )
        )
        or _text(
            row.get(
                "raw_pattern_object",
                "",
            )
        ),
    )


def _same_text(
    left: Any,
    right: Any,
) -> bool:
    return _norm(left) == _norm(right)


def _same_argument_pair(
    left: tuple[str, str, str],
    right: tuple[str, str, str],
) -> bool:
    return (
        _same_text(left[0], right[0])
        and _same_text(left[2], right[2])
    )


def _same_pattern(
    left: tuple[str, str, str],
    right: tuple[str, str, str],
) -> bool:
    return (
        _same_argument_pair(left, right)
        and _same_text(left[1], right[1])
    )


def _automatic_lane(
    row: Mapping[str, str],
) -> str:
    status = _text(
        row.get(
            "automatic_status",
            "",
        )
    ).upper()

    if status in CONFIRMED_STATUSES:
        return "confirmed"

    if status in CANDIDATE_STATUSES:
        return "candidate"

    if status in FATAL_REJECTION_STATUSES:
        return "fatal_rejection"

    raise ValueError(
        "Unknown automatic_status "
        f"{status!r} for candidate "
        f"{row.get('candidate_key')!r}."
    )


def _policy_set_id(
    run_ids: Sequence[str],
) -> str:
    payload = "|".join(
        sorted(set(run_ids))
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:12]


def _evidence_pointer_count(
    row: Mapping[str, str],
) -> int:
    explicit = _int_value(
        row.get(
            "evidence_pointer_count",
            "",
        )
    )

    if explicit:
        return explicit

    return len(
        _json_list(
            row.get(
                "evidence_pointers_json",
                "",
            )
        )
    )


def _reason_codes(
    row: Mapping[str, str],
) -> list[str]:
    values: list[str] = []

    for field in (
        "candidate_reason_codes_json",
        "rejection_reason_codes_json",
    ):
        for value in _json_list(
            row.get(field, "")
        ):
            text = _text(value)

            if text and text not in values:
                values.append(text)

    return values


def _repair_rule_ids(
    row: Mapping[str, str],
) -> list[str]:
    return [
        _text(value)
        for value in _json_list(
            row.get(
                "repair_rule_ids_json",
                "",
            )
        )
        if _text(value)
    ]


def _group_metrics(
    buckets: Mapping[
        str,
        Counter[str],
    ],
) -> dict[str, dict[str, Any]]:
    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for key, bucket in sorted(
        buckets.items()
    ):
        confirmed_tp = int(
            bucket["confirmed_tp"]
        )
        confirmed_fp = int(
            bucket["confirmed_fp"]
        )
        gold_accept = int(
            bucket["gold_accept"]
        )
        candidate_valid = int(
            bucket["candidate_valid"]
        )
        candidate_invalid = int(
            bucket["candidate_invalid"]
        )
        fatal_correct = int(
            bucket["fatal_correct"]
        )
        fatal_fn = int(
            bucket["fatal_fn"]
        )

        candidate_total = (
            candidate_valid
            + candidate_invalid
        )
        fatal_total = (
            fatal_correct
            + fatal_fn
        )

        result[key] = {
            **{
                name: int(value)
                for name, value
                in bucket.items()
            },
            "confirmed_precision": _ratio(
                confirmed_tp,
                confirmed_tp
                + confirmed_fp,
            ),
            "confirmed_recall": _ratio(
                confirmed_tp,
                gold_accept,
            ),
            "candidate_validity": _ratio(
                candidate_valid,
                candidate_total,
            ),
            (
                "confirmed_plus_"
                "candidate_recall"
            ): _ratio(
                confirmed_tp
                + candidate_valid,
                gold_accept,
            ),
            (
                "fatal_rejection_"
                "precision"
            ): _ratio(
                fatal_correct,
                fatal_total,
            ),
            (
                "fatal_rejection_"
                "false_negative_share"
            ): _ratio(
                fatal_fn,
                fatal_total,
            ),
        }

    return result


def evaluate(
    *,
    predictions_path: Path,
    gold_path: Path,
    reports_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Path,
    Path,
]:
    prediction_all = _read_csv(
        predictions_path
    )
    gold_all = _read_csv(
        gold_path
    )

    _require_columns(
        prediction_all,
        (
            "candidate_key",
            "in_relation_calibration",
            "automatic_status",
            "bridge_policy_run_id",
        ),
        source_name="predictions.csv",
    )
    _require_columns(
        gold_all,
        (
            "candidate_key",
            "in_relation_calibration",
            "manual_decision",
            "manual_anchor_correct",
        ),
        source_name="gold.csv",
    )

    prediction_all_by_key = (
        _index_by_candidate(
            prediction_all,
            source_name="predictions.csv",
        )
    )
    gold_all_by_key = (
        _index_by_candidate(
            gold_all,
            source_name="gold.csv",
        )
    )

    prediction_keys = set(
        prediction_all_by_key
    )
    gold_keys = set(
        gold_all_by_key
    )

    if prediction_keys != gold_keys:
        missing_in_predictions = sorted(
            gold_keys - prediction_keys
        )
        missing_in_gold = sorted(
            prediction_keys - gold_keys
        )

        raise ValueError(
            "Prediction/gold candidate sets differ. "
            f"Missing in predictions: "
            f"{missing_in_predictions[:10]!r}; "
            f"missing in gold: "
            f"{missing_in_gold[:10]!r}."
        )

    prediction_rows = _relation_rows(
        prediction_all
    )
    gold_rows = _relation_rows(
        gold_all
    )

    prediction_by_key = (
        _index_by_candidate(
            prediction_rows,
            source_name=(
                "relation predictions"
            ),
        )
    )
    gold_by_key = _index_by_candidate(
        gold_rows,
        source_name="relation gold",
    )

    if set(prediction_by_key) != set(
        gold_by_key
    ):
        raise ValueError(
            "Prediction/gold relation-candidate "
            "sets differ."
        )

    _validate_gold(gold_rows)

    confirmed_tp = 0
    confirmed_fp = 0
    confirmed_fn = 0
    confirmed_tn = 0

    candidate_valid = 0
    candidate_invalid = 0

    fatal_rejection_correct = 0
    fatal_rejection_false_negative = 0

    confirmed_gold_accept_count = 0
    confirmed_relation_correct = 0
    confirmed_subject_correct = 0
    confirmed_object_correct = 0
    confirmed_argument_pair_correct = 0
    confirmed_full_pattern_correct = 0

    candidate_gold_accept_count = 0
    candidate_relation_correct = 0
    candidate_argument_pair_correct = 0
    candidate_full_pattern_correct = 0

    combined_gold_accept_count = 0
    combined_relation_correct = 0
    combined_argument_pair_correct = 0
    combined_full_pattern_correct = 0

    repair_applied = 0
    repair_gold_correct = 0

    anchor_yes = 0
    anchor_no = 0
    anchor_uncertain = 0

    confirmed_without_pointers = 0
    candidate_without_pointers = 0

    status_counts: Counter[str] = (
        Counter()
    )
    error_type_counts: Counter[str] = (
        Counter()
    )
    reason_code_counts: Counter[str] = (
        Counter()
    )
    false_positive_by_relation: (
        Counter[str]
    ) = Counter()

    per_paper: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    per_relation: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    error_rows: list[
        dict[str, Any]
    ] = []

    for key in sorted(gold_by_key):
        prediction = prediction_by_key[key]
        gold = gold_by_key[key]

        lane = _automatic_lane(
            prediction
        )
        status_counts[lane] += 1

        gold_accept = _gold_accepts(
            gold
        )
        gold_tuple = _gold_tuple(gold)
        predicted_tuple = (
            _predicted_tuple(
                prediction
            )
        )

        paper_id = _text(
            prediction.get(
                "paper_id",
                gold.get(
                    "paper_id",
                    "",
                ),
            )
        )
        paper_bucket = per_paper[
            paper_id
        ]
        paper_bucket[
            "relation_candidates"
        ] += 1

        if gold_accept:
            paper_bucket[
                "gold_accept"
            ] += 1
        else:
            paper_bucket[
                "gold_reject"
            ] += 1

        if lane == "confirmed":
            paper_bucket[
                "confirmed"
            ] += 1

            if gold_accept:
                confirmed_tp += 1
                paper_bucket[
                    "confirmed_tp"
                ] += 1
            else:
                confirmed_fp += 1
                paper_bucket[
                    "confirmed_fp"
                ] += 1

        else:
            if gold_accept:
                confirmed_fn += 1
            else:
                confirmed_tn += 1

        if lane == "candidate":
            paper_bucket[
                "candidate"
            ] += 1

            if gold_accept:
                candidate_valid += 1
                paper_bucket[
                    "candidate_valid"
                ] += 1
            else:
                candidate_invalid += 1
                paper_bucket[
                    "candidate_invalid"
                ] += 1

        if lane == "fatal_rejection":
            paper_bucket[
                "fatal_reject"
            ] += 1

            if gold_accept:
                fatal_rejection_false_negative += 1
                paper_bucket[
                    "fatal_fn"
                ] += 1
            else:
                fatal_rejection_correct += 1
                paper_bucket[
                    "fatal_correct"
                ] += 1

        relation_is_correct = False
        subject_is_correct = False
        object_is_correct = False
        argument_pair_is_correct = False
        full_pattern_is_correct = False

        if gold_accept:
            assert gold_tuple is not None

            subject_is_correct = _same_text(
                predicted_tuple[0],
                gold_tuple[0],
            )
            relation_is_correct = _same_text(
                predicted_tuple[1],
                gold_tuple[1],
            )
            object_is_correct = _same_text(
                predicted_tuple[2],
                gold_tuple[2],
            )
            argument_pair_is_correct = (
                _same_argument_pair(
                    predicted_tuple,
                    gold_tuple,
                )
            )
            full_pattern_is_correct = (
                _same_pattern(
                    predicted_tuple,
                    gold_tuple,
                )
            )

            gold_relation = gold_tuple[1]
            relation_bucket = per_relation[
                gold_relation
            ]
            relation_bucket[
                "gold_accept"
            ] += 1

            if lane == "confirmed":
                relation_bucket[
                    "confirmed_predictions"
                ] += 1
                relation_bucket[
                    "confirmed_any"
                ] += 1

                if relation_is_correct:
                    relation_bucket[
                        "confirmed_correct"
                    ] += 1

            elif lane == "candidate":
                relation_bucket[
                    "candidate_predictions"
                ] += 1
                relation_bucket[
                    "candidate_any"
                ] += 1

                if relation_is_correct:
                    relation_bucket[
                        "candidate_correct"
                    ] += 1

            elif lane == "fatal_rejection":
                relation_bucket[
                    "fatal_rejections"
                ] += 1

        if lane == "confirmed" and gold_accept:
            confirmed_gold_accept_count += 1
            confirmed_relation_correct += int(
                relation_is_correct
            )
            confirmed_subject_correct += int(
                subject_is_correct
            )
            confirmed_object_correct += int(
                object_is_correct
            )
            confirmed_argument_pair_correct += int(
                argument_pair_is_correct
            )
            confirmed_full_pattern_correct += int(
                full_pattern_is_correct
            )

        if lane == "candidate" and gold_accept:
            candidate_gold_accept_count += 1
            candidate_relation_correct += int(
                relation_is_correct
            )
            candidate_argument_pair_correct += int(
                argument_pair_is_correct
            )
            candidate_full_pattern_correct += int(
                full_pattern_is_correct
            )

        if (
            lane in {
                "confirmed",
                "candidate",
            }
            and gold_accept
        ):
            combined_gold_accept_count += 1
            combined_relation_correct += int(
                relation_is_correct
            )
            combined_argument_pair_correct += int(
                argument_pair_is_correct
            )
            combined_full_pattern_correct += int(
                full_pattern_is_correct
            )

        pointer_count = (
            _evidence_pointer_count(
                prediction
            )
        )

        if (
            lane == "confirmed"
            and gold_accept
            and pointer_count == 0
        ):
            confirmed_without_pointers += 1

        if (
            lane == "candidate"
            and pointer_count == 0
        ):
            candidate_without_pointers += 1

        if gold_accept:
            anchor = _norm(
                gold.get(
                    "manual_anchor_correct",
                    "",
                )
            )

            if anchor == "yes":
                anchor_yes += 1
            elif anchor == "no":
                anchor_no += 1
            else:
                anchor_uncertain += 1

        repair_ids = _repair_rule_ids(
            prediction
        )

        if repair_ids:
            repair_applied += 1

            if (
                gold_accept
                and full_pattern_is_correct
            ):
                repair_gold_correct += 1

        reason_codes = _reason_codes(
            prediction
        )

        for reason in reason_codes:
            reason_code_counts[reason] += 1

        error_types: list[str] = []

        if (
            lane == "confirmed"
            and not gold_accept
        ):
            error_types.append(
                "CONFIRMED_FALSE_POSITIVE"
            )
            false_positive_by_relation[
                predicted_tuple[1]
            ] += 1

        if (
            lane == "candidate"
            and gold_accept
        ):
            error_types.append(
                (
                    "CONFIRMED_MISS_"
                    "RECOVERED_AS_CANDIDATE"
                )
            )

        if (
            lane == "fatal_rejection"
            and gold_accept
        ):
            error_types.append(
                "FATAL_FALSE_NEGATIVE"
            )

        if (
            lane == "candidate"
            and not gold_accept
        ):
            error_types.append(
                "INVALID_SEMANTIC_CANDIDATE"
            )

        if (
            lane == "confirmed"
            and gold_accept
            and not relation_is_correct
        ):
            error_types.append(
                "RELATION_LABEL_ERROR"
            )

        if (
            lane == "confirmed"
            and gold_accept
            and not argument_pair_is_correct
        ):
            error_types.append(
                "ARGUMENT_TUPLE_ERROR"
            )

        if (
            lane == "confirmed"
            and gold_accept
            and not subject_is_correct
        ):
            error_types.append(
                "ARGUMENT_SUBJECT_ERROR"
            )

        if (
            lane == "confirmed"
            and gold_accept
            and not object_is_correct
        ):
            error_types.append(
                "ARGUMENT_OBJECT_ERROR"
            )

        if (
            repair_ids
            and not (
                gold_accept
                and full_pattern_is_correct
            )
        ):
            error_types.append(
                "REPAIR_NOT_GOLD_CORRECT"
            )

        if (
            gold_accept
            and _norm(
                gold.get(
                    "manual_anchor_correct",
                    "",
                )
            )
            == "no"
        ):
            error_types.append(
                "MANUAL_ANCHOR_ERROR"
            )

        if (
            lane == "confirmed"
            and gold_accept
            and pointer_count == 0
        ):
            error_types.append(
                "MISSING_EVIDENCE_POINTER"
            )

        if (
            lane == "candidate"
            and pointer_count == 0
        ):
            error_types.append(
                (
                    "CANDIDATE_MISSING_"
                    "EVIDENCE_POINTER"
                )
            )

        if not error_types:
            continue

        error_types = list(
            dict.fromkeys(error_types)
        )

        for error_type in error_types:
            error_type_counts[
                error_type
            ] += 1

        error_rows.append({
            "candidate_key": key,
            "paper_id": paper_id,
            "chunk_id": _text(
                prediction.get(
                    "chunk_id",
                    "",
                )
            ),
            "concept_id": _text(
                prediction.get(
                    "concept_id",
                    "",
                )
            ),
            "automatic_status": _text(
                prediction.get(
                    "automatic_status",
                    "",
                )
            ),
            "policy_lane": lane,
            "manual_decision": _text(
                gold.get(
                    "manual_decision",
                    "",
                )
            ),
            "error_types": ";".join(
                error_types
            ),
            "raw_subject": _text(
                prediction.get(
                    "raw_pattern_subject",
                    "",
                )
            ),
            "raw_relation": _text(
                prediction.get(
                    "raw_pattern_relation",
                    "",
                )
            ),
            "raw_object": _text(
                prediction.get(
                    "raw_pattern_object",
                    "",
                )
            ),
            "effective_subject": (
                predicted_tuple[0]
            ),
            "effective_relation": (
                predicted_tuple[1]
            ),
            "effective_object": (
                predicted_tuple[2]
            ),
            "manual_subject": _text(
                gold.get(
                    "manual_pattern_subject",
                    "",
                )
            ),
            "manual_relation": _text(
                gold.get(
                    "manual_pattern_relation",
                    "",
                )
            ),
            "manual_object": _text(
                gold.get(
                    "manual_pattern_object",
                    "",
                )
            ),
            "gold_subject": (
                gold_tuple[0]
                if gold_tuple
                else ""
            ),
            "gold_relation": (
                gold_tuple[1]
                if gold_tuple
                else ""
            ),
            "gold_object": (
                gold_tuple[2]
                if gold_tuple
                else ""
            ),
            "manual_reason": _text(
                gold.get(
                    "manual_reason",
                    "",
                )
            ),
            "source_phrase": _text(
                prediction.get(
                    "source_phrase",
                    "",
                )
            ),
            (
                "candidate_reason_codes_json"
            ): json.dumps(
                _json_list(
                    prediction.get(
                        (
                            "candidate_reason_"
                            "codes_json"
                        ),
                        "",
                    )
                ),
                ensure_ascii=False,
            ),
            (
                "rejection_reason_codes_json"
            ): json.dumps(
                _json_list(
                    prediction.get(
                        (
                            "rejection_reason_"
                            "codes_json"
                        ),
                        "",
                    )
                ),
                ensure_ascii=False,
            ),
            "repair_rule_ids_json": (
                json.dumps(
                    repair_ids,
                    ensure_ascii=False,
                )
            ),
            "evidence_pointer_count": (
                pointer_count
            ),
            "manual_anchor_correct": (
                _text(
                    gold.get(
                        (
                            "manual_anchor_"
                            "correct"
                        ),
                        "",
                    )
                )
            ),
        })

    run_ids = sorted({
        _text(
            row.get(
                "bridge_policy_run_id",
                "",
            )
        )
        for row in prediction_rows
        if _text(
            row.get(
                "bridge_policy_run_id",
                "",
            )
        )
    })

    if not run_ids:
        raise ValueError(
            "No bridge_policy_run_id values "
            "were found in predictions."
        )

    policy_set_id = _policy_set_id(
        run_ids
    )

    relation_count = len(
        prediction_rows
    )
    gold_accept_total = (
        confirmed_tp
        + confirmed_fn
    )
    candidate_total = (
        candidate_valid
        + candidate_invalid
    )
    fatal_total = (
        fatal_rejection_correct
        + fatal_rejection_false_negative
    )

    per_relation_metrics: dict[
        str,
        dict[str, Any],
    ] = {}

    for relation, bucket in sorted(
        per_relation.items()
    ):
        gold_count = int(
            bucket["gold_accept"]
        )
        confirmed_any = int(
            bucket["confirmed_any"]
        )
        candidate_any = int(
            bucket["candidate_any"]
        )

        per_relation_metrics[
            relation
        ] = {
            **{
                name: int(value)
                for name, value
                in bucket.items()
            },
            "confirmed_recall": _ratio(
                confirmed_any,
                gold_count,
            ),
            (
                "confirmed_relation_"
                "label_recall"
            ): _ratio(
                int(
                    bucket[
                        "confirmed_correct"
                    ]
                ),
                gold_count,
            ),
            (
                "confirmed_plus_"
                "candidate_recall"
            ): _ratio(
                confirmed_any
                + candidate_any,
                gold_count,
            ),
            (
                "confirmed_plus_"
                "candidate_relation_"
                "label_recall"
            ): _ratio(
                int(
                    bucket[
                        "confirmed_correct"
                    ]
                )
                + int(
                    bucket[
                        "candidate_correct"
                    ]
                ),
                gold_count,
            ),
        }

    report: dict[str, Any] = {
        "policy_set_id": policy_set_id,
        "bridge_policy_run_ids": run_ids,
        "relation_candidates": (
            relation_count
        ),
        "automatic_lane_counts": {
            "confirmed": int(
                status_counts["confirmed"]
            ),
            "semantic_candidate": int(
                status_counts["candidate"]
            ),
            "fatal_rejection": int(
                status_counts[
                    "fatal_rejection"
                ]
            ),
        },
        "confusion_matrix": {
            "true_positive": (
                confirmed_tp
            ),
            "false_positive": (
                confirmed_fp
            ),
            "false_negative": (
                confirmed_fn
            ),
            "true_negative": (
                confirmed_tn
            ),
        },
        "confirmed_precision": _ratio(
            confirmed_tp,
            confirmed_tp
            + confirmed_fp,
        ),
        "confirmed_recall": _ratio(
            confirmed_tp,
            gold_accept_total,
        ),
        (
            "confirmed_binary_accuracy"
        ): _ratio(
            confirmed_tp
            + confirmed_tn,
            relation_count,
        ),
        # Compatibility aliases.
        "accepted_precision": _ratio(
            confirmed_tp,
            confirmed_tp
            + confirmed_fp,
        ),
        "accepted_recall": _ratio(
            confirmed_tp,
            gold_accept_total,
        ),
        "policy_decision_accuracy": (
            _ratio(
                confirmed_tp
                + confirmed_tn,
                relation_count,
            )
        ),
        "semantic_candidates": {
            "count": candidate_total,
            "gold_valid": (
                candidate_valid
            ),
            "gold_invalid": (
                candidate_invalid
            ),
            "validity": _ratio(
                candidate_valid,
                candidate_total,
            ),
            "recovery_count": (
                candidate_valid
            ),
        },
        (
            "confirmed_plus_"
            "candidate_recall"
        ): _ratio(
            confirmed_tp
            + candidate_valid,
            gold_accept_total,
        ),
        "fatal_rejections": {
            "count": fatal_total,
            "gold_correct": (
                fatal_rejection_correct
            ),
            "false_negative": (
                fatal_rejection_false_negative
            ),
            "precision": _ratio(
                fatal_rejection_correct,
                fatal_total,
            ),
            "false_negative_share": (
                _ratio(
                    (
                        fatal_rejection_false_negative
                    ),
                    fatal_total,
                )
            ),
        },
        (
            "fatal_rejection_precision"
        ): _ratio(
            fatal_rejection_correct,
            fatal_total,
        ),
        (
            "fatal_rejection_false_"
            "negative_count"
        ): (
            fatal_rejection_false_negative
        ),
        (
            "fatal_rejection_false_"
            "negative_share"
        ): _ratio(
            fatal_rejection_false_negative,
            fatal_total,
        ),
        (
            "accepted_relation_"
            "label_accuracy"
        ): _ratio(
            confirmed_relation_correct,
            confirmed_gold_accept_count,
        ),
        (
            "accepted_argument_"
            "subject_accuracy"
        ): _ratio(
            confirmed_subject_correct,
            confirmed_gold_accept_count,
        ),
        (
            "accepted_argument_"
            "object_accuracy"
        ): _ratio(
            confirmed_object_correct,
            confirmed_gold_accept_count,
        ),
        (
            "accepted_argument_"
            "tuple_accuracy"
        ): _ratio(
            confirmed_argument_pair_correct,
            confirmed_gold_accept_count,
        ),
        # Historical alias: the former
        # metric compared the whole argument
        # tuple, not direction alone.
        (
            "accepted_argument_"
            "direction_accuracy"
        ): _ratio(
            confirmed_argument_pair_correct,
            confirmed_gold_accept_count,
        ),
        (
            "accepted_full_pattern_accuracy"
        ): _ratio(
            confirmed_full_pattern_correct,
            confirmed_gold_accept_count,
        ),
        (
            "candidate_relation_"
            "label_accuracy"
        ): _ratio(
            candidate_relation_correct,
            candidate_gold_accept_count,
        ),
        (
            "candidate_argument_"
            "tuple_accuracy"
        ): _ratio(
            candidate_argument_pair_correct,
            candidate_gold_accept_count,
        ),
        (
            "candidate_full_pattern_accuracy"
        ): _ratio(
            candidate_full_pattern_correct,
            candidate_gold_accept_count,
        ),
        (
            "confirmed_plus_candidate_"
            "relation_label_accuracy"
        ): _ratio(
            combined_relation_correct,
            combined_gold_accept_count,
        ),
        (
            "confirmed_plus_candidate_"
            "argument_tuple_accuracy"
        ): _ratio(
            combined_argument_pair_correct,
            combined_gold_accept_count,
        ),
        (
            "confirmed_plus_candidate_"
            "full_pattern_accuracy"
        ): _ratio(
            combined_full_pattern_correct,
            combined_gold_accept_count,
        ),
        "deterministic_repairs": {
            "applied": repair_applied,
            "gold_correct": (
                repair_gold_correct
            ),
            "precision": _ratio(
                repair_gold_correct,
                repair_applied,
            ),
        },
        "manual_anchor_review": {
            "correct": anchor_yes,
            "incorrect": anchor_no,
            "uncertain": anchor_uncertain,
            "accuracy": _ratio(
                anchor_yes,
                anchor_yes + anchor_no,
            ),
        },
        (
            "accepted_without_"
            "evidence_pointers"
        ): confirmed_without_pointers,
        (
            "semantic_candidates_"
            "without_evidence_pointers"
        ): candidate_without_pointers,
        "error_count": len(
            error_rows
        ),
        "error_type_counts": dict(
            sorted(
                error_type_counts.items()
            )
        ),
        "policy_reason_code_counts": (
            dict(
                sorted(
                    reason_code_counts.items()
                )
            )
        ),
        (
            "confirmed_false_"
            "positives_by_relation"
        ): dict(
            sorted(
                false_positive_by_relation
                .items()
            )
        ),
        "per_paper_metrics": (
            _group_metrics(
                per_paper
            )
        ),
        "per_relation_metrics": (
            per_relation_metrics
        ),
    }

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        reports_dir
        / f"report_{policy_set_id}.json"
    )
    errors_path = (
        reports_dir
        / f"errors_{policy_set_id}.csv"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_csv(
        errors_path,
        error_rows,
    )

    return (
        report,
        error_rows,
        report_path,
        errors_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Bridge "
            "semantic calibration gold set "
            "against confirmed, semantic-"
            "candidate, and fatal-rejection "
            "policy lanes."
        )
    )

    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=(
            DEFAULT_CALIBRATION_DIR
        ),
        help=(
            "Directory containing "
            "predictions.csv and gold.csv."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Optional explicit predictions "
            "CSV path."
        ),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help=(
            "Optional explicit gold CSV path."
        ),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for "
            "report JSON and error CSV."
        ),
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    calibration_dir = (
        args.calibration_dir
        .expanduser()
        .resolve()
    )

    predictions_path = (
        args.predictions
        .expanduser()
        .resolve()
        if args.predictions
        else (
            calibration_dir
            / "predictions.csv"
        )
    )
    gold_path = (
        args.gold
        .expanduser()
        .resolve()
        if args.gold
        else (
            calibration_dir
            / "gold.csv"
        )
    )
    reports_dir = (
        args.reports_dir
        .expanduser()
        .resolve()
        if args.reports_dir
        else (
            calibration_dir
            / "reports"
        )
    )

    report, _, report_path, errors_path = (
        evaluate(
            predictions_path=(
                predictions_path
            ),
            gold_path=gold_path,
            reports_dir=reports_dir,
        )
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Report: {report_path}")
    print(f"Errors: {errors_path}")


if __name__ == "__main__":
    main()
