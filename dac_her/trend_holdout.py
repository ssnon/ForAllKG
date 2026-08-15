from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


TREND_HOLDOUT_SPLIT_SEMANTICS_ID = "trend_holdout_split_v1_alpha4c4a"
TREND_HOLDOUT_SELECTION_ALGORITHM = "sha256_namespace_paper_id_rank_v1"


def _clean_unique(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    rows = tuple(str(value).strip() for value in values)
    if any(not value for value in rows):
        raise ValueError(f"{label} must not contain empty paper IDs.")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{label} must contain unique paper IDs.")
    return rows


def deterministic_rank_sha256(*, namespace: str, paper_id: str) -> str:
    namespace = str(namespace).strip()
    paper_id = str(paper_id).strip()
    if not namespace:
        raise ValueError("split namespace must not be empty.")
    if not paper_id:
        raise ValueError("paper_id must not be empty.")
    return hashlib.sha256(
        f"{namespace}|{paper_id}".encode("utf-8")
    ).hexdigest()


def rank_candidate_papers(
    paper_ids: Iterable[str],
    *,
    namespace: str,
) -> tuple[dict[str, str], ...]:
    candidates = _clean_unique(
        paper_ids,
        label="candidate_papers",
    )
    rows = [
        {
            "paper_id": paper_id,
            "rank_sha256": deterministic_rank_sha256(
                namespace=namespace,
                paper_id=paper_id,
            ),
        }
        for paper_id in candidates
    ]
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row["rank_sha256"],
                row["paper_id"],
            ),
        )
    )


@dataclass(frozen=True)
class TrendHoldoutSplit:
    all_paper_ids: tuple[str, ...]
    development_calibration: tuple[str, ...]
    development_seen_regression: tuple[str, ...]
    candidate_papers: tuple[str, ...]
    ranked_candidates: tuple[dict[str, str], ...]
    holdout_papers: tuple[str, ...]
    reserved_future_papers: tuple[str, ...]
    namespace: str
    holdout_count: int

    @property
    def split_sha256(self) -> str:
        payload = {
            "split_semantics_id": TREND_HOLDOUT_SPLIT_SEMANTICS_ID,
            "selection_algorithm": TREND_HOLDOUT_SELECTION_ALGORITHM,
            "namespace": self.namespace,
            "holdout_count": self.holdout_count,
            "all_paper_ids": list(self.all_paper_ids),
            "development_calibration": list(self.development_calibration),
            "development_seen_regression": list(
                self.development_seen_regression
            ),
            "candidate_papers": list(self.candidate_papers),
            "ranked_candidates": list(self.ranked_candidates),
            "holdout_papers": list(self.holdout_papers),
            "reserved_future_papers": list(
                self.reserved_future_papers
            ),
        }
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def build_trend_holdout_split(
    *,
    all_paper_ids: Sequence[str],
    development_calibration: Sequence[str],
    development_seen_regression: Sequence[str],
    namespace: str,
    holdout_count: int,
) -> TrendHoldoutSplit:
    all_ids = _clean_unique(
        all_paper_ids,
        label="all_paper_ids",
    )
    calibration = _clean_unique(
        development_calibration,
        label="development_calibration",
    )
    seen = _clean_unique(
        development_seen_regression,
        label="development_seen_regression",
    )

    development = set(calibration) | set(seen)
    if set(calibration) & set(seen):
        raise ValueError(
            "calibration and seen-regression paper sets must be disjoint."
        )
    if not development.issubset(set(all_ids)):
        raise ValueError(
            "development papers must be drawn from all_paper_ids."
        )

    candidates = tuple(
        paper_id
        for paper_id in all_ids
        if paper_id not in development
    )
    if holdout_count < 1:
        raise ValueError("holdout_count must be >= 1.")
    if holdout_count >= len(candidates):
        raise ValueError(
            "holdout_count must leave at least one future-reserve paper."
        )

    ranked = rank_candidate_papers(
        candidates,
        namespace=namespace,
    )
    holdout = tuple(
        row["paper_id"]
        for row in ranked[:holdout_count]
    )
    reserve = tuple(
        row["paper_id"]
        for row in ranked[holdout_count:]
    )

    return TrendHoldoutSplit(
        all_paper_ids=all_ids,
        development_calibration=calibration,
        development_seen_regression=seen,
        candidate_papers=candidates,
        ranked_candidates=ranked,
        holdout_papers=holdout,
        reserved_future_papers=reserve,
        namespace=str(namespace),
        holdout_count=int(holdout_count),
    )


def validate_protocol_split(
    protocol: Mapping[str, object],
) -> TrendHoldoutSplit:
    selection = protocol.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("protocol.selection must be an object.")

    if (
        selection.get("split_semantics_id")
        != TREND_HOLDOUT_SPLIT_SEMANTICS_ID
    ):
        raise ValueError("trend holdout split semantics drifted.")
    if (
        selection.get("algorithm")
        != TREND_HOLDOUT_SELECTION_ALGORITHM
    ):
        raise ValueError("trend holdout selection algorithm drifted.")
    if selection.get("selection_inputs") != ["paper_id"]:
        raise ValueError(
            "alpha4c.4a selection must depend only on paper_id."
        )
    if selection.get("scientific_content_inspected_for_split") is not False:
        raise ValueError(
            "scientific content must not be used to derive the split."
        )
    if selection.get("trend_outputs_inspected_for_split") is not False:
        raise ValueError(
            "Trend outputs must not be used to derive the split."
        )

    papers = protocol.get("papers")
    if not isinstance(papers, Mapping):
        raise ValueError("protocol.papers must be an object.")

    split = build_trend_holdout_split(
        all_paper_ids=papers.get("curated_corpus", []),
        development_calibration=papers.get(
            "development_calibration", []
        ),
        development_seen_regression=papers.get(
            "development_seen_regression", []
        ),
        namespace=str(selection.get("namespace", "")),
        holdout_count=int(selection.get("holdout_count", 0)),
    )

    expected_candidates = list(split.candidate_papers)
    expected_ranked = list(split.ranked_candidates)
    expected_holdout = list(split.holdout_papers)
    expected_reserve = list(split.reserved_future_papers)

    checks = (
        (
            "candidate_papers",
            papers.get("candidate_papers"),
            expected_candidates,
        ),
        (
            "ranked_candidates",
            papers.get("ranked_candidates"),
            expected_ranked,
        ),
        (
            "frozen_holdout",
            papers.get("frozen_holdout"),
            expected_holdout,
        ),
        (
            "reserved_future",
            papers.get("reserved_future"),
            expected_reserve,
        ),
        (
            "split_sha256",
            selection.get("split_sha256"),
            split.split_sha256,
        ),
    )
    for label, observed, expected in checks:
        if observed != expected:
            raise ValueError(
                f"protocol split mismatch for {label}: "
                f"{observed!r} != {expected!r}"
            )

    acceptance = protocol.get("alpha4c4b_acceptance_policy")
    if not isinstance(acceptance, Mapping):
        raise ValueError(
            "protocol.alpha4c4b_acceptance_policy must be an object."
        )
    forbidden_thresholds = (
        "minimum_trend_evidence_count",
        "minimum_cross_paper_pair_count",
        "minimum_repeated_count",
        "minimum_reversed_count",
        "minimum_context_specific_count",
        "maximum_insufficient_count",
    )
    for key in forbidden_thresholds:
        if acceptance.get(key) is not None:
            raise ValueError(
                f"{key} must remain null; holdout distribution counts "
                "are observations, not success targets."
            )

    if acceptance.get("count_thresholds_used") is not False:
        raise ValueError(
            "count_thresholds_used must remain false."
        )
    return split
