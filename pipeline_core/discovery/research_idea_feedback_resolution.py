from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedbackResolutionCandidate(StrictModel):
    kind: Literal["RESIDUAL", "PROSPECTIVE"]
    path: str
    schema_version: str
    lineage_key: str
    semantic_digest: str


class FeedbackResolutionReport(StrictModel):
    schema_version: Literal[
        "research-idea-feedback-resolution-report-v1"
    ] = "research-idea-feedback-resolution-report-v1"

    source_materialization_report_id: str
    source_context_id: str
    output_portfolio_id: str
    materialized_hypothesis_ids: list[str] = Field(default_factory=list)
    search_roots: list[str] = Field(default_factory=list)

    residual_candidates: list[FeedbackResolutionCandidate] = Field(default_factory=list)
    residual_selected_path: str | None = None
    residual_selected_report_id: str | None = None
    residual_ambiguous: bool = False

    prospective_candidates: list[FeedbackResolutionCandidate] = Field(default_factory=list)
    prospective_selected_paths_by_hypothesis: dict[str, str] = Field(default_factory=dict)
    prospective_ambiguous_hypothesis_ids: list[str] = Field(default_factory=list)
    prospective_unmatched_hypothesis_ids: list[str] = Field(default_factory=list)

    exact_lineage_only: Literal[True] = True
    ambiguous_feedback_attached: Literal[False] = False
    external_prior_art_as_positive_premise: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ResolvedFeedbackArtifacts(StrictModel):
    report: FeedbackResolutionReport
    residual_state: dict[str, Any] | None = None
    prospective_by_hypothesis: dict[str, dict[str, Any]] = Field(default_factory=dict)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _materialization_hypothesis_ids(materialization: Any) -> list[str]:
    return list(dict.fromkeys(
        str(row.hypothesis_id)
        for row in materialization.records
        if str(row.status) == "MATERIALIZED" and row.hypothesis_id is not None
    ))


def _prospective_semantic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "source_hypothesis_id",
        "hypothesis_id",
        "source_context_id",
        "status",
        "contract_integrity_passed",
        "contract_integrity_reasons",
        "current_evidence_status",
        "prospective_identifiability",
        "directionality_mode",
        "measurement_compatibility_mode",
        "would_abstain_if_authoritative",
        "positive_premise_statement_ids",
        "gap_statement_ids",
    )
    return {key: payload.get(key) for key in fields if key in payload}


def _candidate_paths(search_roots: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for raw_root in search_roots:
        root = raw_root.expanduser().resolve()
        if not root.exists():
            continue
        iterator = [root] if root.is_file() else root.rglob("*.json")
        for path in iterator:
            if path in seen or not path.is_file():
                continue
            lower = str(path).lower()
            if "epistemic_state" not in lower and "prospective" not in lower:
                continue
            seen.add(path)
            paths.append(path)
    return sorted(paths)


def resolve_feedback_artifacts(
    *,
    materialization: Any,
    search_roots: Sequence[Path],
) -> ResolvedFeedbackArtifacts:
    output_portfolio_id = str(
        getattr(materialization, "output_portfolio_id", "") or ""
    )
    source_context_id = str(
        getattr(materialization, "source_context_id", "") or ""
    )
    report_id = str(getattr(materialization, "report_id", "") or "")
    hypothesis_ids = _materialization_hypothesis_ids(materialization)
    hypothesis_set = set(hypothesis_ids)

    residual_candidates: list[tuple[FeedbackResolutionCandidate, dict[str, Any]]] = []
    prospective_candidates: dict[
        str, list[tuple[FeedbackResolutionCandidate, dict[str, Any]]]
    ] = {hid: [] for hid in hypothesis_ids}

    for path in _candidate_paths(search_roots):
        payload = _load(path)
        if payload is None:
            continue
        schema = str(payload.get("schema_version") or "")

        if schema == "scientific-portfolio-residual-epistemic-state-v1":
            if not output_portfolio_id:
                continue
            if str(payload.get("source_portfolio_id") or "") != output_portfolio_id:
                continue
            candidate = FeedbackResolutionCandidate(
                kind="RESIDUAL",
                path=str(path),
                schema_version=schema,
                lineage_key=output_portfolio_id,
                semantic_digest=_digest(payload),
            )
            residual_candidates.append((candidate, payload))
            continue

        if schema != "prospective-identification-materialization-shadow-v1":
            continue
        hid = str(payload.get("source_hypothesis_id") or "")
        if hid not in hypothesis_set:
            continue
        payload_context = str(payload.get("source_context_id") or "")
        if source_context_id and payload_context != source_context_id:
            continue
        candidate = FeedbackResolutionCandidate(
            kind="PROSPECTIVE",
            path=str(path),
            schema_version=schema,
            lineage_key=f"{source_context_id}|{hid}",
            semantic_digest=_digest(_prospective_semantic_payload(payload)),
        )
        prospective_candidates[hid].append((candidate, payload))

    residual_selected: dict[str, Any] | None = None
    residual_selected_path: str | None = None
    residual_selected_report_id: str | None = None
    residual_ambiguous = False
    if residual_candidates:
        by_digest: dict[str, list[tuple[FeedbackResolutionCandidate, dict[str, Any]]]] = {}
        for item in residual_candidates:
            by_digest.setdefault(item[0].semantic_digest, []).append(item)
        if len(by_digest) == 1:
            selected_group = next(iter(by_digest.values()))
            selected_group.sort(key=lambda item: item[0].path)
            selected_candidate, residual_selected = selected_group[0]
            residual_selected_path = selected_candidate.path
            residual_selected_report_id = str(residual_selected.get("report_id") or "") or None
        else:
            residual_ambiguous = True

    prospective_selected: dict[str, dict[str, Any]] = {}
    prospective_selected_paths: dict[str, str] = {}
    prospective_ambiguous: list[str] = []
    prospective_unmatched: list[str] = []
    flattened_prospective_candidates: list[FeedbackResolutionCandidate] = []
    for hid in hypothesis_ids:
        rows = prospective_candidates.get(hid, [])
        flattened_prospective_candidates.extend(candidate for candidate, _ in rows)
        if not rows:
            prospective_unmatched.append(hid)
            continue
        by_digest: dict[str, list[tuple[FeedbackResolutionCandidate, dict[str, Any]]]] = {}
        for item in rows:
            by_digest.setdefault(item[0].semantic_digest, []).append(item)
        if len(by_digest) != 1:
            prospective_ambiguous.append(hid)
            continue
        selected_group = next(iter(by_digest.values()))
        selected_group.sort(key=lambda item: item[0].path)
        candidate, payload = selected_group[0]
        prospective_selected[hid] = payload
        prospective_selected_paths[hid] = candidate.path

    resolution_report = FeedbackResolutionReport(
        source_materialization_report_id=report_id,
        source_context_id=source_context_id,
        output_portfolio_id=output_portfolio_id,
        materialized_hypothesis_ids=hypothesis_ids,
        search_roots=[str(path.expanduser().resolve()) for path in search_roots],
        residual_candidates=[row[0] for row in residual_candidates],
        residual_selected_path=residual_selected_path,
        residual_selected_report_id=residual_selected_report_id,
        residual_ambiguous=residual_ambiguous,
        prospective_candidates=flattened_prospective_candidates,
        prospective_selected_paths_by_hypothesis=prospective_selected_paths,
        prospective_ambiguous_hypothesis_ids=prospective_ambiguous,
        prospective_unmatched_hypothesis_ids=prospective_unmatched,
    )
    return ResolvedFeedbackArtifacts(
        report=resolution_report,
        residual_state=residual_selected,
        prospective_by_hypothesis=prospective_selected,
    )


__all__ = [
    "FeedbackResolutionCandidate",
    "FeedbackResolutionReport",
    "ResolvedFeedbackArtifacts",
    "resolve_feedback_artifacts",
]
