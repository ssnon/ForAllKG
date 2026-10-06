from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_search_contracts import (
    IdeaFeedbackCoverageAudit,
    IdeaOutcomeObservation,
    IdeaPriorityBand,
    IdeaSearchDirective,
)


_BAND_RANK = {
    "DEFER": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _stable_id(prefix: str, *parts: object) -> str:
    raw = _canonical(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return dict(value) if isinstance(value, Mapping) else {}


def _residual_index(report: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    return {
        str(row.get("hypothesis_id")): dict(row)
        for row in report.get("hypotheses", [])
        if isinstance(row, Mapping) and row.get("hypothesis_id")
    }


def _prospective_index(
    value: Any,
    *,
    expected_context_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")

    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        if "source_hypothesis_id" in value:
            rows = [value]
        elif isinstance(value.get("rows"), list):
            rows = value.get("rows", [])
        elif isinstance(value.get("artifacts"), list):
            rows = value.get("artifacts", [])
        elif isinstance(value.get("records"), list):
            rows = value.get("records", [])
        else:
            result: dict[str, dict[str, Any]] = {}
            for key, row in value.items():
                payload = _as_dict(row)
                if not payload:
                    continue
                source_context_id = str(
                    payload.get("source_context_id") or ""
                )
                if (
                    expected_context_id
                    and source_context_id
                    and source_context_id != expected_context_id
                ):
                    continue
                result[str(key)] = payload
            return result
    else:
        return {}

    result = {}
    for row in rows:
        payload = _as_dict(row)
        source_context_id = str(payload.get("source_context_id") or "")
        if (
            expected_context_id
            and source_context_id
            and source_context_id != expected_context_id
        ):
            continue
        hid = str(
            payload.get("source_hypothesis_id")
            or payload.get("hypothesis_id")
            or ""
        )
        if hid:
            result[hid] = payload
    return result


def candidate_to_idea_index(
    *,
    pool: Any,
    nodes: Sequence[ResearchIdeaNode],
) -> dict[str, str]:
    idea_by_source_object = {
        node.source_object_id: node.idea_id
        for node in nodes
    }
    result: dict[str, str] = {}
    for candidate in pool.candidates:
        source_object_id = str(candidate.source_object_id)
        idea_id = idea_by_source_object.get(source_object_id)
        if idea_id is None:
            raise ValueError(
                "scientific portfolio candidate cannot be mapped to a "
                "ResearchIdeaNode: " + source_object_id
            )
        result[str(candidate.candidate_id)] = idea_id
    return result


def compile_idea_outcome_observations(
    *,
    pool: Any,
    materialization: Any,
    nodes: Sequence[ResearchIdeaNode],
    residual_state: Mapping[str, Any] | None = None,
    prospective_by_hypothesis: Any = None,
) -> list[IdeaOutcomeObservation]:
    candidate_to_idea = candidate_to_idea_index(pool=pool, nodes=nodes)
    candidate_ids = set(candidate_to_idea)

    expected_portfolio_id = str(
        getattr(materialization, "output_portfolio_id", "") or ""
    )
    residual_portfolio_id = str(
        (residual_state or {}).get("source_portfolio_id") or ""
    )
    if (
        expected_portfolio_id
        and residual_portfolio_id
        and expected_portfolio_id != residual_portfolio_id
    ):
        raise ValueError(
            "residual-state / materialized-portfolio lineage mismatch"
        )

    expected_context_id = str(
        getattr(materialization, "source_context_id", "") or ""
    )
    residual_by_h = _residual_index(residual_state)
    prospective_by_h = _prospective_index(
        prospective_by_hypothesis,
        expected_context_id=expected_context_id or None,
    )

    observations: list[IdeaOutcomeObservation] = []
    for record in materialization.records:
        candidate_id = str(record.candidate_id)
        if candidate_id not in candidate_ids:
            raise ValueError(
                "materialization references candidate outside scientific "
                "portfolio pool: " + candidate_id
            )
        idea_id = candidate_to_idea[candidate_id]
        hypothesis_id = (
            str(record.hypothesis_id)
            if record.hypothesis_id is not None
            else None
        )
        residual = residual_by_h.get(hypothesis_id or "", {})
        prospective = prospective_by_h.get(hypothesis_id or "", {})

        source_report_ids = [str(materialization.report_id)]
        source_systems = ["SCIENTIFIC_PORTFOLIO_MATERIALIZATION"]
        source_versions = [str(getattr(materialization, "schema_version", "scientific-portfolio-materialization-report-v1"))]
        residual_report_id = str(
            (residual_state or {}).get("report_id") or ""
        )
        if residual:
            source_systems.append("RESIDUAL_EPISTEMIC_STATE")
            source_versions.append(str((residual_state or {}).get("schema_version") or "scientific-portfolio-residual-epistemic-state-v1"))
            if residual_report_id:
                source_report_ids.append(residual_report_id)
        prospective_path = str(prospective.get("artifact_path") or "")
        if prospective:
            source_systems.append("PROSPECTIVE_IDENTIFICATION")
            source_versions.append(str(prospective.get("schema_version") or "prospective-identification-materialization-shadow-v1"))
        if prospective_path:
            source_report_ids.append(prospective_path)

        observations.append(
            IdeaOutcomeObservation(
                observation_id=_stable_id(
                    "idea_outcome_observation",
                    idea_id,
                    candidate_id,
                    hypothesis_id,
                    str(record.status),
                    residual.get("final_epistemic_state"),
                    prospective.get("prospective_identifiability"),
                ),
                idea_id=idea_id,
                target_scope="REALIZATION",
                source_systems=list(dict.fromkeys(source_systems)),
                source_versions=list(dict.fromkeys(source_versions)),
                candidate_id=candidate_id,
                hypothesis_id=hypothesis_id,
                materialization_status=str(record.status),
                materialization_issue_codes=list(record.issue_codes),
                residual_epistemic_state=(
                    str(residual.get("final_epistemic_state"))
                    if residual.get("final_epistemic_state") is not None
                    else None
                ),
                residual_state_reason=(
                    str(residual.get("state_reason"))
                    if residual.get("state_reason") is not None
                    else None
                ),
                external_status=(
                    str(residual.get("fresh_external_status"))
                    if residual.get("fresh_external_status") is not None
                    else None
                ),
                prospective_identifiability=(
                    str(prospective.get("prospective_identifiability"))
                    if prospective.get("prospective_identifiability") is not None
                    else None
                ),
                current_evidence_status=(
                    str(prospective.get("current_evidence_status"))
                    if prospective.get("current_evidence_status") is not None
                    else None
                ),
                directionality_mode=(
                    str(prospective.get("directionality_mode"))
                    if prospective.get("directionality_mode") is not None
                    else None
                ),
                measurement_compatibility_mode=(
                    str(prospective.get("measurement_compatibility_mode"))
                    if prospective.get("measurement_compatibility_mode") is not None
                    else None
                ),
                prospective_contract_integrity_passed=(
                    bool(prospective.get("contract_integrity_passed"))
                    if prospective.get("contract_integrity_passed") is not None
                    else None
                ),
                source_report_ids=list(dict.fromkeys(source_report_ids)),
            )
        )

    return observations


def compile_feedback_coverage_audit(
    observations: Sequence[IdeaOutcomeObservation],
) -> IdeaFeedbackCoverageAudit:
    materialized = list(dict.fromkeys(
        row.hypothesis_id
        for row in observations
        if row.materialization_status == "MATERIALIZED" and row.hypothesis_id
    ))
    residual = list(dict.fromkeys(
        row.hypothesis_id
        for row in observations
        if row.hypothesis_id and row.residual_epistemic_state is not None
    ))
    prospective = list(dict.fromkeys(
        row.hypothesis_id
        for row in observations
        if row.hypothesis_id and row.prospective_identifiability is not None
    ))
    materialized_set = set(materialized)
    residual_set = set(residual)
    prospective_set = set(prospective)
    denominator = len(materialized)
    return IdeaFeedbackCoverageAudit(
        materialized_hypothesis_ids=materialized,
        materialized_hypothesis_count=denominator,
        residual_matched_hypothesis_ids=residual,
        residual_matched_hypothesis_count=len(residual),
        prospective_matched_hypothesis_ids=prospective,
        prospective_matched_hypothesis_count=len(prospective),
        residual_unmatched_hypothesis_ids=sorted(materialized_set - residual_set),
        prospective_unmatched_hypothesis_ids=sorted(
            materialized_set - prospective_set
        ),
        residual_coverage_fraction=(
            len(residual) / denominator if denominator else 0.0
        ),
        prospective_coverage_fraction=(
            len(prospective) / denominator if denominator else 0.0
        ),
        any_post_verification_feedback=bool(residual or prospective),
    )


def _upgrade_band(current: IdeaPriorityBand, proposed: IdeaPriorityBand) -> IdeaPriorityBand:
    return proposed if _BAND_RANK[proposed] > _BAND_RANK[current] else current


def _directive_for_observation(
    observation: IdeaOutcomeObservation,
) -> tuple[bool, bool, IdeaPriorityBand, list[str], list[str]]:
    direct = True
    idea = True
    band: IdeaPriorityBand = "MEDIUM"
    actions: list[str] = []
    reasons: list[str] = []

    status = observation.materialization_status
    if status == "MATERIALIZED":
        actions.append("VERIFY")
        reasons.append("REALIZATION_MATERIALIZED")
    elif status in {
        "ABSTAINED",
        "COMPILE_REJECTED",
        "HARD_GATE_REJECTED",
        "MISSING_DRAFT",
        "GENERATION_FAILED",
    }:
        direct = False
        actions.extend(["ALTERNATE_REALIZATION", "REFRAME"])
        reasons.extend([
            "CURRENT_REALIZATION_NOT_VERIFICATION_READY",
            "MATERIALIZATION_FAILURE_DOES_NOT_REJECT_IDEA",
        ])
        if status in {"HARD_GATE_REJECTED", "GENERATION_FAILED"}:
            band = "LOW"

    residual = observation.residual_epistemic_state
    if residual == "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW":
        band = _upgrade_band(band, "HIGH")
        actions.extend(["VERIFY", "SAME_PREMISE_SHARPEN"])
        reasons.append("FRESH_RESIDUAL_CANDIDATE_SURVIVED")
    elif residual == "UNRESOLVED_EVIDENCE_GAP":
        actions.extend(["EVIDENCE_REAXIS", "REQUEST_GRAPH_RETRAVERSAL"])
        reasons.append("EVIDENCE_GAP_REMAINS")
    elif residual == "UNRESOLVED_TOPOLOGY_GAP":
        actions.extend(["AXIS_MUTATION", "REQUEST_GRAPH_RETRAVERSAL"])
        reasons.append("TOPOLOGY_GAP_REMAINS")
    elif residual == "PRIOR_ART_BACKED_OR_NO_RESIDUAL":
        direct = False
        band = "LOW"
        actions.extend(["AXIS_MUTATION", "REFRAME"])
        reasons.extend([
            "CURRENT_AXIS_PRIOR_ART_BACKED_OR_NO_RESIDUAL",
            "IDEA_MAY_REPRODUCE_THROUGH_CONCEPTUAL_CHANGE",
        ])

    prospective = observation.prospective_identifiability
    if prospective == "CURRENTLY_IDENTIFIED":
        band = _upgrade_band(band, "HIGH")
        actions.append("VERIFY")
        reasons.append("CURRENTLY_IDENTIFIED_REALIZATION")
    elif prospective == "PROSPECTIVELY_IDENTIFIABLE":
        band = _upgrade_band(band, "HIGH")
        actions.append("VERIFY")
        reasons.append("PROSPECTIVELY_IDENTIFIABLE_REALIZATION")
    elif prospective == "NOT_OPERATIONALIZABLE":
        direct = False
        band = _upgrade_band(band, "MEDIUM")
        actions.extend([
            "AXIS_MUTATION",
            "REFRAME",
            "REQUEST_GRAPH_RETRAVERSAL",
        ])
        reasons.extend([
            "CURRENT_REALIZATION_NOT_OPERATIONALIZABLE",
            "NOT_OPERATIONALIZABLE_REALIZATION_DOES_NOT_KILL_IDEA",
        ])

    if observation.prospective_contract_integrity_passed is False:
        direct = False
        actions.append("ALTERNATE_REALIZATION")
        reasons.append("PROSPECTIVE_CONTRACT_INTEGRITY_FAILED")

    return (
        direct,
        idea,
        band,
        list(dict.fromkeys(actions)),
        list(dict.fromkeys(reasons)),
    )


def compile_idea_feedback_directives(
    observations: Sequence[IdeaOutcomeObservation],
    *,
    all_idea_ids: Sequence[str] = (),
) -> list[IdeaSearchDirective]:
    grouped: dict[str, list[IdeaOutcomeObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.idea_id].append(observation)

    result: list[IdeaSearchDirective] = []
    for idea_id in list(dict.fromkeys([*all_idea_ids, *grouped.keys()])):
        rows = grouped.get(idea_id, [])
        if not rows:
            result.append(
                IdeaSearchDirective(
                    idea_id=idea_id,
                    observation_ids=[],
                    direct_realization_reproductive=True,
                    idea_reproductive=True,
                    priority_band="MEDIUM",
                    preferred_actions=["ALTERNATE_REALIZATION"],
                    reason_codes=["NO_REALIZATION_OUTCOME_OBSERVED_YET"],
                )
            )
            continue

        direct_any = False
        idea_any = False
        band: IdeaPriorityBand = "DEFER"
        actions: list[str] = []
        reasons: list[str] = []
        for row in rows:
            direct, idea, one_band, one_actions, one_reasons = (
                _directive_for_observation(row)
            )
            direct_any = direct_any or direct
            idea_any = idea_any or idea
            band = _upgrade_band(band, one_band)
            actions.extend(one_actions)
            reasons.extend(one_reasons)

        result.append(
            IdeaSearchDirective(
                idea_id=idea_id,
                observation_ids=[row.observation_id for row in rows],
                direct_realization_reproductive=direct_any,
                idea_reproductive=idea_any,
                priority_band=band,
                preferred_actions=list(dict.fromkeys(actions)),
                reason_codes=list(dict.fromkeys(reasons)),
            )
        )
    return result


__all__ = [
    "candidate_to_idea_index",
    "compile_feedback_coverage_audit",
    "compile_idea_feedback_directives",
    "compile_idea_outcome_observations",
]
