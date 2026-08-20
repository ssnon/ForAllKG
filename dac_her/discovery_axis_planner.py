from __future__ import annotations

import hashlib
import json
from typing import Any

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
    DiscoveryAxisPlan,
    DiscoveryAxisPlannerPolicy,
)
from pipeline_core.discovery.dual_hypothesis_context import DualHypothesisContext


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _continuity_score(value: str) -> float:
    return {
        "high": 1.0,
        "medium": 0.55,
        "low": 0.10,
        "not_applicable": 0.35,
    }.get(str(value), 0.0)


class DiscoveryAxisPlanner:
    """Deterministically turn selected DiscoveryInspirations into generation axes.

    Alpha4 intentionally plans from the already fail-closed DiscoveryBundle.
    It does not reopen traversal or elevate candidate content to evidence.
    Candidate-unit inspirations are preferred because they carry an explicit,
    auditable proposed relation and distinct grounding anchors.
    """

    def __init__(self, policy: DiscoveryAxisPlannerPolicy | None = None) -> None:
        self.policy = policy or DiscoveryAxisPlannerPolicy()
        if self.policy.max_axes < 1:
            raise ValueError("max_axes must be >= 1")

    def _score(self, item: object) -> float:
        exploration = float(getattr(item, "exploration_score", 0.0) or 0.0)
        unit = float(getattr(item, "candidate_unit_score", 0.0) or 0.0)
        continuity = _continuity_score(
            str(getattr(item, "mechanistic_continuity_band", "not_applicable"))
        )
        generic = float(getattr(item, "generic_entity_fraction", 0.0) or 0.0)
        grounding = float(
            getattr(item, "semantic_similarity_to_grounding", 0.0) or 0.0
        )
        reaction = float(
            getattr(item, "reaction_domain_switch_penalty", 0.0) or 0.0
        )
        score = (
            0.42 * exploration
            + 0.28 * unit
            + 0.14 * continuity
            + 0.08 * max(0.0, 1.0 - generic)
            + 0.08 * max(0.0, 1.0 - grounding)
            - 0.20 * reaction
        )
        return max(0.0, float(score))

    def build(self, dual: DualHypothesisContext) -> DiscoveryAxisPlan:
        bundle = dual.discovery_bundle
        if dual.grounded_context.corpus_id != bundle.corpus_id:
            raise ValueError("dual context corpus mismatch")

        eligible: list[tuple[float, int, object]] = []
        excluded: list[str] = []
        seen_units: set[str] = set()

        for index, item in enumerate(bundle.inspirations):
            inspiration_id = str(item.inspiration_id)
            unit_id = str(getattr(item, "candidate_unit_id", "") or "")
            if self.policy.require_candidate_unit and not unit_id:
                excluded.append(inspiration_id)
                continue
            if float(item.exploration_score) < self.policy.min_exploration_score:
                excluded.append(inspiration_id)
                continue
            if unit_id and float(item.candidate_unit_score) < self.policy.min_candidate_unit_score:
                excluded.append(inspiration_id)
                continue
            if (
                float(item.reaction_domain_switch_penalty)
                > self.policy.max_reaction_domain_switch_penalty
            ):
                excluded.append(inspiration_id)
                continue
            if unit_id and unit_id in seen_units:
                excluded.append(inspiration_id)
                continue
            if unit_id:
                seen_units.add(unit_id)
            eligible.append((self._score(item), index, item))

        eligible.sort(key=lambda row: (-row[0], row[1], str(row[2].inspiration_id)))
        selected = eligible[: self.policy.max_axes]
        excluded.extend(str(row[2].inspiration_id) for row in eligible[self.policy.max_axes :])

        axes: list[DiscoveryAxis] = []
        for rank, (planner_score, _index, item) in enumerate(selected, start=1):
            label = str(item.candidate_unit_label or item.rendered_path)
            axis_id = _stable_id(
                "discovery_axis",
                dual.dual_context_sha256,
                item.inspiration_id,
                item.candidate_unit_id,
                rank,
            )
            axes.append(
                DiscoveryAxis(
                    axis_id=axis_id,
                    axis_rank=rank,
                    inspiration_id=str(item.inspiration_id),
                    source_path_id=str(item.source_path_id),
                    candidate_unit_id=str(item.candidate_unit_id),
                    label=label,
                    entry_anchor_id=str(item.candidate_entry_anchor_id),
                    entry_anchor_label=str(item.candidate_entry_anchor_label),
                    exit_anchor_id=str(item.candidate_exit_anchor_id),
                    exit_anchor_label=str(item.candidate_exit_anchor_label),
                    proposed_subject=str(item.candidate_proposed_subject),
                    proposed_relation=str(item.candidate_proposed_relation),
                    proposed_object=str(item.candidate_proposed_object),
                    rendered_path=str(item.rendered_path),
                    source_mode=str(item.source_mode),
                    exploration_score=float(item.exploration_score),
                    candidate_unit_score=float(item.candidate_unit_score),
                    planner_score=float(planner_score),
                    mechanistic_continuity_band=str(item.mechanistic_continuity_band),
                    generic_entity_fraction=float(item.generic_entity_fraction),
                    registry_hop_fraction=float(item.registry_hop_fraction),
                    grounding_semantic_overlap=float(item.semantic_similarity_to_grounding),
                    reaction_domain_switch_penalty=float(
                        item.reaction_domain_switch_penalty
                    ),
                    requires_verification=bool(item.requires_verification),
                    reason_codes=list(item.reason_codes),
                )
            )

        plan_id = _stable_id(
            "discovery_axis_plan",
            dual.dual_context_sha256,
            bundle.bundle_sha256,
            *[axis.axis_id for axis in axes],
        )
        payload = {
            "schema_version": "discovery-axis-plan-v1",
            "plan_id": plan_id,
            "source_dual_context_id": dual.dual_context_id,
            "source_dual_context_sha256": dual.dual_context_sha256,
            "source_bundle_id": bundle.bundle_id,
            "source_bundle_sha256": bundle.bundle_sha256,
            "corpus_id": bundle.corpus_id,
            "axes": [axis.model_dump(mode="json") for axis in axes],
            "excluded_inspiration_ids": sorted(set(excluded)),
            "policy": self.policy.model_dump(mode="json"),
        }
        return DiscoveryAxisPlan(**payload, plan_sha256=_sha256_json(payload))
