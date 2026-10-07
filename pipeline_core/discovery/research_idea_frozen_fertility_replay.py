from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.research_idea_adaptive_fertility import (
    AdaptiveFertilityReport,
    build_adaptive_fertility_report,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _policy_input_manifest(
    *,
    parallel_report: Any,
    lifecycle: Any,
    cycle_generation_index: int,
) -> dict[str, Any]:
    """Canonical frozen state consumed by the v3.1 fertility compiler.

    This deliberately records only the fields that can influence
    ``build_adaptive_fertility_report``.  It therefore provides a stable
    policy-input fingerprint even when unrelated diagnostic fields are added
    to the source artifacts later.
    """

    members_by_idea: dict[str, list[Any]] = {}
    for row in (_get(parallel_report, "members", []) or []):
        members_by_idea.setdefault(str(_get(row, "idea_id")), []).append(row)

    debts_by_idea: dict[str, list[Any]] = {}
    for row in (_get(parallel_report, "epistemic_debts", []) or []):
        debts_by_idea.setdefault(str(_get(row, "idea_id")), []).append(row)

    handoff_by_idea = {
        str(_get(row, "idea_id")): row
        for row in (_get(parallel_report, "evolution_handoffs", []) or [])
    }
    local_by_idea = {
        str(_get(row, "idea_id")): row
        for row in (_get(lifecycle, "local_states", []) or [])
    }

    ideas: list[dict[str, Any]] = []
    for idea_state in (_get(parallel_report, "idea_states", []) or []):
        idea_id = str(_get(idea_state, "idea_id"))
        active_ids = set(_get(idea_state, "active_member_ids", []) or [])
        active_members = [
            row
            for row in members_by_idea.get(idea_id, [])
            if str(_get(row, "member_id")) in active_ids
        ]
        active_members.sort(key=lambda row: str(_get(row, "member_id")))
        debts = list(debts_by_idea.get(idea_id, []))
        debts.sort(key=lambda row: str(_get(row, "debt_id")))
        handoff = handoff_by_idea.get(idea_id)
        local = local_by_idea.get(idea_id)

        ideas.append(
            {
                "idea_id": idea_id,
                "idea_birth_generation_index": int(_get(idea_state, "generation_index", 0) or 0),
                "remains_in_search_population": bool(
                    _get(idea_state, "remains_in_search_population", False)
                ),
                "active_members": [
                    {
                        "member_id": str(_get(row, "member_id")),
                        "epistemic_realization_id": str(
                            _get(row, "epistemic_realization_id", "")
                        ),
                        "epistemic_maturity": str(_get(row, "epistemic_maturity", "UNKNOWN")),
                    }
                    for row in active_members
                ],
                "epistemic_debts": [
                    {
                        "debt_id": str(_get(row, "debt_id")),
                        "requirement_kind": str(_get(row, "requirement_kind", "")),
                        "disposition": str(_get(row, "disposition", "")),
                    }
                    for row in debts
                ],
                "evolution_handoff": (
                    {
                        "handoff_id": str(_get(handoff, "handoff_id")),
                        "recommended_channels": list(
                            _get(handoff, "recommended_channels", []) or []
                        ),
                        "nonbinding_operator_hints": list(
                            _get(handoff, "nonbinding_operator_hints", []) or []
                        ),
                        "reason_codes": list(_get(handoff, "reason_codes", []) or []),
                    }
                    if handoff is not None
                    else None
                ),
                "local_state": (
                    {
                        "realization_count": int(_get(local, "realization_count", 0) or 0),
                        "usable_grounded_realization_count": int(
                            _get(local, "usable_grounded_realization_count", 0) or 0
                        ),
                        "failed_or_abstained_count": int(
                            _get(local, "failed_or_abstained_count", 0) or 0
                        ),
                        "not_operationalizable_count": int(
                            _get(local, "not_operationalizable_count", 0) or 0
                        ),
                        "local_search_budget": int(
                            _get(local, "local_search_budget", 0) or 0
                        ),
                        "local_search_budget_used": int(
                            _get(local, "local_search_budget_used", 0) or 0
                        ),
                        "local_search_exhausted": bool(
                            _get(local, "local_search_exhausted", False)
                        ),
                        "rescued_within_same_idea": bool(
                            _get(local, "rescued_within_same_idea", False)
                        ),
                    }
                    if local is not None
                    else None
                ),
            }
        )

    ideas.sort(key=lambda row: row["idea_id"])
    return {
        "schema_version": "sis-v3-1-frozen-fertility-policy-input-v1",
        "cycle_generation_index": cycle_generation_index,
        "source_parallel_report_id": str(_get(parallel_report, "report_id")),
        "source_lifecycle_report_id": str(_get(lifecycle, "report_id")),
        "ideas": ideas,
    }


class FrozenFertilityDecisionComparison(StrictModel):
    idea_id: str
    replay_disposition: str
    recorded_disposition: str | None = None
    replay_fertile: bool
    recorded_fertile: bool | None = None
    exact_match: bool | None = None


class FrozenFertilityReplayAudit(StrictModel):
    schema_version: Literal[
        "sis-v3-1-frozen-fertility-replay-audit-v1"
    ] = "sis-v3-1-frozen-fertility-replay-audit-v1"
    audit_id: str
    audit_sha256: str
    snapshot_label: str = Field(min_length=1)
    cycle_generation_index: int = Field(ge=2)
    source_parallel_report_id: str
    source_lifecycle_report_id: str
    policy_input_sha256: str = Field(min_length=64, max_length=64)
    source_parallel_payload_sha256: str = Field(min_length=64, max_length=64)
    source_lifecycle_payload_sha256: str = Field(min_length=64, max_length=64)

    replay_fertility_report: AdaptiveFertilityReport
    replay_repeat_exact_match: bool
    replay_disposition_counts: dict[str, int] = Field(default_factory=dict)

    recorded_fertility_report_id: str | None = None
    recorded_report_exact_match: bool | None = None
    recorded_decision_match_count: int = Field(ge=0)
    recorded_decision_mismatch_count: int = Field(ge=0)
    decision_comparisons: list[FrozenFertilityDecisionComparison] = Field(default_factory=list)

    llm_call_count: Literal[0] = 0
    external_search_executed: Literal[False] = False
    evidence_acquisition_executed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    policy_only_replay: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_replay(self) -> "FrozenFertilityReplayAudit":
        if not self.replay_repeat_exact_match:
            raise ValueError("frozen replay is not deterministic on identical input")
        if self.recorded_fertility_report_id is None:
            if self.recorded_report_exact_match is not None:
                raise ValueError("recorded match cannot exist without recorded report")
            if self.recorded_decision_match_count or self.recorded_decision_mismatch_count:
                raise ValueError("recorded decision counts require recorded report")
        return self


def replay_frozen_fertility(
    *,
    snapshot_label: str,
    parallel_report: Any,
    lifecycle: Any,
    cycle_generation_index: int,
    recorded_fertility_report: Any | None = None,
) -> FrozenFertilityReplayAudit:
    manifest = _policy_input_manifest(
        parallel_report=parallel_report,
        lifecycle=lifecycle,
        cycle_generation_index=cycle_generation_index,
    )
    policy_input_sha256 = _sha(manifest)

    replay_a = build_adaptive_fertility_report(
        parallel_report=parallel_report,
        lifecycle=lifecycle,
        cycle_generation_index=cycle_generation_index,
    )
    replay_b = build_adaptive_fertility_report(
        parallel_report=parallel_report,
        lifecycle=lifecycle,
        cycle_generation_index=cycle_generation_index,
    )
    repeat_match = replay_a.model_dump(mode="json") == replay_b.model_dump(mode="json")

    comparisons: list[FrozenFertilityDecisionComparison] = []
    recorded_id = None
    recorded_exact = None
    matches = 0
    mismatches = 0
    if recorded_fertility_report is not None:
        recorded = AdaptiveFertilityReport.model_validate(recorded_fertility_report)
        recorded_id = recorded.report_id
        recorded_exact = recorded.model_dump(mode="json") == replay_a.model_dump(mode="json")
        replay_by_id = {row.idea_id: row for row in replay_a.decisions}
        recorded_by_id = {row.idea_id: row for row in recorded.decisions}
        for idea_id in sorted(set(replay_by_id) | set(recorded_by_id)):
            replay = replay_by_id.get(idea_id)
            prior = recorded_by_id.get(idea_id)
            exact = bool(
                replay is not None
                and prior is not None
                and replay.model_dump(mode="json") == prior.model_dump(mode="json")
            )
            if exact:
                matches += 1
            else:
                mismatches += 1
            comparisons.append(
                FrozenFertilityDecisionComparison(
                    idea_id=idea_id,
                    replay_disposition=(replay.disposition if replay is not None else "MISSING"),
                    recorded_disposition=(prior.disposition if prior is not None else None),
                    replay_fertile=(
                        replay.fertile_for_child_generation if replay is not None else False
                    ),
                    recorded_fertile=(
                        prior.fertile_for_child_generation if prior is not None else None
                    ),
                    exact_match=exact,
                )
            )

    provisional = FrozenFertilityReplayAudit(
        audit_id="pending",
        audit_sha256="pending",
        snapshot_label=snapshot_label,
        cycle_generation_index=cycle_generation_index,
        source_parallel_report_id=str(_get(parallel_report, "report_id")),
        source_lifecycle_report_id=str(_get(lifecycle, "report_id")),
        policy_input_sha256=policy_input_sha256,
        source_parallel_payload_sha256=_sha(parallel_report),
        source_lifecycle_payload_sha256=_sha(lifecycle),
        replay_fertility_report=replay_a,
        replay_repeat_exact_match=repeat_match,
        replay_disposition_counts=dict(
            sorted(Counter(row.disposition for row in replay_a.decisions).items())
        ),
        recorded_fertility_report_id=recorded_id,
        recorded_report_exact_match=recorded_exact,
        recorded_decision_match_count=matches,
        recorded_decision_mismatch_count=mismatches,
        decision_comparisons=comparisons,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("audit_id", None)
    payload.pop("audit_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "audit_id": _stable_id(
                "frozen_fertility_replay",
                snapshot_label,
                cycle_generation_index,
                policy_input_sha256,
                replay_a.report_id,
                recorded_id,
            ),
            "audit_sha256": digest,
        }
    )


def compare_frozen_fertility_audits(
    left: FrozenFertilityReplayAudit,
    right: FrozenFertilityReplayAudit,
) -> dict[str, Any]:
    """Policy-only state comparison for matching idea identities.

    This does not assert causal superiority. It only exposes whether the same
    v3.1 compiler sees materially different frozen inputs and therefore emits
    different fertility decisions.
    """

    left_by_id = {row.idea_id: row for row in left.replay_fertility_report.decisions}
    right_by_id = {row.idea_id: row for row in right.replay_fertility_report.decisions}
    shared = sorted(set(left_by_id) & set(right_by_id))
    transitions = Counter(
        f"{left_by_id[idea_id].disposition}->{right_by_id[idea_id].disposition}"
        for idea_id in shared
    )
    changed = [
        idea_id
        for idea_id in shared
        if left_by_id[idea_id].disposition != right_by_id[idea_id].disposition
        or left_by_id[idea_id].fertile_for_child_generation
        != right_by_id[idea_id].fertile_for_child_generation
    ]
    return {
        "schema_version": "sis-v3-1-frozen-fertility-audit-comparison-v1",
        "left_snapshot_label": left.snapshot_label,
        "right_snapshot_label": right.snapshot_label,
        "same_policy_input": left.policy_input_sha256 == right.policy_input_sha256,
        "shared_idea_count": len(shared),
        "changed_decision_count": len(changed),
        "changed_idea_ids": changed,
        "disposition_transitions": dict(sorted(transitions.items())),
        "policy_only_comparison": True,
        "causal_superiority_asserted": False,
    }


__all__ = [
    "FrozenFertilityDecisionComparison",
    "FrozenFertilityReplayAudit",
    "compare_frozen_fertility_audits",
    "replay_frozen_fertility",
]
