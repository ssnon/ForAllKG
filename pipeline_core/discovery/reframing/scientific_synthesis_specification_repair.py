from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_prompt import HypothesisPrompt
from pipeline_core.discovery.n10_post_generation_continuation_policy import (
    REFINE_NOVELTY_BEARING_SPECIFICATION,
    post_generation_continuation_directive_from_gate_row,
)
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    SYNTHESIS_AUTHORITY_SCOPE,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_ALLOWED_MISSING_FIELDS = frozenset(
    {
        "required_bridge",
        "predicted_observation",
        "falsification_condition",
    }
)


class ScientificSynthesisRepairClaimDiagnostic(StrictModel):
    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    prior_art_identity_terms: list[str] = Field(default_factory=list)
    missing_fields: list[
        Literal[
            "required_bridge",
            "predicted_observation",
            "falsification_condition",
        ]
    ] = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)


class ScientificSynthesisSpecificationRepairContext(StrictModel):
    schema_version: Literal[
        "scientific-synthesis-specification-repair-context-v1"
    ] = "scientific-synthesis-specification-repair-context-v1"

    context_id: str
    source_hypothesis_id: str
    source_query_plan_id: str
    source_query_plan_sha256: str
    source_intake_sha256: str
    source_n10_gate_sha256: str
    source_n10_authority_scope: Literal[
        "scientific_cross_lane_synthesis_candidate"
    ] = "scientific_cross_lane_synthesis_candidate"
    selection_class: Literal["CONDITIONAL"] = "CONDITIONAL"
    repair_action: Literal[
        "REFINE_NOVELTY_BEARING_SPECIFICATION"
    ] = "REFINE_NOVELTY_BEARING_SPECIFICATION"
    continuation_depth: Literal[0] = 0
    next_depth: Literal[1] = 1
    claim_diagnostics: list[
        ScientificSynthesisRepairClaimDiagnostic
    ] = Field(min_length=1)

    diagnostic_only: Literal[True] = True
    production_authority: Literal[False] = False
    scientific_evidence_authority: Literal[False] = False
    external_prior_art_can_be_positive_premise: Literal[False] = False
    absence_is_novelty: Literal[False] = False


class ScientificSynthesisRepairEntry(StrictModel):
    source_hypothesis_id: str
    repair_context_id: str
    repaired_hypothesis_id: str | None = None
    status: Literal[
        "repaired",
        "abstained",
        "compile_rejected",
        "validation_rejected",
    ]
    issues: list[str] = Field(default_factory=list)


class ScientificSynthesisRepairReport(StrictModel):
    schema_version: Literal[
        "scientific-synthesis-bounded-specification-repair-report-v1"
    ] = "scientific-synthesis-bounded-specification-repair-report-v1"

    report_id: str
    source_portfolio_id: str
    output_portfolio_id: str
    eligible_repair_count: int = Field(ge=0)
    repaired_count: int = Field(ge=0)
    entries: list[ScientificSynthesisRepairEntry]

    continuation_depth_in: Literal[0] = 0
    continuation_depth_out: Literal[1] = 1
    further_repair_allowed: Literal[False] = False
    premise_gap_type_locked: Literal[True] = True
    external_prior_art_used_as_positive_premise: Literal[False] = False
    fresh_n10_required_after_repair: Literal[True] = True
    production_authority_created: Literal[False] = False
    legacy_production_selection_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


def _json_safe(value: object) -> object:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _gate_rows(
    production_gate: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if production_gate.get("schema_version") != "scientific-novelty-fallback-gate-v2":
        raise ValueError("scientific synthesis repair requires role-aware N10-v2 gate")
    if production_gate.get("production_authority") is not True:
        raise ValueError("scientific synthesis N10 gate lacks production authority")
    if production_gate.get("authority_scope") != SYNTHESIS_AUTHORITY_SCOPE:
        raise ValueError("scientific synthesis N10 authority scope mismatch")
    if production_gate.get("conditional_is_positive") is not False:
        raise ValueError("CONDITIONAL must remain non-positive")
    if production_gate.get("absence_is_novelty") is not False:
        raise ValueError("search-bounded absence must not become novelty")

    rows = production_gate.get("gates")
    if not isinstance(rows, list):
        raise ValueError("scientific synthesis N10 gate rows must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("scientific synthesis N10 gate row must be an object")
        hypothesis_id = str(row.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            raise ValueError("scientific synthesis N10 gate row lacks hypothesis_id")
        if hypothesis_id in result:
            raise ValueError("duplicate scientific synthesis N10 gate hypothesis_id")
        result[hypothesis_id] = row
    return result


def build_scientific_synthesis_repair_contexts(
    *,
    portfolio: HypothesisPortfolio,
    query_plan: LiteratureQueryPlan,
    intake_shadow: Mapping[str, Any],
    production_gate: Mapping[str, Any],
) -> tuple[ScientificSynthesisSpecificationRepairContext, ...]:
    if query_plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("repair query-plan/source-portfolio mismatch")
    if intake_shadow.get("source_portfolio_id") != portfolio.portfolio_id:
        raise ValueError("repair intake/source-portfolio mismatch")
    if intake_shadow.get("source_query_plan_id") != query_plan.plan_id:
        raise ValueError("repair intake/query-plan mismatch")

    gates = _gate_rows(production_gate)
    expected_ids = {row.hypothesis_id for row in portfolio.hypotheses}
    if set(gates) != expected_ids:
        raise ValueError("repair gate membership does not match synthesis portfolio")

    plan_claims: dict[str, dict[str, Any]] = {}
    for group in query_plan.claims:
        by_id: dict[str, Any] = {}
        for claim in group.claims:
            if claim.hypothesis_id != group.hypothesis_id:
                raise ValueError("query-plan claim hypothesis identity drift")
            if claim.claim_id in by_id:
                raise ValueError("duplicate query-plan claim ID")
            by_id[claim.claim_id] = claim
        plan_claims[group.hypothesis_id] = by_id

    intake_rows: dict[str, Mapping[str, Any]] = {}
    for row in intake_shadow.get("hypotheses", []):
        if not isinstance(row, Mapping):
            raise ValueError("N9 intake hypothesis row must be an object")
        hypothesis_id = str(row.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            raise ValueError("N9 intake hypothesis row lacks hypothesis_id")
        if hypothesis_id in intake_rows:
            raise ValueError("duplicate N9 intake hypothesis row")
        intake_rows[hypothesis_id] = row

    contexts: list[ScientificSynthesisSpecificationRepairContext] = []
    for card in portfolio.hypotheses:
        gate_row = gates[card.hypothesis_id]
        directive = post_generation_continuation_directive_from_gate_row(
            gate_row=gate_row,
            continuation_depth=0,
        )
        if not directive.allow_bounded_continuation:
            continue
        if directive.next_depth != 1 or directive.fresh_post_generation_n10_required is not True:
            raise ValueError("invalid scientific synthesis continuation directive")

        intake_row = intake_rows.get(card.hypothesis_id)
        if intake_row is None:
            raise ValueError(
                f"missing N9 intake row for repair candidate {card.hypothesis_id}"
            )
        canonical_claims = plan_claims.get(card.hypothesis_id, {})
        if not canonical_claims:
            raise ValueError(
                f"query plan has no claims for repair candidate {card.hypothesis_id}"
            )

        diagnostics: list[tuple[int, str, ScientificSynthesisRepairClaimDiagnostic]] = []
        for decision in intake_row.get("claims", []):
            if not isinstance(decision, Mapping):
                raise ValueError("N9 intake claim decision must be an object")
            claim_payload = decision.get("claim")
            specification = decision.get("specification")
            if not isinstance(claim_payload, Mapping) or not isinstance(specification, Mapping):
                raise ValueError("N9 intake repair decision lacks claim/specification payload")
            claim_id = str(claim_payload.get("claim_id") or "").strip()
            canonical = canonical_claims.get(claim_id)
            if canonical is None:
                raise ValueError(
                    f"N9 intake repair claim absent from canonical query plan: {claim_id}"
                )
            if (
                decision.get("shadow_state") != "NEEDS_REFINEMENT"
                or specification.get("status") != "NEEDS_REFINEMENT"
            ):
                continue
            if canonical.novelty_selection_role != "NOVELTY_BEARING":
                continue
            if decision.get("next_action") != "REFINE_HYPOTHESIS_SPECIFICATION":
                raise ValueError("unexpected N9 next action for repairable claim")

            missing = [str(x) for x in (specification.get("missing_fields") or [])]
            if not missing:
                raise ValueError("repairable N9 claim lacks missing_fields")
            unknown = set(missing) - _ALLOWED_MISSING_FIELDS
            if unknown:
                raise ValueError(
                    "unsupported repair missing fields: " + repr(sorted(unknown))
                )
            reasons = [
                str(x)
                for x in (specification.get("reason_codes") or [])
                if str(x).strip()
            ]
            if not reasons:
                raise ValueError("repairable N9 claim lacks reason_codes")

            diagnostics.append(
                (
                    int(canonical.claim_rank),
                    canonical.claim_id,
                    ScientificSynthesisRepairClaimDiagnostic(
                        claim_id=canonical.claim_id,
                        claim_text=canonical.text,
                        prior_art_identity_terms=list(canonical.prior_art_identity_terms),
                        missing_fields=missing,
                        reason_codes=reasons,
                    ),
                )
            )

        if not diagnostics:
            raise ValueError(
                "N10 continuation policy allowed repair but no novelty-bearing "
                f"N9 specification diagnostic exists for {card.hypothesis_id}"
            )
        diagnostics.sort(key=lambda x: (x[0], x[1]))
        context_id = _stable_id(
            "scientific_synthesis_specification_repair_context",
            card.hypothesis_id,
            query_plan.plan_id,
            _sha256_json(intake_shadow),
            _sha256_json(production_gate),
            _canonical_json([x[2] for x in diagnostics]),
        )
        contexts.append(
            ScientificSynthesisSpecificationRepairContext(
                context_id=context_id,
                source_hypothesis_id=card.hypothesis_id,
                source_query_plan_id=query_plan.plan_id,
                source_query_plan_sha256=query_plan.plan_sha256,
                source_intake_sha256=_sha256_json(intake_shadow),
                source_n10_gate_sha256=_sha256_json(production_gate),
                claim_diagnostics=[x[2] for x in diagnostics],
            )
        )

    return tuple(contexts)


def build_scientific_synthesis_repair_prompt(
    *,
    original: HypothesisCard,
    context: HypothesisContext,
    repair_context: ScientificSynthesisSpecificationRepairContext,
) -> HypothesisPrompt:
    if repair_context.source_hypothesis_id != original.hypothesis_id:
        raise ValueError("repair prompt source hypothesis mismatch")
    statement_by_id = {
        statement.statement_id: statement
        for statement in context.evidence_statements
    }
    grounded = []
    for statement_id in original.premise_statement_ids:
        statement = statement_by_id.get(statement_id)
        if statement is None or not statement.eligible_as_premise:
            raise ValueError(
                f"repair source premise is not a grounded positive premise: {statement_id}"
            )
        grounded.append(
            {
                "statement_id": statement.statement_id,
                "text": statement.text,
                "epistemic_role": statement.epistemic_role,
                "paper_ids": list(statement.paper_ids),
            }
        )

    system = """You perform ONE bounded scientific hypothesis specification repair.

Hard epistemic constraints:
- The only positive scientific evidence is the supplied grounded premise statements.
- N9/N10 diagnostics identify missing specification fields; diagnostics are NOT scientific evidence.
- External prior-art identity terms are diagnostic labels only and are NOT positive premises.
- Preserve the exact premise_statement_ids, gap_statement_ids, and hypothesis_type.
- Do not add a new mechanism, material, condition, variable, evidence ID, or literature claim unless it is already supported by the supplied grounded premises.
- Do not claim novelty, priority, absence from literature, or empirical confirmation.
- Repair only missing required_bridge, predicted_observation, and/or falsification_condition structure identified by the diagnostics.
- Every falsifier observable must exactly match one predicted-observation observable.
- Return exactly ONE repaired hypothesis, or abstain if the missing specification cannot be supplied without unsupported content.
- This is the only bounded repair opportunity. The result must undergo a completely fresh external novelty and N10 evaluation.
"""
    payload = {
        "original_hypothesis": {
            "title": original.title,
            "hypothesis_statement": original.hypothesis_statement,
            "hypothesis_type": original.hypothesis_type,
            "premise_statement_ids": list(original.premise_statement_ids),
            "gap_statement_ids": list(original.gap_statement_ids),
            "inferential_bridge": original.inferential_bridge,
            "predicted_observations": [
                row.model_dump(mode="json")
                for row in original.predicted_observations
            ],
            "falsification_criteria": [
                row.model_dump(mode="json")
                for row in original.falsification_criteria
            ],
            "assumptions": list(original.assumptions),
        },
        "grounded_positive_premises": grounded,
        "diagnostic_only_specification_failures": [
            row.model_dump(mode="json")
            for row in repair_context.claim_diagnostics
        ],
        "required_output_contract": {
            "hypothesis_count": 1,
            "preserve_premise_statement_ids": list(original.premise_statement_ids),
            "preserve_gap_statement_ids": list(original.gap_statement_ids),
            "preserve_hypothesis_type": original.hypothesis_type,
        },
    }
    user = (
        "Repair the existing hypothesis only enough to make its novelty-bearing "
        "atomic claims self-contained and testable under the listed missing-field "
        "diagnostics. Do not chase novelty and do not change scientific scope.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    digest = hashlib.sha256((system + "\n---\n" + user).encode("utf-8")).hexdigest()
    return HypothesisPrompt(
        prompt_version="scientific-synthesis-n10-specification-repair-v1",
        system_prompt=system,
        user_prompt=user,
        prompt_sha256=digest,
    )


def lock_repair_provenance(
    *,
    original: HypothesisCard,
    draft: HypothesisPortfolioDraft,
    repair_context: ScientificSynthesisSpecificationRepairContext,
) -> HypothesisPortfolioDraft:
    if not draft.hypotheses:
        return draft
    if len(draft.hypotheses) != 1:
        raise ValueError("bounded synthesis repair must return exactly one hypothesis")
    row = draft.hypotheses[0]
    locked = row.model_copy(
        update={
            "local_id": (
                "bounded_specification_repair_"
                + repair_context.context_id.split(":")[-1]
            ),
            "premise_statement_ids": list(original.premise_statement_ids),
            "gap_statement_ids": list(original.gap_statement_ids),
            "hypothesis_type": original.hypothesis_type,
        }
    )
    return draft.model_copy(
        update={
            "hypotheses": [locked],
            "abstention_reason": None,
        }
    )


__all__ = [
    "ScientificSynthesisRepairClaimDiagnostic",
    "ScientificSynthesisRepairEntry",
    "ScientificSynthesisRepairReport",
    "ScientificSynthesisSpecificationRepairContext",
    "build_scientific_synthesis_repair_contexts",
    "build_scientific_synthesis_repair_prompt",
    "lock_repair_provenance",
]
