from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicClaimKind,
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    CompiledLiteralEndpointBinding,
    RelationalAtomicEndpointBindingReport,
    selected_binding_claims,
)
from pipeline_core.discovery.scientific_relation_ir import (
    NullRelationTypingAdapter,
    RelationTypingAdapter,
    ScientificRelationIRReport,
    compile_atomic_specification_relation_ir,
)
from pipeline_core.domain.domain_profile import ScientificDomainProfile


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ProjectionStatus = Literal[
    "PROJECTED",
    "SKIPPED_ENDPOINT_ABSTENTION",
    "ABSTAINED_SOURCE_BINDING",
]


_ATOMIC_KINDS: frozenset[str] = frozenset(
    {
        "mediator",
        "moderator_interaction",
        "context_condition",
        "pathway_competition",
        "descriptor_interaction",
        "distinctive_prediction",
        "mechanistic_link",
    }
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _exact_text_key(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _normalize_literal(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐-‒–—−-]+", " ", text)
    text = re.sub(r"[^\w\s+*/().,]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _surface_contains(text: str, phrase: str) -> bool:
    needle = _normalize_literal(phrase)
    haystack = _normalize_literal(text)
    return bool(needle and needle in haystack)


def _unique_text(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(str(raw or "").split()).strip()
        key = _normalize_literal(value)
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


class RelationalAtomicProjectionRow(StrictModel):
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    claim_id: str
    source_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_binding_outcome: str

    projection_status: ProjectionStatus
    reason_codes: list[str] = Field(default_factory=list)
    specification: CompiledAtomicSpecification | None = None

    candidate_final_authority_equivalence_inherited: Literal[True] = True
    source_claim_content_preserved: Literal[True] = True
    relation_endpoints_from_literal_binding_only: Literal[True] = True
    observable_from_exact_prediction_falsifier_source_binding_only: Literal[
        True
    ] = True

    scientific_content_added: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_state(self) -> "RelationalAtomicProjectionRow":
        if self.projection_status == "PROJECTED":
            if self.specification is None:
                raise ValueError("PROJECTED row requires specification")
            if self.reason_codes:
                raise ValueError("PROJECTED row cannot carry reason_codes")
        else:
            if self.specification is not None:
                raise ValueError("non-PROJECTED row cannot carry specification")
            if not self.reason_codes:
                raise ValueError("non-PROJECTED row requires reason_codes")
        return self


class RelationalAtomicProjectionReport(StrictModel):
    schema_version: Literal[
        "relational-atomic-projection-report-v1"
    ] = "relational-atomic-projection-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_endpoint_binding_report_id: str

    rows: list[RelationalAtomicProjectionRow]
    selected_claim_count: int = Field(ge=0)
    projected_claim_count: int = Field(ge=0)
    skipped_endpoint_abstention_count: int = Field(ge=0)
    source_binding_abstention_count: int = Field(ge=0)

    source_population_frozen_before_endpoint_binding: Literal[True] = True
    candidate_final_authority_equivalence_required: Literal[True] = True
    exact_prediction_source_binding_required: Literal[True] = True
    exact_falsifier_source_binding_required: Literal[True] = True
    shared_observable_identity_required: Literal[True] = True

    scientific_content_added: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "RelationalAtomicProjectionReport":
        if self.selected_claim_count != len(self.rows):
            raise ValueError("selected_claim_count mismatch")
        if self.projected_claim_count != sum(
            row.projection_status == "PROJECTED"
            for row in self.rows
        ):
            raise ValueError("projected_claim_count mismatch")
        if self.skipped_endpoint_abstention_count != sum(
            row.projection_status == "SKIPPED_ENDPOINT_ABSTENTION"
            for row in self.rows
        ):
            raise ValueError("skipped_endpoint_abstention_count mismatch")
        if self.source_binding_abstention_count != sum(
            row.projection_status == "ABSTAINED_SOURCE_BINDING"
            for row in self.rows
        ):
            raise ValueError("source_binding_abstention_count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("relational atomic projection SHA mismatch")
        if observed_id != "relational_atomic_projection:" + expected_sha[:20]:
            raise ValueError("relational atomic projection ID mismatch")
        return self


def _plan_claim_maps(
    plan: RelationalAtomicBindingPlan,
) -> tuple[
    dict[str, RelationalAtomicBindingHypothesisPlan],
    dict[str, RelationalAtomicBindingClaimPlan],
]:
    hypotheses: dict[str, RelationalAtomicBindingHypothesisPlan] = {}
    claims: dict[str, RelationalAtomicBindingClaimPlan] = {}
    for hypothesis in plan.hypotheses:
        if hypothesis.final_hypothesis_id in hypotheses:
            raise ValueError("duplicate final hypothesis in binding plan")
        hypotheses[hypothesis.final_hypothesis_id] = hypothesis
        for claim in hypothesis.claims:
            if claim.claim_id in claims:
                raise ValueError("duplicate claim ID in binding plan")
            claims[claim.claim_id] = claim
    return hypotheses, claims


def _revalidate_literal_endpoint_binding(
    *,
    claim: RelationalAtomicBindingClaimPlan,
    binding: CompiledLiteralEndpointBinding,
) -> list[str]:
    reasons: list[str] = []
    endpoints = list(binding.relation_endpoint_anchors)
    normalized = [_normalize_literal(value) for value in endpoints]

    if len(endpoints) < 2:
        reasons.append("fewer_than_two_relation_endpoints")
    if len(normalized) != len(set(normalized)):
        reasons.append("duplicate_relation_endpoints")

    for index, endpoint in enumerate(endpoints):
        if not _surface_contains(claim.claim_text, endpoint):
            reasons.append(f"endpoint_not_literal_in_claim:{index}")
        if not _surface_contains(claim.required_bridge, endpoint):
            reasons.append(f"endpoint_not_literal_in_bridge:{index}")

        endpoint_norm = _normalize_literal(endpoint)
        for identity_index, identity in enumerate(
            claim.prior_art_identity_terms
        ):
            identity_norm = _normalize_literal(identity)
            if identity_norm and (
                endpoint_norm == identity_norm
                or identity_norm in endpoint_norm
            ):
                reasons.append(
                    "branch_identity_leaks_into_endpoint:"
                    + str(index)
                    + ":"
                    + str(identity_index)
                )

    for label, spans in (
        ("scope", binding.scope_qualifier_spans),
        ("direction", binding.directional_qualifier_spans),
    ):
        for index, span in enumerate(spans):
            if not _surface_contains(claim.claim_text, span):
                reasons.append(f"{label}_not_literal_in_claim:{index}")
            if not _surface_contains(claim.required_bridge, span):
                reasons.append(f"{label}_not_literal_in_bridge:{index}")

    return list(dict.fromkeys(reasons))


def _candidate_card_and_claim(
    *,
    hypothesis_plan: RelationalAtomicBindingHypothesisPlan,
    claim_plan: RelationalAtomicBindingClaimPlan,
) -> tuple[object, NoveltyClaim]:
    source_portfolio_path = Path(hypothesis_plan.source_candidate_portfolio)
    query_plan_path = Path(hypothesis_plan.source_query_plan)

    if not source_portfolio_path.is_file():
        raise ValueError(
            "candidate source portfolio missing during projection: "
            + str(source_portfolio_path)
        )
    if not query_plan_path.is_file():
        raise ValueError(
            "candidate query plan missing during projection: "
            + str(query_plan_path)
        )

    if _sha256_file(source_portfolio_path) != (
        hypothesis_plan.source_candidate_portfolio_sha256
    ):
        raise ValueError(
            "candidate source portfolio changed after binding-plan freeze"
        )
    if _sha256_file(query_plan_path) != (
        hypothesis_plan.source_query_plan_sha256
    ):
        raise ValueError(
            "candidate query plan changed after binding-plan freeze"
        )

    portfolio = HypothesisPortfolio.model_validate_json(
        source_portfolio_path.read_text(encoding="utf-8")
    )
    cards = [
        row
        for row in portfolio.hypotheses
        if row.hypothesis_id == hypothesis_plan.candidate_hypothesis_id
    ]
    if len(cards) != 1:
        raise ValueError(
            "candidate source portfolio no longer resolves exactly one card"
        )
    candidate_card = cards[0]

    query_plan = LiteratureQueryPlan.model_validate_json(
        query_plan_path.read_text(encoding="utf-8")
    )
    if query_plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError(
            "candidate query-plan/source-portfolio provenance mismatch"
        )

    matching_claims = [
        claim
        for group in query_plan.claims
        if group.hypothesis_id == hypothesis_plan.candidate_hypothesis_id
        for claim in group.claims
        if claim.claim_id == claim_plan.claim_id
    ]
    if len(matching_claims) != 1:
        raise ValueError(
            "candidate query plan no longer resolves exactly one source claim"
        )
    source_claim = matching_claims[0]

    if _sha256_json(source_claim.model_dump(mode="json")) != (
        claim_plan.source_claim_sha256
    ):
        raise ValueError(
            "canonical N10 source claim changed after binding-plan freeze"
        )

    return candidate_card, source_claim


def _exact_observation_binding(
    *,
    candidate_card: object,
    claim: NoveltyClaim,
) -> tuple[object | None, object | None, list[str]]:
    prediction_key = _exact_text_key(claim.predicted_observation)
    falsifier_key = _exact_text_key(claim.falsification_condition)

    # NoveltyClaim.predicted_observation is the predicted observable/result
    # itself. HypothesisCard keeps explanatory prose separately in
    # PredictedObservation.rationale, so provenance must bind to `observable`,
    # not to the rationale text.
    predictions = [
        row
        for row in candidate_card.predicted_observations
        if _exact_text_key(row.observable) == prediction_key
    ]
    falsifiers = [
        row
        for row in candidate_card.falsification_criteria
        if _exact_text_key(row.falsifying_outcome) == falsifier_key
    ]

    reasons: list[str] = []
    if len(predictions) != 1:
        reasons.append(
            "prediction_exact_source_binding_cardinality:"
            + str(len(predictions))
        )
    if len(falsifiers) != 1:
        reasons.append(
            "falsifier_exact_source_binding_cardinality:"
            + str(len(falsifiers))
        )
    if reasons:
        return None, None, reasons

    prediction = predictions[0]
    falsifier = falsifiers[0]
    if _exact_text_key(prediction.observable) != _exact_text_key(
        falsifier.observable
    ):
        reasons.append("prediction_falsifier_observable_identity_mismatch")
        return None, None, reasons

    if not _exact_text_key(prediction.observable):
        reasons.append("observable_empty")
        return None, None, reasons

    return prediction, falsifier, []


def compile_relational_atomic_projection(
    *,
    plan: RelationalAtomicBindingPlan,
    endpoint_report: RelationalAtomicEndpointBindingReport,
) -> RelationalAtomicProjectionReport:
    if endpoint_report.source_binding_plan_id != plan.plan_id:
        raise ValueError("endpoint report/source binding plan ID mismatch")
    if endpoint_report.source_binding_plan_sha256 != plan.plan_sha256:
        raise ValueError("endpoint report/source binding plan SHA mismatch")

    expected_claims = selected_binding_claims(plan)
    expected_ids = [row.claim_id for row in expected_claims]
    observed_ids = [row.claim_id for row in endpoint_report.bindings]
    if expected_ids != observed_ids:
        raise ValueError(
            "endpoint report population/order differs from frozen binding plan"
        )

    hypothesis_by_final, claim_by_id = _plan_claim_maps(plan)
    rows: list[RelationalAtomicProjectionRow] = []

    for binding in endpoint_report.bindings:
        claim_plan = claim_by_id[binding.claim_id]
        if binding.source_claim_sha256 != claim_plan.source_claim_sha256:
            raise ValueError(
                "endpoint binding/source claim SHA mismatch: "
                + binding.claim_id
            )
        if binding.final_hypothesis_id != claim_plan.final_hypothesis_id:
            raise ValueError(
                "endpoint binding/final hypothesis lineage mismatch"
            )
        if binding.candidate_hypothesis_id != (
            claim_plan.candidate_hypothesis_id
        ):
            raise ValueError(
                "endpoint binding/candidate hypothesis lineage mismatch"
            )

        if binding.outcome == "ABSTAINED_UNBINDABLE":
            rows.append(
                RelationalAtomicProjectionRow(
                    candidate_hypothesis_id=(
                        binding.candidate_hypothesis_id
                    ),
                    final_hypothesis_id=binding.final_hypothesis_id,
                    claim_id=binding.claim_id,
                    source_claim_sha256=binding.source_claim_sha256,
                    endpoint_binding_outcome=binding.outcome,
                    projection_status="SKIPPED_ENDPOINT_ABSTENTION",
                    reason_codes=[
                        "literal_endpoint_binding_abstained"
                    ],
                )
            )
            continue

        literal_reasons = _revalidate_literal_endpoint_binding(
            claim=claim_plan,
            binding=binding,
        )
        if literal_reasons:
            raise ValueError(
                "persisted endpoint binding violates literal contract: "
                + binding.claim_id
                + "; "
                + ",".join(literal_reasons)
            )

        hypothesis_plan = hypothesis_by_final[
            binding.final_hypothesis_id
        ]
        candidate_card, source_claim = _candidate_card_and_claim(
            hypothesis_plan=hypothesis_plan,
            claim_plan=claim_plan,
        )

        reasons: list[str] = []
        if source_claim.kind not in _ATOMIC_KINDS:
            reasons.append(
                "unsupported_atomic_claim_kind:" + source_claim.kind
            )
        if source_claim.novelty_selection_role is None:
            reasons.append("missing_novelty_selection_role")

        prediction, falsifier, observation_reasons = (
            _exact_observation_binding(
                candidate_card=candidate_card,
                claim=source_claim,
            )
        )
        reasons.extend(observation_reasons)

        if reasons:
            rows.append(
                RelationalAtomicProjectionRow(
                    candidate_hypothesis_id=(
                        binding.candidate_hypothesis_id
                    ),
                    final_hypothesis_id=binding.final_hypothesis_id,
                    claim_id=binding.claim_id,
                    source_claim_sha256=binding.source_claim_sha256,
                    endpoint_binding_outcome=binding.outcome,
                    projection_status="ABSTAINED_SOURCE_BINDING",
                    reason_codes=list(dict.fromkeys(reasons)),
                )
            )
            continue

        assert prediction is not None
        assert falsifier is not None

        relation_nucleus = _unique_text(
            [
                *binding.relation_endpoint_anchors,
                *binding.directional_qualifier_spans,
            ]
        )

        specification = CompiledAtomicSpecification(
            local_id=(
                "RELATIONAL_ATOMIC_PROJECTION_"
                + str(source_claim.claim_rank)
            ),
            claim_id=source_claim.claim_id,
            kind=source_claim.kind,
            importance=source_claim.importance,
            novelty_selection_role=source_claim.novelty_selection_role,
            text=source_claim.text,
            rationale=source_claim.rationale,
            source_candidate_ids=[
                hypothesis_plan.candidate_hypothesis_id
            ],
            premise_statement_ids=list(
                candidate_card.premise_statement_ids
            ),
            gap_statement_ids=list(
                candidate_card.gap_statement_ids
            ),
            prior_art_identity_terms=list(
                source_claim.prior_art_identity_terms
            ),
            relation_endpoint_anchors=list(
                binding.relation_endpoint_anchors
            ),
            scope_qualifier_spans=list(
                binding.scope_qualifier_spans
            ),
            directional_qualifier_spans=list(
                binding.directional_qualifier_spans
            ),
            relation_nucleus_terms=relation_nucleus,
            distinguishing_terms=list(
                source_claim.distinguishing_terms
            ),
            required_bridge=source_claim.required_bridge,
            observable=prediction.observable,
            predicted_observation=source_claim.predicted_observation,
            falsification_condition=source_claim.falsification_condition,
            prediction_observation_id=prediction.observation_id,
            falsification_criterion_id=falsifier.criterion_id,
            search_concepts=list(source_claim.search_concepts),
            search_queries=list(source_claim.search_queries),
            scientific_structure=source_claim.scientific_structure,
            scientific_structure_reason_codes=list(
                source_claim.scientific_structure_reason_codes
            ),
        )

        rows.append(
            RelationalAtomicProjectionRow(
                candidate_hypothesis_id=binding.candidate_hypothesis_id,
                final_hypothesis_id=binding.final_hypothesis_id,
                claim_id=binding.claim_id,
                source_claim_sha256=binding.source_claim_sha256,
                endpoint_binding_outcome=binding.outcome,
                projection_status="PROJECTED",
                specification=specification,
            )
        )

    body = {
        "schema_version": "relational-atomic-projection-report-v1",
        "source_binding_plan_id": plan.plan_id,
        "source_binding_plan_sha256": plan.plan_sha256,
        "source_endpoint_binding_report_id": endpoint_report.report_id,
        "rows": [row.model_dump(mode="json") for row in rows],
        "selected_claim_count": len(rows),
        "projected_claim_count": sum(
            row.projection_status == "PROJECTED"
            for row in rows
        ),
        "skipped_endpoint_abstention_count": sum(
            row.projection_status == "SKIPPED_ENDPOINT_ABSTENTION"
            for row in rows
        ),
        "source_binding_abstention_count": sum(
            row.projection_status == "ABSTAINED_SOURCE_BINDING"
            for row in rows
        ),
        "source_population_frozen_before_endpoint_binding": True,
        "candidate_final_authority_equivalence_required": True,
        "exact_prediction_source_binding_required": True,
        "exact_falsifier_source_binding_required": True,
        "shared_observable_identity_required": True,
        "scientific_content_added": False,
        "novelty_assessment_performed": False,
        "verifier_result_observed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RelationalAtomicProjectionReport(
        **body,
        report_id="relational_atomic_projection:" + digest[:20],
        report_sha256=digest,
    )


def compile_relational_atomic_projection_relation_ir(
    *,
    report: RelationalAtomicProjectionReport,
    domain_profile: ScientificDomainProfile,
    typing_adapter: RelationTypingAdapter | None = None,
) -> ScientificRelationIRReport:
    adapter: RelationTypingAdapter = (
        typing_adapter
        if typing_adapter is not None
        else NullRelationTypingAdapter()
    )

    relations = [
        compile_atomic_specification_relation_ir(
            hypothesis_id=row.final_hypothesis_id,
            spec=row.specification,
            domain_profile=domain_profile,
            typing_adapter=adapter,
            source_contract="relational-atomic-projection-report-v1",
        )
        for row in report.rows
        if row.projection_status == "PROJECTED"
        and row.specification is not None
    ]

    return ScientificRelationIRReport(
        report_id=_stable_id(
            "scientific_relation_ir_report",
            report.report_id,
            domain_profile.profile_id,
            adapter.adapter_id,
            *[row.relation_ir_id for row in relations],
        ),
        source_atomic_report_id=report.report_id,
        source_contract="relational-atomic-projection-report-v1",
        domain_profile_id=domain_profile.profile_id,
        typing_adapter_id=adapter.adapter_id,
        relations=relations,
        relation_count=len(relations),
        ready_count=sum(
            row.typing_status == "READY"
            for row in relations
        ),
        partial_count=sum(
            row.typing_status == "PARTIAL"
            for row in relations
        ),
        ambiguous_count=sum(
            row.typing_status == "AMBIGUOUS"
            for row in relations
        ),
        structurally_invalid_count=sum(
            row.typing_status == "STRUCTURALLY_INVALID"
            for row in relations
        ),
    )


__all__ = [
    "RelationalAtomicProjectionRow",
    "RelationalAtomicProjectionReport",
    "compile_relational_atomic_projection",
    "compile_relational_atomic_projection_relation_ir",
]
