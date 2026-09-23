from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.nonobviousness_post_generation import (
    assert_candidate_final_authority_equivalent,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ClaimBindingStatus = Literal[
    "READY_FOR_LITERAL_ENDPOINT_BINDING",
    "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION",
]

HypothesisBindingStatus = Literal[
    "READY_FOR_LITERAL_ENDPOINT_BINDING",
    "NO_BINDABLE_CLAIMS",
    "NO_NOVELTY_BEARING_BINDABLE_CLAIM",
]


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


def _normalize(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐-‒–—−-]+", " ", text)
    text = re.sub(r"[^\w\s+*/().,]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _surface_contains(text: str, phrase: str) -> bool:
    needle = _normalize(phrase)
    haystack = _normalize(text)
    return bool(needle and needle in haystack)


class RelationalAtomicBindingClaimPlan(StrictModel):
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    kind: str
    importance: str
    novelty_selection_role: str | None = None

    claim_text: str
    rationale: str
    prior_art_identity_terms: list[str] = Field(default_factory=list)
    relation_nucleus_terms: list[str] = Field(default_factory=list)
    required_bridge: str
    predicted_observation: str
    falsification_condition: str
    search_concepts: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)

    source_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_status: ClaimBindingStatus
    reason_codes: list[str] = Field(default_factory=list)

    source_claim_mutated: Literal[False] = False
    novelty_role_mutated: Literal[False] = False
    scientific_content_added: Literal[False] = False
    relation_endpoints_inferred: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_state(self) -> "RelationalAtomicBindingClaimPlan":
        if (
            self.binding_status
            == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and self.reason_codes
        ):
            raise ValueError(
                "binding-ready claim cannot carry readiness failures"
            )
        if (
            self.binding_status
            == "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
            and not self.reason_codes
        ):
            raise ValueError(
                "ineligible claim requires reason_codes"
            )
        return self


class RelationalAtomicBindingHypothesisPlan(StrictModel):
    original_hypothesis_id: str
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    alpha6_decision: str
    certification_status: str
    n10_selection_class: str

    source_candidate_portfolio: str
    source_candidate_portfolio_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_query_plan: str
    source_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    claim_count: int = Field(ge=0)
    binding_ready_claim_count: int = Field(ge=0)
    novelty_bearing_binding_ready_claim_count: int = Field(ge=0)
    binding_status: HypothesisBindingStatus
    claims: list[RelationalAtomicBindingClaimPlan] = Field(
        default_factory=list
    )

    candidate_final_authority_equivalent: Literal[True] = True
    endpoint_binding_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "RelationalAtomicBindingHypothesisPlan":
        if self.claim_count != len(self.claims):
            raise ValueError("claim_count mismatch")
        ready = sum(
            row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in self.claims
        )
        if self.binding_ready_claim_count != ready:
            raise ValueError("binding_ready_claim_count mismatch")
        novelty_ready = sum(
            row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in self.claims
        )
        if self.novelty_bearing_binding_ready_claim_count != novelty_ready:
            raise ValueError(
                "novelty_bearing_binding_ready_claim_count mismatch"
            )
        if (
            self.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and novelty_ready < 1
        ):
            raise ValueError(
                "binding-ready hypothesis requires a novelty-bearing "
                "binding-ready claim"
            )
        if (
            self.binding_status == "NO_BINDABLE_CLAIMS"
            and ready != 0
        ):
            raise ValueError("NO_BINDABLE_CLAIMS has ready claims")
        if (
            self.binding_status
            == "NO_NOVELTY_BEARING_BINDABLE_CLAIM"
            and not (ready > 0 and novelty_ready == 0)
        ):
            raise ValueError(
                "NO_NOVELTY_BEARING_BINDABLE_CLAIM state mismatch"
            )
        return self


class RelationalAtomicBindingPlan(StrictModel):
    schema_version: Literal[
        "relational-atomic-binding-plan-v1"
    ] = "relational-atomic-binding-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_dir: str

    source_alpha6_candidate_portfolio: str
    source_alpha6_candidate_portfolio_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_certification_report: str
    source_certification_report_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    hypotheses: list[RelationalAtomicBindingHypothesisPlan]
    hypothesis_count: int = Field(ge=0)
    ready_hypothesis_count: int = Field(ge=0)
    not_ready_hypothesis_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    binding_ready_claim_count: int = Field(ge=0)
    novelty_bearing_binding_ready_claim_count: int = Field(ge=0)
    hypothesis_status_counts: dict[str, int]
    claim_status_counts: dict[str, int]

    certification_only_source_required: Literal[True] = True
    candidate_final_authority_equivalence_required: Literal[True] = True
    endpoint_binding_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "RelationalAtomicBindingPlan":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        ready_h = sum(
            row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in self.hypotheses
        )
        if self.ready_hypothesis_count != ready_h:
            raise ValueError("ready_hypothesis_count mismatch")
        if self.not_ready_hypothesis_count != len(self.hypotheses) - ready_h:
            raise ValueError("not_ready_hypothesis_count mismatch")
        if self.claim_count != sum(row.claim_count for row in self.hypotheses):
            raise ValueError("claim_count mismatch")
        if self.binding_ready_claim_count != sum(
            row.binding_ready_claim_count for row in self.hypotheses
        ):
            raise ValueError("binding_ready_claim_count mismatch")
        if self.novelty_bearing_binding_ready_claim_count != sum(
            row.novelty_bearing_binding_ready_claim_count
            for row in self.hypotheses
        ):
            raise ValueError(
                "novelty_bearing_binding_ready_claim_count mismatch"
            )

        expected_h = Counter(row.binding_status for row in self.hypotheses)
        if dict(sorted(expected_h.items())) != dict(
            sorted(self.hypothesis_status_counts.items())
        ):
            raise ValueError("hypothesis_status_counts mismatch")

        expected_c = Counter(
            claim.binding_status
            for row in self.hypotheses
            for claim in row.claims
        )
        if dict(sorted(expected_c.items())) != dict(
            sorted(self.claim_status_counts.items())
        ):
            raise ValueError("claim_status_counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("binding plan SHA mismatch")
        if observed_id != "relational_atomic_binding_plan:" + expected_sha[:20]:
            raise ValueError("binding plan ID mismatch")
        return self


def assess_claim_binding_readiness(
    *,
    claim: NoveltyClaim,
    candidate_hypothesis_id: str,
    final_hypothesis_id: str,
) -> RelationalAtomicBindingClaimPlan:
    reasons: list[str] = []

    if not claim.text.strip():
        reasons.append("missing_claim_text")
    if not claim.required_bridge.strip():
        reasons.append("missing_required_bridge")
    if not claim.predicted_observation.strip():
        reasons.append("missing_predicted_observation")
    if not claim.falsification_condition.strip():
        reasons.append("missing_falsification_condition")
    if not claim.prior_art_identity_terms:
        reasons.append("missing_prior_art_identity_terms")
    if claim.novelty_selection_role is None:
        reasons.append("missing_novelty_selection_role")

    for index, identity in enumerate(claim.prior_art_identity_terms):
        for label, text in (
            ("claim_text", claim.text),
            ("required_bridge", claim.required_bridge),
            ("predicted_observation", claim.predicted_observation),
            ("falsification_condition", claim.falsification_condition),
        ):
            if text.strip() and not _surface_contains(text, identity):
                reasons.append(
                    f"identity_not_literal_in_{label}:{index}"
                )

    reasons = list(dict.fromkeys(reasons))
    status: ClaimBindingStatus = (
        "READY_FOR_LITERAL_ENDPOINT_BINDING"
        if not reasons
        else "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
    )

    payload = claim.model_dump(mode="json")
    return RelationalAtomicBindingClaimPlan(
        candidate_hypothesis_id=candidate_hypothesis_id,
        final_hypothesis_id=final_hypothesis_id,
        claim_id=claim.claim_id,
        claim_rank=claim.claim_rank,
        kind=claim.kind,
        importance=claim.importance,
        novelty_selection_role=claim.novelty_selection_role,
        claim_text=claim.text,
        rationale=claim.rationale,
        prior_art_identity_terms=list(claim.prior_art_identity_terms),
        relation_nucleus_terms=list(claim.relation_nucleus_terms),
        required_bridge=claim.required_bridge,
        predicted_observation=claim.predicted_observation,
        falsification_condition=claim.falsification_condition,
        search_concepts=list(claim.search_concepts),
        search_queries=list(claim.search_queries),
        source_claim_sha256=_sha256_json(payload),
        binding_status=status,
        reason_codes=reasons,
    )


def _load_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_relational_atomic_binding_plan(
    *,
    run_dir: Path,
) -> RelationalAtomicBindingPlan:
    run = run_dir.expanduser().resolve()
    candidate_portfolio_path = (
        run / "novelty_refinement_a6.n10.candidate.portfolio.json"
    )
    certification_path = (
        run / "novelty_refinement_a6.n10.certification.json"
    )

    if not candidate_portfolio_path.is_file():
        raise ValueError(
            "missing Alpha6 N10 candidate portfolio: "
            + str(candidate_portfolio_path)
        )
    if not certification_path.is_file():
        raise ValueError(
            "missing Alpha6 N10 certification report: "
            + str(certification_path)
        )

    final_portfolio = HypothesisPortfolio.model_validate_json(
        candidate_portfolio_path.read_text(encoding="utf-8")
    )
    certification = _load_json_object(certification_path)

    if certification.get("schema_version") != (
        "alpha6-post-generation-novelty-certification-v2"
    ):
        raise ValueError("binding planner requires certification v2")
    if certification.get("authority_mode") != "certification_only":
        raise ValueError("binding planner requires certification_only")
    if certification.get("candidate_portfolio_preserved") is not True:
        raise ValueError("candidate portfolio must be preserved")
    if certification.get("candidate_survival_authority") is not False:
        raise ValueError("certification must not control candidate survival")
    if certification.get("ineligible_deletes_scientific_candidate") is not False:
        raise ValueError("certification must not delete scientific candidates")
    if certification.get("candidate_portfolio_id") != final_portfolio.portfolio_id:
        raise ValueError("certification/final candidate portfolio mismatch")

    final_by_id = {
        row.hypothesis_id: row
        for row in final_portfolio.hypotheses
    }
    if len(final_by_id) != len(final_portfolio.hypotheses):
        raise ValueError("duplicate final hypothesis IDs")

    decisions = certification.get("decisions")
    artifacts = certification.get("candidate_artifacts")
    if not isinstance(decisions, list):
        raise ValueError("certification decisions missing")
    if not isinstance(artifacts, list):
        raise ValueError("candidate_artifacts missing")

    generated_decisions = [
        row
        for row in decisions
        if isinstance(row, dict)
        and row.get("post_generation_n10_required") is True
    ]
    artifact_by_candidate = {
        str(row.get("candidate_id") or ""): row
        for row in artifacts
        if isinstance(row, dict)
    }
    if "" in artifact_by_candidate:
        raise ValueError("candidate artifact missing candidate_id")
    if len(artifact_by_candidate) != len(artifacts):
        raise ValueError("duplicate candidate artifact IDs")

    decision_candidate_ids = {
        str(row.get("candidate_hypothesis_id") or "")
        for row in generated_decisions
    }
    if "" in decision_candidate_ids:
        raise ValueError("generated decision missing candidate hypothesis ID")
    if set(artifact_by_candidate) != decision_candidate_ids:
        raise ValueError(
            "candidate artifact/decision membership mismatch"
        )

    hypothesis_plans: list[RelationalAtomicBindingHypothesisPlan] = []

    for decision in generated_decisions:
        original_id = str(decision.get("original_hypothesis_id") or "")
        candidate_id = str(decision.get("candidate_hypothesis_id") or "")
        final_id = str(decision.get("final_hypothesis_id") or "")
        if not original_id or not candidate_id or not final_id:
            raise ValueError("incomplete Alpha6 lineage decision")

        final_card = final_by_id.get(final_id)
        if final_card is None:
            raise ValueError(
                "final Alpha6 card missing for " + final_id
            )

        artifact = artifact_by_candidate[candidate_id]
        if str(artifact.get("final_hypothesis_id") or "") != final_id:
            raise ValueError(
                "candidate artifact final-hypothesis lineage mismatch"
            )
        if artifact.get("candidate_final_authority_equivalent") is not True:
            raise ValueError(
                "candidate artifact lacks authority-equivalence proof"
            )

        source_portfolio_path = Path(
            str(artifact.get("source_portfolio") or "")
        )
        query_plan_path = Path(
            str(artifact.get("query_plan") or "")
        )
        if not source_portfolio_path.is_file():
            raise ValueError(
                "missing candidate source portfolio: "
                + str(source_portfolio_path)
            )
        if not query_plan_path.is_file():
            raise ValueError(
                "missing candidate query plan: "
                + str(query_plan_path)
            )

        source_portfolio = HypothesisPortfolio.model_validate_json(
            source_portfolio_path.read_text(encoding="utf-8")
        )
        candidate_cards = [
            row
            for row in source_portfolio.hypotheses
            if row.hypothesis_id == candidate_id
        ]
        if len(candidate_cards) != 1:
            raise ValueError(
                "candidate source portfolio must resolve exactly one card"
            )
        candidate_card = candidate_cards[0]

        assert_candidate_final_authority_equivalent(
            candidate=candidate_card,
            final=final_card,
        )

        query_plan = LiteratureQueryPlan.model_validate_json(
            query_plan_path.read_text(encoding="utf-8")
        )
        if query_plan.source_portfolio_id != source_portfolio.portfolio_id:
            raise ValueError(
                "query-plan/source-portfolio provenance mismatch"
            )
        claim_groups = [
            row
            for row in query_plan.claims
            if row.hypothesis_id == candidate_id
        ]
        if len(claim_groups) != 1:
            raise ValueError(
                "query plan must resolve exactly one candidate claim group"
            )

        claim_plans = [
            assess_claim_binding_readiness(
                claim=claim,
                candidate_hypothesis_id=candidate_id,
                final_hypothesis_id=final_id,
            )
            for claim in claim_groups[0].claims
        ]
        ready_count = sum(
            row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in claim_plans
        )
        novelty_ready_count = sum(
            row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in claim_plans
        )

        if ready_count == 0:
            binding_status: HypothesisBindingStatus = "NO_BINDABLE_CLAIMS"
        elif novelty_ready_count == 0:
            binding_status = "NO_NOVELTY_BEARING_BINDABLE_CLAIM"
        else:
            binding_status = "READY_FOR_LITERAL_ENDPOINT_BINDING"

        hypothesis_plans.append(
            RelationalAtomicBindingHypothesisPlan(
                original_hypothesis_id=original_id,
                candidate_hypothesis_id=candidate_id,
                final_hypothesis_id=final_id,
                alpha6_decision=str(
                    decision.get("alpha6_decision") or ""
                ),
                certification_status=str(
                    decision.get("certification_status") or ""
                ),
                n10_selection_class=str(
                    decision.get("n10_selection_class") or ""
                ),
                source_candidate_portfolio=str(source_portfolio_path),
                source_candidate_portfolio_sha256=_sha256_file(
                    source_portfolio_path
                ),
                source_query_plan=str(query_plan_path),
                source_query_plan_sha256=_sha256_file(query_plan_path),
                claim_count=len(claim_plans),
                binding_ready_claim_count=ready_count,
                novelty_bearing_binding_ready_claim_count=novelty_ready_count,
                binding_status=binding_status,
                claims=claim_plans,
            )
        )

    h_counts = Counter(row.binding_status for row in hypothesis_plans)
    c_counts = Counter(
        claim.binding_status
        for row in hypothesis_plans
        for claim in row.claims
    )

    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": str(run),
        "source_alpha6_candidate_portfolio": str(
            candidate_portfolio_path
        ),
        "source_alpha6_candidate_portfolio_sha256": _sha256_file(
            candidate_portfolio_path
        ),
        "source_certification_report": str(certification_path),
        "source_certification_report_sha256": _sha256_file(
            certification_path
        ),
        "hypotheses": [
            row.model_dump(mode="json")
            for row in hypothesis_plans
        ],
        "hypothesis_count": len(hypothesis_plans),
        "ready_hypothesis_count": sum(
            row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in hypothesis_plans
        ),
        "not_ready_hypothesis_count": sum(
            row.binding_status != "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in hypothesis_plans
        ),
        "claim_count": sum(row.claim_count for row in hypothesis_plans),
        "binding_ready_claim_count": sum(
            row.binding_ready_claim_count
            for row in hypothesis_plans
        ),
        "novelty_bearing_binding_ready_claim_count": sum(
            row.novelty_bearing_binding_ready_claim_count
            for row in hypothesis_plans
        ),
        "hypothesis_status_counts": dict(sorted(h_counts.items())),
        "claim_status_counts": dict(sorted(c_counts.items())),
        "certification_only_source_required": True,
        "candidate_final_authority_equivalence_required": True,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


__all__ = [
    "RelationalAtomicBindingClaimPlan",
    "RelationalAtomicBindingHypothesisPlan",
    "RelationalAtomicBindingPlan",
    "assess_claim_binding_readiness",
    "build_relational_atomic_binding_plan",
]
