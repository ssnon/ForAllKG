from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    RelationalAtomicEndpointBindingReport,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    RelationalVerifierArtifactFingerprint,
    fingerprint_file,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ClaimLaneProfile = Literal[
    "RELATIONAL_ONLY",
    "REFRAMING_ONLY",
    "MIXED",
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


class ProspectiveRelationalVerifierPoolCaseSpec(StrictModel):
    case_key: str = Field(min_length=1)
    production_candidate_portfolio: str = Field(min_length=1)
    atomic_synthesis_report: str = Field(min_length=1)
    binding_plan: str = Field(min_length=1)
    endpoint_binding_report: str = Field(min_length=1)


class ProspectiveRelationalVerifierPoolSpec(StrictModel):
    schema_version: Literal[
        "prospective-relational-verifier-pool-spec-v1"
    ] = "prospective-relational-verifier-pool-spec-v1"

    cases: list[ProspectiveRelationalVerifierPoolCaseSpec]
    requested_case_count: int = Field(default=5, ge=1)
    requested_relational_only: int = Field(default=2, ge=0)
    requested_reframing_only: int = Field(default=2, ge=0)
    requested_mixed: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_spec(self) -> "ProspectiveRelationalVerifierPoolSpec":
        keys = [row.case_key for row in self.cases]
        if len(keys) != len(set(keys)):
            raise ValueError("pool case_key values must be unique")
        if (
            self.requested_relational_only
            + self.requested_reframing_only
            + self.requested_mixed
            > self.requested_case_count
        ):
            raise ValueError(
                "requested lane quotas cannot exceed requested_case_count"
            )
        return self


class ProspectiveRelationalVerifierClaimCandidate(StrictModel):
    claim_id: str
    novelty_selection_role: str
    binding_status: str
    endpoint_outcome: str
    source_candidate_ids: list[str]
    source_lanes: list[str]
    lane_profile: ClaimLaneProfile | None = None
    prior_art_identity_term_count: int = Field(ge=0)
    relation_endpoint_count: int = Field(ge=0)


class ProspectiveRelationalVerifierHypothesisCandidate(StrictModel):
    pool_case_key: str

    original_hypothesis_id: str
    candidate_hypothesis_id: str
    final_hypothesis_id: str

    synthesis_kind: str
    hypothesis_source_lanes: list[str]
    claim_count: int = Field(ge=0)
    binding_ready_claim_count: int = Field(ge=0)
    novelty_bearing_binding_ready_claim_count: int = Field(ge=0)
    novelty_bearing_bound_claim_count: int = Field(ge=0)

    novelty_bearing_bound_lane_profiles: list[ClaimLaneProfile]
    primary_lane_profile: ClaimLaneProfile
    claim_candidates: list[ProspectiveRelationalVerifierClaimCandidate]

    input_artifacts: list[RelationalVerifierArtifactFingerprint]

    eligible_for_prospective_verifier: Literal[True] = True
    candidate_final_authority_equivalence_verified: Literal[True] = True
    endpoint_binding_observed_before_selection: Literal[True] = True
    prior_art_outcomes_observed_for_selection: Literal[False] = False
    external_novelty_outcomes_observed_for_selection: Literal[False] = False
    new_verifier_outcomes_observed_for_selection: Literal[False] = False
    old_n10_outcome_fields_used_for_selection: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False


class ProspectiveRelationalVerifierExcludedCandidate(StrictModel):
    pool_case_key: str
    original_hypothesis_id: str
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    reason_codes: list[str]


class FrozenProspectiveRelationalVerifierCase(StrictModel):
    prospective_case_id: str
    pool_case_key: str
    final_hypothesis_id: str
    candidate_hypothesis_id: str
    original_hypothesis_id: str
    primary_lane_profile: ClaimLaneProfile
    synthesis_kind: str
    novelty_bearing_bound_claim_ids: list[str]
    input_artifacts: list[RelationalVerifierArtifactFingerprint]


class ProspectiveRelationalVerifierCohortFreeze(StrictModel):
    schema_version: Literal[
        "prospective-relational-verifier-cohort-freeze-v1"
    ] = "prospective-relational-verifier-cohort-freeze-v1"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_pool_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_case_count: int = Field(ge=1)
    requested_lane_quotas: dict[str, int]

    eligible_candidates: list[
        ProspectiveRelationalVerifierHypothesisCandidate
    ]
    excluded_candidates: list[
        ProspectiveRelationalVerifierExcludedCandidate
    ]
    frozen_cases: list[FrozenProspectiveRelationalVerifierCase]

    eligible_candidate_count: int = Field(ge=0)
    excluded_candidate_count: int = Field(ge=0)
    frozen_case_count: int = Field(ge=0)
    frozen_lane_profile_counts: dict[str, int]

    selection_algorithm: Literal[
        "lane_quota_then_stable_structural_order_v1"
    ] = "lane_quota_then_stable_structural_order_v1"

    pre_verifier_structural_inputs_only: Literal[True] = True
    endpoint_binding_is_pre_verifier_annotation_only: Literal[True] = True
    prior_art_retrieval_used_for_selection: Literal[False] = False
    relation_adjudication_used_for_selection: Literal[False] = False
    external_novelty_outcomes_used_for_selection: Literal[False] = False
    positive_nonobviousness_used_for_selection: Literal[False] = False
    final_certification_used_for_selection: Literal[False] = False
    old_n10_outcome_fields_used_for_selection: Literal[False] = False
    new_verifier_result_observed_before_freeze: Literal[False] = False
    post_freeze_selection_change_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "ProspectiveRelationalVerifierCohortFreeze":
        if self.eligible_candidate_count != len(self.eligible_candidates):
            raise ValueError("eligible_candidate_count mismatch")
        if self.excluded_candidate_count != len(self.excluded_candidates):
            raise ValueError("excluded_candidate_count mismatch")
        if self.frozen_case_count != len(self.frozen_cases):
            raise ValueError("frozen_case_count mismatch")

        expected_case_ids = [
            f"P{index:02d}"
            for index in range(6, 6 + len(self.frozen_cases))
        ]
        if [
            row.prospective_case_id for row in self.frozen_cases
        ] != expected_case_ids:
            raise ValueError("prospective case IDs must be contiguous P06+")

        observed_counts = Counter(
            row.primary_lane_profile
            for row in self.frozen_cases
        )
        if dict(sorted(observed_counts.items())) != dict(
            sorted(self.frozen_lane_profile_counts.items())
        ):
            raise ValueError("frozen_lane_profile_counts mismatch")

        ids = [row.final_hypothesis_id for row in self.frozen_cases]
        if len(ids) != len(set(ids)):
            raise ValueError("frozen final hypothesis IDs must be unique")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective cohort freeze SHA mismatch")
        if observed_id != (
            "prospective_relational_verifier_cohort_freeze:"
            + expected_sha[:20]
        ):
            raise ValueError("prospective cohort freeze ID mismatch")
        return self


def _resolve_artifact(
    path: str,
    *,
    base_dir: Path,
) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = base_dir / value
    value = value.resolve()
    if not value.is_file():
        raise ValueError("missing pool artifact: " + str(value))
    return value


def _lane_profile(lanes: set[str]) -> ClaimLaneProfile:
    if lanes == {"RELATIONAL_DISCOVERY"}:
        return "RELATIONAL_ONLY"
    if lanes == {"SCIENTIFIC_REFRAMING"}:
        return "REFRAMING_ONLY"
    if lanes == {
        "RELATIONAL_DISCOVERY",
        "SCIENTIFIC_REFRAMING",
    }:
        return "MIXED"
    raise ValueError("unsupported source-lane set: " + repr(sorted(lanes)))


def _stable_candidate_key(
    row: ProspectiveRelationalVerifierHypothesisCandidate,
) -> tuple:
    # No novelty/prior-art/verifier outcome fields are allowed here.
    return (
        -row.novelty_bearing_bound_claim_count,
        -row.novelty_bearing_binding_ready_claim_count,
        -row.binding_ready_claim_count,
        row.claim_count,
        row.synthesis_kind,
        row.pool_case_key,
        row.final_hypothesis_id,
    )


def _select_with_quotas(
    rows: list[ProspectiveRelationalVerifierHypothesisCandidate],
    *,
    requested_case_count: int,
    requested_relational_only: int,
    requested_reframing_only: int,
    requested_mixed: int,
) -> list[ProspectiveRelationalVerifierHypothesisCandidate]:
    ordered = sorted(rows, key=_stable_candidate_key)
    selected: list[ProspectiveRelationalVerifierHypothesisCandidate] = []
    used: set[str] = set()

    quotas = [
        ("RELATIONAL_ONLY", requested_relational_only),
        ("REFRAMING_ONLY", requested_reframing_only),
        ("MIXED", requested_mixed),
    ]
    for profile, quota in quotas:
        matches = [
            row
            for row in ordered
            if row.primary_lane_profile == profile
            and row.final_hypothesis_id not in used
        ]
        for row in matches[:quota]:
            selected.append(row)
            used.add(row.final_hypothesis_id)

    for row in ordered:
        if len(selected) >= requested_case_count:
            break
        if row.final_hypothesis_id in used:
            continue
        selected.append(row)
        used.add(row.final_hypothesis_id)

    return selected[:requested_case_count]


def build_prospective_relational_verifier_cohort_freeze(
    *,
    pool_spec: ProspectiveRelationalVerifierPoolSpec,
    pool_spec_sha256: str,
    base_dir: Path,
) -> ProspectiveRelationalVerifierCohortFreeze:
    base_dir = base_dir.expanduser().resolve()

    eligible: list[ProspectiveRelationalVerifierHypothesisCandidate] = []
    excluded: list[ProspectiveRelationalVerifierExcludedCandidate] = []

    for case in pool_spec.cases:
        production_path = _resolve_artifact(
            case.production_candidate_portfolio,
            base_dir=base_dir,
        )
        atomic_path = _resolve_artifact(
            case.atomic_synthesis_report,
            base_dir=base_dir,
        )
        plan_path = _resolve_artifact(
            case.binding_plan,
            base_dir=base_dir,
        )
        endpoint_path = _resolve_artifact(
            case.endpoint_binding_report,
            base_dir=base_dir,
        )

        production = (
            ProductionFacingScientificCandidatePortfolio.model_validate_json(
                production_path.read_text(encoding="utf-8")
            )
        )
        atomic = AtomicCrossLaneSynthesisReport.model_validate_json(
            atomic_path.read_text(encoding="utf-8")
        )
        plan = RelationalAtomicBindingPlan.model_validate_json(
            plan_path.read_text(encoding="utf-8")
        )
        endpoint = RelationalAtomicEndpointBindingReport.model_validate_json(
            endpoint_path.read_text(encoding="utf-8")
        )

        if atomic.source_candidate_portfolio_id != production.portfolio_id:
            raise ValueError(
                case.case_key
                + ": atomic/production candidate portfolio mismatch"
            )
        if endpoint.source_binding_plan_id != plan.plan_id:
            raise ValueError(
                case.case_key + ": endpoint/binding plan ID mismatch"
            )
        if endpoint.source_binding_plan_sha256 != plan.plan_sha256:
            raise ValueError(
                case.case_key + ": endpoint/binding plan SHA mismatch"
            )
        if atomic.novelty_assessment_performed is not False:
            raise ValueError(
                case.case_key
                + ": atomic report unexpectedly performed novelty assessment"
            )
        if endpoint.verifier_result_observed is not False:
            raise ValueError(
                case.case_key
                + ": endpoint binding observed verifier results"
            )

        lane_by_candidate_id = {
            row.candidate_id: row.source_lane
            for row in production.candidates
        }
        atomic_by_hypothesis = {
            row.hypothesis_id: row
            for row in atomic.hypotheses
        }
        if len(atomic_by_hypothesis) != len(atomic.hypotheses):
            raise ValueError(
                case.case_key + ": duplicate atomic hypothesis IDs"
            )

        binding_by_claim = {
            row.claim_id: row
            for row in endpoint.bindings
        }
        if len(binding_by_claim) != len(endpoint.bindings):
            raise ValueError(
                case.case_key + ": duplicate endpoint claim IDs"
            )

        case_fingerprints = [
            fingerprint_file(production_path),
            fingerprint_file(atomic_path),
            fingerprint_file(plan_path),
            fingerprint_file(endpoint_path),
        ]

        for hypothesis in plan.hypotheses:
            original = atomic_by_hypothesis.get(
                hypothesis.original_hypothesis_id
            )
            reasons: list[str] = []
            if original is None:
                reasons.append("original_atomic_hypothesis_missing")
            if (
                hypothesis.binding_status
                != "READY_FOR_LITERAL_ENDPOINT_BINDING"
            ):
                reasons.append("hypothesis_not_binding_ready")
            if (
                hypothesis.novelty_bearing_binding_ready_claim_count < 1
            ):
                reasons.append(
                    "no_novelty_bearing_binding_ready_claim"
                )

            atomic_specs = (
                {
                    row.claim_id: row
                    for row in original.atomic_specifications
                }
                if original is not None
                else {}
            )
            claim_rows: list[
                ProspectiveRelationalVerifierClaimCandidate
            ] = []
            bound_profiles: list[ClaimLaneProfile] = []
            novelty_bound_count = 0

            for claim in hypothesis.claims:
                spec = atomic_specs.get(claim.claim_id)
                endpoint_row = binding_by_claim.get(claim.claim_id)

                lanes: set[str] = set()
                source_candidate_ids: list[str] = []
                profile: ClaimLaneProfile | None = None
                endpoint_outcome = "MISSING"

                if spec is not None:
                    source_candidate_ids = list(spec.source_candidate_ids)
                    missing_sources = sorted(
                        set(source_candidate_ids) - set(lane_by_candidate_id)
                    )
                    if missing_sources:
                        raise ValueError(
                            case.case_key
                            + ": atomic specification references unknown "
                            + "production candidate IDs: "
                            + repr(missing_sources)
                        )
                    lanes = {
                        lane_by_candidate_id[value]
                        for value in source_candidate_ids
                    }
                    profile = _lane_profile(lanes)

                if endpoint_row is not None:
                    if (
                        endpoint_row.final_hypothesis_id
                        != hypothesis.final_hypothesis_id
                    ):
                        raise ValueError(
                            case.case_key
                            + ": endpoint/final hypothesis mismatch"
                        )
                    endpoint_outcome = endpoint_row.outcome

                claim_rows.append(
                    ProspectiveRelationalVerifierClaimCandidate(
                        claim_id=claim.claim_id,
                        novelty_selection_role=(
                            claim.novelty_selection_role or ""
                        ),
                        binding_status=claim.binding_status,
                        endpoint_outcome=endpoint_outcome,
                        source_candidate_ids=source_candidate_ids,
                        source_lanes=sorted(lanes),
                        lane_profile=profile,
                        prior_art_identity_term_count=len(
                            claim.prior_art_identity_terms
                        ),
                        relation_endpoint_count=(
                            len(endpoint_row.relation_endpoint_anchors)
                            if endpoint_row is not None
                            else 0
                        ),
                    )
                )

                if (
                    claim.novelty_selection_role == "NOVELTY_BEARING"
                    and claim.binding_status
                    == "READY_FOR_LITERAL_ENDPOINT_BINDING"
                    and endpoint_row is not None
                    and endpoint_row.outcome
                    == "BOUND_LITERAL_ENDPOINTS"
                ):
                    if profile is None:
                        reasons.append(
                            "bound_novelty_claim_missing_atomic_source_lineage"
                        )
                    else:
                        novelty_bound_count += 1
                        bound_profiles.append(profile)

            if novelty_bound_count < 1:
                reasons.append("no_novelty_bearing_literal_endpoint_binding")

            if reasons:
                excluded.append(
                    ProspectiveRelationalVerifierExcludedCandidate(
                        pool_case_key=case.case_key,
                        original_hypothesis_id=(
                            hypothesis.original_hypothesis_id
                        ),
                        candidate_hypothesis_id=(
                            hypothesis.candidate_hypothesis_id
                        ),
                        final_hypothesis_id=(
                            hypothesis.final_hypothesis_id
                        ),
                        reason_codes=list(dict.fromkeys(reasons)),
                    )
                )
                continue

            # A hypothesis with multiple novelty-bearing bound claims can span
            # multiple claim-level lane profiles. Use MIXED as the hypothesis
            # profile unless every such claim has the same lane provenance.
            unique_profiles = set(bound_profiles)
            primary: ClaimLaneProfile = (
                next(iter(unique_profiles))
                if len(unique_profiles) == 1
                else "MIXED"
            )

            eligible.append(
                ProspectiveRelationalVerifierHypothesisCandidate(
                    pool_case_key=case.case_key,
                    original_hypothesis_id=(
                        hypothesis.original_hypothesis_id
                    ),
                    candidate_hypothesis_id=(
                        hypothesis.candidate_hypothesis_id
                    ),
                    final_hypothesis_id=(
                        hypothesis.final_hypothesis_id
                    ),
                    synthesis_kind=original.synthesis_kind,
                    hypothesis_source_lanes=list(original.source_lanes),
                    claim_count=hypothesis.claim_count,
                    binding_ready_claim_count=(
                        hypothesis.binding_ready_claim_count
                    ),
                    novelty_bearing_binding_ready_claim_count=(
                        hypothesis.novelty_bearing_binding_ready_claim_count
                    ),
                    novelty_bearing_bound_claim_count=novelty_bound_count,
                    novelty_bearing_bound_lane_profiles=sorted(
                        bound_profiles
                    ),
                    primary_lane_profile=primary,
                    claim_candidates=claim_rows,
                    input_artifacts=case_fingerprints,
                )
            )

    selected = _select_with_quotas(
        eligible,
        requested_case_count=pool_spec.requested_case_count,
        requested_relational_only=pool_spec.requested_relational_only,
        requested_reframing_only=pool_spec.requested_reframing_only,
        requested_mixed=pool_spec.requested_mixed,
    )

    if len(selected) < pool_spec.requested_case_count:
        raise ValueError(
            "insufficient eligible prospective verifier hypotheses: "
            f"requested={pool_spec.requested_case_count}; "
            f"eligible={len(selected)}"
        )

    frozen_cases: list[FrozenProspectiveRelationalVerifierCase] = []
    for index, row in enumerate(selected, start=6):
        bound_claim_ids = [
            claim.claim_id
            for claim in row.claim_candidates
            if (
                claim.novelty_selection_role == "NOVELTY_BEARING"
                and claim.endpoint_outcome == "BOUND_LITERAL_ENDPOINTS"
            )
        ]
        frozen_cases.append(
            FrozenProspectiveRelationalVerifierCase(
                prospective_case_id=f"P{index:02d}",
                pool_case_key=row.pool_case_key,
                final_hypothesis_id=row.final_hypothesis_id,
                candidate_hypothesis_id=row.candidate_hypothesis_id,
                original_hypothesis_id=row.original_hypothesis_id,
                primary_lane_profile=row.primary_lane_profile,
                synthesis_kind=row.synthesis_kind,
                novelty_bearing_bound_claim_ids=bound_claim_ids,
                input_artifacts=row.input_artifacts,
            )
        )

    body = {
        "schema_version":
            "prospective-relational-verifier-cohort-freeze-v1",
        "source_pool_spec_sha256": pool_spec_sha256,
        "requested_case_count": pool_spec.requested_case_count,
        "requested_lane_quotas": {
            "RELATIONAL_ONLY": pool_spec.requested_relational_only,
            "REFRAMING_ONLY": pool_spec.requested_reframing_only,
            "MIXED": pool_spec.requested_mixed,
        },
        "eligible_candidates": [
            row.model_dump(mode="json")
            for row in sorted(eligible, key=_stable_candidate_key)
        ],
        "excluded_candidates": [
            row.model_dump(mode="json")
            for row in sorted(
                excluded,
                key=lambda value: (
                    value.pool_case_key,
                    value.final_hypothesis_id,
                ),
            )
        ],
        "frozen_cases": [
            row.model_dump(mode="json")
            for row in frozen_cases
        ],
        "eligible_candidate_count": len(eligible),
        "excluded_candidate_count": len(excluded),
        "frozen_case_count": len(frozen_cases),
        "frozen_lane_profile_counts": dict(
            sorted(
                Counter(
                    row.primary_lane_profile
                    for row in frozen_cases
                ).items()
            )
        ),
        "selection_algorithm":
            "lane_quota_then_stable_structural_order_v1",
        "pre_verifier_structural_inputs_only": True,
        "endpoint_binding_is_pre_verifier_annotation_only": True,
        "prior_art_retrieval_used_for_selection": False,
        "relation_adjudication_used_for_selection": False,
        "external_novelty_outcomes_used_for_selection": False,
        "positive_nonobviousness_used_for_selection": False,
        "final_certification_used_for_selection": False,
        "old_n10_outcome_fields_used_for_selection": False,
        "new_verifier_result_observed_before_freeze": False,
        "post_freeze_selection_change_allowed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRelationalVerifierCohortFreeze(
        **body,
        freeze_id=(
            "prospective_relational_verifier_cohort_freeze:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


__all__ = [
    "FrozenProspectiveRelationalVerifierCase",
    "ProspectiveRelationalVerifierCohortFreeze",
    "ProspectiveRelationalVerifierPoolCaseSpec",
    "ProspectiveRelationalVerifierPoolSpec",
    "build_prospective_relational_verifier_cohort_freeze",
]
