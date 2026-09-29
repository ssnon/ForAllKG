from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ProspectiveCaseRole = Literal[
    "DIRECT_HO_REFERENCE",
    "BROAD_RELATION_CONTROL",
    "MULTIFACTOR_MECHANISM",
    "CONDITIONAL_MECHANISM",
    "STRUCTURE_STABILITY",
    "PERFORMANCE_COMPARISON",
]


class ProspectiveNoveltyValidationTask(StrictModel):
    case_id: str
    case_role: ProspectiveCaseRole

    domain_profile_id: str
    corpus_id: str

    source: str
    stop: str | None = None
    target: str
    question: str
    objective: str = "explain_connection"

    evaluation_focus: list[str] = Field(default_factory=list)
    research_value_capability_expected: bool

    scientific_outcome_expected: Literal["UNSPECIFIED"] = "UNSPECIFIED"
    result_conditioned_case_selection_allowed: Literal[False] = False


class ProspectiveNoveltyValidationCohortSpec(StrictModel):
    schema_version: Literal[
        "prospective-novelty-validation-cohort-spec-v1"
    ] = "prospective-novelty-validation-cohort-spec-v1"

    cohort_name: str
    cases: list[ProspectiveNoveltyValidationTask]

    outcome_blind_freeze_required: Literal[True] = True
    post_freeze_case_mutation_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_cases(self):
        ids = [row.case_id for row in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate prospective case_id")
        if not self.cases:
            raise ValueError("prospective cohort must contain at least one case")
        if any(row.scientific_outcome_expected != "UNSPECIFIED" for row in self.cases):
            raise ValueError(
                "scientific outcomes must remain UNSPECIFIED before prospective freeze"
            )
        return self


class FrozenProspectiveNoveltyValidationCase(StrictModel):
    prospective_case_id: str
    source_case_id: str
    case_role: ProspectiveCaseRole

    domain_profile_id: str
    corpus_id: str

    source: str
    stop: str | None = None
    target: str
    question: str
    objective: str

    evaluation_focus: list[str] = Field(default_factory=list)
    research_value_capability_expected: bool

    scientific_outcome_expected: Literal["UNSPECIFIED"] = "UNSPECIFIED"


class ProspectiveNoveltyValidationCohortFreeze(StrictModel):
    schema_version: Literal[
        "prospective-novelty-validation-cohort-freeze-v1"
    ] = "prospective-novelty-validation-cohort-freeze-v1"

    freeze_id: str
    freeze_sha256: str

    source_spec_sha256: str
    cohort_name: str
    frozen_case_count: int
    domain_counts: dict[str, int]
    role_counts: dict[str, int]

    frozen_cases: list[FrozenProspectiveNoveltyValidationCase]

    prior_results_observed_for_selection: Literal[False] = False
    external_novelty_outcomes_used_for_selection: Literal[False] = False
    conceptual_knownness_outcomes_used_for_selection: Literal[False] = False
    alpha6_recommendations_used_for_selection: Literal[False] = False
    research_value_outcomes_used_for_selection: Literal[False] = False

    post_freeze_case_mutation_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self):
        if self.frozen_case_count != len(self.frozen_cases):
            raise ValueError("frozen case count mismatch")
        return self


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def build_prospective_novelty_validation_cohort_freeze(
    *,
    spec: ProspectiveNoveltyValidationCohortSpec,
    source_spec_sha256: str,
) -> ProspectiveNoveltyValidationCohortFreeze:
    frozen_cases: list[FrozenProspectiveNoveltyValidationCase] = []

    for row in spec.cases:
        frozen_cases.append(
            FrozenProspectiveNoveltyValidationCase(
                prospective_case_id=_stable_id(
                    "prospective_novelty_case",
                    source_spec_sha256,
                    row.case_id,
                    row.domain_profile_id,
                    row.corpus_id,
                    row.source,
                    row.stop or "",
                    row.target,
                    row.question,
                ),
                source_case_id=row.case_id,
                case_role=row.case_role,
                domain_profile_id=row.domain_profile_id,
                corpus_id=row.corpus_id,
                source=row.source,
                stop=row.stop,
                target=row.target,
                question=row.question,
                objective=row.objective,
                evaluation_focus=list(row.evaluation_focus),
                research_value_capability_expected=(
                    row.research_value_capability_expected
                ),
            )
        )

    domain_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}

    for row in frozen_cases:
        domain_counts[row.domain_profile_id] = (
            domain_counts.get(row.domain_profile_id, 0) + 1
        )
        role_counts[row.case_role] = role_counts.get(row.case_role, 0) + 1

    freeze_id = _stable_id(
        "prospective_novelty_validation_freeze",
        source_spec_sha256,
        *[row.prospective_case_id for row in frozen_cases],
    )

    base = {
        "schema_version":
            "prospective-novelty-validation-cohort-freeze-v1",
        "freeze_id":
            freeze_id,
        "source_spec_sha256":
            source_spec_sha256,
        "cohort_name":
            spec.cohort_name,
        "frozen_case_count":
            len(frozen_cases),
        "domain_counts":
            domain_counts,
        "role_counts":
            role_counts,
        "frozen_cases": [
            row.model_dump(mode="json")
            for row in frozen_cases
        ],
        "prior_results_observed_for_selection":
            False,
        "external_novelty_outcomes_used_for_selection":
            False,
        "conceptual_knownness_outcomes_used_for_selection":
            False,
        "alpha6_recommendations_used_for_selection":
            False,
        "research_value_outcomes_used_for_selection":
            False,
        "post_freeze_case_mutation_allowed":
            False,
        "production_selection_authority":
            False,
    }

    return ProspectiveNoveltyValidationCohortFreeze(
        **base,
        freeze_sha256=_sha256_json(base),
    )


__all__ = [
    "ProspectiveNoveltyValidationTask",
    "ProspectiveNoveltyValidationCohortSpec",
    "FrozenProspectiveNoveltyValidationCase",
    "ProspectiveNoveltyValidationCohortFreeze",
    "build_prospective_novelty_validation_cohort_freeze",
]
