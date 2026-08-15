from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.hypothesis_trend_grounding import (
    HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID,
    HypothesisTrendGroundingBundle,
    HypothesisTrendRelationGrounding,
    sha256_file,
)


HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID = (
    "hypothesis_trend_input_contract_v1_alpha4c5b"
)

TrendInputLane = Literal[
    "local_empirical_support",
    "cross_paper_replicated_support",
    "context_dependency_signal",
    "reversal_boundary",
    "replication_gap",
]

TREND_INPUT_LANES = (
    "local_empirical_support",
    "cross_paper_replicated_support",
    "context_dependency_signal",
    "reversal_boundary",
    "replication_gap",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _sorted_unique(values: Any) -> list[str]:
    return sorted({
        str(value)
        for value in values
        if str(value).strip()
    })


def validate_hypothesis_context_sha(
    context: HypothesisContext,
) -> None:
    payload = context.model_dump(mode="json")
    observed = str(payload.pop("context_sha256", ""))
    expected = _sha256_json(payload)
    if observed != expected:
        raise ValueError(
            "HypothesisContext context_sha256 mismatch: "
            f"{observed!r} != {expected!r}."
        )


def validate_trend_grounding_bundle_sha(
    bundle: HypothesisTrendGroundingBundle,
) -> None:
    payload = bundle.model_dump(mode="json")
    observed = str(payload.pop("bundle_sha256", ""))
    expected = _sha256_json(payload)
    if observed != expected:
        raise ValueError(
            "HypothesisTrendGroundingBundle bundle_sha256 mismatch: "
            f"{observed!r} != {expected!r}."
        )


class TrendCorpusBinding(StrictModel):
    corpus_id: str
    corpus_mode: str
    domain_profile_id: str
    trend_id: str
    trend_semantics_id: str
    paper_ids: list[str] = Field(default_factory=list)
    trend_summary_path: str
    trend_summary_sha256: str

    @model_validator(mode="after")
    def _binding_consistency(self) -> "TrendCorpusBinding":
        for name in (
            "corpus_id",
            "corpus_mode",
            "domain_profile_id",
            "trend_id",
            "trend_semantics_id",
            "trend_summary_path",
            "trend_summary_sha256",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(
                    f"TrendCorpusBinding {name} must not be empty."
                )
        if self.paper_ids != sorted(set(self.paper_ids)):
            raise ValueError(
                "TrendCorpusBinding paper_ids must be sorted and unique."
            )
        if not self.paper_ids:
            raise ValueError(
                "TrendCorpusBinding requires source corpus paper_ids."
            )
        return self


class HypothesisTrendInputPolicy(StrictModel):
    explorer_statement_namespace_preserved: Literal[True] = True
    trend_grounding_namespace_separate: Literal[True] = True
    trend_ids_allowed_in_premise_statement_ids: Literal[False] = False
    trend_ids_allowed_in_gap_statement_ids: Literal[False] = False
    maker_consumption_enabled: Literal[False] = False
    prompt_modified: Literal[False] = False
    compiler_modified: Literal[False] = False
    validator_modified: Literal[False] = False
    runtime_modified: Literal[False] = False
    llm_calls_allowed: Literal[False] = False
    zero_trend_yield_is_valid: Literal[True] = True
    zero_trend_yield_creates_gap: Literal[False] = False
    causality_authorized_by_trend_input: Literal[False] = False
    universal_relation_authorized_by_trend_input: Literal[False] = False
    unknown_context_fill_allowed: Literal[False] = False
    majority_direction_vote_allowed: Literal[False] = False


class HypothesisTrendInputView(StrictModel):
    schema_version: Literal["hypothesis-trend-input-view-v1"] = (
        "hypothesis-trend-input-view-v1"
    )
    view_id: str
    contract_semantics_id: str
    input_semantics_id: str
    lane: TrendInputLane

    grounding_id: str
    relation_id: str
    cross_context_status: str
    support_role: str

    independent_variable_key: str
    dependent_observable_key: str
    control_family: str
    observable_semantics: str

    paper_ids: list[str] = Field(default_factory=list)
    local_result_ids: list[str] = Field(default_factory=list)
    member_trend_ids: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    shapes: list[str] = Field(default_factory=list)
    evidence_kinds: list[str] = Field(default_factory=list)
    evidence_bases: list[str] = Field(default_factory=list)

    source_claim_ids: list[str] = Field(default_factory=list)
    source_measurement_ids: list[str] = Field(default_factory=list)
    source_measurement_result_ids: list[str] = Field(default_factory=list)
    source_calculation_ids: list[str] = Field(default_factory=list)
    source_node_ids: list[str] = Field(default_factory=list)

    differentiating_dimensions: list[str] = Field(default_factory=list)
    unresolved_dimensions: list[str] = Field(default_factory=list)
    association_only_result_ids: list[str] = Field(default_factory=list)
    source_asserted_causal_trend_ids: list[str] = Field(
        default_factory=list
    )
    source_requires_verification_trend_ids: list[str] = Field(
        default_factory=list
    )

    requires_context_qualification: bool
    requires_verification: bool
    directional_cross_paper_premise_allowed: bool

    maker_selectable: Literal[False] = False
    causal_use_allowed: Literal[False] = False
    universal_use_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _view_consistency(self) -> "HypothesisTrendInputView":
        if self.contract_semantics_id != (
            HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID
        ):
            raise ValueError(
                "Hypothesis Trend input view contract semantics mismatch."
            )
        if self.lane not in TREND_INPUT_LANES:
            raise ValueError(
                f"Unknown Trend input lane: {self.lane!r}."
            )

        for name in (
            "view_id",
            "input_semantics_id",
            "grounding_id",
            "relation_id",
            "cross_context_status",
            "support_role",
            "independent_variable_key",
            "dependent_observable_key",
            "control_family",
            "observable_semantics",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(
                    f"HypothesisTrendInputView {name} must not be empty."
                )

        for name in (
            "paper_ids",
            "local_result_ids",
            "member_trend_ids",
            "directions",
            "shapes",
            "evidence_kinds",
            "evidence_bases",
            "source_claim_ids",
            "source_measurement_ids",
            "source_measurement_result_ids",
            "source_calculation_ids",
            "source_node_ids",
            "differentiating_dimensions",
            "unresolved_dimensions",
            "association_only_result_ids",
            "source_asserted_causal_trend_ids",
            "source_requires_verification_trend_ids",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(
                    f"{name} must be sorted and unique."
                )

        if not self.paper_ids:
            raise ValueError(
                "Trend input view requires paper_ids."
            )
        if not self.local_result_ids:
            raise ValueError(
                "Trend input view requires local_result_ids."
            )
        if not self.member_trend_ids:
            raise ValueError(
                "Trend input view requires member_trend_ids."
            )

        expected_view_id = _stable_id(
            "hypothesis_trend_input_view",
            self.input_semantics_id,
            self.grounding_id,
            self.lane,
        )
        if self.view_id != expected_view_id:
            raise ValueError(
                "HypothesisTrendInputView view_id is not stable."
            )

        if (
            self.lane == "replication_gap"
            and self.cross_context_status != "insufficient"
        ):
            raise ValueError(
                "replication_gap lane requires status='insufficient'."
            )
        if (
            self.lane == "reversal_boundary"
            and self.cross_context_status != "reversed"
        ):
            raise ValueError(
                "reversal_boundary lane requires status='reversed'."
            )
        return self


def _lanes_for_grounding(
    grounding: HypothesisTrendRelationGrounding,
) -> tuple[TrendInputLane, ...]:
    lanes: list[TrendInputLane] = []
    if grounding.local_empirical_premise_allowed:
        lanes.append("local_empirical_support")
    if grounding.cross_context_replicated_premise_allowed:
        lanes.append("cross_paper_replicated_support")
    if grounding.context_dependency_premise_allowed:
        lanes.append("context_dependency_signal")
    if grounding.reversal_counterevidence_required:
        lanes.append("reversal_boundary")
    if grounding.replication_gap_signal_allowed:
        lanes.append("replication_gap")
    return tuple(lanes)


def project_trend_input_views(
    bundle: HypothesisTrendGroundingBundle,
    *,
    input_semantics_id: str,
) -> list[HypothesisTrendInputView]:
    if bundle.contract_semantics_id != (
        HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID
    ):
        raise ValueError(
            "Unsupported Hypothesis Trend grounding contract semantics."
        )

    views: list[HypothesisTrendInputView] = []
    for grounding in sorted(
        bundle.groundings,
        key=lambda row: (row.relation_id, row.grounding_id),
    ):
        for lane in _lanes_for_grounding(grounding):
            views.append(
                HypothesisTrendInputView(
                    view_id=_stable_id(
                        "hypothesis_trend_input_view",
                        input_semantics_id,
                        grounding.grounding_id,
                        lane,
                    ),
                    contract_semantics_id=
                        HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID,
                    input_semantics_id=input_semantics_id,
                    lane=lane,
                    grounding_id=grounding.grounding_id,
                    relation_id=grounding.relation_id,
                    cross_context_status=
                        grounding.cross_context_status,
                    support_role=grounding.support_role,
                    independent_variable_key=
                        grounding.independent_variable_key,
                    dependent_observable_key=
                        grounding.dependent_observable_key,
                    control_family=grounding.control_family,
                    observable_semantics=
                        grounding.observable_semantics,
                    paper_ids=list(grounding.paper_ids),
                    local_result_ids=list(
                        grounding.local_result_ids
                    ),
                    member_trend_ids=list(
                        grounding.member_trend_ids
                    ),
                    directions=list(grounding.directions),
                    shapes=list(grounding.shapes),
                    evidence_kinds=list(
                        grounding.evidence_kinds
                    ),
                    evidence_bases=list(
                        grounding.evidence_bases
                    ),
                    source_claim_ids=list(
                        grounding.source_claim_ids
                    ),
                    source_measurement_ids=list(
                        grounding.source_measurement_ids
                    ),
                    source_measurement_result_ids=list(
                        grounding.source_measurement_result_ids
                    ),
                    source_calculation_ids=list(
                        grounding.source_calculation_ids
                    ),
                    source_node_ids=list(
                        grounding.source_node_ids
                    ),
                    differentiating_dimensions=list(
                        grounding.differentiating_dimensions
                    ),
                    unresolved_dimensions=list(
                        grounding.unresolved_dimensions
                    ),
                    association_only_result_ids=list(
                        grounding.association_only_result_ids
                    ),
                    source_asserted_causal_trend_ids=list(
                        grounding.source_asserted_causal_trend_ids
                    ),
                    source_requires_verification_trend_ids=list(
                        grounding.source_requires_verification_trend_ids
                    ),
                    requires_context_qualification=
                        grounding.requires_context_qualification,
                    requires_verification=
                        grounding.requires_verification,
                    directional_cross_paper_premise_allowed=
                        grounding.directional_cross_paper_premise_allowed,
                    maker_selectable=False,
                    causal_use_allowed=False,
                    universal_use_allowed=False,
                )
            )

    return sorted(
        views,
        key=lambda row: (
            TREND_INPUT_LANES.index(row.lane),
            row.relation_id,
            row.grounding_id,
            row.view_id,
        ),
    )


def verify_grounding_source_artifacts(
    bundle: HypothesisTrendGroundingBundle,
) -> None:
    seen_roles: set[str] = set()
    for artifact in bundle.source_artifacts:
        if artifact.role in seen_roles:
            raise ValueError(
                "HypothesisTrendGroundingBundle source artifact roles "
                f"must be unique for 5b: {artifact.role!r}."
            )
        seen_roles.add(artifact.role)

        path = Path(artifact.path)
        if not path.exists():
            raise ValueError(
                f"Grounding source artifact missing: {path}"
            )
        observed = sha256_file(path)
        if observed != artifact.sha256:
            raise ValueError(
                "Grounding source artifact SHA mismatch for "
                f"{artifact.role!r}: {observed} != {artifact.sha256}."
            )


def load_trend_corpus_binding(
    bundle: HypothesisTrendGroundingBundle,
) -> TrendCorpusBinding:
    validate_trend_grounding_bundle_sha(bundle)
    verify_grounding_source_artifacts(bundle)

    candidates = [
        artifact
        for artifact in bundle.source_artifacts
        if artifact.role == "trend_summary"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Trend-aware hypothesis input requires exactly one "
            "locked trend_summary source artifact."
        )
    artifact = candidates[0]
    path = Path(artifact.path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(
            "Locked trend_summary must be a JSON object."
        )
    if value.get("structural_gate") is not True:
        raise ValueError(
            "Locked trend_summary structural_gate must be true."
        )

    domain_profile_id = str(
        value.get("domain_profile_id", "")
    )
    trend_semantics_id = str(
        value.get("trend_semantics_id", "")
    )
    corpus_id = str(value.get("corpus_id", ""))
    corpus_mode = str(value.get("corpus_mode", ""))
    trend_id = str(value.get("trend_id", ""))
    paper_ids = _sorted_unique(value.get("paper_ids", []))

    if domain_profile_id != bundle.domain_profile_id:
        raise ValueError(
            "Trend summary / grounding domain mismatch."
        )
    if (
        trend_semantics_id
        != bundle.source_trend_semantics_id
    ):
        raise ValueError(
            "Trend summary / grounding trend semantics mismatch."
        )

    grounding_papers = {
        paper_id
        for row in bundle.groundings
        for paper_id in row.paper_ids
    }
    if not grounding_papers.issubset(set(paper_ids)):
        raise ValueError(
            "Trend grounding references paper IDs outside the "
            "locked Trend corpus."
        )

    return TrendCorpusBinding(
        corpus_id=corpus_id,
        corpus_mode=corpus_mode,
        domain_profile_id=domain_profile_id,
        trend_id=trend_id,
        trend_semantics_id=trend_semantics_id,
        paper_ids=paper_ids,
        trend_summary_path=str(path),
        trend_summary_sha256=artifact.sha256,
    )


class HypothesisTrendInputProjectionAudit(StrictModel):
    schema_version: Literal[
        "hypothesis-trend-input-projection-audit-v1"
    ] = "hypothesis-trend-input-projection-audit-v1"
    contract_semantics_id: str
    input_semantics_id: str
    grounding_bundle_id: str
    grounding_bundle_sha256: str
    corpus_binding: TrendCorpusBinding
    relation_count: int
    view_count: int
    cross_context_status_counts: dict[str, int]
    lane_counts: dict[str, int]
    issues: list[str] = Field(default_factory=list)
    structural_gate: bool


def audit_trend_grounding_for_input(
    bundle: HypothesisTrendGroundingBundle,
    *,
    input_semantics_id: str,
) -> HypothesisTrendInputProjectionAudit:
    issues: list[str] = []
    binding: TrendCorpusBinding | None = None
    views: list[HypothesisTrendInputView] = []

    try:
        binding = load_trend_corpus_binding(bundle)
    except Exception as exc:
        issues.append(f"source_binding:{exc}")

    if binding is not None:
        try:
            views = project_trend_input_views(
                bundle,
                input_semantics_id=input_semantics_id,
            )
        except Exception as exc:
            issues.append(f"view_projection:{exc}")

    if binding is None:
        # Preserve a typed audit even on failure without fabricating
        # scientific lineage.
        binding = TrendCorpusBinding(
            corpus_id="unresolved",
            corpus_mode="unresolved",
            domain_profile_id=bundle.domain_profile_id,
            trend_id="unresolved",
            trend_semantics_id=bundle.source_trend_semantics_id,
            paper_ids=["unresolved"],
            trend_summary_path="unresolved",
            trend_summary_sha256="unresolved",
        )

    return HypothesisTrendInputProjectionAudit(
        contract_semantics_id=
            HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID,
        input_semantics_id=input_semantics_id,
        grounding_bundle_id=bundle.bundle_id,
        grounding_bundle_sha256=bundle.bundle_sha256,
        corpus_binding=binding,
        relation_count=len(bundle.groundings),
        view_count=len(views),
        cross_context_status_counts=dict(sorted(Counter(
            row.cross_context_status
            for row in bundle.groundings
        ).items())),
        lane_counts=dict(sorted(Counter(
            row.lane for row in views
        ).items())),
        issues=issues,
        structural_gate=not issues,
    )


class TrendAwareHypothesisInput(StrictModel):
    schema_version: Literal[
        "trend-aware-hypothesis-input-v1"
    ] = "trend-aware-hypothesis-input-v1"

    input_id: str
    input_sha256: str
    contract_semantics_id: str
    input_semantics_id: str
    domain_profile_id: str
    corpus_id: str

    grounded_context: HypothesisContext
    trend_grounding: HypothesisTrendGroundingBundle
    trend_corpus_binding: TrendCorpusBinding
    trend_views: list[HypothesisTrendInputView] = Field(
        default_factory=list
    )
    lane_counts: dict[str, int] = Field(default_factory=dict)

    policy: HypothesisTrendInputPolicy = Field(
        default_factory=HypothesisTrendInputPolicy
    )

    @model_validator(mode="after")
    def _input_consistency(self) -> "TrendAwareHypothesisInput":
        if self.contract_semantics_id != (
            HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID
        ):
            raise ValueError(
                "Trend-aware hypothesis input contract semantics mismatch."
            )

        validate_hypothesis_context_sha(self.grounded_context)
        validate_trend_grounding_bundle_sha(self.trend_grounding)

        if (
            self.domain_profile_id
            != self.grounded_context.domain_profile_id
        ):
            raise ValueError(
                "Input / HypothesisContext domain mismatch."
            )
        if (
            self.domain_profile_id
            != self.trend_grounding.domain_profile_id
        ):
            raise ValueError(
                "Input / Trend grounding domain mismatch."
            )
        if self.corpus_id != self.grounded_context.corpus_id:
            raise ValueError(
                "Input / HypothesisContext corpus mismatch."
            )
        if (
            self.corpus_id
            != self.trend_corpus_binding.corpus_id
        ):
            raise ValueError(
                "HypothesisContext / locked Trend corpus mismatch."
            )
        if (
            self.domain_profile_id
            != self.trend_corpus_binding.domain_profile_id
        ):
            raise ValueError(
                "Input / Trend corpus binding domain mismatch."
            )
        if (
            self.trend_grounding.source_trend_semantics_id
            != self.trend_corpus_binding.trend_semantics_id
        ):
            raise ValueError(
                "Input / Trend corpus binding semantics mismatch."
            )

        source_artifacts = {
            row.role: row
            for row in self.trend_grounding.source_artifacts
        }
        summary_artifact = source_artifacts.get(
            "trend_summary"
        )
        if summary_artifact is None:
            raise ValueError(
                "Trend grounding lacks trend_summary artifact."
            )
        if (
            self.trend_corpus_binding.trend_summary_path
            != summary_artifact.path
            or self.trend_corpus_binding.trend_summary_sha256
            != summary_artifact.sha256
        ):
            raise ValueError(
                "Trend corpus binding does not match locked "
                "trend_summary artifact."
            )

        grounding_papers = {
            paper_id
            for row in self.trend_grounding.groundings
            for paper_id in row.paper_ids
        }
        if not grounding_papers.issubset(
            set(self.trend_corpus_binding.paper_ids)
        ):
            raise ValueError(
                "Trend input contains paper outside corpus binding."
            )

        expected_views = project_trend_input_views(
            self.trend_grounding,
            input_semantics_id=self.input_semantics_id,
        )
        expected_rows = [
            row.model_dump(mode="json")
            for row in expected_views
        ]
        observed_rows = [
            row.model_dump(mode="json")
            for row in self.trend_views
        ]
        if observed_rows != expected_rows:
            raise ValueError(
                "Trend input views are not the exact deterministic "
                "projection of the locked grounding bundle."
            )

        expected_lane_counts = dict(sorted(Counter(
            row.lane for row in expected_views
        ).items()))
        if self.lane_counts != expected_lane_counts:
            raise ValueError("Trend input lane_counts mismatch.")

        if self.trend_grounding.zero_yield:
            if self.trend_views or self.lane_counts:
                raise ValueError(
                    "Zero-yield Trend grounding must create no views "
                    "and no fabricated gap."
                )

        expected_input_id = _stable_id(
            "trend_aware_hypothesis_input",
            self.input_semantics_id,
            self.grounded_context.context_sha256,
            self.trend_grounding.bundle_sha256,
            self.trend_corpus_binding.trend_summary_sha256,
        )
        if self.input_id != expected_input_id:
            raise ValueError(
                "TrendAwareHypothesisInput input_id is not stable."
            )

        payload = self.model_dump(mode="json")
        observed_sha = str(payload.pop("input_sha256", ""))
        expected_sha = _sha256_json(payload)
        if observed_sha != expected_sha:
            raise ValueError(
                "TrendAwareHypothesisInput input_sha256 mismatch."
            )
        return self


def build_trend_aware_hypothesis_input(
    *,
    grounded_context: HypothesisContext,
    trend_grounding: HypothesisTrendGroundingBundle,
    input_semantics_id: str,
) -> TrendAwareHypothesisInput:
    if not str(input_semantics_id).strip():
        raise ValueError("input_semantics_id must not be empty.")

    validate_hypothesis_context_sha(grounded_context)
    validate_trend_grounding_bundle_sha(trend_grounding)
    binding = load_trend_corpus_binding(trend_grounding)

    if (
        grounded_context.domain_profile_id
        != trend_grounding.domain_profile_id
    ):
        raise ValueError(
            "HypothesisContext and Trend grounding domain profiles "
            "must match."
        )
    if grounded_context.corpus_id != binding.corpus_id:
        raise ValueError(
            "HypothesisContext corpus_id does not match the "
            "SHA-locked Trend summary corpus_id: "
            f"{grounded_context.corpus_id!r} != {binding.corpus_id!r}."
        )

    views = project_trend_input_views(
        trend_grounding,
        input_semantics_id=input_semantics_id,
    )
    lane_counts = dict(sorted(Counter(
        row.lane for row in views
    ).items()))

    input_id = _stable_id(
        "trend_aware_hypothesis_input",
        input_semantics_id,
        grounded_context.context_sha256,
        trend_grounding.bundle_sha256,
        binding.trend_summary_sha256,
    )

    payload = {
        "schema_version": "trend-aware-hypothesis-input-v1",
        "input_id": input_id,
        "contract_semantics_id":
            HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID,
        "input_semantics_id": input_semantics_id,
        "domain_profile_id":
            grounded_context.domain_profile_id,
        "corpus_id": grounded_context.corpus_id,
        "grounded_context":
            grounded_context.model_dump(mode="json"),
        "trend_grounding":
            trend_grounding.model_dump(mode="json"),
        "trend_corpus_binding":
            binding.model_dump(mode="json"),
        "trend_views": [
            row.model_dump(mode="json")
            for row in views
        ],
        "lane_counts": lane_counts,
        "policy": HypothesisTrendInputPolicy().model_dump(
            mode="json"
        ),
    }
    payload["input_sha256"] = _sha256_json(payload)
    return TrendAwareHypothesisInput(**payload)


def verify_trend_aware_input_sources(
    value: TrendAwareHypothesisInput,
) -> None:
    validate_hypothesis_context_sha(value.grounded_context)
    validate_trend_grounding_bundle_sha(value.trend_grounding)
    observed = load_trend_corpus_binding(
        value.trend_grounding
    )
    if (
        observed.model_dump(mode="json")
        != value.trend_corpus_binding.model_dump(mode="json")
    ):
        raise ValueError(
            "Current locked Trend source binding differs from "
            "the binding frozen into TrendAwareHypothesisInput."
        )
