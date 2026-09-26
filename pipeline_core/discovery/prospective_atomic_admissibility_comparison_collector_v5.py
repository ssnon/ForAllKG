from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.atomic_pre_n10_admissibility_shadow import (
    assess_atomic_pre_n10_shadow_row,
)
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    NoveltyClaim,
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)
from pipeline_core.discovery.legacy_atomic_admissibility_adapter import (
    adapt_legacy_atomic_specification_shadow_row,
)
from pipeline_core.discovery.legacy_atomic_specification_shadow import (
    compile_legacy_atomic_specification_shadow_row,
)
from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    PreN10ProspectiveCampaignReportV1,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    PreN10RegenerationReentryReportV2,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ClaimContractRowV1,
    PreN10ScientificContractReportV1,
)
from pipeline_core.discovery.prospective_atomic_admissibility_execution_plan_v5 import (
    ProspectiveAtomicAdmissibilityExecutionPlanV5,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CaseIdV5 = Literal["P29", "P30", "P31", "P32", "P33"]
ComparisonStageV5 = Literal["INITIAL_VPRE", "REGENERATION_REENTRY"]
ComparisonPairV5 = Literal[
    "LEGACY_READY__NEUTRAL_READY",
    "LEGACY_READY__NEUTRAL_NOT_READY",
    "LEGACY_NOT_READY__NEUTRAL_READY",
    "LEGACY_NOT_READY__NEUTRAL_NOT_READY",
]
CaseAccountingStatusV5 = Literal[
    "CLAIMS_COLLECTED",
    "TERMINAL_BEFORE_INITIAL_VPRE",
    "NO_CLAIMS_OBSERVED",
]
ReasonTransitionV5 = Literal[
    "REPRESENTATION_LOSS_ONLY_CANDIDATE",
    "EXACT_TEXT_FAILURE_RECLASSIFIED_WITH_SEMANTIC_BLOCKER",
    "MULTI_DIMENSION_NEUTRAL_BLOCKING",
    "SOURCE_REFERENCE_BLOCKING_REMAINS",
    "SEMANTIC_BLOCKING_EXPOSED",
    "OTHER_NEUTRAL_BLOCKING",
    "NO_NEUTRAL_BLOCKER",
]

EXACT_TEXT_SOURCE_PREFIXES = (
    "prediction_exact_source_binding_cardinality:",
    "falsifier_exact_source_binding_cardinality:",
)
EXACT_TEXT_SOURCE_CODES = {
    "prediction_falsifier_observable_identity_mismatch",
    "observable_empty",
}


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


def sha256_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("missing comparison artifact: " + str(resolved))
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("comparison artifact must be JSON object: " + str(resolved))
    return value


def _slug(value: str) -> str:
    out = str(value).replace(":", "_").replace("/", "_")
    if not out or out in {".", ".."}:
        raise ValueError("invalid lineage slug source: " + repr(value))
    return out


class ProspectiveAtomicAdmissibilityComparisonCaseContractV5(StrictModel):
    case_id: CaseIdV5
    source_task_id: str
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    campaign_root: str
    campaign_report_path: str

    initial_portfolio_path: str
    initial_vpre_query_path: str
    initial_vpre_contract_path: str
    initial_vpre_audit_path: str

    regeneration_reentry_report_path: str
    regeneration_lineage_root: str
    regeneration_audit_filename: Literal[
        "claim_decomposition.sanitization_audit.json"
    ] = "claim_decomposition.sanitization_audit.json"
    regeneration_query_filename: Literal[
        "claims_queries.json"
    ] = "claims_queries.json"
    regeneration_contract_filename: Literal[
        "contract.json"
    ] = "contract.json"

    initial_collection_policy: Literal[
        "REQUIRE_ALL_IF_INITIAL_VPRE_ACTIVE"
    ] = "REQUIRE_ALL_IF_INITIAL_VPRE_ACTIVE"
    regeneration_collection_policy: Literal[
        "COLLECT_EVERY_REENTRY_LINEAGE_WITH_DECOMPOSITION_REQUEST"
    ] = "COLLECT_EVERY_REENTRY_LINEAGE_WITH_DECOMPOSITION_REQUEST"
    terminal_case_policy: Literal[
        "RETAIN_CASE_WITH_ZERO_CLAIM_ROWS"
    ] = "RETAIN_CASE_WITH_ZERO_CLAIM_ROWS"

    @model_validator(mode="after")
    def validate_paths(
        self,
    ) -> "ProspectiveAtomicAdmissibilityComparisonCaseContractV5":
        campaign = Path(self.campaign_root).expanduser().resolve()
        if Path(self.campaign_report_path).expanduser().resolve() != (
            campaign / "campaign.report.json"
        ):
            raise ValueError("comparison campaign report path mismatch")
        initial = campaign / "01_initial_vpre"
        if Path(self.initial_vpre_query_path).expanduser().resolve() != (
            initial / "claims_queries.json"
        ):
            raise ValueError("comparison initial query path mismatch")
        if Path(self.initial_vpre_contract_path).expanduser().resolve() != (
            initial / "contract.report.json"
        ):
            raise ValueError("comparison initial contract path mismatch")
        if Path(self.initial_vpre_audit_path).expanduser().resolve() != (
            initial / "claim_decomposition.sanitization_audit.json"
        ):
            raise ValueError("comparison initial audit path mismatch")
        reentry = campaign / "04_regeneration_reentry"
        if Path(self.regeneration_reentry_report_path).expanduser().resolve() != (
            reentry / "reentry_v2.report.json"
        ):
            raise ValueError("comparison reentry report path mismatch")
        if Path(self.regeneration_lineage_root).expanduser().resolve() != (
            reentry / "lineage"
        ):
            raise ValueError("comparison lineage root mismatch")
        return self


class ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV5(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-comparison-collector-freeze-v5"
    ] = "prospective-atomic-admissibility-comparison-collector-freeze-v5"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_execution_plan_id: str
    source_execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_execution_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    collector_repository_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_tracked_worktree_dirty: Literal[False] = False

    cases: list[ProspectiveAtomicAdmissibilityComparisonCaseContractV5]
    case_ids: list[str]
    case_count: Literal[5] = 5

    comparison_output_path: str

    pair_taxonomy: list[str]
    claim_population_policy: Literal[
        "ALL_AUDITED_CLAIMS_FROM_INITIAL_AND_REENTRY"
    ] = "ALL_AUDITED_CLAIMS_FROM_INITIAL_AND_REENTRY"

    case_denominator_fixed_before_execution: Literal[True] = True
    terminal_before_decomposition_retained_in_case_denominator: Literal[
        True
    ] = True
    terminal_case_claim_rows_synthesized: Literal[False] = False

    missing_required_initial_artifact_is_failure: Literal[True] = True
    missing_required_reentry_artifact_is_failure: Literal[True] = True
    unreported_extra_regeneration_audit_is_failure: Literal[True] = True
    audit_record_subset_selection_allowed: Literal[False] = False
    result_conditioned_artifact_selection_allowed: Literal[False] = False

    frozen_legacy_contract_is_legacy_authority: Literal[True] = True
    neutral_replay_uses_frozen_semantic_binding: Literal[True] = True
    semantic_fidelity_replay_must_match_frozen_audit: Literal[True] = True

    llm_calls_allowed: Literal[False] = False
    artifact_mutation_allowed: Literal[False] = False
    campaign_artifact_writes_allowed: Literal[False] = False

    collector_frozen_before_p29_p33_execution: Literal[True] = True
    p29_p33_outputs_observed_before_collector_freeze: Literal[False] = False
    campaign_execution_may_begin_after_this_freeze: Literal[True] = True

    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV5":
        expected = ["P29", "P30", "P31", "P32", "P33"]
        if self.case_ids != expected:
            raise ValueError("comparison collector case IDs must be P29-P33")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("comparison collector cases must be ordered P29-P33")
        if len({row.source_task_id for row in self.cases}) != 5:
            raise ValueError("comparison collector source task IDs must be unique")
        expected_pairs = [
            "LEGACY_READY__NEUTRAL_READY",
            "LEGACY_READY__NEUTRAL_NOT_READY",
            "LEGACY_NOT_READY__NEUTRAL_READY",
            "LEGACY_NOT_READY__NEUTRAL_NOT_READY",
        ]
        if self.pair_taxonomy != expected_pairs:
            raise ValueError("comparison pair taxonomy mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("comparison collector freeze SHA mismatch")
        if observed_id != (
            "prospective_atomic_admissibility_comparison_collector_freeze_v5:"
            + expected_sha[:20]
        ):
            raise ValueError("comparison collector freeze ID mismatch")
        return self


class ProspectiveAtomicAdmissibilityClaimComparisonV5(StrictModel):
    case_id: CaseIdV5
    stage: ComparisonStageV5
    source_hypothesis_id: str | None = None

    hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    kind: str
    novelty_selection_role: str | None = None

    pair: ComparisonPairV5
    reason_transition: ReasonTransitionV5

    legacy_ready: bool
    legacy_contract_status: str
    legacy_binding_reason_codes: list[str]
    legacy_source_reason_codes: list[str]
    legacy_router_hint: str

    neutral_ready: bool
    neutral_blocking_dimensions: list[str]
    neutral_source_reference_status: str
    neutral_semantic_fidelity_status: str
    neutral_specification_status: str
    neutral_atomic_kind_status: str
    neutral_structural_compilation_status: str

    prediction_observation_id: str | None = None
    falsification_criterion_id: str | None = None

    portfolio_path: str
    query_plan_path: str
    contract_path: str
    audit_path: str

    diagnostic_only: Literal[True] = True
    production_authority: Literal[False] = False


class ProspectiveAtomicAdmissibilityCaseAccountingV5(StrictModel):
    case_id: CaseIdV5
    case_in_denominator: Literal[True] = True
    campaign_final_status: str

    accounting_status: CaseAccountingStatusV5

    initial_vpre_stage_status: str
    initial_claim_comparison_count: int = Field(ge=0)

    regeneration_reentry_stage_status: str
    regeneration_lineage_count: int = Field(ge=0)
    regeneration_decomposition_lineage_count: int = Field(ge=0)
    regeneration_terminal_lineage_count: int = Field(ge=0)
    regeneration_claim_comparison_count: int = Field(ge=0)

    total_claim_comparison_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "ProspectiveAtomicAdmissibilityCaseAccountingV5":
        if self.total_claim_comparison_count != (
            self.initial_claim_comparison_count
            + self.regeneration_claim_comparison_count
        ):
            raise ValueError("case comparison count mismatch")
        if self.accounting_status == "TERMINAL_BEFORE_INITIAL_VPRE":
            if self.total_claim_comparison_count != 0:
                raise ValueError(
                    "terminal-before-initial case cannot carry claim comparisons"
                )
        return self


class ProspectiveAtomicAdmissibilityComparisonReportV5(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-comparison-report-v5"
    ] = "prospective-atomic-admissibility-comparison-report-v5"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_collector_freeze_id: str
    source_collector_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_ids: list[str]
    case_count: Literal[5] = 5
    terminal_before_initial_vpre_case_count: int = Field(ge=0)

    claim_comparison_count: int = Field(ge=0)
    pair_counts: dict[str, int]
    reason_transition_counts: dict[str, int]
    neutral_blocker_dimension_counts: dict[str, int]

    cases: list[ProspectiveAtomicAdmissibilityCaseAccountingV5]
    claims: list[ProspectiveAtomicAdmissibilityClaimComparisonV5]

    case_denominator_preserved: Literal[True] = True
    claim_subset_selection_performed: Literal[False] = False
    llm_calls_performed: Literal[False] = False
    campaign_artifacts_mutated: Literal[False] = False

    diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ProspectiveAtomicAdmissibilityComparisonReportV5":
        expected = ["P29", "P30", "P31", "P32", "P33"]
        if self.case_ids != expected:
            raise ValueError("comparison report case IDs must be P29-P33")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("comparison report cases must be ordered P29-P33")
        if self.claim_comparison_count != len(self.claims):
            raise ValueError("comparison report claim count mismatch")
        if sum(row.total_claim_comparison_count for row in self.cases) != len(
            self.claims
        ):
            raise ValueError("comparison report case/claim accounting mismatch")
        terminal_count = sum(
            row.accounting_status == "TERMINAL_BEFORE_INITIAL_VPRE"
            for row in self.cases
        )
        if self.terminal_before_initial_vpre_case_count != terminal_count:
            raise ValueError("comparison terminal case count mismatch")

        observed_pairs = Counter(row.pair for row in self.claims)
        if dict(sorted(observed_pairs.items())) != dict(
            sorted(self.pair_counts.items())
        ):
            raise ValueError("comparison pair counts mismatch")
        observed_transitions = Counter(
            row.reason_transition for row in self.claims
        )
        if dict(sorted(observed_transitions.items())) != dict(
            sorted(self.reason_transition_counts.items())
        ):
            raise ValueError("comparison transition counts mismatch")
        observed_blockers = Counter(
            blocker
            for row in self.claims
            for blocker in row.neutral_blocking_dimensions
        )
        if dict(sorted(observed_blockers.items())) != dict(
            sorted(self.neutral_blocker_dimension_counts.items())
        ):
            raise ValueError("comparison neutral blocker counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("comparison report SHA mismatch")
        if observed_id != (
            "prospective_atomic_admissibility_comparison_report_v5:"
            + expected_sha[:20]
        ):
            raise ValueError("comparison report ID mismatch")
        return self


def build_prospective_atomic_admissibility_comparison_collector_freeze_v5(
    *,
    execution_plan: ProspectiveAtomicAdmissibilityExecutionPlanV5,
    execution_plan_file_sha256: str,
    collector_repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
    comparison_output_path: Path,
) -> ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV5:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "comparison collector freeze requires a clean tracked worktree"
        )
    if not execution_plan.comparison_collector_contract_required_before_execution:
        raise ValueError("execution plan does not require comparison collector")
    if execution_plan.comparison_collector_contract_frozen_by_this_plan:
        raise ValueError("execution plan unexpectedly claims collector is frozen")
    if execution_plan.execution_authority_granted_by_this_plan:
        raise ValueError("execution plan unexpectedly grants execution authority")

    cases = []
    for row in execution_plan.cases:
        campaign = Path(row.downstream_campaign_output_root).expanduser().resolve()
        initial = campaign / "01_initial_vpre"
        reentry = campaign / "04_regeneration_reentry"
        cases.append(
            ProspectiveAtomicAdmissibilityComparisonCaseContractV5(
                case_id=row.case_id,
                source_task_id=row.source_task_id,
                source_task_sha256=row.source_task_sha256,
                run_dir=str(Path(row.run_dir).expanduser().resolve()),
                campaign_root=str(campaign),
                campaign_report_path=str(campaign / "campaign.report.json"),
                initial_portfolio_path=str(
                    Path(row.initial_portfolio_path).expanduser().resolve()
                ),
                initial_vpre_query_path=str(initial / "claims_queries.json"),
                initial_vpre_contract_path=str(initial / "contract.report.json"),
                initial_vpre_audit_path=str(
                    initial / "claim_decomposition.sanitization_audit.json"
                ),
                regeneration_reentry_report_path=str(
                    reentry / "reentry_v2.report.json"
                ),
                regeneration_lineage_root=str(reentry / "lineage"),
            )
        )

    body = {
        "schema_version": (
            "prospective-atomic-admissibility-comparison-collector-freeze-v5"
        ),
        "source_execution_plan_id": execution_plan.plan_id,
        "source_execution_plan_sha256": execution_plan.plan_sha256,
        "source_execution_plan_file_sha256": execution_plan_file_sha256,
        "collector_repository_head_sha": collector_repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P29", "P30", "P31", "P32", "P33"],
        "case_count": 5,
        "comparison_output_path": str(
            comparison_output_path.expanduser().resolve()
        ),
        "pair_taxonomy": [
            "LEGACY_READY__NEUTRAL_READY",
            "LEGACY_READY__NEUTRAL_NOT_READY",
            "LEGACY_NOT_READY__NEUTRAL_READY",
            "LEGACY_NOT_READY__NEUTRAL_NOT_READY",
        ],
        "claim_population_policy": (
            "ALL_AUDITED_CLAIMS_FROM_INITIAL_AND_REENTRY"
        ),
        "case_denominator_fixed_before_execution": True,
        "terminal_before_decomposition_retained_in_case_denominator": True,
        "terminal_case_claim_rows_synthesized": False,
        "missing_required_initial_artifact_is_failure": True,
        "missing_required_reentry_artifact_is_failure": True,
        "unreported_extra_regeneration_audit_is_failure": True,
        "audit_record_subset_selection_allowed": False,
        "result_conditioned_artifact_selection_allowed": False,
        "frozen_legacy_contract_is_legacy_authority": True,
        "neutral_replay_uses_frozen_semantic_binding": True,
        "semantic_fidelity_replay_must_match_frozen_audit": True,
        "llm_calls_allowed": False,
        "artifact_mutation_allowed": False,
        "campaign_artifact_writes_allowed": False,
        "collector_frozen_before_p29_p33_execution": True,
        "p29_p33_outputs_observed_before_collector_freeze": False,
        "campaign_execution_may_begin_after_this_freeze": True,
        "engineering_diagnostic_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV5(
        **body,
        freeze_id=(
            "prospective_atomic_admissibility_comparison_collector_freeze_v5:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


def _claim_index(plan: LiteratureQueryPlan) -> dict[str, tuple[str, NoveltyClaim]]:
    out: dict[str, tuple[str, NoveltyClaim]] = {}
    for group in plan.claims:
        for claim in group.claims:
            if claim.claim_id in out:
                raise ValueError("duplicate claim ID in query plan: " + claim.claim_id)
            out[claim.claim_id] = (group.hypothesis_id, claim)
    return out


def _card_index(portfolio: HypothesisPortfolio) -> dict[str, HypothesisCard]:
    out: dict[str, HypothesisCard] = {}
    for card in portfolio.hypotheses:
        if card.hypothesis_id in out:
            raise ValueError(
                "duplicate hypothesis ID in portfolio: " + card.hypothesis_id
            )
        out[card.hypothesis_id] = card
    return out


def _contract_claim_index(
    contract: PreN10ScientificContractReportV1,
) -> dict[str, PreN10ClaimContractRowV1]:
    out: dict[str, PreN10ClaimContractRowV1] = {}
    for hypothesis in contract.hypotheses:
        for claim in hypothesis.claims:
            if claim.claim_id in out:
                raise ValueError(
                    "duplicate claim ID in frozen contract: " + claim.claim_id
                )
            out[claim.claim_id] = claim
    return out


def _reconstruct_draft(
    *,
    canonical: NoveltyClaim,
    record: dict,
    claim_id_to_local_id: dict[str, str],
) -> NoveltyClaimDraft:
    fidelity = record.get("semantic_fidelity_shadow")
    if not isinstance(fidelity, dict):
        raise ValueError(
            "audit record lacks semantic_fidelity_shadow for "
            + canonical.claim_id
        )

    component_local_ids = []
    for claim_id in canonical.higher_order_component_claim_ids:
        local_id = claim_id_to_local_id.get(claim_id)
        if local_id is None:
            raise ValueError(
                "cannot recover higher-order component local ID for " + claim_id
            )
        component_local_ids.append(local_id)

    binding = NoveltyClaimSemanticFidelityBindingDraft(
        proposition_basis=str(fidelity.get("proposition_basis") or ""),
        relation_endpoint_anchors=list(
            fidelity.get("relation_endpoint_anchors") or []
        ),
        scope_qualifier_spans=list(
            fidelity.get("scope_qualifier_spans") or []
        ),
        directional_qualifier_spans=list(
            fidelity.get("directional_qualifier_spans") or []
        ),
        prediction_observation_id=(
            str(fidelity.get("prediction_observation_id")).strip()
            if fidelity.get("prediction_observation_id")
            else None
        ),
        falsification_criterion_id=(
            str(fidelity.get("falsification_criterion_id")).strip()
            if fidelity.get("falsification_criterion_id")
            else None
        ),
    )

    return NoveltyClaimDraft(
        local_id=str(record.get("claim_local_id") or "").strip(),
        kind=canonical.kind,
        importance=canonical.importance,
        novelty_selection_role=canonical.novelty_selection_role,
        text=canonical.text,
        rationale=canonical.rationale,
        search_concepts=list(canonical.search_concepts),
        search_queries=list(canonical.search_queries),
        distinguishing_terms=list(canonical.distinguishing_terms),
        prior_art_identity_terms=list(
            record.get("raw_prior_art_identity_terms")
            or canonical.prior_art_identity_terms
        ),
        relation_nucleus_terms=list(canonical.relation_nucleus_terms),
        semantic_fidelity_binding=binding,
        higher_order_relation_basis=list(
            canonical.higher_order_relation_basis
        ),
        higher_order_component_local_ids=component_local_ids,
        required_bridge=str(record.get("raw_required_bridge") or ""),
        predicted_observation=str(
            record.get("raw_predicted_observation") or ""
        ),
        falsification_condition=str(
            record.get("raw_falsification_condition") or ""
        ),
        scientific_structure=canonical.scientific_structure,
        diagnostic_query_kind=canonical.diagnostic_query_kind,
        diagnostic_search_query=canonical.diagnostic_search_query,
        diagnostic_structural_terms=list(
            canonical.diagnostic_structural_terms
        ),
        diagnostic_relation_terms=list(
            canonical.diagnostic_relation_terms
        ),
    )


def classify_pair(
    *,
    legacy_ready: bool,
    neutral_ready: bool,
) -> ComparisonPairV5:
    if legacy_ready and neutral_ready:
        return "LEGACY_READY__NEUTRAL_READY"
    if legacy_ready and not neutral_ready:
        return "LEGACY_READY__NEUTRAL_NOT_READY"
    if not legacy_ready and neutral_ready:
        return "LEGACY_NOT_READY__NEUTRAL_READY"
    return "LEGACY_NOT_READY__NEUTRAL_NOT_READY"


def _legacy_has_exact_text_source_failure(reasons: list[str]) -> bool:
    return any(
        reason in EXACT_TEXT_SOURCE_CODES
        or reason.startswith(EXACT_TEXT_SOURCE_PREFIXES)
        for reason in reasons
    )


def classify_reason_transition(
    *,
    legacy_source_reasons: list[str],
    neutral_source_reference_status: str,
    neutral_semantic_fidelity_status: str,
    neutral_blocking_dimensions: list[str],
    neutral_ready: bool,
) -> ReasonTransitionV5:
    exact_text_failure = _legacy_has_exact_text_source_failure(
        legacy_source_reasons
    )

    if (
        exact_text_failure
        and neutral_source_reference_status == "READY"
        and neutral_ready
    ):
        return "REPRESENTATION_LOSS_ONLY_CANDIDATE"

    if (
        exact_text_failure
        and neutral_source_reference_status == "READY"
        and "SEMANTIC_FIDELITY" in neutral_blocking_dimensions
    ):
        return "EXACT_TEXT_FAILURE_RECLASSIFIED_WITH_SEMANTIC_BLOCKER"

    if (
        neutral_source_reference_status != "READY"
        and neutral_semantic_fidelity_status != "PASS"
    ):
        return "MULTI_DIMENSION_NEUTRAL_BLOCKING"

    if neutral_source_reference_status != "READY":
        return "SOURCE_REFERENCE_BLOCKING_REMAINS"

    if "SEMANTIC_FIDELITY" in neutral_blocking_dimensions:
        return "SEMANTIC_BLOCKING_EXPOSED"

    if neutral_blocking_dimensions:
        return "OTHER_NEUTRAL_BLOCKING"

    return "NO_NEUTRAL_BLOCKER"


def _collect_claim_set(
    *,
    case_id: CaseIdV5,
    stage: ComparisonStageV5,
    source_hypothesis_id: str | None,
    portfolio_path: Path,
    query_path: Path,
    contract_path: Path,
    audit_path: Path,
    expected_audit_schema: str,
) -> list[ProspectiveAtomicAdmissibilityClaimComparisonV5]:
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    query = LiteratureQueryPlan.model_validate_json(
        query_path.read_text(encoding="utf-8")
    )
    contract = PreN10ScientificContractReportV1.model_validate_json(
        contract_path.read_text(encoding="utf-8")
    )
    audit = _load_json_object(audit_path)

    if str(audit.get("schema_version") or "") != expected_audit_schema:
        raise ValueError("unexpected decomposition audit schema: " + str(audit_path))
    if audit.get("diagnostic_only") is not True:
        raise ValueError("decomposition audit lost diagnostic-only boundary")
    if audit.get("production_authority") is not False:
        raise ValueError("decomposition audit gained production authority")

    if query.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("comparison query/portfolio ID mismatch")
    if contract.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("comparison contract/portfolio ID mismatch")
    if contract.source_query_plan_id != query.plan_id:
        raise ValueError("comparison contract/query-plan ID mismatch")
    if contract.source_query_plan_sha256 != sha256_file(query_path):
        raise ValueError("comparison contract/query-plan file SHA mismatch")

    if expected_audit_schema == "pre-n10-specification-sanitization-audit-v1":
        if str(audit.get("source_portfolio_id") or "") != portfolio.portfolio_id:
            raise ValueError("initial audit/source portfolio mismatch")
    else:
        checks = (
            ("regenerated_portfolio_id", portfolio.portfolio_id),
            ("query_plan_id", query.plan_id),
            ("contract_report_id", contract.report_id),
        )
        for field, expected in checks:
            if str(audit.get(field) or "") != expected:
                raise ValueError(
                    "regeneration audit identity mismatch for " + field
                )
        sha_checks = (
            ("regenerated_portfolio_file_sha256", sha256_file(portfolio_path)),
            ("query_plan_file_sha256", sha256_file(query_path)),
            ("contract_report_file_sha256", sha256_file(contract_path)),
        )
        for field, expected in sha_checks:
            if str(audit.get(field) or "") != expected:
                raise ValueError(
                    "regeneration audit file SHA mismatch for " + field
                )

    claims = _claim_index(query)
    cards = _card_index(portfolio)
    frozen_claims = _contract_claim_index(contract)

    records = [
        row for row in (audit.get("records") or [])
        if isinstance(row, dict)
    ]
    record_ids = [str(row.get("claim_id") or "").strip() for row in records]
    if any(not value for value in record_ids):
        raise ValueError("decomposition audit record lacks claim_id")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("decomposition audit contains duplicate claim_id")
    if set(record_ids) != set(claims):
        raise ValueError(
            "decomposition audit must cover every canonical query-plan claim"
        )
    if set(record_ids) != set(frozen_claims):
        raise ValueError(
            "decomposition audit/frozen legacy contract claim population mismatch"
        )

    claim_id_to_local_id = {
        str(row.get("claim_id")): str(row.get("claim_local_id") or "").strip()
        for row in records
    }
    if any(not value for value in claim_id_to_local_id.values()):
        raise ValueError("decomposition audit record lacks claim_local_id")

    rows: list[ProspectiveAtomicAdmissibilityClaimComparisonV5] = []
    for record in records:
        claim_id = str(record["claim_id"])
        hypothesis_id, canonical = claims[claim_id]
        card = cards.get(hypothesis_id)
        if card is None:
            raise ValueError(
                "query-plan hypothesis missing from portfolio: " + hypothesis_id
            )

        draft = _reconstruct_draft(
            canonical=canonical,
            record=record,
            claim_id_to_local_id=claim_id_to_local_id,
        )
        legacy_atomic = compile_legacy_atomic_specification_shadow_row(
            hypothesis=card,
            draft_claim=draft,
            canonical_claim=canonical,
        )

        frozen_fidelity = record.get("semantic_fidelity_shadow")
        frozen_fidelity = (
            frozen_fidelity if isinstance(frozen_fidelity, dict) else {}
        )
        frozen_reasons = list(frozen_fidelity.get("reason_codes") or [])
        if frozen_reasons != list(
            legacy_atomic.semantic_fidelity_reason_codes
        ):
            raise ValueError(
                "semantic-fidelity replay drift for claim " + claim_id
            )

        neutral_assessment = adapt_legacy_atomic_specification_shadow_row(
            legacy_atomic
        )
        neutral = assess_atomic_pre_n10_shadow_row(neutral_assessment)
        legacy = frozen_claims[claim_id]

        legacy_ready = (
            legacy.contract_status == "READY_FOR_N10_CONTRACT"
        )
        neutral_ready = (
            neutral.readiness_status == "READY_FOR_N10_SHADOW"
        )
        blockers = list(neutral.blocking_dimensions)
        source_reasons = list(legacy.source_contract_reason_codes)

        rows.append(
            ProspectiveAtomicAdmissibilityClaimComparisonV5(
                case_id=case_id,
                stage=stage,
                source_hypothesis_id=source_hypothesis_id,
                hypothesis_id=hypothesis_id,
                claim_id=claim_id,
                claim_rank=canonical.claim_rank,
                kind=canonical.kind,
                novelty_selection_role=canonical.novelty_selection_role,
                pair=classify_pair(
                    legacy_ready=legacy_ready,
                    neutral_ready=neutral_ready,
                ),
                reason_transition=classify_reason_transition(
                    legacy_source_reasons=source_reasons,
                    neutral_source_reference_status=(
                        neutral.source_reference_status
                    ),
                    neutral_semantic_fidelity_status=(
                        neutral.semantic_fidelity_status
                    ),
                    neutral_blocking_dimensions=blockers,
                    neutral_ready=neutral_ready,
                ),
                legacy_ready=legacy_ready,
                legacy_contract_status=legacy.contract_status,
                legacy_binding_reason_codes=list(
                    legacy.binding_contract_reason_codes
                ),
                legacy_source_reason_codes=source_reasons,
                legacy_router_hint=legacy.router_hint,
                neutral_ready=neutral_ready,
                neutral_blocking_dimensions=blockers,
                neutral_source_reference_status=(
                    neutral.source_reference_status
                ),
                neutral_semantic_fidelity_status=(
                    neutral.semantic_fidelity_status
                ),
                neutral_specification_status=neutral.specification_status,
                neutral_atomic_kind_status=neutral.atomic_kind_status,
                neutral_structural_compilation_status=(
                    neutral.structural_compilation_status
                ),
                prediction_observation_id=neutral.prediction_observation_id,
                falsification_criterion_id=neutral.falsification_criterion_id,
                portfolio_path=str(portfolio_path.resolve()),
                query_plan_path=str(query_path.resolve()),
                contract_path=str(contract_path.resolve()),
                audit_path=str(audit_path.resolve()),
            )
        )

    return rows


def is_terminal_before_initial_vpre_status(status: str) -> bool:
    return status in {
        "ZERO_HYPOTHESES_AFTER_ALPHA4",
        "INITIAL_SEMANTIC_TERMINAL",
    }


def collect_prospective_atomic_admissibility_comparison_v5(
    freeze: ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV5,
) -> ProspectiveAtomicAdmissibilityComparisonReportV5:
    all_claims: list[ProspectiveAtomicAdmissibilityClaimComparisonV5] = []
    case_rows: list[ProspectiveAtomicAdmissibilityCaseAccountingV5] = []

    for case in freeze.cases:
        campaign_report_path = Path(case.campaign_report_path)
        campaign = PreN10ProspectiveCampaignReportV1.model_validate_json(
            campaign_report_path.read_text(encoding="utf-8")
        )
        stage_map = {row.stage_name: row for row in campaign.stages}
        if len(stage_map) != len(campaign.stages):
            raise ValueError("campaign report contains duplicate stage names")

        initial_stage = stage_map.get("initial_vpre")
        reentry_stage = stage_map.get("regeneration_semantic_reentry")
        if initial_stage is None or reentry_stage is None:
            raise ValueError("campaign report lacks required accounting stages")

        initial_count = 0
        regen_count = 0
        regen_lineage_count = 0
        regen_decomp_count = 0
        regen_terminal_count = 0

        if is_terminal_before_initial_vpre_status(
            campaign.final_status
        ):
            if initial_stage.status != "SKIPPED_TERMINAL":
                raise ValueError(
                    "terminal campaign must mark initial_vpre SKIPPED_TERMINAL"
                )
            forbidden_existing = [
                Path(case.initial_vpre_query_path),
                Path(case.initial_vpre_contract_path),
                Path(case.initial_vpre_audit_path),
            ]
            if any(path.exists() for path in forbidden_existing):
                raise ValueError(
                    "terminal-before-initial case unexpectedly has V_pre artifacts"
                )
            if Path(case.regeneration_reentry_report_path).exists():
                raise ValueError(
                    "terminal-before-initial case unexpectedly has reentry report"
                )
            accounting_status: CaseAccountingStatusV5 = (
                "TERMINAL_BEFORE_INITIAL_VPRE"
            )
        else:
            if initial_stage.status not in {"EXECUTED", "REUSED_VALIDATED"}:
                raise ValueError(
                    "nonterminal campaign must have active initial_vpre stage"
                )
            initial_paths = [
                Path(case.initial_portfolio_path),
                Path(case.initial_vpre_query_path),
                Path(case.initial_vpre_contract_path),
                Path(case.initial_vpre_audit_path),
            ]
            missing = [str(path) for path in initial_paths if not path.is_file()]
            if missing:
                raise ValueError(
                    "missing required initial comparison artifact(s): "
                    + ", ".join(missing)
                )
            initial_rows = _collect_claim_set(
                case_id=case.case_id,
                stage="INITIAL_VPRE",
                source_hypothesis_id=None,
                portfolio_path=initial_paths[0],
                query_path=initial_paths[1],
                contract_path=initial_paths[2],
                audit_path=initial_paths[3],
                expected_audit_schema=(
                    "pre-n10-specification-sanitization-audit-v1"
                ),
            )
            all_claims.extend(initial_rows)
            initial_count = len(initial_rows)

            reentry_path = Path(case.regeneration_reentry_report_path)
            expected_audits: set[Path] = set()
            if reentry_stage.status in {"EXECUTED", "REUSED_VALIDATED"}:
                if not reentry_path.is_file():
                    raise ValueError(
                        "active regeneration reentry stage lacks report"
                    )
                reentry = PreN10RegenerationReentryReportV2.model_validate_json(
                    reentry_path.read_text(encoding="utf-8")
                )
                regen_lineage_count = len(reentry.lineages)

                for lineage in reentry.lineages:
                    if lineage.claim_decomposition_request_count == 0:
                        regen_terminal_count += 1
                        continue

                    regen_decomp_count += 1
                    if (
                        lineage.regenerated_portfolio_path is None
                        or lineage.query_plan_path is None
                        or lineage.contract_report_path is None
                    ):
                        raise ValueError(
                            "decomposition lineage lacks required artifact paths"
                        )

                    query_path = Path(lineage.query_plan_path).expanduser().resolve()
                    lineage_dir = query_path.parent
                    expected_dir = (
                        Path(case.regeneration_lineage_root)
                        / _slug(lineage.source_hypothesis_id)
                    ).expanduser().resolve()
                    if lineage_dir != expected_dir:
                        raise ValueError(
                            "reentry lineage directory does not match frozen rule"
                        )

                    audit_path = (
                        lineage_dir / case.regeneration_audit_filename
                    )
                    expected_audits.add(audit_path.resolve())

                    regen_rows = _collect_claim_set(
                        case_id=case.case_id,
                        stage="REGENERATION_REENTRY",
                        source_hypothesis_id=lineage.source_hypothesis_id,
                        portfolio_path=Path(
                            lineage.regenerated_portfolio_path
                        ).expanduser().resolve(),
                        query_path=query_path,
                        contract_path=Path(
                            lineage.contract_report_path
                        ).expanduser().resolve(),
                        audit_path=audit_path,
                        expected_audit_schema=(
                            "pre-n10-regeneration-decomposition-"
                            "sanitization-audit-v1"
                        ),
                    )
                    all_claims.extend(regen_rows)
                    regen_count += len(regen_rows)

                lineage_root = Path(case.regeneration_lineage_root)
                observed_audits = {
                    path.resolve()
                    for path in lineage_root.glob(
                        "*/claim_decomposition.sanitization_audit.json"
                    )
                    if path.is_file()
                } if lineage_root.is_dir() else set()
                if observed_audits != expected_audits:
                    raise ValueError(
                        "reentry audit population differs from frozen collector rule"
                    )
            else:
                if reentry_stage.status not in {
                    "SKIPPED_NOT_REQUIRED",
                    "SKIPPED_TERMINAL",
                }:
                    raise ValueError(
                        "unexpected regeneration reentry stage status"
                    )
                if reentry_path.exists():
                    raise ValueError(
                        "skipped regeneration reentry unexpectedly has report"
                    )

            accounting_status = (
                "CLAIMS_COLLECTED"
                if initial_count + regen_count > 0
                else "NO_CLAIMS_OBSERVED"
            )

        case_rows.append(
            ProspectiveAtomicAdmissibilityCaseAccountingV5(
                case_id=case.case_id,
                campaign_final_status=campaign.final_status,
                accounting_status=accounting_status,
                initial_vpre_stage_status=initial_stage.status,
                initial_claim_comparison_count=initial_count,
                regeneration_reentry_stage_status=reentry_stage.status,
                regeneration_lineage_count=regen_lineage_count,
                regeneration_decomposition_lineage_count=regen_decomp_count,
                regeneration_terminal_lineage_count=regen_terminal_count,
                regeneration_claim_comparison_count=regen_count,
                total_claim_comparison_count=initial_count + regen_count,
            )
        )

    pair_counts = dict(
        sorted(Counter(row.pair for row in all_claims).items())
    )
    transition_counts = dict(
        sorted(Counter(row.reason_transition for row in all_claims).items())
    )
    blocker_counts = dict(
        sorted(
            Counter(
                blocker
                for row in all_claims
                for blocker in row.neutral_blocking_dimensions
            ).items()
        )
    )

    body = {
        "schema_version": (
            "prospective-atomic-admissibility-comparison-report-v5"
        ),
        "source_collector_freeze_id": freeze.freeze_id,
        "source_collector_freeze_sha256": freeze.freeze_sha256,
        "case_ids": ["P29", "P30", "P31", "P32", "P33"],
        "case_count": 5,
        "terminal_before_initial_vpre_case_count": sum(
            row.accounting_status == "TERMINAL_BEFORE_INITIAL_VPRE"
            for row in case_rows
        ),
        "claim_comparison_count": len(all_claims),
        "pair_counts": pair_counts,
        "reason_transition_counts": transition_counts,
        "neutral_blocker_dimension_counts": blocker_counts,
        "cases": [row.model_dump(mode="json") for row in case_rows],
        "claims": [row.model_dump(mode="json") for row in all_claims],
        "case_denominator_preserved": True,
        "claim_subset_selection_performed": False,
        "llm_calls_performed": False,
        "campaign_artifacts_mutated": False,
        "diagnostic_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveAtomicAdmissibilityComparisonReportV5(
        **body,
        report_id=(
            "prospective_atomic_admissibility_comparison_report_v5:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "ComparisonPairV5",
    "ProspectiveAtomicAdmissibilityCaseAccountingV5",
    "ProspectiveAtomicAdmissibilityClaimComparisonV5",
    "ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV5",
    "ProspectiveAtomicAdmissibilityComparisonReportV5",
    "build_prospective_atomic_admissibility_comparison_collector_freeze_v5",
    "classify_pair",
    "classify_reason_transition",
    "collect_prospective_atomic_admissibility_comparison_v5",
    "is_terminal_before_initial_vpre_status",
    "sha256_file",
]
