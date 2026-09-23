from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.prospective_relational_campaign import (
    ProspectiveRelationalCampaignResult,
    ProspectiveRelationalCaseResult,
)
from pipeline_core.discovery.prospective_relational_execution_plan import (
    ProspectiveRelationalHypothesisSelection,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    RelationalAtomicEndpointBindingReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


DiagnosticClass = Literal[
    "ENDPOINT_LITERAL_ALIGNMENT_FAILURE",
    "ENDPOINT_ATOMICITY_OR_DISTINCTNESS_FAILURE",
    "ENDPOINT_IDENTITY_COLLISION_FAILURE",
    "ENDPOINT_UNGROUNDED_OR_INFERENTIAL_FAILURE",
    "ENDPOINT_UNCLASSIFIED_ABSTENTION",
    "CONTRACT_COMPLETENESS_FAILURE",
    "CONTRACT_LITERAL_IDENTITY_ALIGNMENT_FAILURE",
    "CONTRACT_ROLE_OR_IDENTITY_METADATA_FAILURE",
    "MAIN_E2E_INFRASTRUCTURE_FAILURE",
    "MAIN_E2E_UPSTREAM_STAGE_FAILURE",
    "MAIN_E2E_UNKNOWN_FAILURE",
    "NON_REPAIR_PREVERIFIER_DISPOSITION",
]

RepairAction = Literal[
    "LEXICAL_ALIGNMENT_REPAIR",
    "ATOMIC_SPECIFICATION_REPAIR",
    "IDENTITY_ALIGNMENT_REPAIR",
    "CONTRACT_COMPLETION_REPAIR",
    "METADATA_CONTRACT_REPAIR",
    "NO_SCIENTIFIC_REPAIR_INFRASTRUCTURE",
    "MANUAL_DIAGNOSIS_REQUIRED",
    "NO_AUTOMATIC_PREVERIFIER_REPAIR",
]

ScientificDeltaPolicy = Literal[
    "ZERO_SCIENTIFIC_DELTA_REQUIRED",
    "NO_SCIENTIFIC_REPAIR",
    "ESCALATE_BEFORE_REPAIR",
]


class PreVerifierClaimSnapshot(StrictModel):
    claim_id: str
    novelty_selection_role: str | None = None
    binding_status: str
    reason_codes: list[str]
    claim_text: str
    required_bridge: str
    predicted_observation: str
    falsification_condition: str
    prior_art_identity_terms: list[str]
    relation_nucleus_terms: list[str]


class PreVerifierClaimDiagnostic(StrictModel):
    claim_id: str
    diagnostic_classes: list[DiagnosticClass]
    proposed_repair_actions: list[RepairAction]
    scientific_delta_policy: ScientificDeltaPolicy
    abstention_reason: str | None = None
    source_claim: PreVerifierClaimSnapshot


class PreVerifierCaseDiagnostic(StrictModel):
    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    source_case_result_id: str
    source_case_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_disposition: str
    primary_diagnostic_class: DiagnosticClass
    diagnostic_classes: list[DiagnosticClass]
    proposed_repair_actions: list[RepairAction]
    scientific_delta_policy: ScientificDeltaPolicy
    automatic_repair_eligible: bool
    claim_diagnostics: list[PreVerifierClaimDiagnostic]
    main_e2e_status: str | None = None
    failed_stage_name: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    source_artifact_sha256s: dict[str, str]
    diagnosis_only: Literal[True] = True
    repair_performed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    external_novelty_outcome_used: Literal[False] = False
    verifier_outcome_used: Literal[False] = False
    old_n10_status_used_as_repair_feature: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class PreVerifierSpecificationDiagnosticReport(StrictModel):
    schema_version: Literal[
        "preverifier-specification-diagnostic-report-v1"
    ] = "preverifier-specification-diagnostic-report-v1"
    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_campaign_result_id: str
    source_campaign_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_campaign_result_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[PreVerifierCaseDiagnostic]
    case_count: Literal[5] = 5
    case_ids: list[str]
    primary_diagnostic_counts: dict[str, int]
    automatic_repair_eligible_count: int = Field(ge=0)
    failure_taxonomy_frozen_before_repair: Literal[True] = True
    diagnosis_only: Literal[True] = True
    repair_performed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    external_novelty_outcome_used: Literal[False] = False
    verifier_outcome_used: Literal[False] = False
    old_n10_status_used_as_repair_feature: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreVerifierSpecificationDiagnosticReport":
        expected = ["P06", "P07", "P08", "P09", "P10"]
        if self.case_ids != expected:
            raise ValueError("diagnostic case IDs must be exactly P06-P10")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("diagnostic cases must be ordered P06-P10")
        if len(self.cases) != 5:
            raise ValueError("diagnostic report requires exactly five cases")
        counts = Counter(row.primary_diagnostic_class for row in self.cases)
        if dict(sorted(counts.items())) != dict(sorted(self.primary_diagnostic_counts.items())):
            raise ValueError("primary_diagnostic_counts mismatch")
        eligible = sum(row.automatic_repair_eligible for row in self.cases)
        if eligible != self.automatic_repair_eligible_count:
            raise ValueError("automatic_repair_eligible_count mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("preverifier diagnostic report SHA mismatch")
        if observed_id != "preverifier_specification_diagnostic_report:" + expected_sha[:20]:
            raise ValueError("preverifier diagnostic report ID mismatch")
        return self


def classify_endpoint_abstention(reason: str) -> DiagnosticClass:
    text = " ".join(str(reason or "").casefold().split())
    lexical_markers = (
        "without paraphrase", "paraphrase", "literal", "phrased as", "wording",
        "does not occur", "not present in both", "not shared by",
    )
    if any(marker in text for marker in lexical_markers):
        return "ENDPOINT_LITERAL_ALIGNMENT_FAILURE"
    identity_markers = (
        "prior_art_identity", "branch identity", "identity term", "identity itself",
    )
    if any(marker in text for marker in identity_markers):
        return "ENDPOINT_IDENTITY_COLLISION_FAILURE"
    inferential_markers = (
        "inference", "infer", "implied", "invent", "unstated", "not explicitly stated",
    )
    if any(marker in text for marker in inferential_markers):
        return "ENDPOINT_UNGROUNDED_OR_INFERENTIAL_FAILURE"
    atomicity_markers = (
        "cannot identify two", "cannot select two", "not two scientifically distinct",
        "same endpoint", "nested", "compound relation", "multiple relations",
    )
    if any(marker in text for marker in atomicity_markers):
        return "ENDPOINT_ATOMICITY_OR_DISTINCTNESS_FAILURE"
    return "ENDPOINT_UNCLASSIFIED_ABSTENTION"


def classify_binding_reason(reason_code: str) -> DiagnosticClass:
    value = str(reason_code or "")
    if value.startswith("identity_not_literal_in_"):
        return "CONTRACT_LITERAL_IDENTITY_ALIGNMENT_FAILURE"
    if value in {
        "missing_claim_text", "missing_required_bridge",
        "missing_predicted_observation", "missing_falsification_condition",
    }:
        return "CONTRACT_COMPLETENESS_FAILURE"
    if value in {"missing_prior_art_identity_terms", "missing_novelty_selection_role"}:
        return "CONTRACT_ROLE_OR_IDENTITY_METADATA_FAILURE"
    return "CONTRACT_COMPLETENESS_FAILURE"


def repair_action_for_class(diagnostic_class: DiagnosticClass) -> tuple[RepairAction, ScientificDeltaPolicy, bool]:
    mapping = {
        "ENDPOINT_LITERAL_ALIGNMENT_FAILURE": ("LEXICAL_ALIGNMENT_REPAIR", "ZERO_SCIENTIFIC_DELTA_REQUIRED", True),
        "ENDPOINT_ATOMICITY_OR_DISTINCTNESS_FAILURE": ("ATOMIC_SPECIFICATION_REPAIR", "ZERO_SCIENTIFIC_DELTA_REQUIRED", True),
        "ENDPOINT_IDENTITY_COLLISION_FAILURE": ("IDENTITY_ALIGNMENT_REPAIR", "ZERO_SCIENTIFIC_DELTA_REQUIRED", True),
        "ENDPOINT_UNGROUNDED_OR_INFERENTIAL_FAILURE": ("MANUAL_DIAGNOSIS_REQUIRED", "ESCALATE_BEFORE_REPAIR", False),
        "ENDPOINT_UNCLASSIFIED_ABSTENTION": ("MANUAL_DIAGNOSIS_REQUIRED", "ESCALATE_BEFORE_REPAIR", False),
        "CONTRACT_COMPLETENESS_FAILURE": ("CONTRACT_COMPLETION_REPAIR", "ZERO_SCIENTIFIC_DELTA_REQUIRED", True),
        "CONTRACT_LITERAL_IDENTITY_ALIGNMENT_FAILURE": ("IDENTITY_ALIGNMENT_REPAIR", "ZERO_SCIENTIFIC_DELTA_REQUIRED", True),
        "CONTRACT_ROLE_OR_IDENTITY_METADATA_FAILURE": ("METADATA_CONTRACT_REPAIR", "ZERO_SCIENTIFIC_DELTA_REQUIRED", True),
        "MAIN_E2E_INFRASTRUCTURE_FAILURE": ("NO_SCIENTIFIC_REPAIR_INFRASTRUCTURE", "NO_SCIENTIFIC_REPAIR", False),
        "MAIN_E2E_UPSTREAM_STAGE_FAILURE": ("MANUAL_DIAGNOSIS_REQUIRED", "ESCALATE_BEFORE_REPAIR", False),
        "MAIN_E2E_UNKNOWN_FAILURE": ("MANUAL_DIAGNOSIS_REQUIRED", "ESCALATE_BEFORE_REPAIR", False),
        "NON_REPAIR_PREVERIFIER_DISPOSITION": ("NO_AUTOMATIC_PREVERIFIER_REPAIR", "NO_SCIENTIFIC_REPAIR", False),
    }
    return mapping[diagnostic_class]


def _claim_snapshot(claim: RelationalAtomicBindingClaimPlan) -> PreVerifierClaimSnapshot:
    return PreVerifierClaimSnapshot(
        claim_id=claim.claim_id,
        novelty_selection_role=claim.novelty_selection_role,
        binding_status=claim.binding_status,
        reason_codes=list(claim.reason_codes),
        claim_text=claim.claim_text,
        required_bridge=claim.required_bridge,
        predicted_observation=claim.predicted_observation,
        falsification_condition=claim.falsification_condition,
        prior_art_identity_terms=list(claim.prior_art_identity_terms),
        relation_nucleus_terms=list(claim.relation_nucleus_terms),
    )


def _dedupe_ordered(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _primary_class(classes: list[DiagnosticClass]) -> DiagnosticClass:
    priority = [
        "MAIN_E2E_INFRASTRUCTURE_FAILURE",
        "MAIN_E2E_UPSTREAM_STAGE_FAILURE",
        "MAIN_E2E_UNKNOWN_FAILURE",
        "ENDPOINT_UNGROUNDED_OR_INFERENTIAL_FAILURE",
        "ENDPOINT_UNCLASSIFIED_ABSTENTION",
        "ENDPOINT_ATOMICITY_OR_DISTINCTNESS_FAILURE",
        "ENDPOINT_IDENTITY_COLLISION_FAILURE",
        "CONTRACT_LITERAL_IDENTITY_ALIGNMENT_FAILURE",
        "CONTRACT_COMPLETENESS_FAILURE",
        "CONTRACT_ROLE_OR_IDENTITY_METADATA_FAILURE",
        "ENDPOINT_LITERAL_ALIGNMENT_FAILURE",
        "NON_REPAIR_PREVERIFIER_DISPOSITION",
    ]
    observed = set(classes)
    for value in priority:
        if value in observed:
            return value  # type: ignore[return-value]
    return "NON_REPAIR_PREVERIFIER_DISPOSITION"


def _aggregate_policy(classes: list[DiagnosticClass]) -> tuple[list[RepairAction], ScientificDeltaPolicy, bool]:
    actions = []
    policies = []
    eligibilities = []
    for value in classes:
        action, policy, eligible = repair_action_for_class(value)
        actions.append(action)
        policies.append(policy)
        eligibilities.append(eligible)
    actions = _dedupe_ordered(actions)
    if "NO_SCIENTIFIC_REPAIR" in policies:
        policy = "NO_SCIENTIFIC_REPAIR"
    elif "ESCALATE_BEFORE_REPAIR" in policies:
        policy = "ESCALATE_BEFORE_REPAIR"
    else:
        policy = "ZERO_SCIENTIFIC_DELTA_REQUIRED"
    return actions, policy, bool(eligibilities and all(eligibilities))  # type: ignore[return-value]


def _classify_main_e2e_failure(manifest: dict) -> tuple[DiagnosticClass, str | None, str | None, str | None]:
    failure = manifest.get("failure")
    if not isinstance(failure, dict):
        return "MAIN_E2E_UNKNOWN_FAILURE", None, None, None
    failure_type = str(failure.get("type") or "") or None
    failure_message = str(failure.get("message") or "") or None
    haystack = " ".join([
        str(failure_type or ""), str(failure_message or ""),
        str(failure.get("traceback") or ""),
    ]).casefold()
    infrastructure_markers = (
        "timeout", "timed out", "connection", "connecterror", "ratelimit",
        "rate limit", "429", "502", "503", "504", "authentication",
        "api key", "service unavailable", "network",
    )
    if any(marker in haystack for marker in infrastructure_markers):
        diagnostic = "MAIN_E2E_INFRASTRUCTURE_FAILURE"
    else:
        stages = manifest.get("stages")
        diagnostic = "MAIN_E2E_UPSTREAM_STAGE_FAILURE" if isinstance(stages, list) and stages else "MAIN_E2E_UNKNOWN_FAILURE"
    failed_stage = None
    stages = manifest.get("stages")
    if isinstance(stages, list):
        for row in reversed(stages):
            if isinstance(row, dict) and row.get("status") == "failed":
                failed_stage = str(row.get("name") or "") or None
                break
    return diagnostic, failed_stage, failure_type, failure_message  # type: ignore[return-value]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def diagnose_case(*, campaign_root: Path, case_result: ProspectiveRelationalCaseResult) -> PreVerifierCaseDiagnostic:
    case_root = campaign_root / case_result.case_id
    disposition = case_result.disposition
    classes = []
    claim_diagnostics = []
    artifact_hashes = {}
    main_status = case_result.main_e2e_manifest_status
    failed_stage_name = failure_type = failure_message = None

    if disposition == "ENDPOINT_BINDING_ABSTAINED":
        selected_plan_path = case_root / "relational_atomic_binding_plan.selected.json"
        endpoint_path = case_root / "relational_atomic_endpoint_binding.selected.json"
        if not selected_plan_path.is_file() or not endpoint_path.is_file():
            raise ValueError(case_result.case_id + ": endpoint-abstained case is missing frozen artifacts")
        plan = RelationalAtomicBindingPlan.model_validate_json(selected_plan_path.read_text(encoding="utf-8"))
        endpoint = RelationalAtomicEndpointBindingReport.model_validate_json(endpoint_path.read_text(encoding="utf-8"))
        if endpoint.source_binding_plan_id != plan.plan_id or endpoint.source_binding_plan_sha256 != plan.plan_sha256:
            raise ValueError(case_result.case_id + ": endpoint/plan lineage mismatch")
        claim_by_id = {
            claim.claim_id: claim
            for hypothesis in plan.hypotheses
            for claim in hypothesis.claims
        }
        for binding in endpoint.bindings:
            if binding.outcome != "ABSTAINED_UNBINDABLE":
                continue
            claim = claim_by_id.get(binding.claim_id)
            if claim is None:
                raise ValueError(case_result.case_id + ": endpoint binding claim absent from selected plan")
            diagnostic_class = classify_endpoint_abstention(str(binding.abstention_reason or ""))
            action, policy, _ = repair_action_for_class(diagnostic_class)
            classes.append(diagnostic_class)
            claim_diagnostics.append(PreVerifierClaimDiagnostic(
                claim_id=claim.claim_id,
                diagnostic_classes=[diagnostic_class],
                proposed_repair_actions=[action],
                scientific_delta_policy=policy,
                abstention_reason=binding.abstention_reason,
                source_claim=_claim_snapshot(claim),
            ))
        artifact_hashes = {
            "selected_binding_plan": _sha256_file(selected_plan_path),
            "endpoint_binding_report": _sha256_file(endpoint_path),
        }

    elif disposition == "NO_BINDING_READY_HYPOTHESIS":
        full_plan_path = case_root / "relational_atomic_binding_plan.full.json"
        selection_path = case_root / "prospective_relational_hypothesis_selection.json"
        if not full_plan_path.is_file() or not selection_path.is_file():
            raise ValueError(case_result.case_id + ": no-binding-ready case is missing frozen artifacts")
        plan = RelationalAtomicBindingPlan.model_validate_json(full_plan_path.read_text(encoding="utf-8"))
        selection = ProspectiveRelationalHypothesisSelection.model_validate_json(selection_path.read_text(encoding="utf-8"))
        if selection.source_binding_plan_id != plan.plan_id or selection.source_binding_plan_sha256 != plan.plan_sha256:
            raise ValueError(case_result.case_id + ": selection/plan lineage mismatch")
        if selection.status != "NO_BINDING_READY_HYPOTHESIS":
            raise ValueError(case_result.case_id + ": campaign disposition conflicts with selection status")
        for hypothesis in plan.hypotheses:
            for claim in hypothesis.claims:
                if claim.binding_status != "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION":
                    continue
                claim_classes = _dedupe_ordered([classify_binding_reason(reason) for reason in claim.reason_codes])
                if not claim_classes:
                    claim_classes = ["CONTRACT_COMPLETENESS_FAILURE"]
                claim_actions = []
                claim_policies = []
                for diagnostic_class in claim_classes:
                    action, policy, _ = repair_action_for_class(diagnostic_class)
                    claim_actions.append(action)
                    claim_policies.append(policy)
                    classes.append(diagnostic_class)
                policy = "ESCALATE_BEFORE_REPAIR" if "ESCALATE_BEFORE_REPAIR" in claim_policies else "ZERO_SCIENTIFIC_DELTA_REQUIRED"
                claim_diagnostics.append(PreVerifierClaimDiagnostic(
                    claim_id=claim.claim_id,
                    diagnostic_classes=claim_classes,
                    proposed_repair_actions=_dedupe_ordered(claim_actions),
                    scientific_delta_policy=policy,
                    source_claim=_claim_snapshot(claim),
                ))
        artifact_hashes = {
            "full_binding_plan": _sha256_file(full_plan_path),
            "structural_selection_report": _sha256_file(selection_path),
        }

    elif disposition == "MAIN_E2E_STAGE_FAILED":
        manifest_path = case_root / "e2e_runner.manifest.json"
        if not manifest_path.is_file():
            classes = ["MAIN_E2E_UNKNOWN_FAILURE"]
        else:
            manifest = _load_json(manifest_path)
            diagnostic_class, failed_stage_name, failure_type, failure_message = _classify_main_e2e_failure(manifest)
            classes = [diagnostic_class]
            main_status = str(manifest.get("status") or "") or main_status
            artifact_hashes = {"e2e_runner_manifest": _sha256_file(manifest_path)}
    else:
        classes = ["NON_REPAIR_PREVERIFIER_DISPOSITION"]

    classes = _dedupe_ordered(classes)
    if not classes:
        classes = ["NON_REPAIR_PREVERIFIER_DISPOSITION"]
    actions, policy, eligible = _aggregate_policy(classes)
    primary = _primary_class(classes)
    return PreVerifierCaseDiagnostic(
        case_id=case_result.case_id,
        source_case_result_id=case_result.result_id,
        source_case_result_sha256=case_result.result_sha256,
        original_disposition=case_result.disposition,
        primary_diagnostic_class=primary,
        diagnostic_classes=classes,
        proposed_repair_actions=actions,
        scientific_delta_policy=policy,
        automatic_repair_eligible=eligible,
        claim_diagnostics=claim_diagnostics,
        main_e2e_status=main_status,
        failed_stage_name=failed_stage_name,
        failure_type=failure_type,
        failure_message=failure_message,
        source_artifact_sha256s=artifact_hashes,
    )


def build_preverifier_specification_diagnostic_report(*, campaign_root: Path, campaign_result_path: Path) -> PreVerifierSpecificationDiagnosticReport:
    root = campaign_root.expanduser().resolve()
    result_path = campaign_result_path.expanduser().resolve()
    campaign = ProspectiveRelationalCampaignResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    case_results_dir = root / "P06_P10.case_results"
    cases = []
    expected_case_ids = ["P06", "P07", "P08", "P09", "P10"]
    for index, case_id in enumerate(expected_case_ids):
        case_result_path = case_results_dir / f"{case_id}.json"
        if not case_result_path.is_file():
            raise ValueError("missing immutable case result: " + str(case_result_path))
        case_result = ProspectiveRelationalCaseResult.model_validate_json(case_result_path.read_text(encoding="utf-8"))
        if case_result.case_id != case_id:
            raise ValueError("case-result case ID mismatch")
        if case_result.result_id != campaign.case_result_ids[index]:
            raise ValueError("campaign/case result ID mismatch")
        if case_result.result_sha256 != campaign.case_result_sha256s[index]:
            raise ValueError("campaign/case result SHA mismatch")
        if case_result.disposition != campaign.case_dispositions[case_id]:
            raise ValueError("campaign/case disposition mismatch")
        cases.append(diagnose_case(campaign_root=root, case_result=case_result))
    counts = Counter(row.primary_diagnostic_class for row in cases)
    body = {
        "schema_version": "preverifier-specification-diagnostic-report-v1",
        "source_campaign_result_id": campaign.campaign_result_id,
        "source_campaign_result_sha256": campaign.campaign_result_sha256,
        "source_campaign_result_file_sha256": _sha256_file(result_path),
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_count": 5,
        "case_ids": expected_case_ids,
        "primary_diagnostic_counts": dict(sorted(counts.items())),
        "automatic_repair_eligible_count": sum(row.automatic_repair_eligible for row in cases),
        "failure_taxonomy_frozen_before_repair": True,
        "diagnosis_only": True,
        "repair_performed": False,
        "llm_calls_performed": 0,
        "external_novelty_outcome_used": False,
        "verifier_outcome_used": False,
        "old_n10_status_used_as_repair_feature": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return PreVerifierSpecificationDiagnosticReport(
        **body,
        report_id="preverifier_specification_diagnostic_report:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "DiagnosticClass", "PreVerifierCaseDiagnostic", "PreVerifierClaimDiagnostic",
    "PreVerifierClaimSnapshot", "PreVerifierSpecificationDiagnosticReport",
    "RepairAction", "ScientificDeltaPolicy",
    "build_preverifier_specification_diagnostic_report",
    "classify_binding_reason", "classify_endpoint_abstention",
    "repair_action_for_class",
]
