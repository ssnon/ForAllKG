from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CampaignDisposition = Literal[
    "UPSTREAM_ABSTENTION",
    "ATOMIC_SYNTHESIS_ABSTENTION",
    "VERIFIER_REACHED_COMPLETE",
]


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


class ScientificVerifierProspectiveCampaignCase(StrictModel):
    case_id: str
    run_dir: str
    wrapper_status: str
    disposition: CampaignDisposition
    upstream_pre_n10_status: str | None = None
    candidate_contract_reached: bool
    atomic_cohort_frozen: bool
    verifier_reached: bool
    frozen_hypothesis_count: int = Field(default=0, ge=0)
    frozen_claim_count: int = Field(default=0, ge=0)
    agreement_count: int = Field(default=0, ge=0)
    disagreement_count: int = Field(default=0, ge=0)
    comparison_cells: dict[str, int] = Field(default_factory=dict)
    cohort_changed_after_freeze: bool = False
    old_n10_mutated_by_verifier: bool = False
    verifier_result_consumed_by_production: bool = False
    production_selection_changed: bool = False
    canonical_graph_mutated: bool = False

    @model_validator(mode="after")
    def validate_case(self) -> "ScientificVerifierProspectiveCampaignCase":
        if self.disposition == "VERIFIER_REACHED_COMPLETE":
            if not (self.candidate_contract_reached and self.atomic_cohort_frozen and self.verifier_reached):
                raise ValueError("completed verifier case must reach candidate contract, freeze, and verifier")
            if self.frozen_hypothesis_count < 1:
                raise ValueError("completed verifier case must freeze hypotheses")
            if self.agreement_count + self.disagreement_count != self.frozen_hypothesis_count:
                raise ValueError("old/new comparison count does not match frozen cohort")
        elif self.verifier_reached:
            raise ValueError("abstention case cannot be marked verifier-reached")
        if self.cohort_changed_after_freeze:
            raise ValueError("prospective campaign forbids cohort mutation after freeze")
        if self.old_n10_mutated_by_verifier:
            raise ValueError("prospective campaign forbids verifier mutation of old N10")
        if self.verifier_result_consumed_by_production:
            raise ValueError("prospective campaign forbids verifier result consumption")
        if self.production_selection_changed:
            raise ValueError("prospective campaign forbids production selection mutation")
        if self.canonical_graph_mutated:
            raise ValueError("prospective campaign forbids canonical graph mutation")
        return self


class ScientificVerifierProspectiveCampaignReport(StrictModel):
    schema_version: Literal["scientific-verifier-prospective-campaign-report-v1"] = "scientific-verifier-prospective-campaign-report-v1"
    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_root: str
    case_ids: list[str]
    cases: list[ScientificVerifierProspectiveCampaignCase]
    case_count: int = Field(ge=1)
    disposition_counts: dict[str, int]
    verifier_reached_case_count: int = Field(ge=0)
    verifier_not_reached_case_count: int = Field(ge=0)
    frozen_hypothesis_count: int = Field(ge=0)
    frozen_claim_count: int = Field(ge=0)
    old_new_agreement_count: int = Field(ge=0)
    old_new_disagreement_count: int = Field(ge=0)
    comparison_cells: dict[str, int]
    all_cases_terminal: Literal[True] = True
    no_post_freeze_cohort_mutation: Literal[True] = True
    no_old_n10_mutation_by_verifier: Literal[True] = True
    no_verifier_result_consumed_by_production: Literal[True] = True
    no_production_selection_mutation: Literal[True] = True
    no_canonical_graph_mutation: Literal[True] = True

    @model_validator(mode="after")
    def validate_report(self) -> "ScientificVerifierProspectiveCampaignReport":
        if self.case_count != len(self.cases) or self.case_ids != [row.case_id for row in self.cases]:
            raise ValueError("campaign case metadata mismatch")
        expected_dispositions = Counter(row.disposition for row in self.cases)
        if dict(sorted(expected_dispositions.items())) != dict(sorted(self.disposition_counts.items())):
            raise ValueError("campaign disposition counts mismatch")
        reached = sum(row.verifier_reached for row in self.cases)
        if self.verifier_reached_case_count != reached or self.verifier_not_reached_case_count != self.case_count - reached:
            raise ValueError("campaign verifier reach counts mismatch")
        if self.frozen_hypothesis_count != sum(row.frozen_hypothesis_count for row in self.cases):
            raise ValueError("campaign frozen hypothesis count mismatch")
        if self.frozen_claim_count != sum(row.frozen_claim_count for row in self.cases):
            raise ValueError("campaign frozen claim count mismatch")
        if self.old_new_agreement_count != sum(row.agreement_count for row in self.cases):
            raise ValueError("campaign agreement count mismatch")
        if self.old_new_disagreement_count != sum(row.disagreement_count for row in self.cases):
            raise ValueError("campaign disagreement count mismatch")
        expected_cells: Counter[str] = Counter()
        for row in self.cases:
            expected_cells.update(row.comparison_cells)
        if dict(sorted(expected_cells.items())) != dict(sorted(self.comparison_cells.items())):
            raise ValueError("campaign comparison cells mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("campaign report SHA mismatch")
        if observed_id != "scientific_verifier_prospective_campaign:" + expected_sha[:20]:
            raise ValueError("campaign report ID mismatch")
        return self


def _classify_case(run: Path) -> ScientificVerifierProspectiveCampaignCase:
    wrapper_path = run / "scientific_atomic_verifier_prospective_e2e_manifest.json"
    if not wrapper_path.is_file():
        raise ValueError(f"missing prospective wrapper manifest: {wrapper_path}")
    wrapper = _load_object(wrapper_path)
    status = str(wrapper.get("status") or "")
    if status in {"abstained_upstream_before_candidate_contract", "abstained_no_production_facing_candidates"}:
        disposition: CampaignDisposition = "UPSTREAM_ABSTENTION"
    elif status == "abstained_no_atomic_hypotheses":
        disposition = "ATOMIC_SYNTHESIS_ABSTENTION"
    elif status == "complete":
        disposition = "VERIFIER_REACHED_COMPLETE"
    else:
        raise ValueError(f"prospective case is not terminal: {run.name} status={status!r}")

    candidate_path = run / "scientific_pre_n10_candidate_portfolio.json"
    freeze_path = run / "scientific_atomic_verifier_prospective_cohort.json"
    comparison_path = run / "scientific_atomic_verifier_prospective_comparison.json"

    if disposition != "VERIFIER_REACHED_COMPLETE":
        if freeze_path.exists() or comparison_path.exists():
            raise ValueError(f"abstention case unexpectedly has verifier artifacts: {run.name}")
        return ScientificVerifierProspectiveCampaignCase(
            case_id=run.name,
            run_dir=str(run),
            wrapper_status=status,
            disposition=disposition,
            upstream_pre_n10_status=wrapper.get("upstream_pre_n10_status"),
            candidate_contract_reached=candidate_path.is_file(),
            atomic_cohort_frozen=False,
            verifier_reached=False,
            verifier_result_consumed_by_production=bool(wrapper.get("verifier_result_consumed_by_production", False)),
            production_selection_changed=bool(wrapper.get("production_selection_changed", False)),
            canonical_graph_mutated=bool(wrapper.get("canonical_graph_mutated", False)),
        )

    for path, label in ((candidate_path, "candidate contract"), (freeze_path, "cohort freeze"), (comparison_path, "comparison")):
        if not path.is_file():
            raise ValueError(f"completed verifier case missing {label}: {run.name}")
    freeze = _load_object(freeze_path)
    comparison = _load_object(comparison_path)
    required_freeze_flags = {
        "frozen_before_old_n10": True,
        "frozen_before_new_verifier": True,
        "verifier_results_observed_before_freeze": False,
        "old_n10_results_observed_before_freeze": False,
        "production_selection_authority": False,
        "canonical_graph_mutated": False,
    }
    for key, expected in required_freeze_flags.items():
        if freeze.get(key) is not expected:
            raise ValueError(f"{run.name} freeze invariant failed: {key}")
    if comparison.get("cohort_changed_after_freeze") is not False:
        raise ValueError(f"{run.name} comparison reports post-freeze cohort mutation")
    if comparison.get("old_n10_mutated_by_verifier") is not False:
        raise ValueError(f"{run.name} comparison reports old-N10 mutation")
    if comparison.get("production_selection_changed") is not False:
        raise ValueError(f"{run.name} comparison reports production selection mutation")
    if comparison.get("comparison_is_diagnostic_only") is not True:
        raise ValueError(f"{run.name} comparison is not diagnostic-only")
    hypothesis_count = int(freeze.get("hypothesis_count", -1))
    claim_count = int(freeze.get("claim_count", -1))
    agreement_count = int(comparison.get("agreement_count", -1))
    disagreement_count = int(comparison.get("disagreement_count", -1))
    if int(comparison.get("hypothesis_count", -1)) != hypothesis_count:
        raise ValueError(f"{run.name} comparison/freeze hypothesis count mismatch")
    return ScientificVerifierProspectiveCampaignCase(
        case_id=run.name,
        run_dir=str(run),
        wrapper_status=status,
        disposition=disposition,
        upstream_pre_n10_status=wrapper.get("upstream_pre_n10_status"),
        candidate_contract_reached=True,
        atomic_cohort_frozen=True,
        verifier_reached=True,
        frozen_hypothesis_count=hypothesis_count,
        frozen_claim_count=claim_count,
        agreement_count=agreement_count,
        disagreement_count=disagreement_count,
        comparison_cells={str(k): int(v) for k, v in dict(comparison.get("comparison_cells") or {}).items()},
        cohort_changed_after_freeze=bool(comparison.get("cohort_changed_after_freeze")),
        old_n10_mutated_by_verifier=bool(comparison.get("old_n10_mutated_by_verifier")),
        verifier_result_consumed_by_production=bool(wrapper.get("verifier_result_consumed_by_production", False)),
        production_selection_changed=bool(wrapper.get("production_selection_changed", False)),
        canonical_graph_mutated=bool(wrapper.get("canonical_graph_mutated", False)),
    )


def build_scientific_verifier_prospective_campaign_report(*, root: Path, case_ids: list[str]) -> ScientificVerifierProspectiveCampaignReport:
    resolved_root = root.expanduser().resolve()
    if not case_ids:
        raise ValueError("at least one prospective case is required")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("prospective case IDs must be unique")
    cases = [_classify_case(resolved_root / case_id) for case_id in case_ids]
    dispositions = Counter(row.disposition for row in cases)
    cells: Counter[str] = Counter()
    for row in cases:
        cells.update(row.comparison_cells)
    body = {
        "schema_version": "scientific-verifier-prospective-campaign-report-v1",
        "source_root": str(resolved_root),
        "case_ids": list(case_ids),
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_count": len(cases),
        "disposition_counts": dict(sorted(dispositions.items())),
        "verifier_reached_case_count": sum(row.verifier_reached for row in cases),
        "verifier_not_reached_case_count": sum(not row.verifier_reached for row in cases),
        "frozen_hypothesis_count": sum(row.frozen_hypothesis_count for row in cases),
        "frozen_claim_count": sum(row.frozen_claim_count for row in cases),
        "old_new_agreement_count": sum(row.agreement_count for row in cases),
        "old_new_disagreement_count": sum(row.disagreement_count for row in cases),
        "comparison_cells": dict(sorted(cells.items())),
        "all_cases_terminal": True,
        "no_post_freeze_cohort_mutation": True,
        "no_old_n10_mutation_by_verifier": True,
        "no_verifier_result_consumed_by_production": True,
        "no_production_selection_mutation": True,
        "no_canonical_graph_mutation": True,
    }
    digest = _sha256_json(body)
    return ScientificVerifierProspectiveCampaignReport(
        **body,
        report_id="scientific_verifier_prospective_campaign:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "ScientificVerifierProspectiveCampaignCase",
    "ScientificVerifierProspectiveCampaignReport",
    "build_scientific_verifier_prospective_campaign_report",
]
