from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.hypothesis_trend_compiler import (
    POSITIVE_USES,
    required_companion_uses,
)
from dac_her.hypothesis_trend_directional_compiler import (
    HYPOTHESIS_TREND_DIRECTIONAL_COMPILER_SEMANTICS_ID,
    DirectionAwareTrendHypothesisCompiler,
)
from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwareTrendHypothesisPortfolio,
    DirectionAwareTrendHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_directional_exposure import (
    build_directional_trend_maker_exposure,
)
from dac_her.hypothesis_trend_directional_prompt import (
    PROMPT_VERSION as DIRECTIONAL_PROMPT_VERSION,
    DirectionAwareTrendHypothesisPromptAssembler,
)
from dac_her.hypothesis_trend_directional_run_record import (
    HYPOTHESIS_TREND_DIRECTIONAL_RUNTIME_SEMANTICS_ID,
    DirectionAwareTrendHypothesisMakerRunRecord,
)
from dac_her.hypothesis_trend_directional_validator import (
    HYPOTHESIS_TREND_DIRECTIONAL_VALIDATOR_SEMANTICS_ID,
    DirectionAwareTrendHypothesisValidator,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)


HYPOTHESIS_TREND_EVALUATION_PROTOCOL_SEMANTICS_ID = (
    "trend_aware_hypothesis_evaluation_protocol_v1_alpha4c5e"
)
HYPOTHESIS_TREND_EVALUATOR_SEMANTICS_ID = (
    "trend_aware_hypothesis_evaluator_v1_alpha4c5e"
)
HYPOTHESIS_TREND_RESERVE_MANIFEST_SEMANTICS_ID = (
    "trend_aware_hypothesis_reserve_manifest_v1_alpha4c5e"
)

FATAL_RULE_CODES = (
    "FABRICATED_TREND_VIEW",
    "TREND_SIGN_INVERSION",
    "MISSING_DIRECTION_BINDING",
    "TREND_DIRECTION_MISMATCH",
    "MISSING_REPLICATION_GAP_COMPANION",
    "MISSING_CONTEXT_QUALIFICATION_COMPANION",
    "MISSING_REVERSAL_BOUNDARY",
    "CROSS_PAPER_OVERCLAIM",
    "TREND_CAUSAL_ESCALATION",
    "TREND_UNIVERSAL_ESCALATION",
    "UNSUPPORTED_NUMERIC_PREDICTION",
    "EXTERNAL_NOVELTY_CLAIM",
    "EXPERIMENT_PROTOCOL_LEAKAGE",
    "EVIDENCE_NAMESPACE_COLLISION",
    "PROVENANCE_OR_CORPUS_BINDING_FAILURE",
    "FROZEN_IMPLEMENTATION_DRIFT",
    "RUN_NOT_ACCEPTED",
    "MAKER_SETTING_DRIFT",
    "RECOMPILE_MISMATCH",
    "REVALIDATION_FAILED",
    "RESERVE_BINDING_FAILURE",
)

NONFATAL_OBSERVATION_CODES = (
    "ABSTENTION_ZERO_HYPOTHESES",
    "REPAIR_USED_WITHIN_BOUND",
    "VERIFICATION_REQUIRED",
    "INSUFFICIENT_TREND",
    "CONTEXT_SPECIFIC_TREND",
    "REVERSED_TREND_WITH_BOUNDARIES",
    "REPLICATION_GAP_PRESENT",
    "NO_CROSS_PAPER_SYNTHESIS",
    "NON_MONOTONIC_OR_UNSPECIFIED_DIRECTION",
    "VALIDATION_WARNINGS_PRESENT",
    "ZERO_TREND_YIELD_WITH_ABSTENTION",
)

# The evaluation protocol hash-locks the exact implementation that is allowed
# to consume the reserve. A semantic/code change requires a new protocol epoch.
FROZEN_COMPONENT_PATHS = (
    "dac_her/hypothesis_trend_input.py",
    "dac_her/hypothesis_trend_contracts.py",
    "dac_her/hypothesis_trend_compiler.py",
    "dac_her/hypothesis_trend_validator.py",
    "dac_her/hypothesis_trend_maker_exposure.py",
    "dac_her/hypothesis_trend_prompt.py",
    "dac_her/hypothesis_trend_runtime.py",
    "dac_her/hypothesis_trend_directional_contracts.py",
    "dac_her/hypothesis_trend_directional_compiler.py",
    "dac_her/hypothesis_trend_directional_validator.py",
    "dac_her/hypothesis_trend_directional_exposure.py",
    "dac_her/hypothesis_trend_directional_prompt.py",
    "dac_her/hypothesis_trend_directional_runtime.py",
    "dac_her/hypothesis_trend_evaluation.py",
    "scripts/run_direction_aware_trend_hypothesis_maker.py",
    "scripts/evaluate_direction_aware_trend_hypothesis_run.py",
    "scripts/register_sers_alpha4c5e_reserve.py",
)

_UNIVERSAL_OVERCLAIM_PATTERNS = (
    re.compile(r"\balways\b", re.I),
    re.compile(r"\buniversally\b", re.I),
    re.compile(r"\bin\s+all\s+(?:contexts|conditions|systems)\b", re.I),
    re.compile(r"\bregardless\s+of\s+(?:context|conditions|system)\b", re.I),
    re.compile(r"\bholds?\s+across\s+all\b", re.I),
)
_CROSS_PAPER_OVERCLAIM_PATTERNS = (
    re.compile(r"\breplicat(?:ed|es|ion)\s+across\s+(?:papers|studies)\b", re.I),
    re.compile(r"\bacross\s+independent\s+studies\b", re.I),
    re.compile(r"\bmultiple\s+(?:papers|studies)\s+(?:show|shows|demonstrate|demonstrates|support|supports)\b", re.I),
    re.compile(r"\bcross[- ]paper\s+(?:replication|support|evidence)\s+(?:is|was)\s+(?:established|confirmed)\b", re.I),
)
_CAUSAL_EVIDENCE_ESCALATION_PATTERNS = (
    re.compile(r"\b(?:trend|association|evidence)\b.{0,45}\b(?:proves?|establishes?|demonstrates?|confirms?)\b.{0,80}\bcaus", re.I | re.S),
    re.compile(r"\b(?:proves?|establishes?|demonstrates?|confirms?)\b.{0,80}\bcausal\s+(?:relation|relationship|effect|mechanism)\b", re.I | re.S),
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class TrendHypothesisMakerSettings(StrictModel):
    max_hypotheses: Literal[1] = 1
    max_repairs: Literal[1] = 1
    temperature: Literal[0.0] = 0.0
    prompt_version: str
    backend: str
    model: str
    parse_retries: int | None = None
    backend_mode: str | None = None
    base_url: str | None = None


class SeenSmokeAnchor(StrictModel):
    source_input_id: str
    source_input_sha256: str
    directional_exposure_id: str
    directional_exposure_sha256: str
    prompt_sha256: str
    run_id: str
    portfolio_id: str
    portfolio_sha256: str
    paper_ids: list[str] = Field(default_factory=list)
    generation_attempts: int
    repair_attempts: int
    validation_errors: int
    validation_warnings: int


class TrendHypothesisEvaluationPolicy(StrictModel):
    acceptance_requires_zero_fatal_issues: Literal[True] = True
    count_thresholds_used_for_acceptance: Literal[False] = False
    minimum_hypothesis_count: None = None
    abstention_is_failure: Literal[False] = False
    zero_hypothesis_portfolio_is_failure: Literal[False] = False
    insufficient_trend_is_failure: Literal[False] = False
    context_specific_trend_is_failure: Literal[False] = False
    reversed_trend_is_failure_when_boundaries_preserved: Literal[False] = False
    verification_required_is_failure: Literal[False] = False
    no_cross_paper_synthesis_is_failure: Literal[False] = False
    non_monotonic_or_unspecified_direction_is_failure: Literal[False] = False
    one_repair_then_valid_is_failure: Literal[False] = False

    canonical_independent_change: Literal["increase"] = "increase"
    positive_direction_dependent_change: Literal["increase"] = "increase"
    negative_direction_dependent_change: Literal["decrease"] = "decrease"
    sign_transformation_by_llm_allowed: Literal[False] = False

    reserve_registration_required_before_evaluation: Literal[True] = True
    reserve_rules_may_change_after_registration: Literal[False] = False
    reserve_results_may_change_acceptance_rules: Literal[False] = False
    failed_reserve_campaign_resumable_after_semantic_patch: Literal[False] = False
    new_semantic_patch_requires_new_protocol_epoch: Literal[True] = True
    llm_calls_during_protocol_freeze: Literal[False] = False


class TrendHypothesisEvaluationProtocol(StrictModel):
    schema_version: Literal[
        "trend-hypothesis-evaluation-protocol-v1"
    ] = "trend-hypothesis-evaluation-protocol-v1"

    protocol_id: str
    protocol_sha256: str
    semantics_id: str
    evaluator_semantics_id: str
    domain_profile_id: str

    maker_settings: TrendHypothesisMakerSettings
    seen_smoke_anchor: SeenSmokeAnchor

    frozen_component_sha256: dict[str, str]
    fatal_rule_codes: list[str]
    nonfatal_observation_codes: list[str]
    policy: TrendHypothesisEvaluationPolicy

    reserve_campaign_prefix: str = "sers_alpha4c5e_reserve_"
    reserve_not_yet_registered: Literal[True] = True
    reserve_consumed_by_protocol_freeze: Literal[False] = False

    @model_validator(mode="after")
    def _consistency(self) -> "TrendHypothesisEvaluationProtocol":
        if self.semantics_id != (
            HYPOTHESIS_TREND_EVALUATION_PROTOCOL_SEMANTICS_ID
        ):
            raise ValueError("5e protocol semantics mismatch.")
        if self.evaluator_semantics_id != (
            HYPOTHESIS_TREND_EVALUATOR_SEMANTICS_ID
        ):
            raise ValueError("5e evaluator semantics mismatch.")
        if self.fatal_rule_codes != list(FATAL_RULE_CODES):
            raise ValueError("5e fatal rules changed.")
        if self.nonfatal_observation_codes != list(
            NONFATAL_OBSERVATION_CODES
        ):
            raise ValueError("5e nonfatal observations changed.")
        if list(self.frozen_component_sha256) != sorted(
            self.frozen_component_sha256
        ):
            raise ValueError(
                "frozen_component_sha256 keys must be sorted."
            )

        expected_id = stable_id(
            "trend_hypothesis_evaluation_protocol",
            self.semantics_id,
            self.domain_profile_id,
            self.seen_smoke_anchor.source_input_sha256,
            self.seen_smoke_anchor.portfolio_sha256,
            sha256_json(self.frozen_component_sha256),
            ",".join(self.fatal_rule_codes),
            ",".join(self.nonfatal_observation_codes),
        )
        if self.protocol_id != expected_id:
            raise ValueError("5e protocol_id is not stable.")

        payload = self.model_dump(mode="json")
        observed = str(payload.pop("protocol_sha256", ""))
        expected = sha256_json(payload)
        if observed != expected:
            raise ValueError("5e protocol SHA mismatch.")
        return self


class TrendHypothesisReserveManifest(StrictModel):
    schema_version: Literal[
        "trend-hypothesis-reserve-manifest-v1"
    ] = "trend-hypothesis-reserve-manifest-v1"

    manifest_id: str
    manifest_sha256: str
    semantics_id: str

    reserve_id: str
    protocol_id: str
    protocol_sha256: str
    domain_profile_id: str
    paper_ids: list[str]

    acceptance_rules_frozen_before_registration: Literal[True] = True
    declared_unseen_for_alpha4c5e: Literal[True] = True
    declared_not_inspected_for_trend_hypothesis_semantics_before_registration: Literal[
        True
    ] = True
    reserve_consumed_at_registration: Literal[False] = False
    reserve_results_may_modify_protocol: Literal[False] = False

    @model_validator(mode="after")
    def _consistency(self) -> "TrendHypothesisReserveManifest":
        if self.semantics_id != (
            HYPOTHESIS_TREND_RESERVE_MANIFEST_SEMANTICS_ID
        ):
            raise ValueError("reserve manifest semantics mismatch.")
        if not self.paper_ids:
            raise ValueError("reserve requires at least one paper.")
        if self.paper_ids != sorted(set(self.paper_ids)):
            raise ValueError(
                "reserve paper_ids must be sorted and unique."
            )
        expected_id = stable_id(
            "trend_hypothesis_reserve_manifest",
            self.semantics_id,
            self.protocol_sha256,
            self.reserve_id,
            ",".join(self.paper_ids),
        )
        if self.manifest_id != expected_id:
            raise ValueError("reserve manifest_id is not stable.")
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("manifest_sha256", ""))
        expected = sha256_json(payload)
        if observed != expected:
            raise ValueError("reserve manifest SHA mismatch.")
        return self


class TrendHypothesisEvaluationIssue(StrictModel):
    severity: Literal["fatal", "observation"]
    code: str
    location: str
    message: str
    source_issue_code: str | None = None


class TrendHypothesisEvaluationReport(StrictModel):
    schema_version: Literal[
        "trend-hypothesis-evaluation-report-v1"
    ] = "trend-hypothesis-evaluation-report-v1"

    evaluation_id: str
    evaluator_semantics_id: str
    evaluation_mode: Literal["seen_regression", "reserve"]

    protocol_id: str
    protocol_sha256: str
    reserve_manifest_id: str | None = None
    reserve_manifest_sha256: str | None = None
    reserve_consumed: bool = False

    source_input_id: str
    source_input_sha256: str
    run_id: str
    portfolio_id: str
    portfolio_sha256: str

    fatal_issue_count: int
    observation_count: int
    accepted: bool
    count_thresholds_used_for_acceptance: Literal[False] = False

    hypothesis_count: int
    abstained: bool
    generation_attempts: int
    repair_attempts: int
    revalidation_errors: int
    revalidation_warnings: int

    issues: list[TrendHypothesisEvaluationIssue] = Field(
        default_factory=list
    )


def current_component_hashes(
    root: Path,
) -> dict[str, str]:
    values = {}
    for rel in FROZEN_COMPONENT_PATHS:
        path = root / rel
        if not path.exists():
            raise ValueError(
                f"Frozen 5e component missing: {rel}"
            )
        values[rel] = sha256_file(path)
    return dict(sorted(values.items()))


def verify_protocol_integrity(
    protocol: TrendHypothesisEvaluationProtocol,
    *,
    root: Path,
) -> list[str]:
    issues: list[str] = []
    try:
        observed = current_component_hashes(root)
    except Exception as exc:
        return [str(exc)]
    if observed != protocol.frozen_component_sha256:
        all_paths = sorted(
            set(observed)
            | set(protocol.frozen_component_sha256)
        )
        for rel in all_paths:
            expected = protocol.frozen_component_sha256.get(rel)
            actual = observed.get(rel)
            if expected != actual:
                issues.append(
                    f"{rel}: expected={expected}, observed={actual}"
                )
    return issues


def make_reserve_manifest(
    protocol: TrendHypothesisEvaluationProtocol,
    *,
    reserve_id: str,
    paper_ids: list[str],
) -> TrendHypothesisReserveManifest:
    paper_ids = sorted(set(str(value) for value in paper_ids))
    overlap = sorted(
        set(paper_ids)
        & set(protocol.seen_smoke_anchor.paper_ids)
    )
    if overlap:
        raise ValueError(
            "Reserve overlaps the frozen 5d.1 seen-smoke corpus: "
            f"{overlap}"
        )
    payload = {
        "schema_version":
            "trend-hypothesis-reserve-manifest-v1",
        "manifest_id": stable_id(
            "trend_hypothesis_reserve_manifest",
            HYPOTHESIS_TREND_RESERVE_MANIFEST_SEMANTICS_ID,
            protocol.protocol_sha256,
            reserve_id,
            ",".join(paper_ids),
        ),
        "semantics_id":
            HYPOTHESIS_TREND_RESERVE_MANIFEST_SEMANTICS_ID,
        "reserve_id": reserve_id,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "domain_profile_id": protocol.domain_profile_id,
        "paper_ids": paper_ids,
        "acceptance_rules_frozen_before_registration": True,
        "declared_unseen_for_alpha4c5e": True,
        "declared_not_inspected_for_trend_hypothesis_semantics_before_registration":
            True,
        "reserve_consumed_at_registration": False,
        "reserve_results_may_modify_protocol": False,
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return TrendHypothesisReserveManifest.model_validate(payload)


def _card_text(card: object) -> str:
    parts = [
        str(getattr(card, "title", "")),
        str(getattr(card, "hypothesis_statement", "")),
        str(getattr(card, "inferential_bridge", "")),
    ]
    for row in getattr(card, "predicted_observations", []):
        parts.extend([str(row.observable), str(row.rationale)])
    for row in getattr(card, "falsification_criteria", []):
        parts.extend(
            [str(row.observable), str(row.falsifying_outcome)]
        )
    parts.extend(map(str, getattr(card, "assumptions", [])))
    return "\n".join(parts)


def detect_claim_scope_issues(
    text: str,
    *,
    cross_paper_synthesis: bool,
) -> set[str]:
    issues: set[str] = set()
    if not cross_paper_synthesis and any(
        pattern.search(text)
        for pattern in _CROSS_PAPER_OVERCLAIM_PATTERNS
    ):
        issues.add("CROSS_PAPER_OVERCLAIM")
    if any(
        pattern.search(text)
        for pattern in _UNIVERSAL_OVERCLAIM_PATTERNS
    ):
        issues.add("TREND_UNIVERSAL_ESCALATION")
    if any(
        pattern.search(text)
        for pattern in _CAUSAL_EVIDENCE_ESCALATION_PATTERNS
    ):
        issues.add("TREND_CAUSAL_ESCALATION")
    return issues


def _fatal_from_validation_code(code: str) -> str:
    mapping = {
        "UNKNOWN_TREND_VIEW": "FABRICATED_TREND_VIEW",
        "UNKNOWN_COMPILED_TREND_DIRECTION_VIEW":
            "FABRICATED_TREND_VIEW",
        "NONCANONICAL_TREND_DIRECTION_FRAME":
            "TREND_SIGN_INVERSION",
        "MISSING_COMPILED_TREND_DIRECTION_BINDING":
            "MISSING_DIRECTION_BINDING",
        "COMPILED_DIRECTION_BINDING_MISMATCH":
            "TREND_DIRECTION_MISMATCH",
        "COMPILED_TREND_PREDICTION_DIRECTION_MISMATCH":
            "TREND_DIRECTION_MISMATCH",
        "UNSUPPORTED_NUMERIC_PREDICTION":
            "UNSUPPORTED_NUMERIC_PREDICTION",
        "EXTERNAL_NOVELTY_CLAIM":
            "EXTERNAL_NOVELTY_CLAIM",
        "EXPERIMENT_PROTOCOL_LEAKAGE":
            "EXPERIMENT_PROTOCOL_LEAKAGE",
        "TREND_AUTHORIZATION_ESCALATION":
            "TREND_CAUSAL_ESCALATION",
        "CARD_TREND_AUTHORIZATION_ESCALATION":
            "TREND_CAUSAL_ESCALATION",
        "PORTFOLIO_LINEAGE_MISMATCH":
            "PROVENANCE_OR_CORPUS_BINDING_FAILURE",
        "CARD_LINEAGE_MISMATCH":
            "PROVENANCE_OR_CORPUS_BINDING_FAILURE",
        "TREND_REFERENCE_PROVENANCE_MISMATCH":
            "PROVENANCE_OR_CORPUS_BINDING_FAILURE",
        "PAPER_SCOPE_MISMATCH":
            "PROVENANCE_OR_CORPUS_BINDING_FAILURE",
        "UNKNOWN_PREMISE_STATEMENT":
            "EVIDENCE_NAMESPACE_COLLISION",
        "UNKNOWN_GAP_STATEMENT":
            "EVIDENCE_NAMESPACE_COLLISION",
    }
    return mapping.get(code, "REVALIDATION_FAILED")


def _append_unique(
    issues: list[TrendHypothesisEvaluationIssue],
    issue: TrendHypothesisEvaluationIssue,
) -> None:
    key = (
        issue.severity,
        issue.code,
        issue.location,
        issue.message,
        issue.source_issue_code,
    )
    existing = {
        (
            row.severity,
            row.code,
            row.location,
            row.message,
            row.source_issue_code,
        )
        for row in issues
    }
    if key not in existing:
        issues.append(issue)


def _fatal(
    issues: list[TrendHypothesisEvaluationIssue],
    code: str,
    location: str,
    message: str,
    *,
    source_issue_code: str | None = None,
) -> None:
    _append_unique(
        issues,
        TrendHypothesisEvaluationIssue(
            severity="fatal",
            code=code,
            location=location,
            message=message,
            source_issue_code=source_issue_code,
        ),
    )


def _observe(
    issues: list[TrendHypothesisEvaluationIssue],
    code: str,
    location: str,
    message: str,
) -> None:
    _append_unique(
        issues,
        TrendHypothesisEvaluationIssue(
            severity="observation",
            code=code,
            location=location,
            message=message,
        ),
    )


def evaluate_run(
    *,
    root: Path,
    protocol: TrendHypothesisEvaluationProtocol,
    source: TrendAwareHypothesisInput,
    final_draft: DirectionAwareTrendHypothesisPortfolioDraft,
    run_record: DirectionAwareTrendHypothesisMakerRunRecord,
    portfolio: DirectionAwareTrendHypothesisPortfolio,
    evaluation_mode: Literal["seen_regression", "reserve"],
    reserve_manifest: TrendHypothesisReserveManifest | None = None,
) -> TrendHypothesisEvaluationReport:
    issues: list[TrendHypothesisEvaluationIssue] = []

    protocol_drift = verify_protocol_integrity(
        protocol,
        root=root,
    )
    for row in protocol_drift:
        _fatal(
            issues,
            "FROZEN_IMPLEMENTATION_DRIFT",
            "protocol.frozen_component_sha256",
            row,
        )

    try:
        verify_trend_aware_input_sources(source)
    except Exception as exc:
        _fatal(
            issues,
            "PROVENANCE_OR_CORPUS_BINDING_FAILURE",
            "source",
            str(exc),
        )

    if evaluation_mode == "reserve":
        if reserve_manifest is None:
            _fatal(
                issues,
                "RESERVE_BINDING_FAILURE",
                "reserve_manifest",
                "Reserve evaluation requires a frozen reserve manifest.",
            )
        else:
            if (
                reserve_manifest.protocol_id != protocol.protocol_id
                or reserve_manifest.protocol_sha256
                != protocol.protocol_sha256
            ):
                _fatal(
                    issues,
                    "RESERVE_BINDING_FAILURE",
                    "reserve_manifest.protocol",
                    "Reserve manifest is bound to a different 5e protocol.",
                )
            if (
                reserve_manifest.domain_profile_id
                != source.domain_profile_id
            ):
                _fatal(
                    issues,
                    "RESERVE_BINDING_FAILURE",
                    "reserve_manifest.domain_profile_id",
                    "Reserve/domain profile mismatch.",
                )
            if (
                reserve_manifest.paper_ids
                != source.trend_corpus_binding.paper_ids
            ):
                _fatal(
                    issues,
                    "RESERVE_BINDING_FAILURE",
                    "reserve_manifest.paper_ids",
                    (
                        "Reserve paper set must exactly equal the locked "
                        "Trend corpus paper set."
                    ),
                )
    elif reserve_manifest is not None:
        _fatal(
            issues,
            "RESERVE_BINDING_FAILURE",
            "reserve_manifest",
            "Seen regression must not consume a reserve manifest.",
        )

    settings = protocol.maker_settings
    setting_checks = {
        "runtime_semantics_id":
            HYPOTHESIS_TREND_DIRECTIONAL_RUNTIME_SEMANTICS_ID,
        "directional_compiler_semantics_id":
            HYPOTHESIS_TREND_DIRECTIONAL_COMPILER_SEMANTICS_ID,
        "directional_validator_semantics_id":
            HYPOTHESIS_TREND_DIRECTIONAL_VALIDATOR_SEMANTICS_ID,
        "prompt_version": settings.prompt_version,
        "backend": settings.backend,
        "model": settings.model,
        "max_repairs": settings.max_repairs,
    }
    for field, expected in setting_checks.items():
        actual = getattr(run_record, field)
        if actual != expected:
            _fatal(
                issues,
                "MAKER_SETTING_DRIFT",
                f"run_record.{field}",
                f"expected={expected!r}, actual={actual!r}",
            )
    if run_record.temperature != settings.temperature:
        _fatal(
            issues,
            "MAKER_SETTING_DRIFT",
            "run_record.temperature",
            (
                f"expected={settings.temperature!r}, "
                f"actual={run_record.temperature!r}"
            ),
        )
    if run_record.parse_retries != settings.parse_retries:
        _fatal(
            issues,
            "MAKER_SETTING_DRIFT",
            "run_record.parse_retries",
            (
                f"expected={settings.parse_retries!r}, "
                f"actual={run_record.parse_retries!r}"
            ),
        )
    if run_record.backend_mode != settings.backend_mode:
        _fatal(
            issues,
            "MAKER_SETTING_DRIFT",
            "run_record.backend_mode",
            (
                f"expected={settings.backend_mode!r}, "
                f"actual={run_record.backend_mode!r}"
            ),
        )
    if run_record.base_url != settings.base_url:
        _fatal(
            issues,
            "MAKER_SETTING_DRIFT",
            "run_record.base_url",
            (
                f"expected={settings.base_url!r}, "
                f"actual={run_record.base_url!r}"
            ),
        )

    try:
        exposure = build_directional_trend_maker_exposure(source)
        prompt = DirectionAwareTrendHypothesisPromptAssembler(
            max_hypotheses=settings.max_hypotheses
        ).build(source, exposure=exposure)
        prompt_checks = {
            "source_trend_input_id": source.input_id,
            "source_trend_input_sha256": source.input_sha256,
            "directional_exposure_id": exposure.exposure_id,
            "directional_exposure_sha256":
                exposure.exposure_sha256,
            "source_5d_exposure_id":
                exposure.source_5d_exposure_id,
            "source_5d_exposure_sha256":
                exposure.source_5d_exposure_sha256,
            "prompt_sha256": prompt.prompt_sha256,
        }
        for field, expected in prompt_checks.items():
            actual = getattr(run_record, field)
            if actual != expected:
                _fatal(
                    issues,
                    "PROVENANCE_OR_CORPUS_BINDING_FAILURE",
                    f"run_record.{field}",
                    f"expected={expected!r}, actual={actual!r}",
                )
    except Exception as exc:
        _fatal(
            issues,
            "PROVENANCE_OR_CORPUS_BINDING_FAILURE",
            "directional_exposure_or_prompt",
            str(exc),
        )

    if (
        not run_record.final_validation_passed
        or run_record.failure_stage != "none"
        or run_record.validation_errors != 0
    ):
        _fatal(
            issues,
            "RUN_NOT_ACCEPTED",
            "run_record",
            (
                "Reserve run must finish with final_validation_passed=true, "
                "failure_stage='none', and zero validation errors."
            ),
        )
    if run_record.repair_attempts > settings.max_repairs:
        _fatal(
            issues,
            "MAKER_SETTING_DRIFT",
            "run_record.repair_attempts",
            "Repair count exceeded the frozen 5e bound.",
        )
    if (
        run_record.generation_attempts
        != run_record.repair_attempts + 1
    ):
        _fatal(
            issues,
            "MAKER_SETTING_DRIFT",
            "run_record.generation_attempts",
            (
                "Generation attempts must equal initial generation plus "
                "the bounded repair count."
            ),
        )

    lineage_checks = {
        "source_trend_input_id": source.input_id,
        "source_trend_input_sha256": source.input_sha256,
        "portfolio_id": portfolio.portfolio_id,
        "portfolio_sha256": sha256_json(portfolio),
    }
    for field, expected in lineage_checks.items():
        actual = getattr(run_record, field)
        if actual != expected:
            _fatal(
                issues,
                "PROVENANCE_OR_CORPUS_BINDING_FAILURE",
                f"run_record.{field}",
                f"expected={expected!r}, actual={actual!r}",
            )

    try:
        recompiled = DirectionAwareTrendHypothesisCompiler().compile(
            source,
            final_draft,
        )
        if (
            recompiled.model_dump(mode="json")
            != portfolio.model_dump(mode="json")
        ):
            _fatal(
                issues,
                "RECOMPILE_MISMATCH",
                "portfolio",
                (
                    "Saved portfolio is not the exact deterministic "
                    "compilation of the saved final draft."
                ),
            )
    except Exception as exc:
        _fatal(
            issues,
            "RECOMPILE_MISMATCH",
            "final_draft",
            str(exc),
        )

    revalidation = (
        DirectionAwareTrendHypothesisValidator().validate(
            source,
            portfolio,
        )
    )
    for row in revalidation.issues:
        if row.severity != "error":
            continue
        _fatal(
            issues,
            _fatal_from_validation_code(row.code),
            row.location,
            row.message,
            source_issue_code=row.code,
        )

    view_index = {
        row.view_id: row for row in source.trend_views
    }
    trend_view_ids = set(view_index)

    for h_index, card in enumerate(portfolio.hypotheses):
        hloc = f"portfolio.hypotheses[{h_index}]"

        if (
            trend_view_ids
            & set(card.premise_statement_ids)
        ) or (
            trend_view_ids
            & set(card.gap_statement_ids)
        ):
            _fatal(
                issues,
                "EVIDENCE_NAMESPACE_COLLISION",
                hloc,
                (
                    "Trend view IDs may not appear in Explorer premise/gap "
                    "statement namespaces."
                ),
            )

        refs_by_grounding: dict[str, set[str]] = {}
        for ref in card.trend_references:
            refs_by_grounding.setdefault(
                ref.grounding_id, set()
            ).add(ref.use_role)

        for ref in card.trend_references:
            if ref.use_role not in POSITIVE_USES:
                continue
            view = view_index.get(ref.view_id)
            if view is None:
                _fatal(
                    issues,
                    "FABRICATED_TREND_VIEW",
                    hloc + ".trend_references",
                    f"Unknown Trend view: {ref.view_id}",
                )
                continue
            missing = required_companion_uses(view) - (
                refs_by_grounding.get(view.grounding_id, set())
            )
            for use in sorted(missing):
                code = {
                    "replication_gap":
                        "MISSING_REPLICATION_GAP_COMPANION",
                    "context_qualification":
                        "MISSING_CONTEXT_QUALIFICATION_COMPANION",
                    "counterevidence_boundary":
                        "MISSING_REVERSAL_BOUNDARY",
                }[use]
                _fatal(
                    issues,
                    code,
                    hloc + ".trend_references",
                    (
                        f"{view.grounding_id} status="
                        f"{view.cross_context_status} requires "
                        f"companion {use}."
                    ),
                )

        text = _card_text(card)
        for code in sorted(
            detect_claim_scope_issues(
                text,
                cross_paper_synthesis=
                    card.cross_paper_synthesis,
            )
        ):
            _fatal(
                issues,
                code,
                hloc,
                (
                    "Generated prose exceeds the frozen Trend evidence "
                    "authorization scope."
                ),
            )

        if card.trend_causal_authorization is not False:
            _fatal(
                issues,
                "TREND_CAUSAL_ESCALATION",
                hloc + ".trend_causal_authorization",
                "Trend evidence cannot authorize causality.",
            )
        if card.trend_universal_authorization is not False:
            _fatal(
                issues,
                "TREND_UNIVERSAL_ESCALATION",
                hloc + ".trend_universal_authorization",
                "Trend evidence cannot authorize universality.",
            )

        if card.verification_dependency != "none":
            _observe(
                issues,
                "VERIFICATION_REQUIRED",
                hloc + ".verification_dependency",
                card.verification_dependency,
            )
        if not card.cross_paper_synthesis:
            _observe(
                issues,
                "NO_CROSS_PAPER_SYNTHESIS",
                hloc + ".cross_paper_synthesis",
                "No cross-paper positive-support synthesis was asserted.",
            )

        statuses = {
            ref.cross_context_status
            for ref in card.trend_references
        }
        if "insufficient" in statuses:
            _observe(
                issues,
                "INSUFFICIENT_TREND",
                hloc + ".trend_references",
                "Insufficient Trend evidence is allowed when limitations are preserved.",
            )
        if "context_specific" in statuses:
            _observe(
                issues,
                "CONTEXT_SPECIFIC_TREND",
                hloc + ".trend_references",
                "Context-specific Trend evidence is not an acceptance failure.",
            )
        if "reversed" in statuses:
            _observe(
                issues,
                "REVERSED_TREND_WITH_BOUNDARIES",
                hloc + ".trend_references",
                "Reversed Trend evidence is allowed when required boundaries are preserved.",
            )
        if any(
            ref.use_role == "replication_gap"
            for ref in card.trend_references
        ):
            _observe(
                issues,
                "REPLICATION_GAP_PRESENT",
                hloc + ".trend_references",
                "Replication-gap evidence is a limitation, not a failure.",
            )
        directions = {
            direction
            for ref in card.trend_references
            for direction in ref.directions
        }
        if directions & {"non_monotonic", "unspecified"}:
            _observe(
                issues,
                "NON_MONOTONIC_OR_UNSPECIFIED_DIRECTION",
                hloc + ".trend_references",
                "Non-monotonic/unspecified direction is not a count-based failure.",
            )

    if not portfolio.hypotheses:
        _observe(
            issues,
            "ABSTENTION_ZERO_HYPOTHESES",
            "portfolio",
            (
                "A structurally valid abstention/zero-hypothesis portfolio "
                "is explicitly nonfatal."
            ),
        )
        if source.trend_grounding.zero_yield:
            _observe(
                issues,
                "ZERO_TREND_YIELD_WITH_ABSTENTION",
                "portfolio",
                "Zero Trend yield with valid abstention is nonfatal.",
            )

    if run_record.repair_attempts == 1:
        _observe(
            issues,
            "REPAIR_USED_WITHIN_BOUND",
            "run_record.repair_attempts",
            "One bounded repair followed by a valid portfolio is allowed.",
        )
    if revalidation.warnings:
        _observe(
            issues,
            "VALIDATION_WARNINGS_PRESENT",
            "revalidation",
            f"warnings={revalidation.warnings}",
        )

    fatal_count = sum(
        row.severity == "fatal" for row in issues
    )
    observation_count = sum(
        row.severity == "observation" for row in issues
    )
    portfolio_sha = sha256_json(portfolio)
    reserve_manifest_id = (
        reserve_manifest.manifest_id
        if reserve_manifest is not None
        else None
    )
    reserve_manifest_sha = (
        reserve_manifest.manifest_sha256
        if reserve_manifest is not None
        else None
    )
    evaluation_id = stable_id(
        "trend_hypothesis_evaluation",
        HYPOTHESIS_TREND_EVALUATOR_SEMANTICS_ID,
        protocol.protocol_sha256,
        evaluation_mode,
        source.input_sha256,
        run_record.run_id,
        portfolio_sha,
        reserve_manifest_sha or "seen",
        ",".join(
            sorted(
                row.code
                for row in issues
                if row.severity == "fatal"
            )
        ),
    )
    return TrendHypothesisEvaluationReport(
        evaluation_id=evaluation_id,
        evaluator_semantics_id=
            HYPOTHESIS_TREND_EVALUATOR_SEMANTICS_ID,
        evaluation_mode=evaluation_mode,
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol.protocol_sha256,
        reserve_manifest_id=reserve_manifest_id,
        reserve_manifest_sha256=reserve_manifest_sha,
        reserve_consumed=(
            evaluation_mode == "reserve"
        ),
        source_input_id=source.input_id,
        source_input_sha256=source.input_sha256,
        run_id=run_record.run_id,
        portfolio_id=portfolio.portfolio_id,
        portfolio_sha256=portfolio_sha,
        fatal_issue_count=fatal_count,
        observation_count=observation_count,
        accepted=fatal_count == 0,
        count_thresholds_used_for_acceptance=False,
        hypothesis_count=len(portfolio.hypotheses),
        abstained=not bool(portfolio.hypotheses),
        generation_attempts=run_record.generation_attempts,
        repair_attempts=run_record.repair_attempts,
        revalidation_errors=revalidation.errors,
        revalidation_warnings=revalidation.warnings,
        issues=issues,
    )


def load_protocol(path: Path) -> TrendHypothesisEvaluationProtocol:
    return TrendHypothesisEvaluationProtocol.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def load_reserve_manifest(
    path: Path,
) -> TrendHypothesisReserveManifest:
    return TrendHypothesisReserveManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
