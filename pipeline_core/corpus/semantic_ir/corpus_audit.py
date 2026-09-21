from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityManifest,
    CapabilityRecord,
    CapabilityState,
)
from pipeline_core.corpus.semantic_ir.capability_audit import (
    CapabilityAssessmentMode,
    SemanticIRCapabilityAuditResult,
    audit_semantic_ir_capabilities,
)
from pipeline_core.corpus.semantic_ir.existing_extraction import (
    load_existing_extraction_semantic_ir,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


DuplicatePaperPolicy = Literal["error", "latest_attempt"]
CorpusAssessmentMode = Literal[
    "structural_support",
    "observed_coverage",
    "explicit_gap",
    "mixed",
]


class DiscoveredExtractionAttempt(StrictModel):
    paper_id: str = Field(min_length=1)
    active_chunks_path: str = Field(min_length=1)
    attempt_directory: str = Field(min_length=1)
    run_id: str | None = None
    attempt_id: str | None = None
    graph_materialization_status: str | None = None


class ExtractionAttemptDiscoveryResult(StrictModel):
    schema_version: Literal[
        "extraction-attempt-discovery-v1"
    ] = "extraction-attempt-discovery-v1"

    root: str
    duplicate_paper_policy: DuplicatePaperPolicy
    discovered_active_chunks_count: int = Field(ge=0)
    selected_attempt_count: int = Field(ge=0)
    paper_count: int = Field(ge=0)
    duplicate_paper_count: int = Field(ge=0)
    attempts: list[DiscoveredExtractionAttempt] = Field(default_factory=list)

    filesystem_discovery_only: Literal[True] = True
    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ExtractionAttemptDiscoveryResult":
        if self.selected_attempt_count != len(self.attempts):
            raise ValueError("selected_attempt_count must equal len(attempts)")
        if self.paper_count != len({row.paper_id for row in self.attempts}):
            raise ValueError("paper_count must equal unique selected paper IDs")
        return self


class CorpusCapabilityAggregateMetric(StrictModel):
    capability: str = Field(min_length=1)
    state: CapabilityState
    assessment_mode: CorpusAssessmentMode
    paper_assessed_count: int = Field(ge=0)
    paper_state_counts: dict[str, int] = Field(default_factory=dict)
    applicable_count: int = Field(ge=0)
    supported_count: int = Field(ge=0)
    coverage_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    detail: str = ""
    assessed_from: list[str] = Field(default_factory=list)

    negative_evidence_inferred: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "CorpusCapabilityAggregateMetric":
        if self.supported_count > self.applicable_count:
            raise ValueError("supported_count cannot exceed applicable_count")
        if sum(self.paper_state_counts.values()) != self.paper_assessed_count:
            raise ValueError(
                "paper_state_counts must sum to paper_assessed_count"
            )
        if self.applicable_count == 0:
            if self.coverage_fraction is not None:
                raise ValueError(
                    "coverage_fraction must be null when applicable_count is zero"
                )
        else:
            expected = self.supported_count / self.applicable_count
            if self.coverage_fraction is None:
                raise ValueError(
                    "coverage_fraction is required when applicable_count > 0"
                )
            if abs(self.coverage_fraction - expected) > 1e-12:
                raise ValueError(
                    "coverage_fraction must equal supported/applicable"
                )
        return self


class CorpusPaperCapabilitySummary(StrictModel):
    paper_id: str = Field(min_length=1)
    active_chunks_path: str = Field(min_length=1)
    bundle_id: str = Field(min_length=1)
    manifest_id: str = Field(min_length=1)
    object_counts: dict[str, int] = Field(default_factory=dict)
    capability_states: dict[str, CapabilityState] = Field(default_factory=dict)
    capability_coverage: dict[str, float | None] = Field(default_factory=dict)


class CorpusSemanticCapabilityAuditResult(StrictModel):
    schema_version: Literal[
        "corpus-semantic-capability-audit-v1"
    ] = "corpus-semantic-capability-audit-v1"

    corpus_id: str = Field(min_length=1)
    root: str
    discovery: ExtractionAttemptDiscoveryResult
    manifest: CapabilityManifest
    metrics: dict[str, CorpusCapabilityAggregateMetric]
    papers: list[CorpusPaperCapabilitySummary] = Field(default_factory=list)
    object_counts: dict[str, int] = Field(default_factory=dict)

    audit_only: Literal[True] = True
    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    negative_evidence_inferred: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    selection_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_alignment(self) -> "CorpusSemanticCapabilityAuditResult":
        if self.manifest.scope_kind != "corpus":
            raise ValueError("corpus audit must produce a corpus manifest")
        if self.manifest.scope_id != self.corpus_id:
            raise ValueError("manifest scope_id must equal corpus_id")
        if len(self.papers) != self.discovery.paper_count:
            raise ValueError("paper summaries must match discovery paper_count")
        if set(self.metrics) != set(self.manifest.capabilities):
            raise ValueError(
                "metric and capability mappings must contain identical keys"
            )
        for name, metric in self.metrics.items():
            if name != metric.capability:
                raise ValueError("metric key must equal metric.capability")
            if self.manifest.capabilities[name].state != metric.state:
                raise ValueError(
                    "manifest capability state must equal aggregate metric state"
                )
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _load_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _attempt_from_active_chunks(path: Path) -> DiscoveredExtractionAttempt:
    payload = _load_json_object(path)
    paper_id = str(payload.get("paper_id", "")).strip()
    if not paper_id:
        raise ValueError(f"active_chunks.json is missing paper_id: {path}")
    return DiscoveredExtractionAttempt(
        paper_id=paper_id,
        active_chunks_path=str(path.resolve()),
        attempt_directory=str(path.parent.resolve()),
        run_id=(
            str(payload.get("run_id")).strip()
            if payload.get("run_id") not in (None, "")
            else None
        ),
        attempt_id=(
            str(payload.get("attempt_id")).strip()
            if payload.get("attempt_id") not in (None, "")
            else None
        ),
        graph_materialization_status=(
            str(payload.get("graph_materialization_status")).strip()
            if payload.get("graph_materialization_status") not in (None, "")
            else None
        ),
    )


def discover_extraction_attempts(
    root: str | Path,
    *,
    duplicate_paper_policy: DuplicatePaperPolicy = "error",
) -> ExtractionAttemptDiscoveryResult:
    """Find paper-local extraction attempts below a corpus directory."""

    root_path = Path(root)
    if root_path.is_file():
        if root_path.name != "active_chunks.json":
            raise ValueError(
                "file input must point to active_chunks.json"
            )
        active_paths = [root_path]
    elif root_path.is_dir():
        direct = root_path / "active_chunks.json"
        active_paths = (
            [direct]
            if direct.is_file()
            else sorted(root_path.rglob("active_chunks.json"))
        )
    else:
        raise FileNotFoundError(root_path)

    if not active_paths:
        raise FileNotFoundError(
            f"no active_chunks.json found under {root_path}"
        )

    discovered = [_attempt_from_active_chunks(path) for path in active_paths]
    by_paper: dict[str, list[DiscoveredExtractionAttempt]] = {}
    for row in discovered:
        by_paper.setdefault(row.paper_id, []).append(row)

    duplicate_papers = {
        paper_id: rows
        for paper_id, rows in by_paper.items()
        if len(rows) > 1
    }
    if duplicate_papers and duplicate_paper_policy == "error":
        details = "; ".join(
            f"{paper_id}={len(rows)}"
            for paper_id, rows in sorted(duplicate_papers.items())
        )
        raise ValueError(
            "multiple extraction attempts found for the same paper; "
            "use duplicate_paper_policy='latest_attempt' to choose "
            f"deterministically: {details}"
        )

    if duplicate_paper_policy == "latest_attempt":
        selected = []
        for paper_id, rows in sorted(by_paper.items()):
            selected.append(
                max(
                    rows,
                    key=lambda row: (
                        row.attempt_id or "",
                        row.active_chunks_path,
                    ),
                )
            )
    else:
        selected = sorted(
            discovered,
            key=lambda row: (row.paper_id, row.active_chunks_path),
        )

    return ExtractionAttemptDiscoveryResult(
        root=str(root_path.resolve()),
        duplicate_paper_policy=duplicate_paper_policy,
        discovered_active_chunks_count=len(discovered),
        selected_attempt_count=len(selected),
        paper_count=len({row.paper_id for row in selected}),
        duplicate_paper_count=len(duplicate_papers),
        attempts=sorted(
            selected,
            key=lambda row: (row.paper_id, row.active_chunks_path),
        ),
    )


def _aggregate_mode(
    modes: set[CapabilityAssessmentMode],
) -> CorpusAssessmentMode:
    if len(modes) == 1:
        return next(iter(modes))
    return "mixed"


def _aggregate_state(
    *,
    mode: CorpusAssessmentMode,
    applicable: int,
    supported: int,
    states: list[CapabilityState],
) -> CapabilityState:
    if mode == "explicit_gap":
        return "absent"
    if mode == "structural_support":
        return "complete" if all(state == "complete" for state in states) else "partial"
    if applicable == 0:
        return "unknown"
    if supported == 0:
        return "absent"
    if supported == applicable:
        return "complete"
    return "partial"


def aggregate_semantic_ir_capability_audits(
    *,
    corpus_id: str,
    root: str,
    discovery: ExtractionAttemptDiscoveryResult,
    audits: list[SemanticIRCapabilityAuditResult],
) -> CorpusSemanticCapabilityAuditResult:
    if not corpus_id.strip():
        raise ValueError("corpus_id must not be blank")
    if not audits:
        raise ValueError("at least one paper audit is required")

    paper_ids = [audit.paper_id for audit in audits]
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("paper audits must have unique paper_id values")
    if set(paper_ids) != {row.paper_id for row in discovery.attempts}:
        raise ValueError("paper audit set must match selected discovery attempts")

    attempt_by_paper = {row.paper_id: row for row in discovery.attempts}
    capability_names = sorted(
        {
            name
            for audit in audits
            for name in audit.metrics
        }
    )
    aggregate_metrics: dict[str, CorpusCapabilityAggregateMetric] = {}
    aggregate_capabilities: dict[str, CapabilityRecord] = {}

    for capability in capability_names:
        rows = [
            audit.metrics[capability]
            for audit in audits
            if capability in audit.metrics
        ]
        modes = {row.assessment_mode for row in rows}
        mode = _aggregate_mode(modes)
        applicable = sum(row.applicable_count for row in rows)
        supported = sum(row.supported_count for row in rows)
        states = [row.state for row in rows]
        state = _aggregate_state(
            mode=mode,
            applicable=applicable,
            supported=supported,
            states=states,
        )
        assessed_from = sorted(
            {
                source
                for row in rows
                for source in row.assessed_from
            }
        )
        state_counts = Counter(states)
        metric = CorpusCapabilityAggregateMetric(
            capability=capability,
            state=state,
            assessment_mode=mode,
            paper_assessed_count=len(rows),
            paper_state_counts=dict(sorted(state_counts.items())),
            applicable_count=applicable,
            supported_count=supported,
            coverage_fraction=(
                supported / applicable if applicable > 0 else None
            ),
            detail=(
                f"Corpus aggregate across {len(rows)} paper-local audits. "
                "Coverage counts are weighted by applicable semantic objects, "
                "not averaged across papers."
            ),
            assessed_from=assessed_from,
        )
        aggregate_metrics[capability] = metric
        aggregate_capabilities[capability] = CapabilityRecord(
            name=capability,
            state=state,
            detail=metric.detail,
            assessed_from=assessed_from,
        )

    manifest_payload = {
        name: {
            "state": row.state,
            "assessment_mode": row.assessment_mode,
            "applicable_count": row.applicable_count,
            "supported_count": row.supported_count,
        }
        for name, row in aggregate_metrics.items()
    }
    manifest = CapabilityManifest(
        manifest_id=_stable_id(
            "corpus_semantic_capability_manifest",
            corpus_id,
            _canonical_json(manifest_payload),
        ),
        scope_kind="corpus",
        scope_id=corpus_id,
        capabilities=aggregate_capabilities,
    )

    paper_summaries = []
    object_counts: Counter[str] = Counter()
    for audit in sorted(audits, key=lambda item: item.paper_id):
        object_counts.update(audit.object_counts)
        attempt = attempt_by_paper[audit.paper_id]
        paper_summaries.append(
            CorpusPaperCapabilitySummary(
                paper_id=audit.paper_id,
                active_chunks_path=attempt.active_chunks_path,
                bundle_id=audit.bundle_id,
                manifest_id=audit.manifest.manifest_id,
                object_counts=audit.object_counts,
                capability_states={
                    name: record.state
                    for name, record in sorted(
                        audit.manifest.capabilities.items()
                    )
                },
                capability_coverage={
                    name: metric.coverage_fraction
                    for name, metric in sorted(audit.metrics.items())
                },
            )
        )

    return CorpusSemanticCapabilityAuditResult(
        corpus_id=corpus_id,
        root=str(Path(root).resolve()),
        discovery=discovery,
        manifest=manifest,
        metrics=aggregate_metrics,
        papers=paper_summaries,
        object_counts=dict(sorted(object_counts.items())),
    )


def audit_existing_extraction_corpus(
    root: str | Path,
    *,
    corpus_id: str | None = None,
    duplicate_paper_policy: DuplicatePaperPolicy = "error",
) -> CorpusSemanticCapabilityAuditResult:
    """Discover, import, and aggregate an existing corpus without LLM calls."""

    root_path = Path(root)
    discovery = discover_extraction_attempts(
        root_path,
        duplicate_paper_policy=duplicate_paper_policy,
    )
    resolved_corpus_id = (
        corpus_id.strip()
        if corpus_id is not None and corpus_id.strip()
        else root_path.name or "existing-extraction-corpus"
    )

    audits: list[SemanticIRCapabilityAuditResult] = []
    for attempt in discovery.attempts:
        imported = load_existing_extraction_semantic_ir(
            attempt.active_chunks_path
        )
        if imported.bundle.paper_id != attempt.paper_id:
            raise ValueError(
                "imported paper_id does not match discovery paper_id"
            )
        audits.append(audit_semantic_ir_capabilities(imported.bundle))

    return aggregate_semantic_ir_capability_audits(
        corpus_id=resolved_corpus_id,
        root=str(root_path),
        discovery=discovery,
        audits=audits,
    )
