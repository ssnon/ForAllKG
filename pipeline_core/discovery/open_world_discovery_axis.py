from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
    DiscoveryAxisPlan,
)
from pipeline_core.discovery.dual_hypothesis_context import (
    DualHypothesisContext,
)
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQuery,
    PriorArtWork,
)
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureSearchProvider,
    canonicalize_prior_art_works,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenWorldRetrievalSeed(StrictModel):
    seed_id: str
    source_kind: Literal[
        "FROZEN_RESEARCH_QUESTION",
        "FROZEN_GAP_STATEMENT",
    ]
    source_id: str
    query_text: str = Field(min_length=1)


class OpenWorldRetrievalExecution(StrictModel):
    query_id: str
    seed_id: str
    provider: str
    success: bool
    result_count: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


class OpenWorldRankedWork(StrictModel):
    work: PriorArtWork
    distinct_seed_count: int
    distinct_provider_count: int
    best_provider_rank: int
    citation_count: int
    seed_ids: list[str] = Field(default_factory=list)
    query_ids: list[str] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)
    abstract_available: bool


class OpenWorldRetrievalResult(StrictModel):
    schema_version: Literal[
        "open-world-discovery-retrieval-v1"
    ] = "open-world-discovery-retrieval-v1"
    seeds: list[OpenWorldRetrievalSeed]
    executions: list[OpenWorldRetrievalExecution]
    raw_work_count: int
    canonical_work_count: int
    supplementary_records_collapsed: int
    ranked_works: list[OpenWorldRankedWork]
    complete: bool


class ExternalAxisDraft(StrictModel):
    local_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    proposed_subject: str = Field(min_length=1)
    proposed_relation: str = Field(min_length=1)
    proposed_object: str = Field(min_length=1)
    source_work_ids: list[str] = Field(min_length=1)
    source_evidence_spans: list[str] = Field(min_length=1)
    compatible_grounded_statement_ids: list[str] = Field(min_length=1)
    bounded_synthesis_note: str = ""
    requires_verification: Literal[True] = True


class ExternalAxisDraftBatch(StrictModel):
    axes: list[ExternalAxisDraft] = Field(
        default_factory=list,
        max_length=4,
    )
    interpretation: str = Field(min_length=1)


class ExternalAxisValidationRecord(StrictModel):
    axis: ExternalAxisDraft
    reason_codes: list[str] = Field(default_factory=list)


class ExternalAxisValidationResult(StrictModel):
    schema_version: Literal[
        "open-world-external-axis-validation-v1"
    ] = "open-world-external-axis-validation-v1"
    accepted_axes: list[ExternalAxisDraft] = Field(default_factory=list)
    rejected_axes: list[ExternalAxisValidationRecord] = Field(
        default_factory=list
    )


class ExternalAxisProvenance(StrictModel):
    axis_id: str
    raw_local_id: str
    source_work_ids: list[str]
    source_evidence_spans: list[str]
    compatible_grounded_statement_ids: list[str]
    bounded_synthesis_note: str
    max_control_similarity: float


class OpenWorldExternalAxisBundle(StrictModel):
    schema_version: Literal[
        "open-world-external-axis-bundle-v1"
    ] = "open-world-external-axis-bundle-v1"
    bundle_id: str
    bundle_sha256: str
    source_dual_context_id: str
    source_dual_context_sha256: str
    provenance: list[ExternalAxisProvenance] = Field(default_factory=list)


class ExternalAxisPlanRejection(StrictModel):
    raw_local_id: str
    reason_codes: list[str]
    max_control_similarity: float = -1.0


class OpenWorldAxisPlanResult(StrictModel):
    schema_version: Literal[
        "open-world-axis-plan-result-v1"
    ] = "open-world-axis-plan-result-v1"
    plan: DiscoveryAxisPlan
    bundle: OpenWorldExternalAxisBundle
    rejected_axes: list[ExternalAxisPlanRejection] = Field(
        default_factory=list
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


def _sha_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _norm_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


def _norm_doi(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("https://doi.org/"):
        text = text[len("https://doi.org/") :]
    if text.startswith("doi:"):
        text = text[4:]
    return text or None


def _work_identity(
    *,
    doi: object,
    title: object,
) -> tuple[str, str]:
    normalized_doi = _norm_doi(doi)
    if normalized_doi:
        return ("doi", normalized_doi)
    return ("title", _norm_text(title))


def build_outcome_blind_retrieval_seeds(
    dual: DualHypothesisContext,
) -> list[OpenWorldRetrievalSeed]:
    """Build retrieval seeds without using generated hypotheses or outcomes.

    Only the frozen research question and grounded statements already marked
    eligible_as_gap may create open-world retrieval queries.
    """

    grounded = dual.grounded_context
    seeds = [
        OpenWorldRetrievalSeed(
            seed_id="open_world_seed:question",
            source_kind="FROZEN_RESEARCH_QUESTION",
            source_id=grounded.task_id,
            query_text=grounded.question,
        )
    ]

    gap_index = 0
    for statement in grounded.evidence_statements:
        if not statement.eligible_as_gap:
            continue
        gap_index += 1
        seeds.append(
            OpenWorldRetrievalSeed(
                seed_id=f"open_world_seed:gap_{gap_index}",
                source_kind="FROZEN_GAP_STATEMENT",
                source_id=statement.statement_id,
                query_text=statement.text,
            )
        )
    return seeds


def retrieve_open_world_sources(
    *,
    seeds: list[OpenWorldRetrievalSeed],
    providers: list[LiteratureSearchProvider],
    results_per_query: int = 20,
) -> OpenWorldRetrievalResult:
    """Execute the frozen seed/provider matrix and rank canonical works.

    Provider-set mutation and query rewriting are deliberately outside this
    function. A caller must not use selected literature when `complete=False`.
    """

    if not seeds:
        raise ValueError("at least one open-world retrieval seed is required")
    if not providers:
        raise ValueError("at least one literature provider is required")
    if results_per_query <= 0:
        raise ValueError("results_per_query must be positive")

    raw_works: list[PriorArtWork] = []
    rank_rows: list[dict[str, Any]] = []
    executions: list[OpenWorldRetrievalExecution] = []

    for seed_index, seed in enumerate(seeds, start=1):
        query = LiteratureQuery(
            query_id=f"open_world_query:{seed_index}",
            hypothesis_id="open_world_discovery_axis",
            claim_id=None,
            query_kind="hypothesis_composite",
            query_text=seed.query_text,
        )

        for provider in providers:
            started = time.perf_counter()
            try:
                works = provider.search(
                    query,
                    limit=results_per_query,
                )
                executions.append(
                    OpenWorldRetrievalExecution(
                        query_id=query.query_id,
                        seed_id=seed.seed_id,
                        provider=provider.provider_name,
                        success=True,
                        result_count=len(works),
                        elapsed_seconds=(
                            time.perf_counter() - started
                        ),
                    )
                )
                for provider_rank, work in enumerate(
                    works,
                    start=1,
                ):
                    raw_works.append(work)
                    rank_rows.append(
                        {
                            "query_id": query.query_id,
                            "seed_id": seed.seed_id,
                            "provider": provider.provider_name,
                            "provider_rank": provider_rank,
                            "identity": _work_identity(
                                doi=work.doi,
                                title=work.title,
                            ),
                        }
                    )
            except Exception as exc:
                executions.append(
                    OpenWorldRetrievalExecution(
                        query_id=query.query_id,
                        seed_id=seed.seed_id,
                        provider=provider.provider_name,
                        success=False,
                        result_count=0,
                        elapsed_seconds=(
                            time.perf_counter() - started
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    canonical, supplementary_collapsed = canonicalize_prior_art_works(
        raw_works
    )
    ranked: list[OpenWorldRankedWork] = []

    for work in canonical:
        identity = _work_identity(
            doi=work.doi,
            title=work.title,
        )
        refs = [
            row
            for row in rank_rows
            if row["identity"] == identity
        ]
        seed_ids = sorted(
            {str(row["seed_id"]) for row in refs}
        )
        query_ids = sorted(
            {str(row["query_id"]) for row in refs}
        )
        provider_names = sorted(
            {str(row["provider"]) for row in refs}
        )
        best_rank = min(
            [int(row["provider_rank"]) for row in refs]
            or [10**9]
        )
        abstract_available = bool(
            isinstance(work.abstract, str)
            and work.abstract.strip()
        )
        ranked.append(
            OpenWorldRankedWork(
                work=work,
                distinct_seed_count=len(seed_ids),
                distinct_provider_count=len(provider_names),
                best_provider_rank=best_rank,
                citation_count=int(work.citation_count or 0),
                seed_ids=seed_ids,
                query_ids=query_ids,
                providers=provider_names,
                abstract_available=abstract_available,
            )
        )

    ranked.sort(
        key=lambda row: (
            -row.distinct_seed_count,
            -row.distinct_provider_count,
            row.best_provider_rank,
            -row.citation_count,
            row.work.title.lower(),
            row.work.work_id,
        )
    )

    return OpenWorldRetrievalResult(
        seeds=seeds,
        executions=executions,
        raw_work_count=len(raw_works),
        canonical_work_count=len(canonical),
        supplementary_records_collapsed=supplementary_collapsed,
        ranked_works=ranked,
        complete=all(row.success for row in executions),
    )


def select_open_world_abstracts(
    result: OpenWorldRetrievalResult,
    *,
    max_selected: int = 12,
) -> list[OpenWorldRankedWork]:
    if not result.complete:
        raise RuntimeError(
            "open-world retrieval is incomplete; do not synthesize axes"
        )
    if max_selected <= 0:
        raise ValueError("max_selected must be positive")
    return [
        row
        for row in result.ranked_works
        if row.abstract_available
    ][:max_selected]


def build_external_axis_prompt_payload(
    *,
    dual: DualHypothesisContext,
    control_plan: DiscoveryAxisPlan,
    selected_works: list[OpenWorldRankedWork],
) -> dict[str, Any]:
    grounded = dual.grounded_context
    positive = [
        statement
        for statement in grounded.evidence_statements
        if statement.eligible_as_premise
    ]
    gaps = [
        statement
        for statement in grounded.evidence_statements
        if statement.eligible_as_gap
    ]

    return {
        "research_question": grounded.question,
        "selected_positive_premises": [
            {
                "statement_id": row.statement_id,
                "text": row.text,
                "claim_kind": row.claim_kind,
            }
            for row in positive
        ],
        "research_gaps": [
            {
                "statement_id": row.statement_id,
                "text": row.text,
            }
            for row in gaps
        ],
        "current_control_axes": [
            {
                "axis_id": axis.axis_id,
                "label": axis.label,
                "proposed_subject": axis.proposed_subject,
                "proposed_relation": axis.proposed_relation,
                "proposed_object": axis.proposed_object,
            }
            for axis in control_plan.axes
        ],
        "retrieved_abstract_backed_works": [
            {
                "work_id": row.work.work_id,
                "title": row.work.title,
                "year": row.work.year,
                "doi": row.work.doi,
                "abstract": row.work.abstract,
                "providers": row.providers,
                "retrieval_seed_ids": row.seed_ids,
            }
            for row in selected_works
        ],
    }


def validate_external_axis_drafts(
    *,
    batch: ExternalAxisDraftBatch,
    dual: DualHypothesisContext,
    selected_works: list[OpenWorldRankedWork],
    max_source_span_words: int = 40,
) -> ExternalAxisValidationResult:
    if max_source_span_words <= 0:
        raise ValueError("max_source_span_words must be positive")

    work_by_id = {
        row.work.work_id: row.work
        for row in selected_works
    }
    allowed_premise_ids = {
        row.statement_id
        for row in dual.grounded_context.evidence_statements
        if row.eligible_as_premise
    }
    seen_local_ids: set[str] = set()
    accepted: list[ExternalAxisDraft] = []
    rejected: list[ExternalAxisValidationRecord] = []

    for axis in batch.axes:
        reasons: list[str] = []

        if axis.local_id in seen_local_ids:
            reasons.append("DUPLICATE_LOCAL_ID")
        seen_local_ids.add(axis.local_id)

        unknown_works = [
            work_id
            for work_id in axis.source_work_ids
            if work_id not in work_by_id
        ]
        if unknown_works:
            reasons.append(
                "UNKNOWN_SOURCE_WORK_IDS:"
                + ",".join(sorted(unknown_works))
            )

        invalid_premises = [
            statement_id
            for statement_id in axis.compatible_grounded_statement_ids
            if statement_id not in allowed_premise_ids
        ]
        if invalid_premises:
            reasons.append(
                "INVALID_GROUNDED_STATEMENT_IDS:"
                + ",".join(sorted(invalid_premises))
            )

        cited_texts = []
        for work_id in axis.source_work_ids:
            work = work_by_id.get(work_id)
            if work is None:
                continue
            cited_texts.append(
                work.title + "\n" + str(work.abstract or "")
            )

        for span in axis.source_evidence_spans:
            if len(span.split()) > max_source_span_words:
                reasons.append(
                    "SOURCE_SPAN_TOO_LONG:" + span[:80]
                )
                continue
            if not any(span in text for text in cited_texts):
                reasons.append(
                    "SOURCE_SPAN_NOT_EXACT:" + span[:80]
                )

        if reasons:
            rejected.append(
                ExternalAxisValidationRecord(
                    axis=axis,
                    reason_codes=reasons,
                )
            )
        else:
            accepted.append(axis)

    return ExternalAxisValidationResult(
        accepted_axes=accepted,
        rejected_axes=rejected,
    )


AxisSimilarity = Callable[[str, str], float]


def _axis_text_from_parts(
    subject: str,
    relation: str,
    obj: str,
) -> str:
    return " | ".join([subject, relation, obj])


def build_external_axis_plan(
    *,
    dual: DualHypothesisContext,
    control_plan: DiscoveryAxisPlan,
    selected_works: list[OpenWorldRankedWork],
    validated_axes: list[ExternalAxisDraft],
    similarity: AxisSimilarity,
    max_control_similarity: float = 0.85,
) -> OpenWorldAxisPlanResult:
    """Convert validated external drafts into ordinary DiscoveryAxis records.

    External literature remains inspiration-only. The resulting plan reuses the
    control planner policy and keeps source evidence in a separate provenance
    bundle referenced by source_bundle_id/source_bundle_sha256.
    """

    if not 0.0 <= max_control_similarity <= 1.0:
        raise ValueError(
            "max_control_similarity must be within [0, 1]"
        )

    selected_by_id = {
        row.work.work_id: row.work
        for row in selected_works
    }
    statement_by_id = {
        row.statement_id: row
        for row in dual.grounded_context.evidence_statements
    }

    control_rows = []
    control_triples = set()
    for axis in control_plan.axes:
        text = _axis_text_from_parts(
            axis.proposed_subject,
            axis.proposed_relation,
            axis.proposed_object,
        )
        control_rows.append((axis.axis_id, text))
        control_triples.add(_norm_text(text))

    accepted_axes: list[DiscoveryAxis] = []
    provenance_rows: list[ExternalAxisProvenance] = []
    rejected: list[ExternalAxisPlanRejection] = []

    for raw in validated_axes:
        ext_text = _axis_text_from_parts(
            raw.proposed_subject,
            raw.proposed_relation,
            raw.proposed_object,
        )
        reasons: list[str] = []

        if _norm_text(ext_text) in control_triples:
            reasons.append("EXACT_CONTROL_TRIPLE_DUPLICATE")

        similarities = [
            float(similarity(ext_text, control_text))
            for _, control_text in control_rows
        ]
        max_similarity = max(similarities, default=-1.0)
        if max_similarity > max_control_similarity:
            reasons.append("CONTROL_AXIS_SEMANTIC_DUPLICATE")

        compatible_ids = [
            statement_id
            for statement_id in raw.compatible_grounded_statement_ids
            if statement_id in statement_by_id
            and statement_by_id[statement_id].eligible_as_premise
        ]
        if not compatible_ids:
            reasons.append(
                "NO_VALID_COMPATIBLE_GROUNDED_STATEMENT"
            )

        source_work_ids = sorted(set(raw.source_work_ids))
        if (
            not source_work_ids
            or any(
                work_id not in selected_by_id
                for work_id in source_work_ids
            )
        ):
            reasons.append("MISSING_SELECTED_SOURCE_WORK")

        if reasons:
            rejected.append(
                ExternalAxisPlanRejection(
                    raw_local_id=raw.local_id,
                    reason_codes=sorted(set(reasons)),
                    max_control_similarity=max_similarity,
                )
            )
            continue

        axis_rank = len(accepted_axes) + 1
        source_hash = _sha_json(source_work_ids)
        axis_id = _stable_id(
            "discovery_axis",
            dual.dual_context_sha256,
            "external_open_world",
            raw.local_id,
            source_hash,
            axis_rank,
        )
        inspiration_id = _stable_id(
            "external_inspiration",
            raw.local_id,
            source_hash,
        )
        candidate_unit_id = _stable_id(
            "external_candidate_unit",
            raw.proposed_subject,
            raw.proposed_relation,
            raw.proposed_object,
            source_hash,
        )
        source_path_id = _stable_id(
            "external_work_set",
            *source_work_ids,
        )

        entry_id = compatible_ids[0]
        entry = statement_by_id[entry_id]
        first_work = selected_by_id[source_work_ids[0]]
        evidence_preview = " ; ".join(
            raw.source_evidence_spans[:3]
        )

        accepted_axes.append(
            DiscoveryAxis(
                axis_id=axis_id,
                axis_rank=axis_rank,
                inspiration_id=inspiration_id,
                source_path_id=source_path_id,
                candidate_unit_id=candidate_unit_id,
                label=raw.label,
                entry_anchor_id=entry_id,
                entry_anchor_label=entry.text,
                exit_anchor_id=source_work_ids[0],
                exit_anchor_label=first_work.title,
                proposed_subject=raw.proposed_subject,
                proposed_relation=raw.proposed_relation,
                proposed_object=raw.proposed_object,
                rendered_path=(
                    "EXTERNAL_OPEN_WORLD_INSPIRATION_ONLY :: "
                    + raw.proposed_subject
                    + " --"
                    + raw.proposed_relation
                    + "--> "
                    + raw.proposed_object
                    + " :: source_evidence="
                    + evidence_preview
                ),
                source_mode="external_open_world",
                exploration_score=1.0,
                candidate_unit_score=1.0,
                planner_score=max(
                    0.0,
                    1.0 - 0.01 * (axis_rank - 1),
                ),
                mechanistic_continuity_band=(
                    "external_requires_verification"
                ),
                generic_entity_fraction=0.0,
                registry_hop_fraction=0.0,
                grounding_semantic_overlap=0.0,
                reaction_domain_switch_penalty=0.0,
                requires_verification=True,
                reason_codes=[
                    "OPEN_WORLD_DISCOVERY_ESCAPE",
                    "EXTERNAL_INSPIRATION_ONLY",
                    "NOT_POSITIVE_PREMISE",
                    "EXACT_SOURCE_SPANS_VALIDATED",
                ],
            )
        )
        provenance_rows.append(
            ExternalAxisProvenance(
                axis_id=axis_id,
                raw_local_id=raw.local_id,
                source_work_ids=source_work_ids,
                source_evidence_spans=list(
                    raw.source_evidence_spans
                ),
                compatible_grounded_statement_ids=compatible_ids,
                bounded_synthesis_note=raw.bounded_synthesis_note,
                max_control_similarity=max_similarity,
            )
        )

    bundle_body = {
        "schema_version":
            "open-world-external-axis-bundle-v1",
        "source_dual_context_id": dual.dual_context_id,
        "source_dual_context_sha256":
            dual.dual_context_sha256,
        "provenance": [
            row.model_dump(mode="json")
            for row in provenance_rows
        ],
    }
    bundle_sha = _sha_json(bundle_body)
    bundle_id = _stable_id(
        "external_axis_bundle",
        dual.dual_context_sha256,
        bundle_sha,
    )
    bundle = OpenWorldExternalAxisBundle(
        **bundle_body,
        bundle_id=bundle_id,
        bundle_sha256=bundle_sha,
    )

    plan_body = {
        "schema_version": "discovery-axis-plan-v1",
        "plan_id": _stable_id(
            "discovery_axis_plan",
            dual.dual_context_sha256,
            bundle_id,
            *[axis.axis_id for axis in accepted_axes],
        ),
        "source_dual_context_id": dual.dual_context_id,
        "source_dual_context_sha256":
            dual.dual_context_sha256,
        "source_bundle_id": bundle.bundle_id,
        "source_bundle_sha256": bundle.bundle_sha256,
        "corpus_id": dual.grounded_context.corpus_id,
        "axes": [
            axis.model_dump(mode="json")
            for axis in accepted_axes
        ],
        "excluded_inspiration_ids": [],
        "policy": control_plan.policy.model_dump(mode="json"),
    }
    plan = DiscoveryAxisPlan(
        **plan_body,
        plan_sha256=_sha_json(plan_body),
    )

    return OpenWorldAxisPlanResult(
        plan=plan,
        bundle=bundle,
        rejected_axes=rejected,
    )
