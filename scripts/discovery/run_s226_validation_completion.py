from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.registry import get_domain_profile
from pipeline_core.discovery.direct_higher_order_conceptual_knownness_contracts import (
    ConceptualKnownnessLevelResult,
    DirectHigherOrderConceptualKnownnessRecord,
    build_conceptual_knownness_profile,
)
from pipeline_core.discovery.direct_higher_order_shadow_bundle import (
    DirectHigherOrderShadowBundle,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtPacket,
)
from pipeline_core.discovery.node_mapping import (
    DEFAULT_EMBED_MODEL,
    SentenceTransformerEncoder,
)
from pipeline_core.discovery.prior_art_matching import PriorArtRanker
from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
)
from pipeline_core.discovery.prior_art_retrieval import LiteratureRetriever
from pipeline_core.discovery.prospective_novelty_validation_completion_s226 import (
    audit_external_report_payload,
    first_gap_level,
    l3_status_from_external_card,
    research_value_detail_payload,
)
from pipeline_core.discovery.prospective_novelty_validation_execution import (
    ProspectiveNoveltyExecutionPlan,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntermediateAbstractionDraft(StrictModel):
    proposed_relation: str = ""
    retrieval_query: str = ""
    generalized_or_removed_terms: list[str] = Field(default_factory=list)
    preserved_relation_invariants: list[str] = Field(default_factory=list)
    added_scientific_content: bool = False
    added_content_description: str = ""
    abstraction_supported: bool = True
    unsupported_reason: str = ""

    @model_validator(mode="after")
    def validate_row(self):
        if self.abstraction_supported:
            if not self.proposed_relation.strip():
                raise ValueError(
                    "supported abstraction requires proposed_relation"
                )
            if not self.retrieval_query.strip():
                raise ValueError(
                    "supported abstraction requires retrieval_query"
                )
        else:
            if (
                self.proposed_relation.strip()
                or self.retrieval_query.strip()
            ):
                raise ValueError(
                    "unsupported abstraction must emit empty relation/query"
                )
        if (
            self.added_scientific_content
            and not self.added_content_description.strip()
        ):
            raise ValueError(
                "added scientific content requires description"
            )
        return self


class ConceptualMatchDraft(StrictModel):
    work_id: str
    relationship: Literal[
        "DIRECT_RELATION",
        "PARTIAL_RELATION",
        "COMPONENT_ONLY",
        "TITLE_ONLY_NEIGHBOR",
        "CONTEXT_MISMATCH",
        "CONFLICTING_RELATION",
        "UNRELATED",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class ConceptualReviewDraft(StrictModel):
    matches: list[ConceptualMatchDraft] = Field(default_factory=list)
    interpretation: str = Field(min_length=1)


ABSTRACTION_SYSTEM = """You propose ONE intermediate conceptual abstraction of a generated higher-order scientific relation.

This is a SHADOW REPRESENTATION TASK ONLY.
Do NOT judge novelty, prior art, truth, importance, plausibility, or research value.

INPUTS:
- a recorded BASE_TASK_RELATION with confirmed-known structural authority,
- a MODIFIER_COMPONENT_RELATION with candidate-inspiration authority,
- the generated FULL_HIGHER_ORDER_CLAIM.

Produce only L2_INTERMEDIATE.

Rules:
- Preserve the higher-order relational form of the FULL claim.
- Generalize only overly specific material/system/modifier identity when a
  scientifically meaningful one-step parent class is defensible from the text.
- Preserve the same dependent relation and interaction/moderation structure.
- Do not add a mechanism, sign, threshold, reversal, regime, material family,
  or causal proposition not licensed by the supplied text.
- Do not import scientific facts from memory.
- If no safe one-step abstraction exists, set abstraction_supported=false.
- retrieval_query must search exactly the proposed relation, with no Boolean
  syntax, title, author, year, DOI, or literature-knownness language.
- added_scientific_content must be true if anything scientifically stronger
  than the supplied relation is introduced. Prefer unsupported over stronger.
"""


REVIEW_SYSTEM = """You review bounded literature candidates against ONE conceptual scientific relation.

This is a conceptual-knownness SHADOW diagnostic, not novelty certification.

Labels:
- DIRECT_RELATION: the abstract materially establishes the supplied target
  relation at this conceptual level or a compatible more-specific instance.
- PARTIAL_RELATION: the abstract establishes a substantial portion of the
  relation, but not the complete target. Co-occurrence alone is insufficient.
- COMPONENT_ONLY: components appear but the relation nucleus is not established.
- TITLE_ONLY_NEIGHBOR: the title suggests material relevance but the abstract
  is unavailable or insufficient to verify the relation.
- CONTEXT_MISMATCH: similar relation in a materially incompatible domain/system.
- CONFLICTING_RELATION: substantially same relation with incompatible result.
- UNRELATED: no material bearing on the target.

For moderation/interaction targets, DIRECT requires evidence that the moderator
changes the underlying X-to-Y relation, not merely moderator-to-Y plus X-to-Y.

Never infer detailed results from title alone.
Never infer literature-wide absence from this bounded candidate set.
Return only work IDs from ALLOWED_WORK_IDS.
"""


def clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def norm(value: object) -> str:
    text = clean(value).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def sha256_json(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def human_relation(value: str) -> str:
    mapping = {
        "VARIES_WITH": "varies with",
        "CORRELATES_WITH": "correlates with",
        "MODULATES": "modulates",
        "AFFECTS": "affects",
        "ASSOCIATED_WITH": "is associated with",
        "DEPENDS_ON": "depends on",
        "PROMOTES": "promotes",
        "INHIBITS": "inhibits",
    }
    key = clean(value).upper()
    return mapping.get(
        key,
        clean(value).replace("_", " ").lower(),
    )


def dedup_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = clean(value)
        key = norm(value)
        if not value or not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def get_client(
    *,
    api_key: str,
    base_url: str | None,
    instructor_mode: str,
    timeout: float,
):
    try:
        import instructor
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "S226 requires installed openai and instructor packages"
        ) from exc

    mode = getattr(instructor.Mode, instructor_mode.upper(), None)
    if mode is None:
        raise ValueError(
            f"unknown instructor mode: {instructor_mode}"
        )

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return instructor.from_openai(
        OpenAI(**kwargs),
        mode=mode,
    )


def build_abstraction_prompt(view) -> str:
    return "\n".join(
        [
            "BASE_TASK_RELATION",
            "==================",
            f"subject: {view.base_subject}",
            f"relation: {view.base_relation}",
            f"object: {view.base_object}",
            f"authority: {view.base_authority}",
            "",
            "MODIFIER_COMPONENT_RELATION",
            "===========================",
            f"subject: {view.modifier_subject}",
            f"relation: {view.modifier_relation}",
            f"object: {view.modifier_object}",
            f"authority: {view.modifier_authority}",
            "",
            "FULL_HIGHER_ORDER_CLAIM",
            "=======================",
            f"hypothesis_id: {view.hypothesis_id}",
            f"text: {view.full_higher_order_claim}",
            "",
            "Return the L2_INTERMEDIATE abstraction only.",
        ]
    )


def propose_l2(
    *,
    client,
    model: str,
    view,
    temperature: float,
    max_retries: int,
    telemetry_path: str | None,
) -> tuple[IntermediateAbstractionDraft, Any]:
    prompt = build_abstraction_prompt(view)
    result, event = run_instructor_structured_call(
        client.chat.completions,
        model=model,
        response_model=IntermediateAbstractionDraft,
        messages=[
            {"role": "system", "content": ABSTRACTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_retries=max_retries,
        telemetry_path=telemetry_path,
        telemetry_context={
            "pipeline":
                "prospective_novelty_validation_completion_s226",
            "stage": "conceptual_abstraction",
            "hypothesis_id": view.hypothesis_id,
        },
    )
    if not isinstance(result, IntermediateAbstractionDraft):
        result = IntermediateAbstractionDraft.model_validate(result)
    return result, event


def abstraction_admissible(
    draft: IntermediateAbstractionDraft,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not draft.abstraction_supported:
        reasons.append("MODEL_ABSTRACTION_UNSUPPORTED")
    if draft.added_scientific_content:
        reasons.append("ADDED_SCIENTIFIC_CONTENT")
    prohibited = {
        "novel",
        "unknown",
        "known",
        "established",
        "unprecedented",
    }
    joined = norm(
        draft.proposed_relation + " " + draft.retrieval_query
    )
    if any(token in joined.split() for token in prohibited):
        reasons.append("KNOWNNESS_LANGUAGE_IN_ABSTRACTION")
    return not reasons, reasons


def make_synthetic_claim(
    *,
    target_id: str,
    relation: str,
    queries: list[str],
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=target_id,
        hypothesis_id=stable_id("s226_hypothesis", target_id),
        claim_rank=1,
        kind="context_condition",
        importance="supporting",
        novelty_selection_role=None,
        text=relation,
        rationale=(
            "S226 conceptual-knownness retrieval target only; "
            "no novelty or scientific authority."
        ),
        search_concepts=[relation],
        search_queries=list(queries),
    )


def make_query_plan(
    *,
    target_id: str,
    relation: str,
    queries: list[str],
) -> LiteratureQueryPlan:
    claim = make_synthetic_claim(
        target_id=target_id,
        relation=relation,
        queries=queries,
    )
    query_rows = [
        LiteratureQuery(
            query_id=stable_id(
                "s226_query",
                target_id,
                index,
                query,
            ),
            hypothesis_id=claim.hypothesis_id,
            claim_id=claim.claim_id,
            query_kind=(
                "claim_primary"
                if index == 1
                else "claim_variant"
            ),
            query_text=query,
        )
        for index, query in enumerate(queries, start=1)
    ]
    group = HypothesisNoveltyClaims(
        hypothesis_id=claim.hypothesis_id,
        title="S226 conceptual knownness target",
        claims=[claim],
        decomposition_notes=(
            "Synthetic validation-only conceptual target."
        ),
    )
    body = {
        "schema_version": "literature-query-plan-v1",
        "plan_id": stable_id(
            "s226_query_plan",
            target_id,
            *queries,
        ),
        "source_portfolio_id":
            "s226_conceptual_knownness_validation",
        "queries": [
            row.model_dump(mode="json")
            for row in query_rows
        ],
        "claims": [group.model_dump(mode="json")],
        "policy_version":
            "external-novelty-query-policy-v1",
    }
    body["plan_sha256"] = sha256_json(body)
    return LiteratureQueryPlan.model_validate(body)


def ranked_work_payloads(
    *,
    packet: PriorArtPacket,
    ranked,
) -> list[dict[str, Any]]:
    works = {row.work_id: row for row in packet.works}
    result = []
    for ranking in ranked.ranked_works:
        work = works[ranking.work_id]
        result.append(
            {
                "work_id": work.work_id,
                "title": work.title,
                "year": work.year,
                "doi": work.doi,
                "url": work.url,
                "abstract": work.abstract,
                "providers": list(work.providers),
                "relevance_score": ranking.relevance_score,
                "semantic_similarity": ranking.semantic_similarity,
                "lexical_coverage": ranking.lexical_coverage,
                "reaction_domain_relevance":
                    ranking.reaction_domain_relevance,
                "catalyst_scope_relevance":
                    ranking.catalyst_scope_relevance,
                "abstract_available":
                    ranking.abstract_available,
            }
        )
    return result


def review_target(
    *,
    client,
    model: str,
    target: dict[str, Any],
    works: list[dict[str, Any]],
    temperature: float,
    max_retries: int,
    telemetry_path: str | None,
) -> tuple[ConceptualReviewDraft, Any]:
    lines = [
        "CONCEPTUAL TARGET",
        "=================",
        f"target_id: {target['target_id']}",
        f"level: {target['level']}",
        f"relation: {target['relation']}",
        "",
        "RETRIEVAL QUERIES",
        "=================",
        *["- " + row for row in target["queries"]],
        "",
        "RETRIEVED CANDIDATES",
        "====================",
    ]
    if not works:
        lines.append("- NONE")

    for index, work in enumerate(works, start=1):
        abstract = clean(work.get("abstract"))
        if len(abstract) > 1600:
            abstract = abstract[:1599].rstrip() + "…"
        lines.extend(
            [
                f"[{index}] work_id={work['work_id']}",
                f"title: {work['title']}",
                f"year: {work['year']}",
                f"doi: {work['doi']}",
                (
                    "semantic_similarity: "
                    f"{work['semantic_similarity']:.4f}"
                ),
                (
                    "lexical_coverage: "
                    f"{work['lexical_coverage']:.4f}"
                ),
                (
                    "abstract: "
                    + (
                        abstract
                        if abstract
                        else "[NO ABSTRACT AVAILABLE]"
                    )
                ),
                "",
            ]
        )

    allowed = [row["work_id"] for row in works]
    lines.extend(
        [
            "ALLOWED_WORK_IDS",
            "================",
            *allowed,
            "",
            "Return only IDs above.",
        ]
    )

    result, event = run_instructor_structured_call(
        client.chat.completions,
        model=model,
        response_model=ConceptualReviewDraft,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ],
        temperature=temperature,
        max_retries=max_retries,
        telemetry_path=telemetry_path,
        telemetry_context={
            "pipeline":
                "prospective_novelty_validation_completion_s226",
            "stage": "conceptual_review",
            "target_id": target["target_id"],
            "level": target["level"],
        },
    )
    if not isinstance(result, ConceptualReviewDraft):
        result = ConceptualReviewDraft.model_validate(result)
    return result, event


def compile_target_result(
    *,
    target: dict[str, Any],
    packet: PriorArtPacket,
    ranked_works: list[dict[str, Any]],
    draft: ConceptualReviewDraft,
    min_confidence: float,
    min_unique_works: int,
    min_abstract_works: int,
    min_successful_query_variants: int,
) -> dict[str, Any]:
    allowed = {row["work_id"]: row for row in ranked_works}
    compiled = []

    for row in draft.matches:
        work = allowed.get(row.work_id)
        if work is None:
            continue
        relationship = row.relationship
        if (
            not work["abstract_available"]
            and relationship
            in {
                "DIRECT_RELATION",
                "PARTIAL_RELATION",
                "CONFLICTING_RELATION",
            }
        ):
            relationship = "TITLE_ONLY_NEIGHBOR"

        compiled.append(
            {
                "work_id": row.work_id,
                "relationship": relationship,
                "confidence": row.confidence,
                "rationale": row.rationale,
                "title": work["title"],
                "year": work["year"],
                "doi": work["doi"],
                "url": work["url"],
                "abstract_available": work["abstract_available"],
                "relevance_score": work["relevance_score"],
            }
        )

    successful_query_ids = {
        row.query_id
        for row in packet.executions
        if row.success
    }
    unique_work_count = len(packet.works)
    abstract_work_count = sum(
        bool(row.abstract)
        for row in packet.works
    )
    coverage_sufficient = (
        len(successful_query_ids)
        >= min_successful_query_variants
        and unique_work_count >= min_unique_works
        and abstract_work_count >= min_abstract_works
    )

    direct = [
        row for row in compiled
        if row["relationship"] == "DIRECT_RELATION"
        and row["confidence"] >= min_confidence
    ]
    partial = [
        row for row in compiled
        if row["relationship"] == "PARTIAL_RELATION"
        and row["confidence"] >= min_confidence
    ]
    conflicting = [
        row for row in compiled
        if row["relationship"] == "CONFLICTING_RELATION"
        and row["confidence"] >= min_confidence
    ]
    title_only = [
        row for row in compiled
        if row["relationship"] == "TITLE_ONLY_NEIGHBOR"
        and row["confidence"] >= min_confidence
    ]

    if direct:
        status = "RELATION_BACKED"
        reason = "DIRECT_CONCEPTUAL_RELATION_FOUND"
    elif conflicting:
        status = "CONFLICTING_RELATION"
        reason = "CONFLICTING_CONCEPTUAL_RELATION_FOUND"
    elif partial:
        status = "PARTIAL_RELATION_BACKED"
        reason = "PARTIAL_CONCEPTUAL_RELATION_FOUND"
    elif title_only:
        status = "INSUFFICIENT_COVERAGE"
        reason = "MATERIAL_TITLE_ONLY_PRIOR_ART_UNRESOLVED"
        coverage_sufficient = False
    elif coverage_sufficient:
        status = "SEARCH_BOUNDED_GAP"
        reason = "SEARCH_BOUNDED_NO_RELATION_MATCH"
    else:
        status = "INSUFFICIENT_COVERAGE"
        reason = "SEARCH_COVERAGE_INSUFFICIENT"

    return {
        **target,
        "status": status,
        "reason_code": reason,
        "coverage": {
            "query_variant_count": len(target["queries"]),
            "successful_query_variant_count":
                len(successful_query_ids),
            "provider_execution_success_count": sum(
                row.success
                for row in packet.executions
            ),
            "unique_work_count": unique_work_count,
            "abstract_work_count": abstract_work_count,
            "sufficient_for_search_bounded_gap":
                coverage_sufficient,
        },
        "review": {
            "matches": compiled,
            "interpretation": draft.interpretation,
            "material_title_only_count": len(title_only),
        },
        "packet_id": packet.packet_id,
        "packet_sha256": packet.packet_sha256,
        "authority": {
            "conceptual_knownness_shadow_only": True,
            "absence_is_not_novelty_proof": True,
            "novelty_certification_authority": False,
            "production_selection_authority": False,
        },
    }


def external_pairs_for_case(
    *,
    run_dir: Path,
    bundle: DirectHigherOrderShadowBundle | None,
) -> list[tuple[str, Path, Path | None]]:
    pairs: list[tuple[str, Path, Path | None]] = []

    main_report = run_dir / "external_novelty_a52.report.json"
    main_packet = run_dir / "external_novelty_a52.prior_art.json"
    if main_report.is_file():
        pairs.append(
            (
                "MAIN_EXTERNAL_NOVELTY",
                main_report,
                main_packet if main_packet.is_file() else None,
            )
        )

    if bundle is not None:
        for arm in bundle.arms:
            if (
                arm.external_novelty.status != "COMPLETED"
                or not arm.external_novelty.report_path
            ):
                continue
            report = Path(
                arm.external_novelty.report_path
            ).expanduser().resolve()
            packet = (
                Path(arm.external_novelty.prior_art_path)
                .expanduser()
                .resolve()
                if arm.external_novelty.prior_art_path
                else None
            )
            pairs.append(
                (
                    f"DIRECT_HO_ARM_{arm.arm_index}",
                    report,
                    packet if packet and packet.is_file() else None,
                )
            )

    return pairs


def audit_external_case(
    *,
    run_dir: Path,
    bundle: DirectHigherOrderShadowBundle | None,
) -> dict[str, Any]:
    rows = []
    for scope, report_path, packet_path in external_pairs_for_case(
        run_dir=run_dir,
        bundle=bundle,
    ):
        report = load_json(report_path)
        packet = (
            load_json(packet_path)
            if packet_path is not None and packet_path.is_file()
            else None
        )
        audit = audit_external_report_payload(
            report,
            packet=packet,
        )
        rows.append(
            {
                "scope": scope,
                "report_path": str(report_path),
                "prior_art_path": (
                    str(packet_path)
                    if packet_path is not None
                    else None
                ),
                **audit,
            }
        )

    if not rows:
        disposition = "NOT_AVAILABLE"
    elif any(row["disposition"] == "RESOLUTION_RISK" for row in rows):
        disposition = "RESOLUTION_RISK"
    elif any(row["disposition"] == "COVERAGE_INCOMPLETE" for row in rows):
        disposition = "COVERAGE_INCOMPLETE"
    else:
        disposition = "NO_OBSERVED_RESOLUTION_RISK"

    return {
        "measurement_status": (
            "AVAILABLE" if rows else "NOT_AVAILABLE"
        ),
        "overall_disposition": disposition,
        "reports": rows,
        "retrieval_recall_completeness":
            "UNOBSERVABLE_WITHOUT_EXTERNAL_REFERENCE_SET",
        "production_selection_authority": False,
    }


def exact_cards_by_hypothesis(
    bundle: DirectHigherOrderShadowBundle,
) -> dict[str, tuple[dict[str, Any], str]]:
    result: dict[str, tuple[dict[str, Any], str]] = {}
    for arm in bundle.arms:
        if (
            arm.external_novelty.status != "COMPLETED"
            or not arm.external_novelty.report_path
        ):
            continue
        path = Path(
            arm.external_novelty.report_path
        ).expanduser().resolve()
        report = ExternalNoveltyReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for card in report.cards:
            result[card.hypothesis_id] = (
                card.model_dump(mode="json"),
                str(path),
            )
    return result


def run_conceptual_for_case(
    *,
    case_id: str,
    bundle: DirectHigherOrderShadowBundle,
    case_out: Path,
    retriever: LiteratureRetriever,
    ranker: PriorArtRanker,
    client,
    model: str,
    temperature: float,
    max_retries: int,
    telemetry_path: str | None,
    min_match_confidence: float,
    min_unique_works: int,
    min_abstract_works: int,
    min_successful_query_variants: int,
) -> dict[str, Any]:
    views = [
        view
        for arm in bundle.arms
        for view in arm.structural_views
    ]

    if not views:
        return {
            "measurement_status":
                "NOT_APPLICABLE_NO_STRUCTURAL_VIEW",
            "record_count": 0,
            "first_gap_levels": {},
        }

    exact_cards = exact_cards_by_hypothesis(bundle)
    packets_dir = case_out / "conceptual_packets"
    packets_dir.mkdir(parents=True, exist_ok=True)

    proposals: dict[str, dict[str, Any]] = {}
    l1_targets: dict[str, dict[str, Any]] = {}
    l2_targets: dict[str, dict[str, Any]] = {}

    for view in views:
        l1_relation = clean(
            f"{view.base_subject} "
            f"{human_relation(view.base_relation)} "
            f"{view.base_object}"
        )
        l1_queries = dedup_texts(
            [
                l1_relation,
                clean(
                    f"{view.base_subject} "
                    f"{view.base_object} "
                    f"{human_relation(view.base_relation)}"
                ),
            ]
        )
        l1_id = stable_id(
            "s226_target",
            "L1_BROAD",
            norm(l1_relation),
        )
        bucket = l1_targets.setdefault(
            l1_id,
            {
                "target_id": l1_id,
                "level": "L1_BROAD",
                "relation": l1_relation,
                "queries": l1_queries,
                "hypothesis_ids": [],
            },
        )
        bucket["hypothesis_ids"].append(
            view.hypothesis_id
        )

        draft, event = propose_l2(
            client=client,
            model=model,
            view=view,
            temperature=temperature,
            max_retries=max_retries,
            telemetry_path=telemetry_path,
        )
        admissible, reasons = abstraction_admissible(draft)
        proposals[view.hypothesis_id] = {
            "hypothesis_id": view.hypothesis_id,
            "direct_context_id": view.direct_context_id,
            "direct_topology_id": view.direct_topology_id,
            "l1_target_id": l1_id,
            "l2_draft": draft.model_dump(mode="json"),
            "l2_admissible": admissible,
            "l2_rejection_reasons": reasons,
            "telemetry_event_id": (
                getattr(event, "event_id", None)
                if event is not None
                else None
            ),
        }

        if admissible:
            l2_relation = clean(draft.proposed_relation)
            l2_queries = dedup_texts(
                [
                    draft.retrieval_query,
                    draft.proposed_relation,
                ]
            )
            l2_id = stable_id(
                "s226_target",
                "L2_INTERMEDIATE",
                norm(l2_relation),
            )
            proposals[view.hypothesis_id][
                "l2_target_id"
            ] = l2_id
            bucket2 = l2_targets.setdefault(
                l2_id,
                {
                    "target_id": l2_id,
                    "level": "L2_INTERMEDIATE",
                    "relation": l2_relation,
                    "queries": l2_queries,
                    "hypothesis_ids": [],
                },
            )
            bucket2["hypothesis_ids"].append(
                view.hypothesis_id
            )

    target_results: dict[str, dict[str, Any]] = {}

    for target_id, target in sorted(
        {**l1_targets, **l2_targets}.items(),
        key=lambda item: (
            item[1]["level"],
            item[0],
        ),
    ):
        plan = make_query_plan(
            target_id=target_id,
            relation=target["relation"],
            queries=target["queries"],
        )
        outcome = retriever.retrieve(plan)
        packet = outcome.packet

        synthetic_claim = plan.claims[0].claims[0]
        ranked = ranker.rank(
            synthetic_claim,
            packet,
            plan,
        )
        works = ranked_work_payloads(
            packet=packet,
            ranked=ranked,
        )

        packet_path = (
            packets_dir
            / (
                target_id.replace(":", "_")
                + ".prior_art.json"
            )
        )
        packet_path.write_text(
            packet.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        draft, event = review_target(
            client=client,
            model=model,
            target=target,
            works=works,
            temperature=temperature,
            max_retries=max_retries,
            telemetry_path=telemetry_path,
        )

        compiled = compile_target_result(
            target=target,
            packet=packet,
            ranked_works=works,
            draft=draft,
            min_confidence=min_match_confidence,
            min_unique_works=min_unique_works,
            min_abstract_works=min_abstract_works,
            min_successful_query_variants=
                min_successful_query_variants,
        )
        compiled["prior_art_packet_path"] = str(packet_path)
        compiled["telemetry_event_id"] = (
            getattr(event, "event_id", None)
            if event is not None
            else None
        )
        target_results[target_id] = compiled

    records = []
    detailed_records = []

    for view in views:
        proposal = proposals[view.hypothesis_id]
        l1 = target_results[
            proposal["l1_target_id"]
        ]

        if proposal["l2_admissible"]:
            l2 = target_results[
                proposal["l2_target_id"]
            ]
        else:
            l2 = {
                "level": "L2_INTERMEDIATE",
                "relation": "",
                "status": "INSUFFICIENT_COVERAGE",
                "reason_code":
                    "ABSTRACTION_REJECTED_BEFORE_RETRIEVAL",
                "coverage": {
                    "sufficient_for_search_bounded_gap": False,
                },
                "review": {
                    "matches": [],
                    "interpretation":
                        "No admissible intermediate abstraction.",
                },
            }

        exact_entry = exact_cards.get(
            view.hypothesis_id
        )
        if exact_entry is None:
            l3 = {
                "status": "INSUFFICIENT_COVERAGE",
                "source_external_status": None,
                "sufficient_coverage": False,
                "material_title_only_matches": [],
                "reason_code":
                    "EXACT_EXTERNAL_CARD_NOT_AVAILABLE",
            }
            exact_report_path = None
        else:
            card_payload, exact_report_path = exact_entry
            l3 = l3_status_from_external_card(
                card_payload,
                min_confidence=min_match_confidence,
            )

        levels = [
            ("L1_BROAD", l1["status"]),
            ("L2_INTERMEDIATE", l2["status"]),
            ("L3_EXACT", l3["status"]),
        ]
        first_gap = first_gap_level(levels)

        level_models = [
            ConceptualKnownnessLevelResult(
                level="L1_BROAD",
                abstraction_text=l1["relation"],
                knownness_class=l1["status"],
                retrieval_status="COMPLETED",
                search_bounded=True,
                sufficient_coverage=bool(
                    l1["coverage"][
                        "sufficient_for_search_bounded_gap"
                    ]
                ),
                source_profile_path=str(
                    case_out
                    / "conceptual_knownness.details.json"
                ),
            ),
            ConceptualKnownnessLevelResult(
                level="L2_INTERMEDIATE",
                abstraction_text=l2.get("relation", ""),
                knownness_class=l2["status"],
                retrieval_status=(
                    "COMPLETED"
                    if proposal["l2_admissible"]
                    else "ABSTRACTION_REJECTED"
                ),
                search_bounded=True,
                sufficient_coverage=bool(
                    (l2.get("coverage") or {}).get(
                        "sufficient_for_search_bounded_gap",
                        False,
                    )
                ),
                source_profile_path=str(
                    case_out
                    / "conceptual_knownness.details.json"
                ),
            ),
            ConceptualKnownnessLevelResult(
                level="L3_EXACT",
                abstraction_text=view.full_higher_order_claim,
                knownness_class=l3["status"],
                retrieval_status=(
                    "REUSED_EXISTING_EXTERNAL_NOVELTY"
                    if exact_entry is not None
                    else "NOT_AVAILABLE"
                ),
                search_bounded=True,
                sufficient_coverage=bool(
                    l3.get("sufficient_coverage", False)
                ),
                source_profile_path=exact_report_path,
            ),
        ]

        records.append(
            DirectHigherOrderConceptualKnownnessRecord(
                hypothesis_id=view.hypothesis_id,
                direct_context_id=view.direct_context_id,
                direct_topology_id=view.direct_topology_id,
                first_gap_level=first_gap,
                levels=level_models,
            )
        )
        detailed_records.append(
            {
                "hypothesis_id": view.hypothesis_id,
                "direct_context_id": view.direct_context_id,
                "direct_topology_id": view.direct_topology_id,
                "full_higher_order_claim":
                    view.full_higher_order_claim,
                "proposal": proposal,
                "L1_BROAD": l1,
                "L2_INTERMEDIATE": l2,
                "L3_EXACT": l3,
                "first_gap_level": first_gap,
            }
        )

    profile = build_conceptual_knownness_profile(
        source_bundle_id=bundle.bundle_id,
        records=records,
    )

    profile_path = (
        case_out / "conceptual_knownness.profile.json"
    )
    details_path = (
        case_out / "conceptual_knownness.details.json"
    )
    write_json(profile_path, profile)
    write_json(
        details_path,
        {
            "schema_version":
                "prospective-conceptual-knownness-validation-s226-v1",
            "source_bundle_id": bundle.bundle_id,
            "case_id": case_id,
            "records": detailed_records,
            "targets": list(target_results.values()),
            "authority": {
                "diagnostic_only": True,
                "search_bounded_only": True,
                "novelty_authority_created": False,
                "production_selection_authority": False,
            },
        },
    )

    return {
        "measurement_status": "AVAILABLE",
        "profile_path": str(profile_path),
        "details_path": str(details_path),
        "record_count": len(records),
        "first_gap_levels": {
            row.hypothesis_id:
                row.first_gap_level
            for row in records
        },
    }


def load_bundle_from_manifest(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
) -> DirectHigherOrderShadowBundle | None:
    info = manifest.get(
        "direct_higher_order_downstream_shadow"
    ) or {}
    raw = info.get("bundle")
    if raw:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            return (
                DirectHigherOrderShadowBundle
                .model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            )

    fallback = (
        run_dir
        / "direct_higher_order_shadow"
        / "downstream"
        / "direct_higher_order.shadow_bundle.json"
    )
    if fallback.is_file():
        return (
            DirectHigherOrderShadowBundle
            .model_validate_json(
                fallback.read_text(encoding="utf-8")
            )
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "S226 post-hoc validation completion over the already-frozen "
            "prospective novelty cohort. No case selection, generation, "
            "or production authority is changed."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("OPENROUTER_AGENT_MODEL")
            or "openai/gpt-5.6-luna"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=(
            os.getenv("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
    )
    parser.add_argument(
        "--providers",
        default="auto",
        help=(
            "auto or comma-separated subset of "
            "openalex,crossref,semantic_scholar"
        ),
    )
    parser.add_argument(
        "--instructor-mode",
        default="JSON",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-ranked-works",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min-match-confidence",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--min-unique-works",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--min-abstract-works",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-successful-query-variants",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve()

    if not plan_path.is_file():
        raise RuntimeError(
            "missing prospective plan: " + str(plan_path)
        )
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(
            "S226 output root must be fresh/nonexistent: "
            + str(out_root)
        )
    out_root.mkdir(parents=True, exist_ok=True)

    plan = ProspectiveNoveltyExecutionPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"missing API key in {args.api_key_env}"
        )

    requested = None
    if args.providers.strip().lower() != "auto":
        requested = [
            row.strip()
            for row in args.providers.split(",")
            if row.strip()
        ]

    provider_plan = resolve_literature_provider_plan(
        requested=requested,
    )
    if requested is None:
        require_standard_or_full_auto_plan(
            provider_plan
        )
    providers = build_literature_providers(
        provider_plan
    )
    retriever = LiteratureRetriever(
        providers,
        results_per_query=args.results_per_query,
    )

    encoder = SentenceTransformerEncoder(
        args.embed_model,
        device=args.device,
    )

    client = get_client(
        api_key=api_key,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        timeout=args.timeout,
    )

    rows = []
    failures = 0

    print("=== S226 VALIDATION COMPLETION ===")
    print("source plan:", plan.plan_id)
    print("provider mode:", provider_plan.mode)
    print("cases:", len(plan.cases))
    print(
        "result-conditioned scientific routing: false"
    )

    for case in plan.cases:
        print("\n" + "=" * 92)
        print(case.source_case_id)
        print("=" * 92)

        run_dir = Path(case.run_dir).expanduser().resolve()
        case_out = out_root / case.source_case_id.lower()
        case_out.mkdir(parents=True, exist_ok=False)

        manifest_path = run_dir / "e2e_runner.manifest.json"
        if not manifest_path.is_file():
            row = {
                "source_case_id": case.source_case_id,
                "run_status": "NOT_RUN",
                "external_retrieval_completion": {
                    "measurement_status": "NOT_AVAILABLE",
                },
                "conceptual_knownness": {
                    "measurement_status": "NOT_RUN",
                },
                "research_value": {
                    "measurement_status": "NOT_RUN",
                },
            }
            rows.append(row)
            write_json(case_out / "case.validation.json", row)
            continue

        manifest = load_json(manifest_path)
        bundle = load_bundle_from_manifest(
            run_dir=run_dir,
            manifest=manifest,
        )

        external = audit_external_case(
            run_dir=run_dir,
            bundle=bundle,
        )

        rv_path = run_dir / "research_value.shadow.json"
        rv = research_value_detail_payload(
            load_json(rv_path)
            if rv_path.is_file()
            else None
        )
        if rv_path.is_file():
            rv["artifact"] = str(rv_path)
        rv["capability_expected"] = (
            case.research_value_capability_expected
        )

        try:
            if bundle is None:
                conceptual = {
                    "measurement_status": (
                        "NOT_RUN_SOURCE_RUN_FAILED"
                        if manifest.get("status") == "failed"
                        else "NOT_APPLICABLE_NO_DIRECT_HO_BUNDLE"
                    ),
                    "record_count": 0,
                    "first_gap_levels": {},
                }
            else:
                ranker = PriorArtRanker(
                    encoder,
                    max_ranked_works_per_claim=
                        args.max_ranked_works,
                    domain_profile=get_domain_profile(
                        case.domain_profile_id
                    ),
                )
                conceptual = run_conceptual_for_case(
                    case_id=case.source_case_id,
                    bundle=bundle,
                    case_out=case_out,
                    retriever=retriever,
                    ranker=ranker,
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    telemetry_path=str(
                        case_out / "conceptual.telemetry.jsonl"
                    ),
                    min_match_confidence=
                        args.min_match_confidence,
                    min_unique_works=args.min_unique_works,
                    min_abstract_works=args.min_abstract_works,
                    min_successful_query_variants=
                        args.min_successful_query_variants,
                )
        except Exception as exc:
            failures += 1
            conceptual = {
                "measurement_status": "ERROR",
                "error": repr(exc),
                "record_count": 0,
                "first_gap_levels": {},
            }
            if not args.continue_on_error:
                raise

        row = {
            "source_case_id": case.source_case_id,
            "prospective_case_id":
                case.prospective_case_id,
            "domain_profile_id":
                case.domain_profile_id,
            "case_role":
                case.case_role,
            "run_status":
                manifest.get("status", "UNKNOWN"),
            "external_retrieval_completion":
                external,
            "conceptual_knownness":
                conceptual,
            "research_value":
                rv,
            "authority": {
                "posthoc_validation_only": True,
                "scientific_case_selection_changed": False,
                "generation_changed": False,
                "novelty_authority_created": False,
                "research_value_selection_authority": False,
                "production_selection_authority": False,
            },
        }
        rows.append(row)
        write_json(case_out / "case.validation.json", row)

        print(
            "run=", row["run_status"],
            "| external=",
            external["overall_disposition"],
            "| conceptual=",
            conceptual["measurement_status"],
            "| RV=",
            rv["measurement_status"],
        )
        if conceptual.get("first_gap_levels"):
            print(
                " first gaps:",
                conceptual["first_gap_levels"],
            )

    matrix = {
        "schema_version":
            "prospective-novelty-validation-completion-s226-v1",
        "source_plan_id": plan.plan_id,
        "source_freeze_id": plan.source_freeze_id,
        "provider_plan":
            provider_plan.model_dump(mode="json"),
        "case_count": len(plan.cases),
        "cases": rows,
        "validation_failures": failures,
        "interpretation_policy": {
            "posthoc_measurement_changes_original_prospective_result":
                False,
            "missing_measurement_is_negative": False,
            "title_only_neighbor_is_direct_prior_art": False,
            "material_title_only_allows_search_bounded_gap": False,
            "retrieval_recall_claimed_complete": False,
            "overall_winner_computed": False,
            "production_selection_changed": False,
        },
    }
    matrix_path = out_root / "validation.matrix.json"
    write_json(matrix_path, matrix)

    print("\nS226 validation completion finished")
    print("conceptual failures:", failures)
    print("original prospective results mutated: false")
    print("overall winner computed: false")
    print("production selection changed: false")
    print("artifact:", matrix_path)

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
