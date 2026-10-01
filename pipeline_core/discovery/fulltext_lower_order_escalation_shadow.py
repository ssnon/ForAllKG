
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.literature.acquisition.access_contracts import SourceAcquisitionPolicy
from pipeline_core.literature.acquisition.artifact_acquisition import MainArtifactDownloader
from pipeline_core.literature.acquisition.materialization_contracts import MaterializationPolicy
from pipeline_core.literature.acquisition.materialization_package import materialize_artifact, stable_paper_id
from pipeline_core.literature.acquisition.oa_resolution import OpenAccessResolver
from pipeline_core.literature.catalog_contracts import CatalogWork
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FullTextExactReviewDraft(_StrictModel):
    relationship: Literal[
        "DIRECT_PRIOR_ART",
        "PARTIAL_PRIOR_ART",
        "NOT_RELATION_BACKED",
        "INSUFFICIENT_FULLTEXT_EVIDENCE",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_spans: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


_FULLTEXT_SYSTEM = """
You perform bounded FULL-TEXT exact verification for one already-retrieved
scientific prior-art candidate and one atomic scientific claim.

This is NOT a new literature search and NOT a literature-wide novelty judgment.

DIRECT_PRIOR_ART requires materially relevant system/structural identity and
the exact atomic relation/direction to be explicitly supported by the supplied
full-text excerpts.

PARTIAL_PRIOR_ART requires substantial system/structural identity and the same
atomic relation, with at most one bounded scope/detail omission that does not
replace a constitutive material, structural carrier, endpoint, or relation.

NOT_RELATION_BACKED means relevant components or neighboring science may be
present but the exact atomic relation is not established.

INSUFFICIENT_FULLTEXT_EVIDENCE means the supplied extraction is inadequate to
decide responsibly.

Do not create a relation by combining independent statements unless the text
itself explicitly links them. Do not treat generic analogues from a different
architecture/material system as DIRECT or PARTIAL merely because relation
vocabulary is similar.

For DIRECT_PRIOR_ART or PARTIAL_PRIOR_ART, return 1-3 exact contiguous
evidence_spans copied verbatim from FULL-TEXT EXCERPTS. If no exact supporting
span is available, do not return DIRECT/PARTIAL.

The work is prior-art evidence only and never becomes a positive premise.
"""


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐‑‒–—−_/]+", " ", text)
    text = re.sub(r"[^\w\s+*.]", " ", text)
    return " ".join(text.split())


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+*.]{1,}", _norm(value))
        if len(token) >= 3
    }


def _claim_map(query_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(claim.get("claim_id")): claim
        for group in query_plan.get("claims", [])
        for claim in group.get("claims", [])
        if claim.get("claim_id")
    }


def _review_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("reviews")
    if not isinstance(rows, list):
        rows = payload.get("claim_reviews")
    if not isinstance(rows, list):
        rows = []
    return {
        str(row.get("claim_id")): row
        for row in rows
        if row.get("claim_id")
    }


def escalation_target_claim_ids(topology_report: dict[str, Any]) -> list[str]:
    topology = topology_report.get("topology_residual", topology_report)
    result: list[str] = []
    seen: set[str] = set()
    for row in topology.get("composites", []):
        if row.get("topology_state") != "EXPLICIT":
            continue
        if row.get("full_relation_status") in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}:
            continue
        if row.get("residual_state") not in {"UNSATURATED_BASE", "PARTIAL_BASE_SATURATION"}:
            continue
        components = [
            str(x)
            for x in row.get("effective_component_claim_ids_shadow", [])
            if str(x)
        ]
        backed = {
            str(x)
            for x in row.get("relation_backed_component_claim_ids", [])
            if str(x)
        }
        for claim_id in components:
            if claim_id not in backed and claim_id not in seen:
                seen.add(claim_id)
                result.append(claim_id)
    return result


def _candidate_score(
    claim: dict[str, Any],
    work: dict[str, Any],
    match: dict[str, Any] | None,
) -> tuple[float, float, int, int, str]:
    reviewed = 1.0 if match else 0.0
    match_score = (
        float(match.get("confidence") or 0.0)
        * max(float(match.get("relevance_score") or 0.0), 0.001)
        if match
        else 0.0
    )
    claim_tokens = _tokens(
        " ".join(
            [
                str(claim.get("text") or ""),
                *map(str, claim.get("prior_art_identity_terms", [])),
                *map(str, claim.get("relation_nucleus_terms", [])),
            ]
        )
    )
    work_tokens = _tokens(
        f"{work.get('title', '')} {work.get('abstract', '')}"
    )
    overlap = len(claim_tokens & work_tokens)
    access_hint = int(
        bool(work.get("open_access_url"))
        or bool(work.get("doi"))
        or bool((work.get("provider_ids") or {}).get("openalex"))
    )
    combined = match_score + min(overlap, 12) * 0.06
    return (
        combined,
        float(overlap),
        access_hint,
        reviewed,
        str(work.get("work_id") or ""),
    )


def select_existing_candidates(
    *,
    claim_id: str,
    claim: dict[str, Any],
    review: dict[str, Any] | None,
    packet: dict[str, Any],
    max_candidates: int,
) -> list[dict[str, Any]]:
    works = {
        str(work.get("work_id")): work
        for work in packet.get("works", [])
        if work.get("work_id")
    }
    matches: dict[str, dict[str, Any]] = {}
    if review:
        for match in review.get("matches", []):
            wid = str(match.get("work_id") or "")
            if not wid:
                continue
            if match.get("relationship") in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}:
                continue
            matches[wid] = match

    ids = set(matches)
    for work in packet.get("works", []):
        if claim_id in {str(x) for x in work.get("retrieval_claim_ids", [])}:
            wid = str(work.get("work_id") or "")
            if wid:
                ids.add(wid)

    rows = [works[wid] for wid in ids if wid in works]
    rows.sort(
        key=lambda work: _candidate_score(
            claim,
            work,
            matches.get(str(work.get("work_id") or "")),
        ),
        reverse=True,
    )
    return rows[: max(1, int(max_candidates))]


def _catalog_work(work: dict[str, Any]) -> CatalogWork:
    return CatalogWork(
        work_id=str(work.get("work_id") or ""),
        title=str(work.get("title") or ""),
        year=work.get("year"),
        publication_date=work.get("publication_date"),
        doi=work.get("doi"),
        url=work.get("url"),
        open_access_url=work.get("open_access_url"),
        abstract=work.get("abstract"),
        authors=list(work.get("authors") or []),
        venue=work.get("venue"),
        citation_count=work.get("citation_count"),
        providers=list(work.get("providers") or []),
        provider_ids=dict(work.get("provider_ids") or {}),
        retrieval_query_ids=list(work.get("retrieval_query_ids") or []),
        retrieval_axis_ids=[],
    )


def _fallback_pdf_text(path: Path) -> tuple[str, str]:
    # Text extraction only; OCR is intentionally not used.
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n\n".join(str(page.extract_text() or "") for page in reader.pages).strip()
        if text:
            return text, "pypdf_text_fallback"
    except Exception:
        pass

    try:
        import fitz
        document = fitz.open(str(path))
        try:
            text = "\n\n".join(page.get_text("text") for page in document).strip()
        finally:
            document.close()
        if text:
            return text, "pymupdf_text_fallback"
    except Exception:
        pass

    executable = shutil.which("pdftotext")
    if executable:
        target = path.with_suffix(".fallback.txt")
        try:
            completed = subprocess.run(
                [executable, "-layout", str(path), str(target)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
            if completed.returncode == 0 and target.exists():
                text = target.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return text, "pdftotext_fallback"
        finally:
            if target.exists():
                target.unlink()
    return "", "no_text_extractor_available"


def _extract_fulltext(work: CatalogWork, output_root: Path) -> dict[str, Any]:
    source_policy = SourceAcquisitionPolicy(
        policy_id="lower-order-fulltext-escalation-shadow-v1",
        use_unpaywall=True,
        use_openalex=True,
        openalex_require_api_key=False,
        use_pmc_aws=True,
        use_catalog_open_access_url=True,
        auto_download_main=True,
        try_all_direct_pdf_locations=True,
    )
    resolution = OpenAccessResolver(source_policy).resolve(work)
    artifact = MainArtifactDownloader(source_policy).acquire(
        work=work,
        resolution=resolution,
        output_root=output_root,
    )
    result = {
        "resolution": resolution.model_dump(mode="json"),
        "artifact": artifact.model_dump(mode="json"),
        "text": "",
        "text_source": None,
        "materialization": None,
    }
    if artifact.status != "downloaded" or not artifact.local_path:
        return result

    policy = MaterializationPolicy(
        policy_id="lower-order-fulltext-escalation-materialization-v1"
    )
    paper_id = stable_paper_id(prefix="ftshadow", work_id=work.work_id)
    materialized = materialize_artifact(
        materialization_id="lower_order_fulltext_escalation_shadow_v1",
        paper_id=paper_id,
        work=work,
        document_id=f"{paper_id}_main",
        role="main",
        artifact=artifact,
        package_dir=output_root / "materialized" / paper_id / "main",
        policy=policy,
        project_root=Path.cwd(),
    )
    result["materialization"] = materialized.model_dump(mode="json")

    if materialized.status == "materialized" and materialized.markdown_path:
        path = Path(materialized.markdown_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                result["text"] = text
                result["text_source"] = materialized.materializer
                return result

    text, source = _fallback_pdf_text(Path(artifact.local_path))
    result["text"] = text
    result["text_source"] = source
    return result


def _excerpt(fulltext: str, claim: dict[str, Any], max_chars: int) -> str:
    if len(fulltext) <= max_chars:
        return fulltext

    query_tokens = _tokens(
        " ".join(
            [
                str(claim.get("text") or ""),
                *map(str, claim.get("prior_art_identity_terms", [])),
                *map(str, claim.get("relation_nucleus_terms", [])),
            ]
        )
    )
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n+", fulltext)
        if len(p.strip()) >= 40
    ]
    scored = [
        (len(query_tokens & _tokens(paragraph)), index)
        for index, paragraph in enumerate(paragraphs)
    ]
    scored = [row for row in scored if row[0] > 0]
    scored.sort(reverse=True)

    if not scored:
        return fulltext[:max_chars]

    selected: set[int] = set()
    for _, index in scored[:20]:
        for neighbor in (index - 1, index, index + 1):
            if 0 <= neighbor < len(paragraphs):
                selected.add(neighbor)

    chunks: list[str] = []
    total = 0
    for index in sorted(selected):
        paragraph = paragraphs[index]
        if chunks and total + len(paragraph) + 8 > max_chars:
            break
        chunks.append(paragraph)
        total += len(paragraph) + 8
    return "\n\n---\n\n".join(chunks)[:max_chars]


def _instructor_client(api_key_env: str, base_url: str | None):
    import instructor
    from openai import OpenAI

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key environment variable: {api_key_env}")
    return instructor.from_openai(
        OpenAI(api_key=api_key, base_url=base_url or None, timeout=180.0),
        mode=instructor.Mode.JSON,
    )


def _review_candidate(
    *,
    model: str,
    api_key_env: str,
    base_url: str | None,
    claim: dict[str, Any],
    work: dict[str, Any],
    excerpt: str,
    parse_retries: int,
) -> FullTextExactReviewDraft:
    user = "\n".join(
        [
            "ATOMIC CLAIM",
            "============",
            f"claim_id: {claim.get('claim_id')}",
            f"text: {claim.get('text')}",
            f"prior_art_identity_terms: {claim.get('prior_art_identity_terms', [])!r}",
            f"relation_nucleus_terms: {claim.get('relation_nucleus_terms', [])!r}",
            "",
            "WORK",
            "====",
            f"work_id: {work.get('work_id')}",
            f"title: {work.get('title')}",
            f"doi: {work.get('doi')}",
            "",
            "FULL-TEXT EXCERPTS",
            "==================",
            excerpt,
        ]
    )
    result, _event = run_instructor_structured_call(
        _instructor_client(api_key_env, base_url).chat.completions,
        model=model,
        response_model=FullTextExactReviewDraft,
        messages=[
            {"role": "system", "content": _FULLTEXT_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_retries=parse_retries,
        telemetry_path=None,
        telemetry_context={
            "pipeline": "lower_order_fulltext_escalation_shadow",
            "stage": "fulltext_exact_review",
            "claim_id": claim.get("claim_id"),
            "work_id": work.get("work_id"),
        },
    )
    if not isinstance(result, FullTextExactReviewDraft):
        result = FullTextExactReviewDraft.model_validate(result)
    return result


def run_escalation(
    *,
    query_plan: dict[str, Any],
    claim_reviews: dict[str, Any],
    packet: dict[str, Any],
    topology_report: dict[str, Any],
    output_root: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    max_candidates_per_claim: int = 4,
    max_excerpt_chars: int = 24000,
    parse_retries: int = 3,
) -> dict[str, Any]:
    claims = _claim_map(query_plan)
    reviews = _review_map(claim_reviews)
    targets = escalation_target_claim_ids(topology_report)
    target_rows: list[dict[str, Any]] = []

    for claim_id in targets:
        claim = claims.get(claim_id)
        if claim is None:
            target_rows.append(
                {
                    "claim_id": claim_id,
                    "status": "CLAIM_NOT_FOUND",
                    "candidate_count": 0,
                    "reviews": [],
                    "fulltext_relation_backed_work_ids": [],
                }
            )
            continue

        candidates = select_existing_candidates(
            claim_id=claim_id,
            claim=claim,
            review=reviews.get(claim_id),
            packet=packet,
            max_candidates=max_candidates_per_claim,
        )
        reviews_out: list[dict[str, Any]] = []
        backed_ids: list[str] = []

        for work in candidates:
            catalog = _catalog_work(work)
            acquired = _extract_fulltext(catalog, output_root)
            fulltext = str(acquired.get("text") or "")
            row = {
                "work_id": catalog.work_id,
                "title": catalog.title,
                "doi": catalog.doi,
                "access_status": acquired["resolution"].get("status"),
                "artifact_status": acquired["artifact"].get("status"),
                "text_source": acquired.get("text_source"),
                "relationship": "INSUFFICIENT_FULLTEXT_EVIDENCE",
                "confidence": 0.0,
                "evidence_spans": [],
                "rationale": "full text was not available or extractable",
            }

            if fulltext:
                excerpt = _excerpt(fulltext, claim, max_excerpt_chars)
                draft = _review_candidate(
                    model=model,
                    api_key_env=api_key_env,
                    base_url=base_url,
                    claim=claim,
                    work=work,
                    excerpt=excerpt,
                    parse_retries=parse_retries,
                )
                valid_spans = [
                    span for span in draft.evidence_spans if span and span in excerpt
                ]
                relationship = draft.relationship
                if relationship in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"} and not valid_spans:
                    relationship = "INSUFFICIENT_FULLTEXT_EVIDENCE"
                row.update(
                    {
                        "relationship": relationship,
                        "confidence": draft.confidence,
                        "evidence_spans": valid_spans,
                        "rationale": draft.rationale,
                        "excerpt_sha256": hashlib.sha256(
                            excerpt.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                if relationship in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}:
                    backed_ids.append(catalog.work_id)

            reviews_out.append(row)

        if backed_ids:
            status = "FULLTEXT_RELATION_BACKED"
        elif any(
            row.get("artifact_status") == "downloaded" and row.get("text_source")
            for row in reviews_out
        ):
            status = "FULLTEXT_REVIEWED_NOT_BACKED"
        else:
            status = "FULLTEXT_UNAVAILABLE"

        target_rows.append(
            {
                "claim_id": claim_id,
                "claim_text": claim.get("text"),
                "status": status,
                "candidate_count": len(candidates),
                "reviews": reviews_out,
                "fulltext_relation_backed_work_ids": sorted(set(backed_ids)),
            }
        )

    return {
        "schema_version": "lower-order-fulltext-escalation-shadow-v1",
        "shadow_only": True,
        "new_search_performed": False,
        "positive_premise_authority": False,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
        "target_claim_count": len(target_rows),
        "fulltext_relation_backed_target_count": sum(
            row["status"] == "FULLTEXT_RELATION_BACKED"
            for row in target_rows
        ),
        "targets": target_rows,
    }
