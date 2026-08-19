from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dac_her.domains import get_domain_profile
from dac_her.external_novelty_contracts import (
    ExternalNoveltyPolicy,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.node_mapping import DEFAULT_EMBED_MODEL, SentenceTransformerEncoder
from dac_her.prior_art_matching import PriorArtRanker


SEMANTICS_ID = "sers_standard2_ranker_only_dev_v1"
DOMAIN_PROFILE_ID = "sers_au_ag"
DEVICE = "cpu"

DEFAULT_DIAGNOSTIC_ROOT = Path.home() / "GraphAgentsDAC"
SOURCE_QUERY_PLAN = Path(
    "evaluation/sers_alpha4c5k/dev_e2e_v2/"
    "external_novelty.claims_queries.json"
)
SOURCE_CANONICAL_PACKET = Path(
    "evaluation/sers_provider_reliability/"
    "canonicalization_only_dev_recheck_v2/"
    "canonical_prior_art_v2.json"
)
SOURCE_CANONICAL_RECHECK = Path(
    "evaluation/sers_provider_reliability/"
    "canonicalization_only_dev_recheck_v2/"
    "RECHECK_PASS.json"
)

DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "ranker_only_dev_spec_v1"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "ranker_only_dev_run_v1"
)

EXPECTED_CANONICAL_RECHECK_RUN_ID = (
    "sers_canonicalization_doi_conflict_recheck:"
    "4bf58f736ebd42192c8b"
)
EXPECTED_CANONICAL_WORK_COUNT = 430
EXPECTED_CLAIM_COUNT = 12

SOURCE_FILES_TO_FREEZE = (
    Path("dac_her/prior_art_matching.py"),
    Path("dac_her/node_mapping.py"),
    Path("dac_her/external_novelty_contracts.py"),
    Path("dac_her/domain_profile.py"),
    Path("dac_her/domains/registry.py"),
    Path("dac_her/domains/sers_au_ag.py"),
)

MODEL_SENTINELS = (
    "SERS Au Ag plasmonic nanogap electromagnetic enhancement",
    "surface enhanced Raman scattering gold silver bimetallic substrate",
    "charge transfer and localized surface plasmon resonance",
)


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def force_offline_model_mode() -> None:
    # Hugging Face / transformers honor these environment controls.
    # The validation intentionally fails if the already-used embedding model
    # is not locally available; it never downloads a model during this stage.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "sentence-transformers",
        "transformers",
        "torch",
    )
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def _source_hashes(repo_root: Path) -> dict[str, str]:
    result = {}
    for rel in SOURCE_FILES_TO_FREEZE:
        path = repo_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(rel)] = sha256_file(path)
    return result


def _load_encoder(
    model_name: str,
    *,
    device: str = DEVICE,
) -> SentenceTransformerEncoder:
    force_offline_model_mode()
    try:
        return SentenceTransformerEncoder(
            model_name,
            device=device,
        )
    except Exception as exc:
        raise RuntimeError(
            "Embedding model could not be loaded in strict offline mode. "
            "No model download is authorized in ranker-only validation. "
            f"model={model_name!r}; "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def model_behavior_fingerprint(
    encoder: SentenceTransformerEncoder,
) -> dict[str, Any]:
    q = np.asarray(
        encoder.encode_query(MODEL_SENTINELS[0]),
        dtype=np.float32,
    )
    docs = np.asarray(
        encoder.encode_documents(
            list(MODEL_SENTINELS[1:]),
            batch_size=2,
        ),
        dtype=np.float32,
    )
    if q.ndim != 1 or docs.ndim != 2:
        raise RuntimeError(
            f"Unexpected encoder sentinel shapes: q={q.shape}, docs={docs.shape}"
        )
    digest = hashlib.sha256()
    digest.update(q.tobytes(order="C"))
    digest.update(docs.tobytes(order="C"))
    return {
        "query_dimension": int(q.shape[0]),
        "document_shape": [int(x) for x in docs.shape],
        "float_dtype": str(q.dtype),
        "sentinel_sha256": digest.hexdigest(),
    }


class CachingEncoder:
    """Exact-input cache around the frozen production encoder.

    This changes no vectors or ranking semantics. It only avoids recomputing
    the same claim/document encodings during deterministic validation passes.
    """

    def __init__(self, base: SentenceTransformerEncoder) -> None:
        self.base = base
        self._query: dict[str, np.ndarray] = {}
        self._docs: dict[tuple[str, ...], np.ndarray] = {}

    def encode_query(self, text: str) -> np.ndarray:
        if text not in self._query:
            self._query[text] = np.asarray(
                self.base.encode_query(text),
                dtype=np.float32,
            )
        return self._query[text].copy()

    def encode_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        key = tuple(texts)
        if key not in self._docs:
            self._docs[key] = np.asarray(
                self.base.encode_documents(
                    texts,
                    batch_size=batch_size,
                ),
                dtype=np.float32,
            )
        return self._docs[key].copy()


def _load_and_validate_inputs(
    *,
    repo_root: Path,
    diagnostic_root: Path,
) -> tuple[LiteratureQueryPlan, PriorArtPacket, dict[str, Any]]:
    query_path = diagnostic_root / SOURCE_QUERY_PLAN
    packet_path = repo_root / SOURCE_CANONICAL_PACKET
    marker_path = repo_root / SOURCE_CANONICAL_RECHECK

    if not query_path.is_file():
        raise FileNotFoundError(query_path)
    if not packet_path.is_file():
        raise FileNotFoundError(packet_path)
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)

    marker = read_json(marker_path)
    if marker.get("status") != "recheck_pass":
        raise ValueError("Canonicalization v2 RECHECK_PASS status missing.")
    if marker.get("run_id") != EXPECTED_CANONICAL_RECHECK_RUN_ID:
        raise ValueError(
            "Canonicalization v2 recheck run ID mismatch."
        )
    if marker.get(
        "canonical_packet_eligible_for_dev_ranker_validation"
    ) is not True:
        raise ValueError(
            "Canonical packet is not marked ranker-eligible."
        )

    plan = LiteratureQueryPlan.model_validate_json(
        query_path.read_text(encoding="utf-8")
    )
    packet = PriorArtPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )

    if packet.source_query_plan_id != plan.plan_id:
        raise ValueError(
            "Canonical packet/query-plan lineage mismatch."
        )
    if packet.canonical_work_count != EXPECTED_CANONICAL_WORK_COUNT:
        raise ValueError(
            "Expected canonical_work_count "
            f"{EXPECTED_CANONICAL_WORK_COUNT}, observed "
            f"{packet.canonical_work_count}."
        )

    claim_count = sum(
        len(row.claims) for row in plan.claims
    )
    if claim_count != EXPECTED_CLAIM_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLAIM_COUNT} claims, observed {claim_count}."
        )

    packet_body = packet.model_dump(mode="json")
    packet_sha = packet_body.pop("packet_sha256")
    if sha256_json(packet_body) != packet_sha:
        raise ValueError("Canonical packet internal SHA mismatch.")
    if marker.get("canonical_packet_id") != packet.packet_id:
        raise ValueError("RECHECK_PASS packet ID mismatch.")
    if marker.get("canonical_packet_sha256") != packet.packet_sha256:
        raise ValueError("RECHECK_PASS packet SHA mismatch.")

    return (
        plan,
        packet,
        {
            "query_plan_file_sha256": sha256_file(query_path),
            "canonical_packet_file_sha256": sha256_file(packet_path),
            "canonical_recheck_marker": marker,
        },
    )


def build_spec(
    *,
    repo_root: Path,
    diagnostic_root: Path,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> dict[str, Any]:
    plan, packet, input_meta = _load_and_validate_inputs(
        repo_root=repo_root,
        diagnostic_root=diagnostic_root,
    )

    domain_profile = get_domain_profile(DOMAIN_PROFILE_ID)
    if domain_profile.profile_id != DOMAIN_PROFILE_ID:
        raise ValueError("SERS domain-profile resolution mismatch.")

    encoder = _load_encoder(
        embed_model,
        device=DEVICE,
    )
    fingerprint = model_behavior_fingerprint(encoder)
    policy = ExternalNoveltyPolicy()

    claims = [
        {
            "hypothesis_id": claim.hypothesis_id,
            "claim_id": claim.claim_id,
            "claim_rank": claim.claim_rank,
            "importance": claim.importance,
            "kind": claim.kind,
            "text_sha256": hashlib.sha256(
                claim.text.encode("utf-8")
            ).hexdigest(),
        }
        for group in plan.claims
        for claim in group.claims
    ]

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-ranker-only-dev-spec-v1",
        "semantics_id": SEMANTICS_ID,
        "source_query_plan_id": plan.plan_id,
        "source_query_plan_sha256": plan.plan_sha256,
        "source_query_plan_file_sha256":
            input_meta["query_plan_file_sha256"],
        "source_canonical_packet_id": packet.packet_id,
        "source_canonical_packet_sha256":
            packet.packet_sha256,
        "source_canonical_packet_file_sha256":
            input_meta["canonical_packet_file_sha256"],
        "source_canonical_recheck_run_id":
            input_meta["canonical_recheck_marker"]["run_id"],
        "source_canonical_recheck_run_sha256":
            input_meta["canonical_recheck_marker"]["run_sha256"],
        "canonical_work_count": len(packet.works),
        "claim_count": len(claims),
        "core_claim_count": sum(
            row["importance"] == "core"
            for row in claims
        ),
        "claims": claims,
        "ranker": {
            "class":
                "dac_her.prior_art_matching.PriorArtRanker",
            "domain_profile_id": domain_profile.profile_id,
            "max_ranked_works_per_claim":
                policy.max_ranked_works_per_claim,
            "min_abstract_works_per_core_claim":
                policy.min_abstract_works_per_core_claim,
            "embed_model": embed_model,
            "device": DEVICE,
            "model_behavior_fingerprint": fingerprint,
            "package_versions": _package_versions(),
        },
        "source_hashes": _source_hashes(repo_root),
        "validation_policy": {
            "require_all_claims_have_candidates": True,
            "require_exact_topn_determinism": True,
            "require_production_topn_equals_full_ranking_prefix": True,
            "require_unique_topn_work_ids": True,
            "require_core_claim_topn_minimum_abstract_coverage": True,
        },
        "epistemic_policy": {
            "mechanical_ranker_validation_only": True,
            "scientific_relevance_ground_truth_available": False,
            "scientific_relevance_status":
                "MANUAL_REVIEW_REQUIRED",
            "llm_calls": 0,
            "network_calls": 0,
            "claim_review_used": False,
            "novelty_verdict_used": False,
            "automatic_claim_level_review_authorized": False,
            "fresh_reserve_consumed": False,
        },
    }
    body["spec_sha256"] = sha256_json(body)
    body["spec_id"] = (
        "sers_standard2_ranker_only_dev_spec:"
        + body["spec_sha256"][:20]
    )
    return body


def verify_spec(
    *,
    repo_root: Path,
    diagnostic_root: Path,
    spec_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not spec_path.is_file():
        return ["spec missing"], {}

    stored = read_json(spec_path)
    issues: list[str] = []

    body = dict(stored)
    spec_id = body.pop("spec_id", None)
    spec_sha = body.pop("spec_sha256", None)
    observed = sha256_json(body)
    if spec_sha != observed:
        issues.append("spec SHA mismatch")
    if spec_id != (
        "sers_standard2_ranker_only_dev_spec:"
        + observed[:20]
    ):
        issues.append("spec ID mismatch")

    try:
        recomputed = build_spec(
            repo_root=repo_root,
            diagnostic_root=diagnostic_root,
            embed_model=str(stored["ranker"]["embed_model"]),
        )
        if canonical_json(recomputed) != canonical_json(stored):
            issues.append(
                "deterministic spec/model/source recomputation mismatch"
            )
    except Exception as exc:
        issues.append(
            f"spec recomputation failed: {type(exc).__name__}: {exc}"
        )

    return sorted(set(issues)), stored


def _ranking_rows(
    ranked: Any,
    packet_index: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for position, item in enumerate(
        ranked.ranked_works,
        start=1,
    ):
        work = packet_index[item.work_id]
        abstract = work.abstract or ""
        excerpt = " ".join(abstract.split())[:700]
        rows.append(
            {
                "rank": position,
                "work_id": item.work_id,
                "title": work.title,
                "year": work.year,
                "doi": work.doi,
                "providers": list(work.providers),
                "abstract_available": bool(work.abstract),
                "abstract_excerpt": excerpt,
                "relevance_score": item.relevance_score,
                "semantic_similarity": item.semantic_similarity,
                "lexical_coverage": item.lexical_coverage,
                "reaction_domain_relevance":
                    item.reaction_domain_relevance,
                "catalyst_scope_relevance":
                    item.catalyst_scope_relevance,
                "retrieval_query_ids":
                    list(work.retrieval_query_ids),
                "retrieval_claim_ids":
                    list(work.retrieval_claim_ids),
            }
        )
    return rows


def run_ranker_validation(
    *,
    repo_root: Path,
    diagnostic_root: Path,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    plan, packet, input_meta = _load_and_validate_inputs(
        repo_root=repo_root,
        diagnostic_root=diagnostic_root,
    )

    if input_meta["query_plan_file_sha256"] != (
        spec["source_query_plan_file_sha256"]
    ):
        raise RuntimeError("Frozen query-plan file SHA drift.")
    if input_meta["canonical_packet_file_sha256"] != (
        spec["source_canonical_packet_file_sha256"]
    ):
        raise RuntimeError("Frozen canonical-packet file SHA drift.")
    if _source_hashes(repo_root) != spec["source_hashes"]:
        raise RuntimeError("Frozen ranker/domain source hash drift.")

    embed_model = str(spec["ranker"]["embed_model"])
    base_encoder = _load_encoder(
        embed_model,
        device=str(spec["ranker"]["device"]),
    )
    observed_fingerprint = model_behavior_fingerprint(
        base_encoder
    )
    if observed_fingerprint != (
        spec["ranker"]["model_behavior_fingerprint"]
    ):
        raise RuntimeError(
            "Frozen embedding-model behavior fingerprint drift."
        )

    encoder = CachingEncoder(base_encoder)
    profile = get_domain_profile(
        str(spec["ranker"]["domain_profile_id"])
    )
    topn = int(
        spec["ranker"]["max_ranked_works_per_claim"]
    )
    minimum_core_abstracts = int(
        spec["ranker"]["min_abstract_works_per_core_claim"]
    )

    ranker = PriorArtRanker(
        encoder,
        max_ranked_works_per_claim=topn,
        domain_profile=profile,
    )
    full_ranker = PriorArtRanker(
        encoder,
        max_ranked_works_per_claim=max(
            len(packet.works),
            topn,
        ),
        domain_profile=profile,
    )

    packet_index = {
        row.work_id: row
        for row in packet.works
    }

    claim_reports = []
    mechanical_checks = []
    for group in plan.claims:
        for claim in group.claims:
            top_first = ranker.rank(
                claim,
                packet,
                plan,
            )
            top_second = ranker.rank(
                claim,
                packet,
                plan,
            )
            full = full_ranker.rank(
                claim,
                packet,
                plan,
            )

            first_json = canonical_json(top_first)
            second_json = canonical_json(top_second)
            deterministic = first_json == second_json

            top_ids = [
                row.work_id
                for row in top_first.ranked_works
            ]
            full_ids = [
                row.work_id
                for row in full.ranked_works
            ]
            prefix_match = (
                top_ids
                == full_ids[: len(top_ids)]
            )
            unique_ids = (
                len(top_ids)
                == len(set(top_ids))
            )
            all_known = all(
                work_id in packet_index
                for work_id in top_ids
            )

            abstract_count = sum(
                bool(packet_index[work_id].abstract)
                for work_id in top_ids
            )
            has_candidates = bool(full_ids)
            core_abstract_pass = (
                claim.importance != "core"
                or abstract_count >= minimum_core_abstracts
            )

            rows = _ranking_rows(
                top_first,
                packet_index,
            )
            candidate_abstract_count = sum(
                bool(packet_index[work_id].abstract)
                for work_id in full_ids
            )

            checks = {
                "has_candidates": has_candidates,
                "exact_topn_deterministic": deterministic,
                "production_topn_equals_full_ranking_prefix":
                    prefix_match,
                "unique_topn_work_ids": unique_ids,
                "all_topn_work_ids_known": all_known,
                "core_topn_minimum_abstract_coverage":
                    core_abstract_pass,
            }
            mechanical_checks.append(
                all(checks.values())
            )

            scores = [
                row["relevance_score"]
                for row in rows
            ]
            claim_reports.append(
                {
                    "hypothesis_id": claim.hypothesis_id,
                    "claim_id": claim.claim_id,
                    "claim_rank": claim.claim_rank,
                    "importance": claim.importance,
                    "kind": claim.kind,
                    "claim_text": claim.text,
                    "candidate_pool_count": len(full_ids),
                    "candidate_pool_abstract_count":
                        candidate_abstract_count,
                    "topn_count": len(top_ids),
                    "topn_abstract_count": abstract_count,
                    "topn_abstract_fraction": (
                        abstract_count / len(top_ids)
                        if top_ids
                        else 0.0
                    ),
                    "top1_relevance_score":
                        scores[0] if scores else None,
                    "topn_min_relevance_score":
                        min(scores) if scores else None,
                    "topn_mean_relevance_score": (
                        sum(scores) / len(scores)
                        if scores
                        else None
                    ),
                    "checks": checks,
                    "scientific_relevance_status":
                        "MANUAL_REVIEW_REQUIRED",
                    "top_ranked_works": rows,
                }
            )

    core_rows = [
        row for row in claim_reports
        if row["importance"] == "core"
    ]
    all_core_coverage = all(
        row["checks"][
            "core_topn_minimum_abstract_coverage"
        ]
        for row in core_rows
    )

    mechanical_pass = (
        bool(claim_reports)
        and len(claim_reports) == EXPECTED_CLAIM_COUNT
        and all(mechanical_checks)
        and all_core_coverage
    )

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-ranker-only-dev-run-v1",
        "semantics_id": SEMANTICS_ID,
        "source_spec_id": spec["spec_id"],
        "source_spec_sha256": spec["spec_sha256"],
        "source_query_plan_id": plan.plan_id,
        "source_canonical_packet_id": packet.packet_id,
        "source_canonical_packet_sha256":
            packet.packet_sha256,
        "mechanical_outcome": (
            "RANKER_MECHANICAL_DEV_PASS"
            if mechanical_pass
            else "RANKER_MECHANICAL_DEV_FAIL"
        ),
        "scientific_relevance_outcome":
            "MANUAL_REVIEW_REQUIRED",
        "summary": {
            "claim_count": len(claim_reports),
            "core_claim_count": len(core_rows),
            "claims_with_candidates": sum(
                row["candidate_pool_count"] > 0
                for row in claim_reports
            ),
            "core_claims_meeting_min_topn_abstract_coverage":
                sum(
                    row["checks"][
                        "core_topn_minimum_abstract_coverage"
                    ]
                    for row in core_rows
                ),
            "minimum_core_topn_abstract_requirement":
                minimum_core_abstracts,
            "topn": topn,
            "canonical_work_count":
                len(packet.works),
        },
        "claim_reports": claim_reports,
        "ranker_source_hashes": spec["source_hashes"],
        "embed_model": embed_model,
        "model_behavior_fingerprint":
            observed_fingerprint,
        "domain_profile_id": profile.profile_id,
        "network_calls": 0,
        "llm_calls": 0,
        "claim_review_used": False,
        "novelty_verdict_used": False,
        "fresh_reserve_consumed": False,
        "automatic_claim_level_review_authorized": False,
        "human_relevance_review_required": True,
    }
    body["run_sha256"] = sha256_json(body)
    body["run_id"] = (
        "sers_standard2_ranker_only_dev_run:"
        + body["run_sha256"][:20]
    )

    md = render_human_audit_markdown(body)
    return body, md


def render_human_audit_markdown(
    report: Mapping[str, Any],
) -> str:
    lines = [
        "# SERS Ranker-only DEV Human Relevance Audit",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Mechanical outcome: **{report['mechanical_outcome']}**",
        "- Scientific relevance outcome: **MANUAL_REVIEW_REQUIRED**",
        f"- Domain profile: `{report['domain_profile_id']}`",
        f"- Embedding model: `{report['embed_model']}`",
        "- LLM calls: `0`",
        "- Network calls: `0`",
        "",
        (
            "This artifact does not assign prior-art relationships or novelty "
            "statuses. It exposes the production ranker's top candidates for "
            "human relevance inspection."
        ),
        "",
    ]

    for index, row in enumerate(
        report["claim_reports"],
        start=1,
    ):
        lines.extend(
            [
                f"## Claim {index} — {row['importance']} / {row['kind']}",
                "",
                row["claim_text"],
                "",
                (
                    f"Candidate pool: {row['candidate_pool_count']} | "
                    f"Top-N: {row['topn_count']} | "
                    f"Top-N abstracts: {row['topn_abstract_count']} | "
                    f"Top-1 score: "
                    f"{row['top1_relevance_score']:.4f}"
                    if row["top1_relevance_score"] is not None
                    else "Top-1 score: n/a"
                ),
                "",
            ]
        )
        for work in row["top_ranked_works"]:
            lines.extend(
                [
                    (
                        f"### #{work['rank']} — "
                        f"{work['title']}"
                    ),
                    "",
                    (
                        f"`rel={work['relevance_score']:.4f}` | "
                        f"`sem={work['semantic_similarity']:.4f}` | "
                        f"`lex={work['lexical_coverage']:.4f}` | "
                        f"`domain={work['reaction_domain_relevance']:.4f}` | "
                        f"`scope={work['catalyst_scope_relevance']:.4f}` | "
                        f"`abstract={work['abstract_available']}`"
                    ),
                    "",
                    (
                        f"DOI: `{work['doi']}`"
                        if work["doi"]
                        else "DOI: `None`"
                    ),
                    "",
                ]
            )
            if work["abstract_excerpt"]:
                lines.extend(
                    [
                        "Abstract excerpt:",
                        "",
                        work["abstract_excerpt"],
                        "",
                    ]
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def verify_run(
    *,
    repo_root: Path,
    diagnostic_root: Path,
    spec: Mapping[str, Any],
    report_path: Path,
    audit_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not report_path.is_file():
        return ["ranker report missing"], {}
    if not audit_path.is_file():
        return ["human audit markdown missing"], {}

    stored = read_json(report_path)
    issues: list[str] = []

    body = dict(stored)
    run_id = body.pop("run_id", None)
    run_sha = body.pop("run_sha256", None)
    observed = sha256_json(body)
    if run_sha != observed:
        issues.append("run SHA mismatch")
    if run_id != (
        "sers_standard2_ranker_only_dev_run:"
        + observed[:20]
    ):
        issues.append("run ID mismatch")

    if stored.get("source_spec_id") != spec.get("spec_id"):
        issues.append("run/spec ID mismatch")
    if stored.get("source_spec_sha256") != spec.get(
        "spec_sha256"
    ):
        issues.append("run/spec SHA mismatch")
    if stored.get("mechanical_outcome") != (
        "RANKER_MECHANICAL_DEV_PASS"
    ):
        issues.append("mechanical ranker outcome is not PASS")
    if stored.get("scientific_relevance_outcome") != (
        "MANUAL_REVIEW_REQUIRED"
    ):
        issues.append("scientific relevance status was overclaimed")
    if stored.get("automatic_claim_level_review_authorized") is not False:
        issues.append("claim-level review was unexpectedly authorized")
    if stored.get("network_calls") != 0:
        issues.append("unexpected network calls recorded")
    if stored.get("llm_calls") != 0:
        issues.append("unexpected LLM calls recorded")
    if stored.get("claim_review_used") is not False:
        issues.append("claim-review-use violation")
    if stored.get("novelty_verdict_used") is not False:
        issues.append("novelty-verdict-use violation")
    if stored.get("fresh_reserve_consumed") is not False:
        issues.append("fresh-reserve consumption violation")

    expected_md = render_human_audit_markdown(stored)
    actual_md = audit_path.read_text(encoding="utf-8")
    if expected_md != actual_md:
        issues.append("human audit markdown mismatch")

    try:
        recomputed, recomputed_md = run_ranker_validation(
            repo_root=repo_root,
            diagnostic_root=diagnostic_root,
            spec=spec,
        )
        if canonical_json(recomputed) != canonical_json(stored):
            issues.append("offline ranker recomputation mismatch")
        if recomputed_md != actual_md:
            issues.append("offline audit recomputation mismatch")
    except Exception as exc:
        issues.append(
            f"offline recomputation failed: {type(exc).__name__}: {exc}"
        )

    return sorted(set(issues)), stored
