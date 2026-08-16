from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from dac_her.external_novelty_contracts import PriorArtPacket, PriorArtWork
from dac_her.literature_retrieval import canonicalize_prior_art_packet


SOURCE_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "canonicalization_only_dev_run_v1"
)
SOURCE_RAW_PACKET = SOURCE_RUN_ROOT / "raw_prior_art.json"
SOURCE_REPORT = SOURCE_RUN_ROOT / "canonicalization_report.json"

OUTPUT_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "canonicalization_only_dev_recheck_v2"
)

EXPECTED_SOURCE_SPEC_ID = (
    "sers_standard2_canonicalization_dev_spec:854bb80d14d8699d7c2c"
)
EXPECTED_SOURCE_OUTCOME = "CANONICALIZATION_DEV_FAIL"
EXPECTED_RAW_WORK_COUNT = 618
EXPECTED_V1_CANONICAL_WORK_COUNT = 421
EXPECTED_COLLISION_GROUP_COUNT = 5

SEMANTICS_ID = "sers_canonicalization_doi_conflict_recheck_v2"


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


def _norm_doi(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("https://doi.org/"):
        text = text[len("https://doi.org/") :]
    if text.startswith("doi:"):
        text = text[4:]
    return text or None


_SUPPLEMENTARY_DOI_RE = re.compile(r"\.s\d+$", re.I)


def _doi_family(value: Any) -> str | None:
    doi = _norm_doi(value)
    if not doi:
        return None
    return _SUPPLEMENTARY_DOI_RE.sub("", doi)


def _norm_title(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


def collision_groups(
    works: list[PriorArtWork],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[PriorArtWork]] = defaultdict(list)
    for work in works:
        title = _norm_title(work.title)
        if len(title) >= 20:
            groups[title].append(work)

    result: dict[str, dict[str, Any]] = {}
    for title, rows in groups.items():
        families = sorted(
            {
                family
                for family in (_doi_family(row.doi) for row in rows)
                if family
            }
        )
        if len(families) < 2:
            continue
        title_hash = hashlib.sha256(
            title.encode("utf-8")
        ).hexdigest()
        result[title_hash] = {
            "raw_record_count": len(rows),
            "doi_families": families,
            "doi_family_hashes": [
                hashlib.sha256(
                    family.encode("utf-8")
                ).hexdigest()
                for family in families
            ],
        }
    return result


def validate_source(root: Path) -> tuple[PriorArtPacket, dict[str, Any]]:
    raw_path = root / SOURCE_RAW_PACKET
    report_path = root / SOURCE_REPORT
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    raw_packet = PriorArtPacket.model_validate_json(
        raw_path.read_text(encoding="utf-8")
    )
    report = read_json(report_path)

    if report.get("source_spec_id") != EXPECTED_SOURCE_SPEC_ID:
        raise ValueError("source canonicalization spec ID mismatch")
    if report.get("outcome") != EXPECTED_SOURCE_OUTCOME:
        raise ValueError("source run is not the expected FAIL artifact")

    failed = sorted(
        key
        for key, value in report.get("checks", {}).items()
        if value is not True
    )
    if failed != ["no_exact_title_multiple_doi_family_collision"]:
        raise ValueError(
            f"unexpected v1 failed checks: {failed}"
        )

    counts = report.get("counts", {})
    if counts.get("raw_work_count") != EXPECTED_RAW_WORK_COUNT:
        raise ValueError("source raw work count mismatch")
    if counts.get("canonical_work_count") != EXPECTED_V1_CANONICAL_WORK_COUNT:
        raise ValueError("source v1 canonical work count mismatch")
    if (
        counts.get("title_cross_doi_collision_group_count")
        != EXPECTED_COLLISION_GROUP_COUNT
    ):
        raise ValueError("source collision group count mismatch")

    return raw_packet, report


def run_recheck(root: Path) -> tuple[PriorArtPacket, dict[str, Any]]:
    raw_packet, source_report = validate_source(root)
    raw_works = list(raw_packet.works)
    raw_collisions = collision_groups(raw_works)

    first = canonicalize_prior_art_packet(raw_packet)
    second = canonicalize_prior_art_packet(raw_packet)

    deterministic = (
        canonical_json(first)
        == canonical_json(second)
    )

    canonical_groups: dict[str, list[PriorArtWork]] = defaultdict(list)
    for work in first.works:
        title = _norm_title(work.title)
        if len(title) >= 20:
            canonical_groups[
                hashlib.sha256(title.encode("utf-8")).hexdigest()
            ].append(work)

    preservation_rows = []
    all_conflicts_preserved = True
    for title_hash, raw_group in sorted(raw_collisions.items()):
        rows = canonical_groups.get(title_hash, [])
        canonical_families = sorted(
            {
                family
                for family in (_doi_family(row.doi) for row in rows)
                if family
            }
        )
        raw_families = sorted(raw_group["doi_families"])
        preserved = (
            set(raw_families).issubset(set(canonical_families))
            and len(rows) >= len(raw_families)
        )
        all_conflicts_preserved = (
            all_conflicts_preserved and preserved
        )
        preservation_rows.append(
            {
                "normalized_title_sha256": title_hash,
                "raw_distinct_doi_family_count": len(raw_families),
                "canonical_distinct_doi_family_count": len(
                    canonical_families
                ),
                "canonical_record_count_for_title": len(rows),
                "all_raw_doi_families_preserved": preserved,
                "raw_doi_family_hashes": [
                    hashlib.sha256(
                        family.encode("utf-8")
                    ).hexdigest()
                    for family in raw_families
                ],
                "canonical_doi_family_hashes": [
                    hashlib.sha256(
                        family.encode("utf-8")
                    ).hexdigest()
                    for family in canonical_families
                ],
            }
        )

    canonical_ids = [row.work_id for row in first.works]
    query_ids_raw = {
        qid
        for row in raw_packet.works
        for qid in row.retrieval_query_ids
    }
    query_ids_canonical = {
        qid
        for row in first.works
        for qid in row.retrieval_query_ids
    }
    claim_ids_raw = {
        cid
        for row in raw_packet.works
        for cid in row.retrieval_claim_ids
    }
    claim_ids_canonical = {
        cid
        for row in first.works
        for cid in row.retrieval_claim_ids
    }
    providers_raw = {
        provider
        for row in raw_packet.works
        for provider in row.providers
    }
    providers_canonical = {
        provider
        for row in first.works
        for provider in row.providers
    }

    checks = {
        "offline_recanonicalization_only": True,
        "deterministic_recanonicalization": deterministic,
        "raw_collision_group_count_stable":
            len(raw_collisions) == EXPECTED_COLLISION_GROUP_COUNT,
        "conflicting_doi_families_preserved":
            all_conflicts_preserved,
        "canonical_work_ids_unique":
            len(canonical_ids) == len(set(canonical_ids)),
        "query_provenance_preserved":
            query_ids_raw == query_ids_canonical,
        "claim_provenance_preserved":
            claim_ids_raw == claim_ids_canonical,
        "provider_provenance_preserved":
            providers_raw == providers_canonical,
        "canonical_count_not_less_than_v1":
            len(first.works) >= EXPECTED_V1_CANONICAL_WORK_COUNT,
    }
    passed = all(checks.values())

    body: dict[str, Any] = {
        "schema_version":
            "sers-canonicalization-doi-conflict-recheck-v2",
        "semantics_id": SEMANTICS_ID,
        "source_v1_run_id": source_report["run_id"],
        "source_v1_run_sha256": source_report["run_sha256"],
        "source_raw_packet_id": raw_packet.packet_id,
        "source_raw_packet_sha256": raw_packet.packet_sha256,
        "canonical_packet_id": first.packet_id,
        "canonical_packet_sha256": first.packet_sha256,
        "outcome": (
            "CANONICALIZATION_DOI_CONFLICT_HARDENING_PASS"
            if passed
            else "CANONICALIZATION_DOI_CONFLICT_HARDENING_FAIL"
        ),
        "checks": checks,
        "counts": {
            "raw_work_count": len(raw_packet.works),
            "v1_canonical_work_count":
                source_report["counts"]["canonical_work_count"],
            "v2_canonical_work_count": len(first.works),
            "v1_deduplicated_work_count":
                source_report["counts"]["deduplicated_work_count"],
            "v2_deduplicated_work_count":
                len(raw_packet.works) - len(first.works),
            "collision_group_count": len(raw_collisions),
            "v2_multi_provider_work_count": sum(
                len(set(row.providers)) >= 2
                for row in first.works
            ),
            "v2_multi_query_work_count": sum(
                len(set(row.retrieval_query_ids)) >= 2
                for row in first.works
            ),
            "v2_multi_claim_work_count": sum(
                len(set(row.retrieval_claim_ids)) >= 2
                for row in first.works
            ),
        },
        "collision_preservation": preservation_rows,
        "network_calls": 0,
        "ranker_used": False,
        "llm_calls": 0,
        "claim_review_used": False,
        "scientific_result_interpretation": False,
        "novelty_status_change_authorized": False,
        "fresh_reserve_consumed": False,
        "canonical_packet_eligible_for_dev_ranker_validation": passed,
        "automatic_next_stage_authorized": False,
    }
    body["run_sha256"] = sha256_json(body)
    body["run_id"] = (
        "sers_canonicalization_doi_conflict_recheck:"
        + body["run_sha256"][:20]
    )
    return first, body
