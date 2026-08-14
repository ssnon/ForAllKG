from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from dac_her.literature_catalog import doi_family, normalize_title
from dac_her.literature_catalog_contracts import (
    CatalogQuery,
    CatalogQueryExecution,
    CatalogWork,
    LiteratureCatalogPacket,
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


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _identity_keys(work: CatalogWork) -> tuple[str, ...]:
    keys: list[str] = []
    family = doi_family(work.doi)
    if family:
        keys.append(f"doi:{family}")
    title = normalize_title(work.title)
    # Keep the same conservative exact-title spirit as M1 canonicalization,
    # while avoiding extremely short-title collisions during expansion.
    if len(title) >= 20:
        keys.append(f"title:{title}")
    return tuple(keys)


def _unique_by_id(rows: Iterable, attr: str) -> list:
    result = []
    seen: set[str] = set()
    for row in rows:
        key = str(getattr(row, attr))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


@dataclass(frozen=True)
class CatalogExpansionResult:
    packet: LiteratureCatalogPacket
    report: dict[str, object]


def append_catalog_expansion(
    *,
    base: LiteratureCatalogPacket,
    incoming: LiteratureCatalogPacket,
    expansion_id: str,
) -> CatalogExpansionResult:
    """Append genuinely new canonical works while freezing all base works.

    Existing CatalogWork rows are intentionally not enriched or rewritten.
    That gives the expansion a strong append-only property: every prior work
    keeps the same work_id *and* the same metadata payload.  Overlapping works
    found by the deeper/new-provider discovery are counted as overlaps and are
    skipped.  Only previously unseen works are appended.
    """

    if base.acquisition_profile_id != incoming.acquisition_profile_id:
        raise ValueError(
            "Catalog expansion profile mismatch: "
            f"{base.acquisition_profile_id!r} != "
            f"{incoming.acquisition_profile_id!r}"
        )

    base_ids = [row.work_id for row in base.works]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("Base catalog contains duplicate work_id")

    identity_owner: dict[str, str] = {}
    for row in base.works:
        for key in _identity_keys(row):
            owner = identity_owner.get(key)
            if owner is not None and owner != row.work_id:
                raise ValueError(
                    "Base catalog has ambiguous canonical identity: "
                    f"{key!r} -> {owner!r}, {row.work_id!r}"
                )
            identity_owner[key] = row.work_id

    appended: list[CatalogWork] = []
    appended_ids: set[str] = set()
    overlap_ids: list[str] = []
    collision_ids: list[str] = []

    for row in incoming.works:
        keys = _identity_keys(row)
        matches = {
            identity_owner[key]
            for key in keys
            if key in identity_owner
        }
        if len(matches) > 1:
            raise ValueError(
                "Incoming work bridges multiple frozen base identities: "
                f"{row.work_id!r} -> {sorted(matches)!r}"
            )
        if matches:
            overlap_ids.append(row.work_id)
            continue

        if row.work_id in set(base_ids) or row.work_id in appended_ids:
            collision_ids.append(row.work_id)
            raise ValueError(
                "New work_id collision without canonical identity match: "
                f"{row.work_id!r}"
            )

        appended.append(row)
        appended_ids.add(row.work_id)
        for key in keys:
            owner = identity_owner.get(key)
            if owner is not None and owner != row.work_id:
                # Incoming packets are expected to be internally canonical.
                # Fail instead of silently merging two new identities here.
                raise ValueError(
                    "Incoming catalog contains duplicate canonical identity: "
                    f"{key!r}"
                )
            identity_owner[key] = row.work_id

    works = [*base.works, *appended]
    if [row.work_id for row in works[: len(base.works)]] != base_ids:
        raise RuntimeError("Append-only invariant violated: base prefix changed")
    for before, after in zip(base.works, works[: len(base.works)]):
        if before.model_dump(mode="json") != after.model_dump(mode="json"):
            raise RuntimeError(
                "Append-only invariant violated: base work metadata changed"
            )

    queries: list[CatalogQuery] = _unique_by_id(
        [*base.queries, *incoming.queries],
        "query_id",
    )
    executions: list[CatalogQueryExecution] = [
        *base.executions,
        *incoming.executions,
    ]
    providers = list(
        dict.fromkeys(
            [*base.providers_requested, *incoming.providers_requested]
        )
    )

    catalog_id = _stable_id(
        "literature_catalog",
        base.acquisition_profile_id,
        "append_expansion_v1",
        expansion_id,
        base.catalog_id,
        incoming.catalog_id,
        *[row.work_id for row in works],
    )
    packet_without_sha = {
        "schema_version": "literature-catalog-packet-v1",
        "catalog_id": catalog_id,
        "acquisition_profile_id": base.acquisition_profile_id,
        "searched_at_utc": incoming.searched_at_utc,
        "providers_requested": providers,
        "queries": [row.model_dump(mode="json") for row in queries],
        "works": [row.model_dump(mode="json") for row in works],
        "executions": [row.model_dump(mode="json") for row in executions],
        "raw_work_count": base.raw_work_count + incoming.raw_work_count,
        "canonical_work_count": len(works),
        "deduplicated_work_count": max(
            0,
            base.raw_work_count
            + incoming.raw_work_count
            - len(works),
        ),
        "supplementary_records_collapsed": (
            base.supplementary_records_collapsed
            + incoming.supplementary_records_collapsed
        ),
        "epistemic_usage": "candidate_source_only_not_positive_premise",
    }
    catalog_sha256 = _sha256_json(packet_without_sha)
    packet = LiteratureCatalogPacket.model_validate(
        {
            **packet_without_sha,
            "catalog_sha256": catalog_sha256,
        }
    )

    report: dict[str, object] = {
        "schema_version": "catalog-expansion-report-v1",
        "expansion_id": expansion_id,
        "profile_id": base.acquisition_profile_id,
        "base_catalog_id": base.catalog_id,
        "base_catalog_sha256": base.catalog_sha256,
        "incoming_catalog_id": incoming.catalog_id,
        "incoming_catalog_sha256": incoming.catalog_sha256,
        "expanded_catalog_id": packet.catalog_id,
        "expanded_catalog_sha256": packet.catalog_sha256,
        "base_work_count": len(base.works),
        "incoming_work_count": len(incoming.works),
        "overlap_work_count": len(overlap_ids),
        "new_work_count": len(appended),
        "expanded_work_count": len(works),
        "base_prefix_preserved": True,
        "base_metadata_preserved": True,
        "new_work_ids": [row.work_id for row in appended],
        "overlap_incoming_work_ids": overlap_ids,
        "work_id_collision_count": len(collision_ids),
        "positive_evidence_promotion_performed": False,
    }
    return CatalogExpansionResult(packet=packet, report=report)
