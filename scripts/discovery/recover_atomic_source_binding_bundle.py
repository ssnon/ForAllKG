#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    AtomicScientificSourceBindingRecord,
    build_atomic_scientific_source_binding_bundle,
)
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def safe_load_json(path: Path) -> Any | None:
    try:
        return load_json(path)
    except Exception:
        return None


def candidate_json_files(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            paths = [root]
        else:
            paths = root.rglob("*.json")
        for path in paths:
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            # Skip very large provider packets unless their name suggests provenance.
            name = path.name.lower()
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > 25_000_000 and not any(
                key in name
                for key in (
                    "binding",
                    "decomposition",
                    "sanitization",
                    "source",
                    "claim",
                )
            ):
                continue
            out.append(path)
    return sorted(out)


def find_existing_bundle(
    files: list[Path],
    *,
    query_plan: LiteratureQueryPlan,
) -> tuple[Path | None, dict[str, Any] | None]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        payload = safe_load_json(path)
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != "atomic-scientific-source-binding-bundle-v1":
            continue
        if payload.get("source_query_plan_id") != query_plan.plan_id:
            continue
        if payload.get("source_query_plan_sha256") != query_plan.plan_sha256:
            continue
        matches.append((path, payload))
    if not matches:
        return None, None
    matches.sort(key=lambda row: (len(row[0].parts), str(row[0])))
    return matches[0]


def collect_provenance_candidates(
    files: list[Path],
    *,
    claim_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {cid: [] for cid in claim_ids}

    for path in files:
        payload = safe_load_json(path)
        if payload is None:
            continue

        for row in walk_dicts(payload):
            cid = str(row.get("claim_id") or "").strip()
            if cid not in claim_ids:
                continue

            # Direct source-binding record shape.
            if (
                "proposition_basis" in row
                and "relation_endpoint_anchors" in row
            ):
                candidate = {
                    "source_path": str(path),
                    "claim_id": cid,
                    "claim_rank": row.get("claim_rank"),
                    "claim_local_id": row.get("claim_local_id"),
                    "proposition_basis": row.get("proposition_basis", ""),
                    "relation_endpoint_anchors": row.get(
                        "relation_endpoint_anchors", []
                    ),
                    "scope_qualifier_spans": row.get(
                        "scope_qualifier_spans", []
                    ),
                    "directional_qualifier_spans": row.get(
                        "directional_qualifier_spans", []
                    ),
                    "prediction_observation_id": row.get(
                        "prediction_observation_id"
                    ),
                    "falsification_criterion_id": row.get(
                        "falsification_criterion_id"
                    ),
                }
                out[cid].append(candidate)
                continue

            # specification_sanitization_record shape with nested source binding.
            sf = row.get("semantic_fidelity_shadow")
            if isinstance(sf, dict):
                binding = sf.get("binding")
                if isinstance(binding, dict) and (
                    "proposition_basis" in binding
                    or "relation_endpoint_anchors" in binding
                ):
                    candidate = {
                        "source_path": str(path),
                        "claim_id": cid,
                        "claim_rank": row.get("claim_rank"),
                        "claim_local_id": row.get("claim_local_id"),
                        "proposition_basis": binding.get(
                            "proposition_basis", ""
                        ),
                        "relation_endpoint_anchors": binding.get(
                            "relation_endpoint_anchors", []
                        ),
                        "scope_qualifier_spans": binding.get(
                            "scope_qualifier_spans", []
                        ),
                        "directional_qualifier_spans": binding.get(
                            "directional_qualifier_spans", []
                        ),
                        "prediction_observation_id": binding.get(
                            "prediction_observation_id"
                        ),
                        "falsification_criterion_id": binding.get(
                            "falsification_criterion_id"
                        ),
                    }
                    out[cid].append(candidate)

            # raw decomposition draft shape:
            # semantic_fidelity_binding is nested directly on a claim-like row.
            binding = row.get("semantic_fidelity_binding")
            if isinstance(binding, dict):
                candidate = {
                    "source_path": str(path),
                    "claim_id": cid,
                    "claim_rank": row.get("claim_rank"),
                    "claim_local_id": row.get("local_id")
                    or row.get("claim_local_id"),
                    "proposition_basis": binding.get(
                        "proposition_basis", ""
                    ),
                    "relation_endpoint_anchors": binding.get(
                        "relation_endpoint_anchors", []
                    ),
                    "scope_qualifier_spans": binding.get(
                        "scope_qualifier_spans", []
                    ),
                    "directional_qualifier_spans": binding.get(
                        "directional_qualifier_spans", []
                    ),
                    "prediction_observation_id": binding.get(
                        "prediction_observation_id"
                    ),
                    "falsification_criterion_id": binding.get(
                        "falsification_criterion_id"
                    ),
                }
                out[cid].append(candidate)

    return out


def normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if str(x).strip()]


def candidate_score(candidate: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(bool(str(candidate.get("claim_local_id") or "").strip())),
        int(bool(str(candidate.get("proposition_basis") or "").strip())),
        len(normalize_list(candidate.get("relation_endpoint_anchors"))),
        int(bool(candidate.get("prediction_observation_id")))
        + int(bool(candidate.get("falsification_criterion_id"))),
        str(candidate.get("source_path") or ""),
    )


def choose_candidate(
    candidates: list[dict[str, Any]],
    *,
    expected_rank: int,
) -> dict[str, Any] | None:
    filtered = []
    for row in candidates:
        rank = row.get("claim_rank")
        if rank is not None:
            try:
                if int(rank) != int(expected_rank):
                    continue
            except Exception:
                pass
        filtered.append(row)

    if not filtered:
        filtered = list(candidates)
    if not filtered:
        return None

    filtered.sort(key=candidate_score, reverse=True)
    best = filtered[0]

    # Fail closed on ambiguity among equally informative provenance records.
    best_key = candidate_score(best)[:4]
    tied = [row for row in filtered if candidate_score(row)[:4] == best_key]
    distinct_payloads = {
        canonical_json(
            {
                "claim_local_id": row.get("claim_local_id"),
                "proposition_basis": row.get("proposition_basis", ""),
                "relation_endpoint_anchors": normalize_list(
                    row.get("relation_endpoint_anchors")
                ),
                "scope_qualifier_spans": normalize_list(
                    row.get("scope_qualifier_spans")
                ),
                "directional_qualifier_spans": normalize_list(
                    row.get("directional_qualifier_spans")
                ),
                "prediction_observation_id": row.get(
                    "prediction_observation_id"
                ),
                "falsification_criterion_id": row.get(
                    "falsification_criterion_id"
                ),
            }
        )
        for row in tied
    }
    if len(distinct_payloads) > 1:
        return None
    return best


def rebuild_bundle(
    *,
    query_plan: LiteratureQueryPlan,
    candidates_by_claim: dict[str, list[dict[str, Any]]],
):
    claim_rows = [
        claim
        for group in query_plan.claims
        for claim in group.claims
    ]

    records: list[AtomicScientificSourceBindingRecord] = []
    diagnostics: list[dict[str, Any]] = []

    for claim in claim_rows:
        candidate = choose_candidate(
            candidates_by_claim.get(claim.claim_id, []),
            expected_rank=claim.claim_rank,
        )
        if candidate is None:
            diagnostics.append(
                {
                    "claim_id": claim.claim_id,
                    "status": "MISSING_OR_AMBIGUOUS_PROVENANCE",
                    "candidate_count": len(
                        candidates_by_claim.get(claim.claim_id, [])
                    ),
                }
            )
            continue

        local_id = str(candidate.get("claim_local_id") or "").strip()
        if not local_id:
            diagnostics.append(
                {
                    "claim_id": claim.claim_id,
                    "status": "MISSING_CLAIM_LOCAL_ID",
                    "source_path": candidate.get("source_path"),
                }
            )
            continue

        record = AtomicScientificSourceBindingRecord(
            hypothesis_id=claim.hypothesis_id,
            claim_id=claim.claim_id,
            claim_rank=claim.claim_rank,
            claim_local_id=local_id,
            source_claim_sha256=sha256_json(
                claim.model_dump(mode="json")
            ),
            proposition_basis=str(
                candidate.get("proposition_basis") or ""
            ),
            relation_endpoint_anchors=normalize_list(
                candidate.get("relation_endpoint_anchors")
            ),
            scope_qualifier_spans=normalize_list(
                candidate.get("scope_qualifier_spans")
            ),
            directional_qualifier_spans=normalize_list(
                candidate.get("directional_qualifier_spans")
            ),
            prediction_observation_id=(
                str(candidate.get("prediction_observation_id")).strip()
                if candidate.get("prediction_observation_id")
                else None
            ),
            falsification_criterion_id=(
                str(candidate.get("falsification_criterion_id")).strip()
                if candidate.get("falsification_criterion_id")
                else None
            ),
        )
        records.append(record)
        diagnostics.append(
            {
                "claim_id": claim.claim_id,
                "status": "RECOVERED",
                "source_path": candidate.get("source_path"),
                "proposition_basis_present": bool(
                    record.proposition_basis.strip()
                ),
                "endpoint_anchor_count": len(
                    record.relation_endpoint_anchors
                ),
            }
        )

    if len(records) != len(claim_rows):
        return None, diagnostics

    bundle = build_atomic_scientific_source_binding_bundle(
        source_portfolio_id=query_plan.source_portfolio_id,
        query_plan=query_plan,
        records=records,
    )
    return bundle, diagnostics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--abl", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument(
        "--query-plan",
        default="B.claims_queries.json",
    )
    p.add_argument(
        "--output-bundle",
        default="B.atomic_scientific_source_binding_bundle.recovered.json",
    )
    p.add_argument(
        "--diagnostics",
        default="B.atomic_scientific_source_binding_recovery.diagnostics.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    abl = args.abl.expanduser().resolve()
    run = args.run.expanduser().resolve()
    repo = args.repo.expanduser().resolve()

    query_plan = LiteratureQueryPlan.model_validate_json(
        (abl / args.query_plan).read_text(encoding="utf-8")
    )

    roots = [
        abl,
        run,
        repo / "evaluation",
    ]
    files = candidate_json_files(roots)
    print("JSON artifacts scanned:", len(files))

    existing_path, existing_payload = find_existing_bundle(
        files,
        query_plan=query_plan,
    )
    if existing_payload is not None:
        output = abl / args.output_bundle
        output.write_text(
            json.dumps(
                existing_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print("Existing matching source-binding bundle found:", existing_path)
        print("Copied canonical bundle to:", output)
        print("RECOVERY_MODE=EXISTING_BUNDLE")
        return 0

    claim_ids = {
        claim.claim_id
        for group in query_plan.claims
        for claim in group.claims
    }
    candidates = collect_provenance_candidates(
        files,
        claim_ids=claim_ids,
    )

    print(
        "Claims with provenance candidates:",
        sum(bool(rows) for rows in candidates.values()),
        "/",
        len(claim_ids),
    )

    bundle, diagnostics = rebuild_bundle(
        query_plan=query_plan,
        candidates_by_claim=candidates,
    )

    diag_path = abl / args.diagnostics
    diag_path.write_text(
        json.dumps(
            {
                "schema_version":
                    "atomic-source-binding-recovery-diagnostics-v1",
                "source_query_plan_id": query_plan.plan_id,
                "source_query_plan_sha256": query_plan.plan_sha256,
                "records": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    if bundle is None:
        print("Could not safely rebuild complete bundle.")
        print("Diagnostics:", diag_path)
        print("RECOVERY_MODE=FAILED_CLOSED")
        return 2

    output = abl / args.output_bundle
    output.write_text(
        bundle.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print("Rebuilt complete source-binding bundle:", output)
    print("Diagnostics:", diag_path)
    print("RECOVERY_MODE=REBUILT_FROM_PROVENANCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
