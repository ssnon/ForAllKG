from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from domains.registry import get_domain_profile
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.node_mapping import (
    DEFAULT_EMBED_MODEL,
    SentenceTransformerEncoder,
)
from pipeline_core.discovery.prior_art_matching import (
    PriorArtRanker,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
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


def source_prefix_from_prior_art(path: Path) -> Path:
    suffix = ".prior_art.json"
    if not path.name.endswith(suffix):
        raise RuntimeError(
            "expected .prior_art.json packet: " + str(path)
        )
    return path.with_name(path.name[: -len(suffix)])


def select_pre_review_missing_abstract_targets(
    *,
    plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
    ranker: PriorArtRanker,
) -> dict[str, Any]:
    works = {row.work_id: row for row in packet.works}

    selected_ids: set[str] = set()
    per_claim: list[dict[str, Any]] = []

    claims = [
        claim
        for group in plan.claims
        for claim in group.claims
        if str(claim.importance) == "core"
    ]

    for claim in claims:
        ranked = ranker.rank(
            claim,
            packet,
            plan,
        )

        unresolved = [
            row.work_id
            for row in ranked.ranked_works
            if (
                row.work_id in works
                and not works[row.work_id].abstract
            )
        ]
        selected_ids.update(unresolved)

        per_claim.append(
            {
                "hypothesis_id": claim.hypothesis_id,
                "claim_id": claim.claim_id,
                "claim_text": claim.text,
                "ranked_work_count":
                    len(ranked.ranked_works),
                "missing_abstract_ranked_work_ids":
                    unresolved,
            }
        )

    return {
        "core_claim_count": len(claims),
        "selected_work_ids": sorted(selected_ids),
        "selected_work_count": len(selected_ids),
        "per_claim": per_claim,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S228 pre-review metadata-resolution target-selector audit. "
            "No provider, LLM, or external literature calls are performed."
        )
    )
    parser.add_argument(
        "--s226-matrix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--s227-summary",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--max-ranked-works",
        type=int,
        default=8,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    s226_path = args.s226_matrix.expanduser().resolve()
    s227_path = args.s227_summary.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    for path in (s226_path, s227_path):
        if not path.is_file():
            raise RuntimeError("missing input: " + str(path))

    if output_path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: "
            + str(output_path)
        )

    s226 = load_json(s226_path)
    s227 = load_json(s227_path)

    domain_by_case = {
        str(row.get("source_case_id")):
            str(row.get("domain_profile_id"))
        for row in s226.get("cases", [])
    }

    encoder = SentenceTransformerEncoder(
        args.embed_model,
        device=args.device,
    )

    rows: list[dict[str, Any]] = []

    print("=== S228 PRE-REVIEW RESOLUTION TARGET SELECTOR AUDIT ===")
    print("provider calls: 0")
    print("LLM calls: 0")
    print("scientific query plan changed: false")
    print("review outcome consumed by selector: false")

    for row in s227.get("rows", []):
        if row.get("measurement_status") != "AVAILABLE":
            continue

        case_id = str(row["source_case_id"])
        scope = str(row["scope"])
        domain_profile_id = domain_by_case.get(case_id)
        if not domain_profile_id:
            raise RuntimeError(
                "missing domain profile: " + case_id
            )

        packet_path = Path(
            row["source_packet_path"]
        ).expanduser().resolve()
        audit_path = Path(
            row["audit_path"]
        ).expanduser().resolve()

        prefix = source_prefix_from_prior_art(
            packet_path
        )
        query_path = prefix.with_name(
            prefix.name + ".claims_queries.json"
        )

        for path in (
            packet_path,
            audit_path,
            query_path,
        ):
            if not path.is_file():
                raise RuntimeError(
                    "missing source artifact: " + str(path)
                )

        packet = PriorArtPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        plan = LiteratureQueryPlan.model_validate_json(
            query_path.read_text(encoding="utf-8")
        )
        audit = load_json(audit_path)

        if packet.source_query_plan_id != plan.plan_id:
            raise RuntimeError(
                "packet/query plan mismatch: " + scope
            )

        ranker = PriorArtRanker(
            encoder,
            max_ranked_works_per_claim=
                args.max_ranked_works,
            domain_profile=get_domain_profile(
                domain_profile_id
            ),
        )

        selection = (
            select_pre_review_missing_abstract_targets(
                plan=plan,
                packet=packet,
                ranker=ranker,
            )
        )

        selected = set(
            selection["selected_work_ids"]
        )

        s227_targets = {
            str(item["work_id"])
            for item in audit.get("works", [])
        }
        recovered = {
            str(item["work_id"])
            for item in audit.get("works", [])
            if item.get("abstract_recovered") is True
        }

        target_overlap = selected & s227_targets
        recovered_overlap = selected & recovered

        target_recall = (
            len(target_overlap) / len(s227_targets)
            if s227_targets
            else 1.0
        )
        recovery_capture = (
            len(recovered_overlap) / len(recovered)
            if recovered
            else 1.0
        )
        selected_overlap_fraction = (
            len(target_overlap) / len(selected)
            if selected
            else 1.0
        )

        result = {
            "source_case_id": case_id,
            "scope": scope,
            "domain_profile_id": domain_profile_id,
            "source_packet_path": str(packet_path),
            "query_plan_path": str(query_path),
            "s227_audit_path": str(audit_path),
            "core_claim_count":
                selection["core_claim_count"],
            "pre_review_selected_count":
                selection["selected_work_count"],
            "s227_material_title_only_target_count":
                len(s227_targets),
            "s227_recovered_target_count":
                len(recovered),
            "target_overlap_count":
                len(target_overlap),
            "recovered_overlap_count":
                len(recovered_overlap),
            "material_target_recall":
                target_recall,
            "recovered_target_capture":
                recovery_capture,
            "selected_overlap_fraction":
                selected_overlap_fraction,
            "missed_material_target_ids":
                sorted(s227_targets - selected),
            "missed_recovered_target_ids":
                sorted(recovered - selected),
            "extra_pre_review_target_ids":
                sorted(selected - s227_targets),
            "selected_work_ids":
                sorted(selected),
            "per_claim":
                selection["per_claim"],
            "authority": {
                "pre_review_deterministic_only": True,
                "review_outcome_consumed_by_selector":
                    False,
                "provider_calls": 0,
                "llm_calls": 0,
                "novelty_authority_created": False,
                "production_selection_authority": False,
            },
        }
        rows.append(result)

        print(
            case_id,
            "|",
            scope,
            "| selected=",
            len(selected),
            "| S227-targets=",
            len(s227_targets),
            "| target-recall=",
            f"{target_recall:.3f}",
            "| recovered-capture=",
            f"{recovery_capture:.3f}",
        )

    total_selected = set()
    total_targets: set[str] = set()
    total_recovered: set[str] = set()
    total_target_overlap: set[str] = set()
    total_recovered_overlap: set[str] = set()

    # Work IDs may repeat across packets, so cohort metrics below are
    # packet-scoped pairs rather than globally de-duplicated work IDs.
    packet_scoped_selected: set[tuple[int, str]] = set()
    packet_scoped_targets: set[tuple[int, str]] = set()
    packet_scoped_recovered: set[tuple[int, str]] = set()

    for idx, row in enumerate(rows):
        selected = {
            (idx, work_id)
            for work_id in row["selected_work_ids"]
        }
        targets = {
            (idx, work_id)
            for work_id in (
                set(row["selected_work_ids"])
                - set(row["extra_pre_review_target_ids"])
            )
        }
        # Reconstruct exact S227 target set from overlap + misses.
        targets |= {
            (idx, work_id)
            for work_id in row["missed_material_target_ids"]
        }
        recovered = {
            (idx, work_id)
            for work_id in (
                set(row["selected_work_ids"])
                - set(row["extra_pre_review_target_ids"])
            )
            if work_id not in row["missed_recovered_target_ids"]
        }
        # Exact recovered set comes from overlap + missed recovered.
        recovered = {
            (idx, work_id)
            for work_id in (
                set(
                    x
                    for x in row["selected_work_ids"]
                    if x not in row["extra_pre_review_target_ids"]
                )
                & set(
                    x
                    for x in row["selected_work_ids"]
                    if x not in row["missed_recovered_target_ids"]
                )
            )
        } | {
            (idx, work_id)
            for work_id in row["missed_recovered_target_ids"]
        }

        packet_scoped_selected |= selected
        packet_scoped_targets |= targets

        # Simpler exact recovered reconstruction from counts is unsafe.
        # Read the audit again for authoritative recovered IDs.
        audit = load_json(Path(row["s227_audit_path"]))
        recovered_exact = {
            (idx, str(item["work_id"]))
            for item in audit.get("works", [])
            if item.get("abstract_recovered") is True
        }
        packet_scoped_recovered |= recovered_exact

    target_overlap = (
        packet_scoped_selected
        & packet_scoped_targets
    )
    recovered_overlap = (
        packet_scoped_selected
        & packet_scoped_recovered
    )

    cohort_target_recall = (
        len(target_overlap) / len(packet_scoped_targets)
        if packet_scoped_targets
        else 1.0
    )
    cohort_recovery_capture = (
        len(recovered_overlap)
        / len(packet_scoped_recovered)
        if packet_scoped_recovered
        else 1.0
    )

    summary = {
        "schema_version":
            "pre-review-resolution-target-selector-audit-s228-v1",
        "source_s226_matrix": str(s226_path),
        "source_s227_summary": str(s227_path),
        "packet_count": len(rows),
        "pre_review_selected_count":
            len(packet_scoped_selected),
        "s227_material_target_count":
            len(packet_scoped_targets),
        "s227_recovered_target_count":
            len(packet_scoped_recovered),
        "material_target_overlap_count":
            len(target_overlap),
        "recovered_target_overlap_count":
            len(recovered_overlap),
        "material_target_recall":
            cohort_target_recall,
        "recovered_target_capture":
            cohort_recovery_capture,
        "rows": rows,
        "interpretation_policy": {
            "selector_basis":
                "top-ranked missing-abstract works for core claims",
            "review_outcome_used_for_selection": False,
            "provider_calls": 0,
            "llm_calls": 0,
            "scientific_query_plan_changed": False,
            "production_selection_changed": False,
        },
    }

    write_json(output_path, summary)

    print("\nS228 selector audit complete")
    print(
        "packet-scoped pre-review selected:",
        summary["pre_review_selected_count"],
    )
    print(
        "S227 material targets:",
        summary["s227_material_target_count"],
    )
    print(
        "material target recall:",
        f'{summary["material_target_recall"]:.3f}',
    )
    print(
        "S227 recovered targets:",
        summary["s227_recovered_target_count"],
    )
    print(
        "recovered target capture:",
        f'{summary["recovered_target_capture"]:.3f}',
    )
    print("provider calls: 0")
    print("LLM calls: 0")
    print("artifact:", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
