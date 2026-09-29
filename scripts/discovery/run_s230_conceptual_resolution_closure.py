
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pipeline_core.discovery.prior_art_retrieval as _s230_prior_art_retrieval
import pipeline_core.literature.acquisition.provider_resilience as _s230_resilience
from pipeline_core.literature.acquisition.provider_resilience import (
    resilient_request_json as _s230_resilient_request_json,
)

from domains.registry import get_domain_profile
from pipeline_core.discovery.direct_higher_order_conceptual_knownness_contracts import (
    ConceptualKnownnessLevelResult,
    DirectHigherOrderConceptualKnownnessRecord,
    build_conceptual_knownness_profile,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    PriorArtPacket,
)
from pipeline_core.discovery.node_mapping import (
    DEFAULT_EMBED_MODEL,
    SentenceTransformerEncoder,
)
from pipeline_core.discovery.prior_art_matching import PriorArtRanker
from pipeline_core.discovery.prior_art_metadata_resolution_s227 import (
    PriorArtMetadataResolver,
)
from pipeline_core.discovery.prior_art_provider_plan import (
    LiteratureProviderPlan,
    build_literature_providers,
)
from pipeline_core.discovery.prospective_novelty_validation_completion_s226 import (
    first_gap_level,
    l3_status_from_external_card,
)
from scripts.discovery.run_s226_validation_completion import (
    compile_target_result,
    get_client,
    make_query_plan,
    ranked_work_payloads,
    review_target,
)



_s230_original_parse_retry_after_seconds = (
    _s230_resilience.parse_retry_after_seconds
)


def _s230_capped_retry_after(
    value,
    *,
    now_utc=None,
):
    seconds = _s230_original_parse_retry_after_seconds(
        value,
        now_utc=now_utc,
    )
    if seconds is None:
        return None
    return min(float(seconds), 10.0)


# Validation-local only:
# never honor a provider Retry-After longer than 10 seconds.
_s230_resilience.parse_retry_after_seconds = (
    _s230_capped_retry_after
)


def _s230_fail_fast_request_json(
    url: str,
    *,
    headers=None,
    timeout: float = 30.0,
    retries: int = 2,
    retry_backoff: float = 1.0,
    pacer=None,
    telemetry=None,
):
    # S230 bounded provider policy:
    # - <= 10 s transport timeout
    # - one retry
    # - Retry-After capped at 10 s
    # - provider pacer retained
    return _s230_resilient_request_json(
        url,
        headers=headers,
        timeout=min(float(timeout), 10.0),
        retries=1,
        retry_backoff=1.0,
        pacer=pacer,
        telemetry=telemetry,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def conceptual_disposition(levels: list[tuple[str, str]]) -> str:
    gap = first_gap_level(levels)
    mapping = {
        "L1_BROAD": "GAP_AT_L1",
        "L2_INTERMEDIATE": "GAP_AT_L2",
        "L3_EXACT": "GAP_AT_L3",
    }
    if gap is not None:
        return mapping.get(gap, "GAP_AT_UNKNOWN_LEVEL")
    statuses = [status for _, status in levels]
    if "CONFLICTING_RELATION" in statuses:
        return "CONFLICTING_EVIDENCE"
    if "INSUFFICIENT_COVERAGE" in statuses:
        return "INDETERMINATE_COVERAGE"
    return "NO_GAP_OBSERVED"


def exact_cards_from_s229(summary: dict[str, Any]) -> dict[str, tuple[dict[str, Any], str]]:
    result: dict[str, tuple[dict[str, Any], str]] = {}
    for row in summary.get("rows", []):
        if row.get("measurement_status") != "AVAILABLE":
            continue
        raw = row.get("integrated_report_path")
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError("missing S229 integrated report: " + str(path))
        report = ExternalNoveltyReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for card in report.cards:
            payload = card.model_dump(mode="json")
            old = result.get(card.hypothesis_id)
            if old is not None and old[0] != payload:
                raise RuntimeError(
                    "conflicting S229 exact cards for " + card.hypothesis_id
                )
            result[card.hypothesis_id] = (payload, str(path))
    return result


def resolve_target(
    *,
    target: dict[str, Any],
    ranker: PriorArtRanker,
    doi_resolver: PriorArtMetadataResolver,
    title_resolver: PriorArtMetadataResolver,
    client,
    model: str,
    temperature: float,
    max_retries: int,
    min_match_confidence: float,
    min_unique_works: int,
    min_abstract_works: int,
    min_successful_query_variants: int,
    output_dir: Path,
    telemetry_path: str,
) -> dict[str, Any]:
    packet_path = Path(target["prior_art_packet_path"]).expanduser().resolve()
    packet = PriorArtPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    plan = make_query_plan(
        target_id=target["target_id"],
        relation=target["relation"],
        queries=list(target["queries"]),
    )
    if packet.source_query_plan_id != plan.plan_id:
        raise RuntimeError(
            f"conceptual packet/query-plan mismatch: "
            f"{packet.source_query_plan_id} != {plan.plan_id}"
        )

    claim = plan.claims[0].claims[0]
    ranked_before = ranker.rank(claim, packet, plan)
    works_by_id = {row.work_id: row for row in packet.works}
    selected = {
        row.work_id
        for row in ranked_before.ranked_works
        if row.work_id in works_by_id and not works_by_id[row.work_id].abstract
    }

    if selected:
        ordered_targets = sorted(selected)
        original_work_by_id = {
            row.work_id: row
            for row in packet.works
        }
        resolved_packet = packet

        target_audits = []
        all_attempts = []
        all_work_audits = []
        recovered_total = 0
        provider_failures = 0
        matched_total = 0

        print(
            f"  [{target['target_id']}] metadata resolution: "
            f"{len(ordered_targets)} work(s)",
            flush=True,
        )

        for work_index, work_id in enumerate(ordered_targets, start=1):
            source_work = original_work_by_id.get(work_id)
            if source_work is None:
                print(
                    f"    [{work_index}/{len(ordered_targets)}] "
                    f"{work_id}: source work missing; skip",
                    flush=True,
                )
                continue

            has_doi = bool(str(source_work.doi or "").strip())
            resolver = doi_resolver if has_doi else title_resolver
            mode = "DOI_FAMILY_ONLY" if has_doi else "EXACT_TITLE_FALLBACK"

            current_ids = {row.work_id for row in resolved_packet.works}
            if work_id not in current_ids:
                print(
                    f"    [{work_index}/{len(ordered_targets)}] "
                    f"{work_id}: already canonicalized away; skip",
                    flush=True,
                )
                continue

            print(
                f"    [{work_index}/{len(ordered_targets)}] "
                f"{mode} | {source_work.title[:90]}",
                flush=True,
            )
            started = time.perf_counter()

            resolved_packet, one_audit = resolver.resolve(
                resolved_packet,
                target_work_ids={work_id},
            )

            elapsed = time.perf_counter() - started
            recovered = int(one_audit.get("target_abstract_recovered_count", 0) or 0)
            failures = int(one_audit.get("provider_failure_count", 0) or 0)

            recovered_total += recovered
            provider_failures += failures
            matched_total += int(one_audit.get("matched_target_work_count", 0) or 0)
            all_attempts.extend(one_audit.get("attempts", []))
            all_work_audits.extend(one_audit.get("works", []))
            target_audits.append(one_audit)

            print(
                f"      done in {elapsed:.1f}s | "
                f"abstract_recovered={recovered} | "
                f"provider_failures={failures}",
                flush=True,
            )

            if failures:
                for attempt in one_audit.get("attempts", []):
                    if attempt.get("success") is False:
                        print(
                            "        failure:",
                            attempt.get("provider"),
                            "|",
                            attempt.get("error"),
                            flush=True,
                        )

        audit = {
            "schema_version": "prior-art-metadata-resolution-s230-progress-v1",
            "source_packet_id": packet.packet_id,
            "resolved_packet_id": resolved_packet.packet_id,
            "requested_target_work_count": len(ordered_targets),
            "matched_target_work_count": matched_total,
            "target_abstract_recovered_count": recovered_total,
            "provider_attempt_count": len(all_attempts),
            "provider_failure_count": provider_failures,
            "attempts": all_attempts,
            "works": all_work_audits,
            "per_work_resolution_audits": target_audits,
            "lookup_policy": {
                "doi_present": "DOI_FAMILY_ONLY",
                "doi_absent": "EXACT_TITLE_FALLBACK",
                "title_fallback_when_doi_present": False,
            },
        }

        print(
            f"  [{target['target_id']}] metadata resolution complete | "
            f"recovered={recovered_total}/{len(ordered_targets)} | "
            f"provider_failures={provider_failures}",
            flush=True,
        )
    else:
        resolved_packet = packet
        audit = {
            "schema_version": "prior-art-metadata-resolution-s227-v1",
            "source_packet_id": packet.packet_id,
            "resolved_packet_id": packet.packet_id,
            "requested_target_work_count": 0,
            "matched_target_work_count": 0,
            "target_abstract_recovered_count": 0,
            "provider_attempt_count": 0,
            "provider_failure_count": 0,
        }
        print(
            f"  [{target['target_id']}] metadata resolution: "
            "0 work(s); proceeding to conceptual review",
            flush=True,
        )

    ranked_after = ranker.rank(claim, resolved_packet, plan)
    works = ranked_work_payloads(
        packet=resolved_packet,
        ranked=ranked_after,
    )
    print(
        f"  [{target['target_id']}] starting conceptual LLM review",
        flush=True,
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
        packet=resolved_packet,
        ranked_works=works,
        draft=draft,
        min_confidence=min_match_confidence,
        min_unique_works=min_unique_works,
        min_abstract_works=min_abstract_works,
        min_successful_query_variants=min_successful_query_variants,
    )

    stem = slug(target["target_id"])
    packet_out = output_dir / f"{stem}.prior_art.resolved.json"
    audit_out = output_dir / f"{stem}.metadata_resolution.json"
    write_json(packet_out, resolved_packet)
    audit["selector"] = {
        "selected_work_ids": sorted(selected),
        "selected_work_count": len(selected),
        "basis": "conceptual top-ranked missing-abstract works",
        "review_outcome_consumed": False,
    }
    audit["authority"] = {
        "validation_only": True,
        "scientific_query_plan_changed": False,
        "novelty_authority_created": False,
        "production_selection_authority": False,
    }
    write_json(audit_out, audit)

    compiled["source_prior_art_packet_path"] = str(packet_path)
    compiled["resolved_prior_art_packet_path"] = str(packet_out)
    compiled["metadata_resolution_path"] = str(audit_out)
    compiled["telemetry_event_id"] = getattr(event, "event_id", None)
    compiled["selected_resolution_target_count"] = len(selected)
    compiled["abstract_recovered_count"] = audit.get(
        "target_abstract_recovered_count", 0
    )
    return compiled


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s226-matrix", required=True, type=Path)
    p.add_argument("--s229-summary", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--model", default=os.getenv("OPENROUTER_AGENT_MODEL") or "openai/gpt-5.6-luna")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--device", default=None)
    p.add_argument("--max-ranked-works", type=int, default=8)
    p.add_argument("--lookup-limit", type=int, default=5)
    p.add_argument("--min-match-confidence", type=float, default=0.65)
    p.add_argument("--min-unique-works", type=int, default=10)
    p.add_argument("--min-abstract-works", type=int, default=5)
    p.add_argument("--min-successful-query-variants", type=int, default=2)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    s226_path = args.s226_matrix.expanduser().resolve()
    s229_path = args.s229_summary.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError("S230 output root must be fresh: " + str(out_root))
    out_root.mkdir(parents=True, exist_ok=True)

    s226 = load_json(s226_path)
    s229 = load_json(s229_path)
    provider_plan = LiteratureProviderPlan.model_validate(s226["provider_plan"])

    # S230 validation only: fail fast on provider stalls/rate limits.
    # Production retrieval behavior is unchanged.
    _s230_prior_art_retrieval._request_json = (
        _s230_fail_fast_request_json
    )

    providers = build_literature_providers(provider_plan)

    # S230 validation only:
    # OpenAlex is currently rate-limited, so avoid immediately
    # hammering the next DOI lookup after a successful/failed request.
    for provider in providers:
        if getattr(provider, "provider_name", "") == "openalex":
            provider.minimum_interval_seconds = max(
                float(
                    getattr(
                        provider,
                        "minimum_interval_seconds",
                        0.0,
                    )
                ),
                2.0,
            )
            if hasattr(provider, "_request_pacer"):
                provider._request_pacer.minimum_interval_seconds = (
                    provider.minimum_interval_seconds
                )

    # S230 is a validation-side metadata-completion pass.
    # Keep provider stalls bounded; normal production provider policy is unchanged.
    for provider in providers:
        if hasattr(provider, "timeout"):
            provider.timeout = 10.0

    exact_cards = exact_cards_from_s229(s229)

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key in {args.api_key_env}")
    client = get_client(
        api_key=api_key,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        timeout=args.timeout,
    )
    encoder = SentenceTransformerEncoder(args.embed_model, device=args.device)
    doi_resolver = PriorArtMetadataResolver(
        providers,
        lookup_limit=args.lookup_limit,
        enable_title_fallback=False,
    )
    title_resolver = PriorArtMetadataResolver(
        providers,
        lookup_limit=args.lookup_limit,
        enable_title_fallback=True,
    )

    case_rows: list[dict[str, Any]] = []
    failures = 0

    print("=== S230 CONCEPTUAL-KNOWNNESS RESOLUTION CLOSURE ===")
    print("provider mode:", provider_plan.mode)
    print("new scientific literature queries: false")
    print("S226 conceptual targets/packets reused: true")
    print("S229 exact integrated reports reused: true")
    print("production selection changed: false")

    for case in s226.get("cases", []):
        case_id = str(case.get("source_case_id") or "UNKNOWN")
        conceptual = case.get("conceptual_knownness") or {}
        print("\n" + "=" * 96)
        print(case_id)
        print("=" * 96)

        if conceptual.get("measurement_status") != "AVAILABLE":
            row = {
                "source_case_id": case_id,
                "measurement_status": "NOT_APPLICABLE",
                "reason": conceptual.get("measurement_status"),
            }
            case_rows.append(row)
            print("skip:", row["reason"])
            continue

        try:
            details_path = Path(conceptual["details_path"]).expanduser().resolve()
            details = load_json(details_path)
            target_by_id = {
                str(row["target_id"]): row
                for row in details.get("targets", [])
            }

            case_out = out_root / slug(case_id)
            target_out = case_out / "targets"
            target_out.mkdir(parents=True, exist_ok=False)

            ranker = PriorArtRanker(
                encoder,
                max_ranked_works_per_claim=args.max_ranked_works,
                domain_profile=get_domain_profile(case["domain_profile_id"]),
            )

            needed: set[str] = set()
            for record in details.get("records", []):
                proposal = record["proposal"]
                needed.add(str(proposal["l1_target_id"]))
                if proposal.get("l2_admissible") and proposal.get("l2_target_id"):
                    needed.add(str(proposal["l2_target_id"]))

            resolved_targets: dict[str, dict[str, Any]] = {}
            for target_id in sorted(needed):
                target = target_by_id[target_id]
                resolved_targets[target_id] = resolve_target(
                    target=target,
                    ranker=ranker,
                    doi_resolver=doi_resolver,
                    title_resolver=title_resolver,
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    min_match_confidence=args.min_match_confidence,
                    min_unique_works=args.min_unique_works,
                    min_abstract_works=args.min_abstract_works,
                    min_successful_query_variants=args.min_successful_query_variants,
                    output_dir=target_out,
                    telemetry_path=str(case_out / "conceptual.telemetry.jsonl"),
                )

            records_out = []
            normalized = []

            for old in details.get("records", []):
                proposal = old["proposal"]
                hypothesis_id = str(old["hypothesis_id"])
                l1 = resolved_targets[proposal["l1_target_id"]]
                l2 = (
                    resolved_targets[proposal["l2_target_id"]]
                    if proposal.get("l2_admissible")
                    else old["L2_INTERMEDIATE"]
                )

                exact = exact_cards.get(hypothesis_id)
                if exact is None:
                    l3 = {
                        "status": "INSUFFICIENT_COVERAGE",
                        "source_external_status": None,
                        "sufficient_coverage": False,
                        "material_title_only_matches": [],
                        "reason_code": "S229_EXACT_CARD_NOT_AVAILABLE",
                    }
                    exact_path = None
                else:
                    payload, exact_path = exact
                    l3 = l3_status_from_external_card(
                        payload,
                        min_confidence=args.min_match_confidence,
                    )

                before_levels = [
                    ("L1_BROAD", old["L1_BROAD"]["status"]),
                    ("L2_INTERMEDIATE", old["L2_INTERMEDIATE"]["status"]),
                    ("L3_EXACT", old["L3_EXACT"]["status"]),
                ]
                after_levels = [
                    ("L1_BROAD", l1["status"]),
                    ("L2_INTERMEDIATE", l2["status"]),
                    ("L3_EXACT", l3["status"]),
                ]
                before_disp = conceptual_disposition(before_levels)
                after_disp = conceptual_disposition(after_levels)
                after_gap = first_gap_level(after_levels)

                ambiguity_resolved = (
                    before_disp == "INDETERMINATE_COVERAGE"
                    and after_disp
                    in {"GAP_AT_L1", "GAP_AT_L2", "GAP_AT_L3", "NO_GAP_OBSERVED"}
                )

                level_models = [
                    ConceptualKnownnessLevelResult(
                        level="L1_BROAD",
                        abstraction_text=l1["relation"],
                        knownness_class=l1["status"],
                        retrieval_status="S226_PACKET_REVIEWED_AFTER_METADATA_RESOLUTION",
                        search_bounded=True,
                        sufficient_coverage=bool(
                            (l1.get("coverage") or {}).get(
                                "sufficient_for_search_bounded_gap", False
                            )
                        ),
                        source_profile_path=l1.get("resolved_prior_art_packet_path"),
                    ),
                    ConceptualKnownnessLevelResult(
                        level="L2_INTERMEDIATE",
                        abstraction_text=l2.get("relation", ""),
                        knownness_class=l2["status"],
                        retrieval_status=(
                            "S226_PACKET_REVIEWED_AFTER_METADATA_RESOLUTION"
                            if proposal.get("l2_admissible")
                            else "ABSTRACTION_REJECTED"
                        ),
                        search_bounded=True,
                        sufficient_coverage=bool(
                            (l2.get("coverage") or {}).get(
                                "sufficient_for_search_bounded_gap", False
                            )
                        ),
                        source_profile_path=l2.get("resolved_prior_art_packet_path"),
                    ),
                    ConceptualKnownnessLevelResult(
                        level="L3_EXACT",
                        abstraction_text=old["full_higher_order_claim"],
                        knownness_class=l3["status"],
                        retrieval_status=(
                            "REUSED_S229_INTEGRATED_EXTERNAL_NOVELTY"
                            if exact is not None
                            else "NOT_AVAILABLE"
                        ),
                        search_bounded=True,
                        sufficient_coverage=bool(l3.get("sufficient_coverage", False)),
                        source_profile_path=exact_path,
                    ),
                ]
                normalized.append(
                    DirectHigherOrderConceptualKnownnessRecord(
                        hypothesis_id=hypothesis_id,
                        direct_context_id=old["direct_context_id"],
                        direct_topology_id=old["direct_topology_id"],
                        first_gap_level=after_gap,
                        levels=level_models,
                    )
                )
                records_out.append(
                    {
                        "hypothesis_id": hypothesis_id,
                        "full_higher_order_claim": old["full_higher_order_claim"],
                        "before": {
                            "L1_BROAD": old["L1_BROAD"]["status"],
                            "L2_INTERMEDIATE": old["L2_INTERMEDIATE"]["status"],
                            "L3_EXACT": old["L3_EXACT"]["status"],
                            "first_gap_level": old.get("first_gap_level"),
                            "disposition": before_disp,
                        },
                        "after": {
                            "L1_BROAD": l1,
                            "L2_INTERMEDIATE": l2,
                            "L3_EXACT": l3,
                            "first_gap_level": after_gap,
                            "disposition": after_disp,
                        },
                        "ambiguity_resolved": ambiguity_resolved,
                        "disposition_changed": before_disp != after_disp,
                    }
                )

            profile = build_conceptual_knownness_profile(
                source_bundle_id=details["source_bundle_id"],
                records=normalized,
            )
            profile_path = case_out / "conceptual_knownness.profile.json"
            details_out = case_out / "conceptual_knownness.details.json"
            write_json(profile_path, profile)
            write_json(
                details_out,
                {
                    "schema_version": "conceptual-knownness-resolution-closure-s230-v1",
                    "source_s226_details": str(details_path),
                    "source_bundle_id": details["source_bundle_id"],
                    "source_case_id": case_id,
                    "records": records_out,
                    "resolved_targets": list(resolved_targets.values()),
                    "authority": {
                        "validation_only": True,
                        "new_scientific_literature_queries": False,
                        "s226_conceptual_targets_changed": False,
                        "s229_exact_reports_reused": True,
                        "novelty_authority_created": False,
                        "production_selection_authority": False,
                    },
                },
            )

            before_counts = Counter(x["before"]["disposition"] for x in records_out)
            after_counts = Counter(x["after"]["disposition"] for x in records_out)
            row = {
                "source_case_id": case_id,
                "measurement_status": "AVAILABLE",
                "profile_path": str(profile_path),
                "details_path": str(details_out),
                "record_count": len(records_out),
                "ambiguity_resolved_count": sum(x["ambiguity_resolved"] for x in records_out),
                "disposition_changed_count": sum(x["disposition_changed"] for x in records_out),
                "before_disposition_counts": dict(sorted(before_counts.items())),
                "after_disposition_counts": dict(sorted(after_counts.items())),
                "first_gap_levels": {
                    x["hypothesis_id"]: x["after"]["first_gap_level"]
                    for x in records_out
                },
                "conceptual_resolution_targets": sum(
                    int(x.get("selected_resolution_target_count") or 0)
                    for x in resolved_targets.values()
                ),
                "conceptual_abstracts_recovered": sum(
                    int(x.get("abstract_recovered_count") or 0)
                    for x in resolved_targets.values()
                ),
            }
            case_rows.append(row)
            print(
                "records=", row["record_count"],
                "| ambiguity resolved=", row["ambiguity_resolved_count"],
                "| targets=", row["conceptual_resolution_targets"],
                "| abstracts recovered=", row["conceptual_abstracts_recovered"],
            )
            print(" before:", row["before_disposition_counts"])
            print(" after :", row["after_disposition_counts"])
            print(" first gaps:", row["first_gap_levels"])

        except Exception as exc:
            failures += 1
            row = {
                "source_case_id": case_id,
                "measurement_status": "ERROR",
                "error": repr(exc),
            }
            case_rows.append(row)
            print("ERROR:", repr(exc))
            if not args.continue_on_error:
                raise

    available = [x for x in case_rows if x.get("measurement_status") == "AVAILABLE"]
    before_total = Counter()
    after_total = Counter()
    for row in available:
        before_total.update(row["before_disposition_counts"])
        after_total.update(row["after_disposition_counts"])

    summary = {
        "schema_version": "conceptual-knownness-resolution-closure-s230-v1",
        "source_s226_matrix": str(s226_path),
        "source_s229_summary": str(s229_path),
        "provider_plan": provider_plan.model_dump(mode="json"),
        "completed_case_count": len(available),
        "failed_case_count": failures,
        "record_count": sum(x["record_count"] for x in available),
        "ambiguity_resolved_count": sum(x["ambiguity_resolved_count"] for x in available),
        "disposition_changed_count": sum(x["disposition_changed_count"] for x in available),
        "conceptual_resolution_targets": sum(x["conceptual_resolution_targets"] for x in available),
        "conceptual_abstracts_recovered": sum(x["conceptual_abstracts_recovered"] for x in available),
        "before_disposition_counts": dict(sorted(before_total.items())),
        "after_disposition_counts": dict(sorted(after_total.items())),
        "cases": case_rows,
        "interpretation_policy": {
            "GAP_AT_L1": "first search-bounded conceptual gap at broad level",
            "GAP_AT_L2": "first search-bounded conceptual gap at intermediate level",
            "GAP_AT_L3": "broad/intermediate relation-backed; exact relation gap-like",
            "NO_GAP_OBSERVED": "no search-bounded gap and no unresolved/conflicting level",
            "INDETERMINATE_COVERAGE": "at least one level remains insufficiently covered",
            "CONFLICTING_EVIDENCE": "at least one level contains conflicting relation evidence",
            "new_scientific_literature_queries": False,
            "production_selection_changed": False,
        },
    }
    summary_path = out_root / "closure.summary.json"
    write_json(summary_path, summary)

    print("\nS230 conceptual closure complete")
    print("completed cases:", len(available))
    print("failed cases:", failures)
    print("hypothesis records:", summary["record_count"])
    print("conceptual resolution targets:", summary["conceptual_resolution_targets"])
    print("conceptual abstracts recovered:", summary["conceptual_abstracts_recovered"])
    print("ambiguity resolved:", summary["ambiguity_resolved_count"])
    print("before:", summary["before_disposition_counts"])
    print("after :", summary["after_disposition_counts"])
    print("new scientific queries: false")
    print("production selection changed: false")
    print("artifact:", summary_path)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
