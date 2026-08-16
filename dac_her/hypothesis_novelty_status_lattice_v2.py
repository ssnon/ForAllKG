from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from dac_her.external_novelty import ExternalNoveltyAssessor
from dac_her.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimPriorArtStatus,
    ExternalNoveltyPolicy,
    HypothesisSearchCoverage,
    LiteratureQueryPlan,
    PriorArtPacket,
)


SEMANTICS_ID = "sers_hypothesis_novelty_status_lattice_v2"
EXPECTED_FREEZE_COMMIT = "664b40c0ca1348d953f16b90a13aadd681d31c72"
EXPECTED_PARENT_SPEC_ID = (
    "sers_hypothesis_novelty_synthesis_dev_spec:"
    "c09a87371711f764b3d9"
)
EXPECTED_PARENT_RUN_ID = (
    "sers_hypothesis_novelty_synthesis_dev_run:"
    "f0451b5748b68b32dbc8"
)
EXPECTED_CLAIM_REVIEW_V3_RUN_ID = (
    "sers_standard2_claim_review_only_dev_run_v3:"
    "4bdbec786ff05fc1b99b"
)

FROZEN_INPUT_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "hypothesis_novelty_frozen_input_v1"
)
PARENT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "hypothesis_novelty_synthesis_dev_spec_v1"
)
PARENT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "hypothesis_novelty_synthesis_dev_run_v1"
)
DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "hypothesis_novelty_status_lattice_v2_spec"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "hypothesis_novelty_status_lattice_v2_run"
)

CLAIM_STATUSES: tuple[ClaimPriorArtStatus, ...] = (
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "TITLE_ONLY_NEIGHBORS",
    "COMPONENTS_ONLY",
    "NO_DIRECT_MATCH_FOUND",
    "CONFLICTING_PRIOR_ART",
    "INSUFFICIENT_METADATA",
)

SOURCE_FILES_TO_FREEZE = (
    Path("dac_her/external_novelty.py"),
    Path("dac_her/external_novelty_contracts.py"),
)


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
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
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _validate_internal_sha(
    value: Mapping[str, Any],
    *,
    id_key: str,
    sha_key: str,
    id_prefix: str,
) -> None:
    body = dict(value)
    stored_id = body.pop(id_key, None)
    stored_sha = body.pop(sha_key, None)
    observed = sha256_json(body)
    if stored_sha != observed:
        raise ValueError(f"{sha_key} mismatch")
    if stored_id != id_prefix + observed[:20]:
        raise ValueError(f"{id_key} mismatch")


def source_hashes(repo_root: Path) -> dict[str, str]:
    result = {}
    for rel in SOURCE_FILES_TO_FREEZE:
        path = repo_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(rel)] = sha256_file(path)
    return result


def load_frozen_inputs(
    repo_root: Path,
) -> tuple[
    LiteratureQueryPlan,
    PriorArtPacket,
    dict[str, ClaimPriorArtReview],
    dict[str, Any],
]:
    root = repo_root / FROZEN_INPUT_ROOT
    required = {
        "claim_report": root / "claim_review_report_v3.json",
        "plan": root / "frozen_query_plan.json",
        "packet": root / "canonical_prior_art_v2.json",
        "handoff_manifest": root / "handoff_manifest.json",
        "handoff_marker": root / "HANDOFF_PASS.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    claim_report = read_json(required["claim_report"])
    _validate_internal_sha(
        claim_report,
        id_key="run_id",
        sha_key="run_sha256",
        id_prefix="sers_standard2_claim_review_only_dev_run_v3:",
    )
    if claim_report.get("run_id") != EXPECTED_CLAIM_REVIEW_V3_RUN_ID:
        raise ValueError("unexpected frozen claim-review v3 run ID")
    if claim_report.get("structural_outcome") != (
        "CLAIM_REVIEW_V3_STRUCTURAL_DEV_PASS"
    ):
        raise ValueError("frozen claim-review v3 is not structural PASS")
    if claim_report.get("scientific_relationship_outcome") != (
        "MANUAL_REVIEW_REQUIRED"
    ):
        raise ValueError("frozen claim-review scientific outcome drift")

    plan = LiteratureQueryPlan.model_validate_json(
        required["plan"].read_text(encoding="utf-8")
    )
    packet = PriorArtPacket.model_validate_json(
        required["packet"].read_text(encoding="utf-8")
    )

    reviews_raw = claim_report.get("claim_reviews")
    if not isinstance(reviews_raw, list) or len(reviews_raw) != 12:
        raise ValueError("expected exactly 12 frozen claim reviews")
    reviews_list = [
        ClaimPriorArtReview.model_validate(row)
        for row in reviews_raw
    ]
    reviews = {row.claim_id: row for row in reviews_list}
    if len(reviews) != 12:
        raise ValueError("duplicate frozen claim-review claim ID")
    if any(row.reviewer_unknown_work_ids for row in reviews.values()):
        raise ValueError("unknown reviewer work IDs in frozen claim reviews")

    plan_claim_ids = [
        claim.claim_id
        for group in plan.claims
        for claim in group.claims
    ]
    if set(plan_claim_ids) != set(reviews):
        raise ValueError("frozen query-plan / claim-review mismatch")
    if len(plan.claims) != 3:
        raise ValueError("expected exactly 3 frozen hypotheses")
    if len(packet.works) != 430:
        raise ValueError("frozen canonical work count drift")
    if packet.source_query_plan_id != plan.plan_id:
        raise ValueError("packet/query-plan lineage mismatch")

    manifest = read_json(required["handoff_manifest"])
    marker = read_json(required["handoff_marker"])
    manifest_body = dict(manifest)
    manifest_sha = manifest_body.pop("manifest_sha256", None)
    if sha256_json(manifest_body) != manifest_sha:
        raise ValueError("handoff manifest SHA mismatch")
    if marker.get("status") != "handoff_pass":
        raise ValueError("handoff marker status mismatch")
    if marker.get("manifest_sha256") != manifest_sha:
        raise ValueError("handoff marker/manifest mismatch")

    return plan, packet, reviews, claim_report


def load_parent_v1(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_path = repo_root / PARENT_SPEC_ROOT / "novelty_synthesis_spec.json"
    run_path = (
        repo_root / PARENT_RUN_ROOT /
        "hypothesis_novelty_synthesis_report.json"
    )
    marker_path = repo_root / PARENT_RUN_ROOT / "STRUCTURAL_PASS.json"
    for path in (spec_path, run_path, marker_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    spec = read_json(spec_path)
    run = read_json(run_path)
    marker = read_json(marker_path)

    _validate_internal_sha(
        spec,
        id_key="spec_id",
        sha_key="spec_sha256",
        id_prefix="sers_hypothesis_novelty_synthesis_dev_spec:",
    )
    _validate_internal_sha(
        run,
        id_key="run_id",
        sha_key="run_sha256",
        id_prefix="sers_hypothesis_novelty_synthesis_dev_run:",
    )
    if spec.get("spec_id") != EXPECTED_PARENT_SPEC_ID:
        raise ValueError("unexpected parent v1 spec ID")
    if run.get("run_id") != EXPECTED_PARENT_RUN_ID:
        raise ValueError("unexpected parent v1 run ID")
    if run.get("structural_outcome") != (
        "HYPOTHESIS_NOVELTY_SYNTHESIS_STRUCTURAL_DEV_PASS"
    ):
        raise ValueError("parent v1 is not structural PASS")
    if run.get("scientific_novelty_status_outcome") != (
        "MANUAL_REVIEW_REQUIRED"
    ):
        raise ValueError("parent v1 scientific outcome drift")
    if marker.get("status") != "structural_pass":
        raise ValueError("parent v1 marker status mismatch")
    if marker.get("run_id") != run.get("run_id"):
        raise ValueError("parent v1 marker run ID mismatch")
    if run.get("fresh_reserve_consumed") is not False:
        raise ValueError("parent v1 consumed Fresh Reserve")
    if run.get("automatic_next_stage_authorized") is not False:
        raise ValueError("parent v1 authorized next stage")
    return spec, run


def make_assessor(
    policy: ExternalNoveltyPolicy | None = None,
) -> ExternalNoveltyAssessor:
    assessor = object.__new__(ExternalNoveltyAssessor)
    assessor.policy = policy or ExternalNoveltyPolicy()
    return assessor


def _synthetic_review(
    status: ClaimPriorArtStatus,
    index: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        importance="core",
        status=status,
        claim_id=f"policy_probe:{index}",
    )


def _synthetic_coverage(
    sufficient: bool,
) -> HypothesisSearchCoverage:
    return HypothesisSearchCoverage(
        hypothesis_id="policy_probe",
        query_count=2,
        successful_query_count=(2 if sufficient else 0),
        provider_success_count=(2 if sufficient else 0),
        unique_work_count=(10 if sufficient else 0),
        abstract_work_count=(5 if sufficient else 0),
        core_claim_count=1,
        core_claims_with_minimum_abstract_coverage=(1 if sufficient else 0),
        sufficient_for_absence_based_novelty=sufficient,
    )


def characterize_status_lattice(
    *,
    max_core_claims: int = 4,
) -> dict[str, Any]:
    assessor = make_assessor()
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for n in range(1, max_core_claims + 1):
        for statuses in product(CLAIM_STATUSES, repeat=n):
            reviews = [
                _synthetic_review(status, index)
                for index, status in enumerate(statuses, start=1)
            ]
            for sufficient in (False, True):
                first = assessor._status(
                    reviews,
                    _synthetic_coverage(sufficient),
                )
                second = assessor._status(
                    reviews,
                    _synthetic_coverage(sufficient),
                )
                if first != second:
                    raise RuntimeError("non-deterministic _status result")
                status, reasons, interpretation = first
                counts[status] += 1
                rows.append(
                    {
                        "core_statuses": list(statuses),
                        "coverage_sufficient": sufficient,
                        "status": status,
                        "reason_codes": list(reasons),
                        "interpretation": interpretation,
                    }
                )

    def probe(
        name: str,
        statuses: Iterable[ClaimPriorArtStatus],
        sufficient: bool,
    ) -> dict[str, Any]:
        seq = list(statuses)
        status, reasons, interpretation = assessor._status(
            [
                _synthetic_review(row, i)
                for i, row in enumerate(seq, start=1)
            ],
            _synthetic_coverage(sufficient),
        )
        return {
            "name": name,
            "core_statuses": seq,
            "coverage_sufficient": sufficient,
            "observed_status": status,
            "reason_codes": list(reasons),
            "interpretation": interpretation,
        }

    probes = [
        probe(
            "all_direct_sufficient",
            ["DIRECT_PRIOR_ART", "DIRECT_PRIOR_ART"],
            True,
        ),
        probe(
            "direct_partial_sufficient",
            ["DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"],
            True,
        ),
        probe(
            "partial_components_sufficient",
            ["PARTIAL_PRIOR_ART", "COMPONENTS_ONLY"],
            True,
        ),
        probe(
            "all_components_sufficient",
            ["COMPONENTS_ONLY", "COMPONENTS_ONLY"],
            True,
        ),
        probe(
            "components_no_direct_sufficient",
            ["COMPONENTS_ONLY", "NO_DIRECT_MATCH_FOUND"],
            True,
        ),
        probe(
            "title_components_sufficient",
            ["TITLE_ONLY_NEIGHBORS", "COMPONENTS_ONLY"],
            True,
        ),
        probe(
            "insufficient_metadata_sufficient",
            ["COMPONENTS_ONLY", "INSUFFICIENT_METADATA"],
            True,
        ),
        probe(
            "all_no_direct_sufficient",
            ["NO_DIRECT_MATCH_FOUND", "NO_DIRECT_MATCH_FOUND"],
            True,
        ),
        probe(
            "all_no_direct_insufficient",
            ["NO_DIRECT_MATCH_FOUND", "NO_DIRECT_MATCH_FOUND"],
            False,
        ),
    ]

    body = {
        "schema_version":
            "hypothesis-novelty-status-lattice-audit-v2",
        "semantics_id": SEMANTICS_ID,
        "claim_status_domain": list(CLAIM_STATUSES),
        "max_core_claims": max_core_claims,
        "case_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "manual_semantic_review_probes": probes,
        "rows": rows,
        "scientific_semantic_outcome": "MANUAL_REVIEW_REQUIRED",
        "case_specific_sers_rules_used": False,
    }
    body["audit_sha256"] = sha256_json(body)
    return body


def compute_real_synthesis(
    *,
    plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
    reviews: Mapping[str, ClaimPriorArtReview],
) -> list[dict[str, Any]]:
    assessor = make_assessor()
    rows = []
    for group in plan.claims:
        grouped = [reviews[row.claim_id] for row in group.claims]
        if any(row.hypothesis_id != group.hypothesis_id for row in grouped):
            raise ValueError("claim/hypothesis lineage mismatch")
        hypothesis = SimpleNamespace(
            hypothesis_id=group.hypothesis_id,
            title=group.title,
        )
        coverage_1 = assessor._coverage(
            hypothesis,
            grouped,
            packet,
            plan,
        )
        coverage_2 = assessor._coverage(
            hypothesis,
            grouped,
            packet,
            plan,
        )
        if canonical_json(coverage_1) != canonical_json(coverage_2):
            raise RuntimeError("non-deterministic _coverage result")
        status_1 = assessor._status(grouped, coverage_1)
        status_2 = assessor._status(grouped, coverage_2)
        if status_1 != status_2:
            raise RuntimeError("non-deterministic _status result")
        status, reasons, interpretation = status_1
        core = [row for row in grouped if row.importance == "core"] or grouped
        rows.append(
            {
                "hypothesis_id": group.hypothesis_id,
                "title": group.title,
                "claim_ids": [row.claim_id for row in grouped],
                "core_claim_statuses": [row.status for row in core],
                "coverage": coverage_1.model_dump(mode="json"),
                "status": status,
                "reason_codes": list(reasons),
                "interpretation": interpretation,
            }
        )
    return rows


def build_spec(repo_root: Path) -> dict[str, Any]:
    parent_spec, parent_run = load_parent_v1(repo_root)
    plan, packet, reviews, claim_report = load_frozen_inputs(repo_root)
    lattice = characterize_status_lattice()

    body: dict[str, Any] = {
        "schema_version":
            "sers-hypothesis-novelty-status-lattice-v2-spec",
        "semantics_id": SEMANTICS_ID,
        "freeze_commit": EXPECTED_FREEZE_COMMIT,
        "parent_v1_spec_id": parent_spec["spec_id"],
        "parent_v1_run_id": parent_run["run_id"],
        "source_claim_review_v3_run_id": claim_report["run_id"],
        "source_query_plan_id": plan.plan_id,
        "source_query_plan_sha256": plan.plan_sha256,
        "source_canonical_packet_id": packet.packet_id,
        "source_canonical_packet_sha256": packet.packet_sha256,
        "claim_count": len(reviews),
        "hypothesis_count": len(plan.claims),
        "source_hashes": source_hashes(repo_root),
        "status_lattice_audit_sha256": lattice["audit_sha256"],
        "status_lattice_case_count": lattice["case_count"],
        "controlled_change": {
            "external_novelty_status_enum_addition":
                "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
            "title_only_core_fail_closed": True,
            "new_combination_requires_relation_backed_core_claim": True,
            "all_components_relational_gap_split": True,
            "claim_reviewer_changed": False,
            "ranker_changed": False,
            "provider_changed": False,
            "canonicalizer_changed": False,
            "coverage_policy_changed": False,
            "case_specific_sers_rules_used": False,
        },
        "epistemic_policy": {
            "title_only_is_unresolved_not_absence_evidence": True,
            "component_only_is_not_relation_backed_evidence": True,
            "new_combination_requires_positive_relation_evidence": True,
            "known_components_relational_gap_is_not_novelty_proof": True,
            "no_direct_match_is_not_literature_wide_absence": True,
            "scientific_novelty_status_outcome":
                "MANUAL_REVIEW_REQUIRED",
            "automatic_next_stage_authorized": False,
            "fresh_reserve_consumed": False,
        },
        "llm_calls_during_spec_freeze": 0,
        "network_calls_during_spec_freeze": 0,
    }
    body["spec_sha256"] = sha256_json(body)
    body["spec_id"] = (
        "sers_hypothesis_novelty_status_lattice_v2_spec:"
        + body["spec_sha256"][:20]
    )
    return body


def verify_spec(
    repo_root: Path,
    spec_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not spec_path.is_file():
        return ["spec missing"], {}
    stored = read_json(spec_path)
    issues = []
    body = dict(stored)
    spec_id = body.pop("spec_id", None)
    spec_sha = body.pop("spec_sha256", None)
    observed = sha256_json(body)
    if spec_sha != observed:
        issues.append("spec SHA mismatch")
    if spec_id != (
        "sers_hypothesis_novelty_status_lattice_v2_spec:"
        + observed[:20]
    ):
        issues.append("spec ID mismatch")
    try:
        load_parent_v1(repo_root)
        load_frozen_inputs(repo_root)
        if stored.get("source_hashes") != source_hashes(repo_root):
            issues.append("production source hash drift")
        lattice = characterize_status_lattice()
        if stored.get("status_lattice_audit_sha256") != (
            lattice["audit_sha256"]
        ):
            issues.append("status-lattice audit drift")
    except Exception as exc:
        issues.append(
            f"source/input verification failed: "
            f"{type(exc).__name__}: {exc}"
        )
    return sorted(set(issues)), stored


def render_human_audit(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SERS Hypothesis Novelty Status-Lattice v2 Audit",
        "",
        "- Input: **frozen claim-review v3**",
        "- Production logic: **hardened _coverage + _status**",
        "- LLM calls: **0**",
        "- Network calls: **0**",
        "- Scientific novelty-status outcome: **MANUAL_REVIEW_REQUIRED**",
        "",
        "`KNOWN_COMPONENTS_WITH_RELATIONAL_GAP` means that relevant "
        "components are represented but no core relation is positively "
        "represented as direct/partial prior art. It is not proof of novelty.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        coverage = row["coverage"]
        lines.extend(
            [
                f"## Hypothesis {index} — {row['status']}",
                "",
                row["title"],
                "",
                f"`hypothesis_id={row['hypothesis_id']}`",
                "",
                f"- core statuses: `{row['core_claim_statuses']}`",
                f"- coverage sufficient: "
                f"**{coverage['sufficient_for_absence_based_novelty']}**",
                f"- successful queries: "
                f"{coverage['successful_query_count']}/"
                f"{coverage['query_count']}",
                f"- unique works: {coverage['unique_work_count']}",
                f"- abstract works: {coverage['abstract_work_count']}",
                f"- reason codes: `{', '.join(row['reason_codes'])}`",
                f"- interpretation: {row['interpretation']}",
                "",
                "### Human audit",
                "",
                "- semantic verdict: **PENDING**",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
