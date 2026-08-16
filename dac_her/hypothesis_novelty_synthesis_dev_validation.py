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


SEMANTICS_ID = "sers_hypothesis_novelty_synthesis_only_dev_v1"

EXPECTED_FREEZE_COMMIT = "664b40c0ca1348d953f16b90a13aadd681d31c72"
EXPECTED_CLAIM_REVIEW_V3_SPEC_ID = (
    "sers_standard2_claim_review_only_dev_spec_v3:"
    "1b2368c3dae2f569068b"
)
EXPECTED_CLAIM_REVIEW_V3_RUN_ID = (
    "sers_standard2_claim_review_only_dev_run_v3:"
    "4bdbec786ff05fc1b99b"
)
EXPECTED_RANKER_RUN_ID = (
    "sers_standard2_ranker_only_dev_run:"
    "28b2c16ed3c9befb6bc0"
)
EXPECTED_CANONICAL_WORK_COUNT = 430
EXPECTED_CLAIM_COUNT = 12
EXPECTED_CORE_CLAIM_COUNT = 10
EXPECTED_HYPOTHESIS_COUNT = 3
EXPECTED_V3_STATUS_COUNTS = {
    "COMPONENTS_ONLY": 5,
    "PARTIAL_PRIOR_ART": 7,
}

FROZEN_INPUT_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "hypothesis_novelty_frozen_input_v1"
)
DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "hypothesis_novelty_synthesis_dev_spec_v1"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "hypothesis_novelty_synthesis_dev_run_v1"
)

SOURCE_FILES_TO_FREEZE = (
    Path("dac_her/external_novelty.py"),
    Path("dac_her/external_novelty_contracts.py"),
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

HANDOFF_SOURCE_FILES = {
    "claim_review_spec_v3.json": Path(
        "evaluation/sers_provider_reliability/"
        "claim_review_only_dev_spec_v3/claim_review_spec_v3.json"
    ),
    "claim_review_report_v3.json": Path(
        "evaluation/sers_provider_reliability/"
        "claim_review_only_dev_run_v3/claim_review_report_v3.json"
    ),
    "claim_review_structural_pass_v3.json": Path(
        "evaluation/sers_provider_reliability/"
        "claim_review_only_dev_run_v3/STRUCTURAL_PASS.json"
    ),
    "human_relationship_audit_v3.md": Path(
        "evaluation/sers_provider_reliability/"
        "claim_review_only_dev_run_v3/human_relationship_audit_v3.md"
    ),
    "frozen_query_plan.json": Path(
        "evaluation/sers_provider_reliability/"
        "ranker_only_dev_spec_v1/frozen_query_plan.json"
    ),
    "canonical_prior_art_v2.json": Path(
        "evaluation/sers_provider_reliability/"
        "canonicalization_only_dev_recheck_v2/canonical_prior_art_v2.json"
    ),
}


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


def source_hashes(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for rel in SOURCE_FILES_TO_FREEZE:
        path = repo_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(rel)] = sha256_file(path)
    return result


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


def _claim_review_map(
    report: Mapping[str, Any],
) -> dict[str, ClaimPriorArtReview]:
    rows = report.get("claim_reviews")
    if not isinstance(rows, list):
        raise ValueError("claim-review v3 report lacks claim_reviews")
    reviews = [ClaimPriorArtReview.model_validate(row) for row in rows]
    if len(reviews) != EXPECTED_CLAIM_COUNT:
        raise ValueError("claim-review v3 claim count drift")
    result = {row.claim_id: row for row in reviews}
    if len(result) != len(reviews):
        raise ValueError("duplicate claim_id in claim-review v3 report")
    return result


def validate_frozen_source(
    source_root: Path,
    *,
    flat: bool = False,
) -> dict[str, Any]:
    paths = {
        name: source_root / (Path(name) if flat else rel)
        for name, rel in HANDOFF_SOURCE_FILES.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    spec = read_json(paths["claim_review_spec_v3.json"])
    report = read_json(paths["claim_review_report_v3.json"])
    marker = read_json(paths["claim_review_structural_pass_v3.json"])
    plan = LiteratureQueryPlan.model_validate_json(
        paths["frozen_query_plan.json"].read_text(encoding="utf-8")
    )
    packet = PriorArtPacket.model_validate_json(
        paths["canonical_prior_art_v2.json"].read_text(encoding="utf-8")
    )

    _validate_internal_sha(
        spec,
        id_key="spec_id",
        sha_key="spec_sha256",
        id_prefix="sers_standard2_claim_review_only_dev_spec_v3:",
    )
    _validate_internal_sha(
        report,
        id_key="run_id",
        sha_key="run_sha256",
        id_prefix="sers_standard2_claim_review_only_dev_run_v3:",
    )

    if spec.get("spec_id") != EXPECTED_CLAIM_REVIEW_V3_SPEC_ID:
        raise ValueError("unexpected claim-review v3 spec ID")
    if report.get("run_id") != EXPECTED_CLAIM_REVIEW_V3_RUN_ID:
        raise ValueError("unexpected claim-review v3 run ID")
    if report.get("source_spec_id") != spec.get("spec_id"):
        raise ValueError("v3 run/spec lineage mismatch")
    if report.get("source_ranker_run_id") != EXPECTED_RANKER_RUN_ID:
        raise ValueError("unexpected source ranker run ID")
    if spec.get("source_ranker_run_id") != EXPECTED_RANKER_RUN_ID:
        raise ValueError("unexpected v3 spec ranker lineage")

    if report.get("structural_outcome") != (
        "CLAIM_REVIEW_V3_STRUCTURAL_DEV_PASS"
    ):
        raise ValueError("claim-review v3 is not structural PASS")
    if report.get("scientific_relationship_outcome") != (
        "MANUAL_REVIEW_REQUIRED"
    ):
        raise ValueError("claim-review v3 scientific outcome drift")
    if marker.get("status") != "structural_pass":
        raise ValueError("claim-review v3 structural marker mismatch")
    if marker.get("run_id") != report.get("run_id"):
        raise ValueError("claim-review v3 marker run ID mismatch")

    for key in (
        "ranker_recomputed",
        "compiler_changed_from_v2",
        "invalid_id_guess_mapping_used",
        "case_specific_expected_statuses_used",
        "hypothesis_level_novelty_status_computed",
        "automatic_next_stage_authorized",
        "fresh_reserve_consumed",
    ):
        if report.get(key) is not False:
            raise ValueError(f"claim-review v3 policy drift: {key}")

    diagnostics = report.get("diagnostics", {})
    if diagnostics.get("compiled_status_counts") != EXPECTED_V3_STATUS_COUNTS:
        raise ValueError("claim-review v3 status-count drift")

    reviews = _claim_review_map(report)
    if any(row.reviewer_unknown_work_ids for row in reviews.values()):
        raise ValueError("claim-review v3 contains unknown reviewer work IDs")
    if sum(row.importance == "core" for row in reviews.values()) != (
        EXPECTED_CORE_CLAIM_COUNT
    ):
        raise ValueError("claim-review v3 core-claim count drift")

    if plan.plan_id != spec.get("source_query_plan_id"):
        raise ValueError("query-plan ID does not match v3 spec")
    if plan.plan_sha256 != spec.get("source_query_plan_sha256"):
        raise ValueError("query-plan SHA does not match v3 spec")
    if packet.packet_id != spec.get("source_canonical_packet_id"):
        raise ValueError("canonical packet ID does not match v3 spec")
    if packet.packet_sha256 != spec.get("source_canonical_packet_sha256"):
        raise ValueError("canonical packet SHA does not match v3 spec")
    if len(packet.works) != EXPECTED_CANONICAL_WORK_COUNT:
        raise ValueError("canonical work count drift")
    if packet.source_query_plan_id != plan.plan_id:
        raise ValueError("canonical packet/query-plan lineage mismatch")

    packet_body = packet.model_dump(mode="json")
    packet_sha = packet_body.pop("packet_sha256")
    if sha256_json(packet_body) != packet_sha:
        raise ValueError("canonical packet internal SHA mismatch")

    plan_claim_ids = [
        claim.claim_id
        for group in plan.claims
        for claim in group.claims
    ]
    if len(plan_claim_ids) != EXPECTED_CLAIM_COUNT:
        raise ValueError("query-plan claim count drift")
    if len(set(plan_claim_ids)) != EXPECTED_CLAIM_COUNT:
        raise ValueError("duplicate query-plan claim ID")
    if set(plan_claim_ids) != set(reviews):
        raise ValueError("query-plan / claim-review claim-ID mismatch")
    if len(plan.claims) != EXPECTED_HYPOTHESIS_COUNT:
        raise ValueError("hypothesis count drift")

    return {
        "spec": spec,
        "report": report,
        "marker": marker,
        "plan": plan,
        "packet": packet,
        "reviews": reviews,
        "paths": paths,
    }


def validate_handoff(repo_root: Path) -> dict[str, Any]:
    root = repo_root / FROZEN_INPUT_ROOT
    manifest_path = root / "handoff_manifest.json"
    marker_path = root / "HANDOFF_PASS.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)

    manifest = read_json(manifest_path)
    marker = read_json(marker_path)
    if marker.get("status") != "handoff_pass":
        raise ValueError("HANDOFF_PASS status mismatch")
    manifest_body = dict(manifest)
    stored_manifest_sha = manifest_body.pop("manifest_sha256", None)
    if stored_manifest_sha != sha256_json(manifest_body):
        raise ValueError("handoff manifest internal SHA mismatch")
    if marker.get("manifest_sha256") != stored_manifest_sha:
        raise ValueError("handoff marker/manifest SHA mismatch")

    source = validate_frozen_source(root, flat=True)
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("handoff manifest lacks files")
    for name in HANDOFF_SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        row = expected_files.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"handoff manifest lacks {name}")
        observed = sha256_file(path)
        if row.get("target_sha256") != observed:
            raise ValueError(f"handoff target SHA drift: {name}")
        if row.get("source_sha256") != observed:
            raise ValueError(f"handoff source/target SHA mismatch: {name}")
    return source | {"manifest": manifest, "handoff_marker": marker}


def make_policy_assessor(
    policy: ExternalNoveltyPolicy | None = None,
) -> ExternalNoveltyAssessor:
    assessor = object.__new__(ExternalNoveltyAssessor)
    assessor.policy = policy or ExternalNoveltyPolicy()
    return assessor


def compute_hypothesis_synthesis(
    *,
    plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
    reviews: Mapping[str, ClaimPriorArtReview],
    policy: ExternalNoveltyPolicy | None = None,
) -> list[dict[str, Any]]:
    assessor = make_policy_assessor(policy)
    rows: list[dict[str, Any]] = []

    for group in plan.claims:
        ordered_reviews = [reviews[claim.claim_id] for claim in group.claims]
        if any(
            row.hypothesis_id != group.hypothesis_id
            for row in ordered_reviews
        ):
            raise ValueError(
                f"claim-review hypothesis lineage mismatch: "
                f"{group.hypothesis_id}"
            )

        hypothesis = SimpleNamespace(
            hypothesis_id=group.hypothesis_id,
            title=group.title,
        )
        coverage_1 = assessor._coverage(
            hypothesis,
            ordered_reviews,
            packet,
            plan,
        )
        coverage_2 = assessor._coverage(
            hypothesis,
            ordered_reviews,
            packet,
            plan,
        )
        if canonical_json(coverage_1) != canonical_json(coverage_2):
            raise RuntimeError(
                f"non-deterministic coverage: {group.hypothesis_id}"
            )

        status_1 = assessor._status(ordered_reviews, coverage_1)
        status_2 = assessor._status(ordered_reviews, coverage_2)
        if status_1 != status_2:
            raise RuntimeError(
                f"non-deterministic hypothesis status: "
                f"{group.hypothesis_id}"
            )
        status, reason_codes, interpretation = status_1

        core_reviews = [
            row for row in ordered_reviews if row.importance == "core"
        ] or ordered_reviews
        rows.append(
            {
                "hypothesis_id": group.hypothesis_id,
                "title": group.title,
                "claim_ids": [row.claim_id for row in ordered_reviews],
                "claim_statuses": [
                    {
                        "claim_id": row.claim_id,
                        "importance": row.importance,
                        "status": row.status,
                    }
                    for row in ordered_reviews
                ],
                "core_claim_ids": [row.claim_id for row in core_reviews],
                "core_claim_statuses": [row.status for row in core_reviews],
                "coverage": coverage_1.model_dump(mode="json"),
                "status": status,
                "reason_codes": list(reason_codes),
                "interpretation": interpretation,
            }
        )
    return rows


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
    policy: ExternalNoveltyPolicy | None = None,
    *,
    max_core_claims: int = 4,
) -> dict[str, Any]:
    assessor = make_policy_assessor(policy)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for core_count in range(1, max_core_claims + 1):
        for statuses in product(CLAIM_STATUSES, repeat=core_count):
            reviews = [
                _synthetic_review(status, i)
                for i, status in enumerate(statuses, start=1)
            ]
            for sufficient in (False, True):
                coverage = _synthetic_coverage(sufficient)
                observed = assessor._status(reviews, coverage)
                repeated = assessor._status(reviews, coverage)
                if observed != repeated:
                    raise RuntimeError(
                        "non-deterministic status-lattice result"
                    )
                status, reasons, interpretation = observed
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
        reviews = [
            _synthetic_review(status, i)
            for i, status in enumerate(seq, start=1)
        ]
        status, reasons, interpretation = assessor._status(
            reviews,
            _synthetic_coverage(sufficient),
        )
        return {
            "name": name,
            "core_statuses": seq,
            "coverage_sufficient": sufficient,
            "observed_status": status,
            "reason_codes": list(reasons),
            "interpretation": interpretation,
            "automatic_scientific_approval": False,
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
        "schema_version": "hypothesis-novelty-status-lattice-audit-v1",
        "semantics_id": SEMANTICS_ID,
        "claim_status_domain": list(CLAIM_STATUSES),
        "max_core_claims": max_core_claims,
        "case_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "manual_semantic_review_probes": probes,
        "rows": rows,
        "scientific_semantic_outcome": "MANUAL_REVIEW_REQUIRED",
        "case_specific_sers_expected_statuses_used": False,
    }
    body["audit_sha256"] = sha256_json(body)
    return body


def build_spec(repo_root: Path) -> dict[str, Any]:
    source = validate_handoff(repo_root)
    plan: LiteratureQueryPlan = source["plan"]
    packet: PriorArtPacket = source["packet"]
    reviews: Mapping[str, ClaimPriorArtReview] = source["reviews"]
    manifest = source["manifest"]

    policy = ExternalNoveltyPolicy()
    lattice = characterize_status_lattice(policy)

    body: dict[str, Any] = {
        "schema_version":
            "sers-hypothesis-novelty-synthesis-dev-spec-v1",
        "semantics_id": SEMANTICS_ID,
        "freeze_commit": EXPECTED_FREEZE_COMMIT,
        "source_claim_review_v3_spec_id":
            EXPECTED_CLAIM_REVIEW_V3_SPEC_ID,
        "source_claim_review_v3_run_id":
            EXPECTED_CLAIM_REVIEW_V3_RUN_ID,
        "source_ranker_run_id": EXPECTED_RANKER_RUN_ID,
        "source_query_plan_id": plan.plan_id,
        "source_query_plan_sha256": plan.plan_sha256,
        "source_canonical_packet_id": packet.packet_id,
        "source_canonical_packet_sha256": packet.packet_sha256,
        "handoff_manifest_sha256": manifest["manifest_sha256"],
        "claim_count": len(reviews),
        "core_claim_count": sum(
            row.importance == "core" for row in reviews.values()
        ),
        "hypothesis_count": len(plan.claims),
        "policy": policy.model_dump(mode="json"),
        "source_hashes": source_hashes(repo_root),
        "status_lattice_audit_sha256": lattice["audit_sha256"],
        "status_lattice_case_count": lattice["case_count"],
        "hypothesis_level_novelty_status_computed_during_spec_freeze": False,
        "validation_policy": {
            "frozen_claim_review_v3_only": True,
            "production_coverage_method_reused": True,
            "production_status_method_reused": True,
            "ranker_recomputed": False,
            "claim_reviewer_recomputed": False,
            "claim_decomposition_recomputed": False,
            "literature_retrieval_recomputed": False,
            "canonicalization_recomputed": False,
            "llm_calls": 0,
            "network_calls": 0,
            "require_repeat_exact_determinism": True,
            "automatic_scientific_status_approval": False,
            "automatic_next_stage_authorization": False,
        },
        "epistemic_policy": {
            "component_only_is_not_novelty_proof": True,
            "no_direct_match_is_not_literature_wide_absence": True,
            "partial_prior_art_is_not_hypothesis_established": True,
            "coverage_gate_required_for_absence_dependent_statuses": True,
            "scientific_novelty_status_outcome":
                "MANUAL_REVIEW_REQUIRED",
            "fresh_reserve_consumed": False,
        },
        "network_calls_during_spec_freeze": 0,
        "llm_calls_during_spec_freeze": 0,
    }
    body["spec_sha256"] = sha256_json(body)
    body["spec_id"] = (
        "sers_hypothesis_novelty_synthesis_dev_spec:"
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
    issues: list[str] = []

    body = dict(stored)
    spec_id = body.pop("spec_id", None)
    spec_sha = body.pop("spec_sha256", None)
    observed = sha256_json(body)
    if spec_sha != observed:
        issues.append("spec SHA mismatch")
    if spec_id != (
        "sers_hypothesis_novelty_synthesis_dev_spec:"
        + observed[:20]
    ):
        issues.append("spec ID mismatch")

    try:
        source = validate_handoff(repo_root)
        if stored.get("handoff_manifest_sha256") != (
            source["manifest"]["manifest_sha256"]
        ):
            issues.append("handoff manifest drift")
        if stored.get("source_hashes") != source_hashes(repo_root):
            issues.append("production source hash drift")
        if stored.get("source_query_plan_id") != source["plan"].plan_id:
            issues.append("query plan ID drift")
        if stored.get("source_canonical_packet_id") != (
            source["packet"].packet_id
        ):
            issues.append("canonical packet ID drift")
        lattice = characterize_status_lattice(ExternalNoveltyPolicy())
        if stored.get("status_lattice_audit_sha256") != (
            lattice["audit_sha256"]
        ):
            issues.append("status-lattice characterization drift")
    except Exception as exc:
        issues.append(
            f"input/source verification failed: "
            f"{type(exc).__name__}: {exc}"
        )
    return sorted(set(issues)), stored


def render_manual_audit(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SERS Hypothesis-level Novelty Synthesis-only DEV Audit",
        "",
        "- Claim-review source: **frozen v3 compiled reviews only**",
        "- Production logic: **ExternalNoveltyAssessor._coverage + _status**",
        "- LLM calls: **0**",
        "- Literature/network calls: **0**",
        "- Scientific novelty-status outcome: **MANUAL_REVIEW_REQUIRED**",
        "",
        "`NEW_COMBINATION_OF_KNOWN_EFFECTS` and `PLAUSIBLY_NOVEL` are "
        "search-bounded policy labels, not literature-wide novelty proof.",
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
                "### Core claim statuses",
                "",
            ]
        )
        for claim in row["claim_statuses"]:
            if claim["importance"] == "core":
                lines.append(
                    f"- `{claim['claim_id']}` — `{claim['status']}`"
                )
        lines.extend(["", "### Supporting claim statuses", ""])
        supporting = [
            claim
            for claim in row["claim_statuses"]
            if claim["importance"] != "core"
        ]
        if supporting:
            for claim in supporting:
                lines.append(
                    f"- `{claim['claim_id']}` — `{claim['status']}`"
                )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "### Coverage",
                "",
                f"- queries: {coverage['successful_query_count']}/"
                f"{coverage['query_count']} successful",
                f"- providers with successful executions: "
                f"{coverage['provider_success_count']}",
                f"- unique works: {coverage['unique_work_count']}",
                f"- abstract works: {coverage['abstract_work_count']}",
                f"- core claims with minimum abstract coverage: "
                f"{coverage['core_claims_with_minimum_abstract_coverage']}/"
                f"{coverage['core_claim_count']}",
                f"- sufficient for absence-based novelty: "
                f"**{coverage['sufficient_for_absence_based_novelty']}**",
                "",
                "### Production synthesis",
                "",
                f"- status: **{row['status']}**",
                f"- reason codes: `{', '.join(row['reason_codes'])}`",
                f"- interpretation: {row['interpretation']}",
                "",
                "### Human audit",
                "",
                "- semantic verdict: **PENDING**",
                "- check for overclaim/underclaim relative to the frozen "
                "claim-level evidence and coverage gate.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
