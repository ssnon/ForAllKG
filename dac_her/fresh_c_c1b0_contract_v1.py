from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SEMANTICS_ID = "sers_fresh_c_c1b0_input_contract_v1"
PROTOCOL_PREFIX = "sers_fresh_c_c1b0_contract_protocol_v1"

R2_REPORT = Path(
    "evaluation/sers_novelty_gap/"
    "r2_final_reassessment_run_v1/r2_report.json"
)
R2_FREEZE = Path(
    "evaluation/sers_novelty_gap/"
    "r2_final_reassessment_freeze_v1/freeze_manifest.json"
)
C1AR1_CORPUS = Path(
    "evaluation/sers_fresh_c/"
    "c1a_r1_recovery_run_v1/recovered_corpus_manifest.json"
)
C1AR1_RESULT_FREEZE = Path(
    "evaluation/sers_fresh_c/"
    "c1a_r1_recovery_result_freeze_v1/freeze_manifest.json"
)

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_c1b0_contract_v1_protocol.json"
)
DEFAULT_PROTOCOL_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1b0_contract_protocol_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c1b0_contract_audit_run_v1"
)
DEFAULT_RESULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1b0_contract_result_freeze_v1"
)

EXPECTED_R2_REPORT_ID = (
    "sers_r2_final_reassessment_report_v1:e9a9502cbfaa7566d457"
)
EXPECTED_R2_REPORT_SHA256 = (
    "e9a9502cbfaa7566d45753f636b23b71bc25d492877c49fc60d37d45609447d7"
)
EXPECTED_R2_FREEZE_ID = (
    "sers_r2_final_reassessment_freeze_v1:aa2f75aa46fb82284db0"
)
EXPECTED_R2_FREEZE_SHA256 = (
    "aa2f75aa46fb82284db04c5031b40227b4b2bdd12f4044ccdb8acfa84a1d3831"
)

EXPECTED_C1AR1_FREEZE_ID = (
    "sers_fresh_c_c1a_r1_recovery_result_freeze_v1:"
    "87b7254c15c383ccaa4c"
)
EXPECTED_C1AR1_FREEZE_SHA256 = (
    "b7da11545050a4562457e65ec2da80e8015482b9e3757a911a86a305e26187c5"
)
EXPECTED_C1AR1_RUN_ID = (
    "sers_fresh_c_c1a_r1_recovery_run_v1:f5ad9b566df23038b543"
)
EXPECTED_C1AR1_RUN_SHA256 = (
    "f375c190e0ee3a96751ad6cf1bd874b9a66a7be9c344aec0afa36c8ec490aa6f"
)
EXPECTED_C1AR1_CORPUS_SHA256 = (
    "cffb7eab1465258b61ea28d64b1a703cb5a2b0cb940da0342bc7c1929db89e19"
)

H1 = "direction_aware_trend_hypothesis:ad13dac8334238124899"
H2 = "direction_aware_trend_hypothesis:8507f8cadfc46d8d80de"
H3 = "direction_aware_trend_hypothesis:1cf889e57332402d88c9"

EXPECTED_R2_DECISIONS = {
    H1: "KEEP_BOUNDED_EXTENSION",
    H2: "REJECT_AS_FORMULATED",
    H3: "KEEP_RELATIONAL_GAP_CANDIDATE",
}
PRIMARY_REMAINING = H3


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def payload_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return canonical_json_sha256(value)


def expected_protocol_id(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("protocol_id", None)
    value.pop("protocol_sha256", None)
    return PROTOCOL_PREFIX + ":" + canonical_json_sha256(value)[:20]


def validate_protocol(path: Path) -> dict[str, Any]:
    p = load_object(path)
    required = {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "semantics_id",
        "stage",
        "r2_report_id",
        "r2_report_sha256",
        "r2_freeze_id",
        "r2_freeze_manifest_sha256",
        "c1ar1_result_freeze_id",
        "c1ar1_result_freeze_manifest_sha256",
        "c1ar1_run_id",
        "c1ar1_run_sha256",
        "c1ar1_corpus_sha256",
        "source_identity_count",
        "scientific_target_hypothesis_ids",
        "terminal_rejected_hypothesis_ids",
        "primary_remaining_candidate_hypothesis_id",
        "fresh_c_text_semantic_read_allowed",
        "fresh_c_text_hash_verification_allowed",
        "network_calls_allowed",
        "llm_calls",
        "automatic_c1b1_transition_allowed",
        "stop_after_audit",
    }
    if set(p) != required:
        raise ValueError(
            "C1B.0 protocol key set mismatch: "
            f"missing={sorted(required - set(p))}, "
            f"extra={sorted(set(p) - required)}"
        )
    if p["schema_version"] != "sers-fresh-c-c1b0-contract-protocol-v1":
        raise ValueError("C1B.0 protocol schema mismatch.")
    if p["semantics_id"] != SEMANTICS_ID or p["stage"] != "C1B.0":
        raise ValueError("C1B.0 protocol semantics/stage mismatch.")
    if p["protocol_id"] != expected_protocol_id(p):
        raise ValueError("C1B.0 protocol ID mismatch.")
    if p["protocol_sha256"] != payload_sha(p, "protocol_sha256"):
        raise ValueError("C1B.0 protocol SHA mismatch.")

    expected = {
        "r2_report_id": EXPECTED_R2_REPORT_ID,
        "r2_report_sha256": EXPECTED_R2_REPORT_SHA256,
        "r2_freeze_id": EXPECTED_R2_FREEZE_ID,
        "r2_freeze_manifest_sha256": EXPECTED_R2_FREEZE_SHA256,
        "c1ar1_result_freeze_id": EXPECTED_C1AR1_FREEZE_ID,
        "c1ar1_result_freeze_manifest_sha256": EXPECTED_C1AR1_FREEZE_SHA256,
        "c1ar1_run_id": EXPECTED_C1AR1_RUN_ID,
        "c1ar1_run_sha256": EXPECTED_C1AR1_RUN_SHA256,
        "c1ar1_corpus_sha256": EXPECTED_C1AR1_CORPUS_SHA256,
        "source_identity_count": 25,
        "scientific_target_hypothesis_ids": [H1, H3],
        "terminal_rejected_hypothesis_ids": [H2],
        "primary_remaining_candidate_hypothesis_id": H3,
        "fresh_c_text_semantic_read_allowed": False,
        "fresh_c_text_hash_verification_allowed": True,
        "network_calls_allowed": False,
        "llm_calls": 0,
        "automatic_c1b1_transition_allowed": False,
        "stop_after_audit": True,
    }
    for key, value in expected.items():
        if p[key] != value:
            raise ValueError(f"C1B.0 protocol field drifted: {key}")
    return p


def _contains_pair(value: Any, needle_a: str, needle_b: str) -> bool:
    """Metadata-only recursive check; does not print arbitrary scientific text."""
    if isinstance(value, dict):
        scalar_values = [
            item for item in value.values()
            if isinstance(item, (str, int, float, bool)) or item is None
        ]
        if needle_a in scalar_values and needle_b in scalar_values:
            return True
        return any(_contains_pair(v, needle_a, needle_b) for v in value.values())
    if isinstance(value, list):
        return any(_contains_pair(v, needle_a, needle_b) for v in value)
    return False


def _schema_shape(value: Any, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {
            key: _schema_shape(value[key], depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, list):
        if not value:
            return ["EMPTY"]
        # Schema only; never persist list scientific contents.
        shapes = []
        seen = set()
        for item in value[:50]:
            shape = _schema_shape(item, depth + 1)
            marker = json.dumps(shape, sort_keys=True, ensure_ascii=False)
            if marker not in seen:
                seen.add(marker)
                shapes.append(shape)
        return shapes
    return type(value).__name__


def validate_r2_lineage(root: Path) -> dict[str, Any]:
    report = load_object(root / R2_REPORT)
    freeze = load_object(root / R2_FREEZE)

    if freeze.get("freeze_id") != EXPECTED_R2_FREEZE_ID:
        raise ValueError("R2 freeze ID drifted.")
    if freeze.get("manifest_sha256") != EXPECTED_R2_FREEZE_SHA256:
        raise ValueError("R2 freeze SHA drifted.")
    if freeze.get("r2_report_id") != EXPECTED_R2_REPORT_ID:
        raise ValueError("R2 report ID binding drifted.")
    if freeze.get("r2_report_sha256") != EXPECTED_R2_REPORT_SHA256:
        raise ValueError("R2 report SHA binding drifted.")
    if freeze.get("r2_complete") is not True:
        raise ValueError("R2 is not complete.")
    if freeze.get("primary_remaining_candidate_hypothesis_id") != H3:
        raise ValueError("R2 primary remaining candidate drifted.")
    if freeze.get("secondary_bounded_extension_hypothesis_id") != H1:
        raise ValueError("R2 bounded extension identity drifted.")
    if freeze.get("rejected_as_formulated_hypothesis_ids") != [H2]:
        raise ValueError("R2 terminal rejected set drifted.")
    if freeze.get("hypothesis_rewrite_called") is not False:
        raise ValueError("R2 unexpectedly rewrote hypothesis.")
    if freeze.get("stop_after_freeze") is not True:
        raise ValueError("R2 freeze STOP drifted.")

    # Validate known decision pairs are represented somewhere in the frozen report
    # without depending on a guessed card schema.
    for hypothesis_id, decision in EXPECTED_R2_DECISIONS.items():
        if not _contains_pair(report, hypothesis_id, decision):
            raise ValueError(
                f"R2 report does not bind expected decision: "
                f"{hypothesis_id} => {decision}"
            )

    return {
        "report_top_level_keys": sorted(report),
        "report_schema_shape": _schema_shape(report),
        "r2_report_id": EXPECTED_R2_REPORT_ID,
        "r2_report_sha256": EXPECTED_R2_REPORT_SHA256,
        "r2_freeze_id": EXPECTED_R2_FREEZE_ID,
        "r2_freeze_manifest_sha256": EXPECTED_R2_FREEZE_SHA256,
        "expected_decisions": EXPECTED_R2_DECISIONS,
        "primary_remaining_candidate_hypothesis_id": H3,
        "scientific_target_hypothesis_ids": [H1, H3],
        "terminal_rejected_hypothesis_ids": [H2],
    }


def validate_c1ar1_lineage(root: Path, *, hash_text_files: bool) -> dict[str, Any]:
    corpus = load_object(root / C1AR1_CORPUS)
    freeze = load_object(root / C1AR1_RESULT_FREEZE)

    expected_fields = {
        "freeze_id": EXPECTED_C1AR1_FREEZE_ID,
        "manifest_sha256": EXPECTED_C1AR1_FREEZE_SHA256,
        "source_run_id": EXPECTED_C1AR1_RUN_ID,
        "source_run_sha256": EXPECTED_C1AR1_RUN_SHA256,
        "recovered_corpus_sha256": EXPECTED_C1AR1_CORPUS_SHA256,
        "source_identity_count": 25,
        "direct_original_count": 24,
        "structurally_repaired_derivative_count": 1,
        "fresh_reserve_c_already_consumed": True,
        "scientific_reviewer_read_performed": False,
        "scientific_adjudication_performed": False,
        "c1b_authorized": False,
        "stop": True,
    }
    for key, value in expected_fields.items():
        if freeze.get(key) != value:
            raise ValueError(f"C1A-R1 result freeze drifted: {key}")

    if corpus.get("materialized_source_count") != 25:
        raise ValueError("C1A-R1 corpus is not 25/25.")
    if corpus.get("direct_original_count") != 24:
        raise ValueError("C1A-R1 direct count drifted.")
    if corpus.get("structurally_repaired_derivative_count") != 1:
        raise ValueError("C1A-R1 repaired count drifted.")
    if corpus.get("corpus_sha256") != EXPECTED_C1AR1_CORPUS_SHA256:
        raise ValueError("C1A-R1 corpus SHA drifted.")
    if corpus.get("scientific_reviewer_read_performed") is not False:
        raise ValueError("C1A-R1 corpus was scientifically read.")
    if corpus.get("scientific_adjudication_performed") is not False:
        raise ValueError("C1A-R1 corpus was scientifically adjudicated.")

    records = corpus.get("records")
    if not isinstance(records, list) or len(records) != 25:
        raise ValueError("C1A-R1 record count drifted.")

    frozen_text_hashes = freeze.get("materialized_text_sha256")
    if not isinstance(frozen_text_hashes, dict) or len(frozen_text_hashes) != 25:
        raise ValueError("C1A-R1 frozen text hash map drifted.")

    result_records = []
    seen_indexes = set()
    seen_ids = set()
    repaired_indexes = []
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("C1A-R1 record is not an object.")
        idx = row.get("reserve_index")
        cid = row.get("canonical_id")
        path_text = row.get("materialized_text_path")
        text_sha = row.get("materialized_text_sha256")
        mode = row.get("materialization_mode")
        if not isinstance(idx, int) or not (1 <= idx <= 25):
            raise ValueError("C1A-R1 reserve index invalid.")
        if idx in seen_indexes or cid in seen_ids:
            raise ValueError("C1A-R1 duplicate reserve index/canonical ID.")
        seen_indexes.add(idx)
        seen_ids.add(cid)
        if frozen_text_hashes.get(cid) != text_sha:
            raise ValueError("C1A-R1 text hash map/record mismatch.")
        if row.get("negative_absence_inference_allowed") is not False:
            raise ValueError("C1A-R1 negative absence policy drifted.")

        if mode == "STRUCTURALLY_REPAIRED_DERIVATIVE":
            repaired_indexes.append(idx)
            if row.get("completeness_claim_allowed") is not False:
                raise ValueError("Repaired artifact completeness policy drifted.")
        elif mode != "DIRECT_ORIGINAL":
            raise ValueError("Unknown C1A-R1 materialization mode.")

        if hash_text_files:
            if not isinstance(path_text, str):
                raise ValueError("Materialized text path missing.")
            actual = sha256_file(root / path_text)
            if actual != text_sha:
                raise ValueError(
                    f"Materialized text SHA drifted at reserve index {idx}."
                )

        result_records.append({
            "reserve_index": idx,
            "canonical_id": cid,
            "materialization_mode": mode,
            "materialized_text_path": path_text,
            "materialized_text_sha256": text_sha,
            "page_count": row.get("page_count"),
            "negative_absence_inference_allowed": False,
            "completeness_claim_allowed": row.get(
                "completeness_claim_allowed"
            ),
        })

    if seen_indexes != set(range(1, 26)):
        raise ValueError("C1A-R1 reserve index set is not exactly 1..25.")
    if repaired_indexes != [14]:
        raise ValueError("Expected exactly reserve #14 as repaired derivative.")

    result_records.sort(key=lambda row: row["reserve_index"])
    return {
        "source_identity_count": 25,
        "direct_original_count": 24,
        "structurally_repaired_derivative_count": 1,
        "repaired_reserve_indexes": repaired_indexes,
        "records": result_records,
        "materialized_text_hash_verification_performed": hash_text_files,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "negative_absence_inference_from_any_single_paper_allowed": False,
        "repaired_derivative_completeness_claim_allowed": False,
    }
