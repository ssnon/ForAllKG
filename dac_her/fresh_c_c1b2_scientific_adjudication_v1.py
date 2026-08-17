from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from dac_her.fresh_c_c1b1_reviewer_contract_v1 import (
    FINAL_ADJUDICATOR_SYSTEM_PROMPT,
    PAPER_REVIEW_SYSTEM_PROMPT,
    FreshCFinalAdjudication,
    FreshCPaperReview,
    H1,
    H2,
    H3,
    canonical_json_sha256,
    sha256_text,
)


STAGE = "C1B.2"
SEMANTICS_ID = "sers_fresh_c_c1b2_scientific_adjudication_v1"
PROTOCOL_PREFIX = "sers_fresh_c_c1b2_scientific_protocol_v1"

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
C1B0_RESULT_FREEZE = Path(
    "evaluation/sers_fresh_c/"
    "c1b0_contract_result_freeze_v1/freeze_manifest.json"
)
C1B1_REVIEWER_FREEZE = Path(
    "evaluation/sers_fresh_c/"
    "c1b1_reviewer_protocol_freeze_v1/freeze_manifest.json"
)
C1B1_R1_TRANSPORT_RESULT_FREEZE = Path(
    "evaluation/sers_fresh_c/"
    "c1b1_r1_transport_result_freeze_v1/freeze_manifest.json"
)

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_c1b2_scientific_protocol_v1.json"
)
DEFAULT_SCHEMA_QUALIFICATION_DIR = Path(
    "evaluation/sers_fresh_c/c1b2_exact_schema_qualification_v1"
)
DEFAULT_PROTOCOL_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1b2_scientific_protocol_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c1b2_scientific_adjudication_run_v1"
)
DEFAULT_RESULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1b2_scientific_result_freeze_v1"
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
EXPECTED_C1AR1_CORPUS_SHA256 = (
    "cffb7eab1465258b61ea28d64b1a703cb5a2b0cb940da0342bc7c1929db89e19"
)

EXPECTED_C1B0_FREEZE_ID = (
    "sers_fresh_c_c1b0_contract_result_freeze_v1:"
    "accdc6f461b02c13dfc0"
)
EXPECTED_C1B0_FREEZE_SHA256 = (
    "a89f6149177431a1db00d8798657b2504e455937926e7ed7fc75813d9c07fb40"
)

EXPECTED_C1B1_FREEZE_ID = (
    "sers_fresh_c_c1b1_reviewer_protocol_freeze_v1:"
    "30fc8ea1d36ec3503c21"
)
EXPECTED_C1B1_FREEZE_SHA256 = (
    "31ba6d47570935bcb9809bebe8b727b65bd4aab5831106ad96f5eb8b3cb650be"
)

EXPECTED_TRANSPORT_RESULT_FREEZE_ID = (
    "sers_fresh_c_c1b1_r1_transport_result_freeze_v1:"
    "dbbdc7f8c7f08310c9a2"
)
EXPECTED_TRANSPORT_RESULT_FREEZE_SHA256 = (
    "56678366f7749c94f5d1e6b3fdc6aecb72a3f253905563319028fc39351b3aaa"
)

EXPECTED_MODEL = "openai/gpt-5.6-luna"
EXPECTED_BASE_URL = "https://openrouter.ai/api/v1"

EXPECTED_PAPER_PROMPT_SHA256 = (
    "800f582d9c1e97a2622439d13ee70130333244dd894b52efc4edec782d69a0e8"
)
EXPECTED_FINAL_PROMPT_SHA256 = (
    "fd2d5e896b12f023a30283d6281c0e7ac3d29e5266df674ce156f2e2fe43e420"
)
EXPECTED_PAPER_SCHEMA_SHA256 = (
    "0e15619d5bf2122b586cac329c5118e1e015c0a8d0404b4c2f1a0882c2e4253b"
)
EXPECTED_FINAL_SCHEMA_SHA256 = (
    "b5ce3c3b962e63407828b5daf85c21441dfd1d899edea377308e23e98015a451"
)

EROSION_LABELS = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "CONTRADICTORY_OR_DISCONFIRMING",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def protocol_expected_id(payload: dict[str, Any]) -> str:
    tmp = dict(payload)
    tmp.pop("protocol_id", None)
    tmp.pop("protocol_sha256", None)
    return PROTOCOL_PREFIX + ":" + canonical_json_sha256(tmp)[:20]


def validate_protocol(path: Path) -> dict[str, Any]:
    p = load_object(path)
    if p.get("schema_version") != "sers-fresh-c-c1b2-scientific-protocol-v1":
        raise ValueError("C1B.2 protocol schema mismatch")
    if p.get("stage") != STAGE or p.get("semantics_id") != SEMANTICS_ID:
        raise ValueError("C1B.2 stage/semantics mismatch")
    if p.get("protocol_id") != protocol_expected_id(p):
        raise ValueError("C1B.2 protocol ID mismatch")
    tmp = dict(p)
    stored = tmp.pop("protocol_sha256", None)
    if stored != canonical_json_sha256(tmp):
        raise ValueError("C1B.2 protocol SHA mismatch")

    exact = {
        "r2_report_id": EXPECTED_R2_REPORT_ID,
        "r2_report_sha256": EXPECTED_R2_REPORT_SHA256,
        "r2_freeze_id": EXPECTED_R2_FREEZE_ID,
        "r2_freeze_sha256": EXPECTED_R2_FREEZE_SHA256,
        "c1ar1_result_freeze_id": EXPECTED_C1AR1_FREEZE_ID,
        "c1ar1_result_freeze_sha256": EXPECTED_C1AR1_FREEZE_SHA256,
        "c1ar1_corpus_sha256": EXPECTED_C1AR1_CORPUS_SHA256,
        "c1b0_result_freeze_id": EXPECTED_C1B0_FREEZE_ID,
        "c1b0_result_freeze_sha256": EXPECTED_C1B0_FREEZE_SHA256,
        "c1b1_reviewer_freeze_id": EXPECTED_C1B1_FREEZE_ID,
        "c1b1_reviewer_freeze_sha256": EXPECTED_C1B1_FREEZE_SHA256,
        "transport_result_freeze_id": EXPECTED_TRANSPORT_RESULT_FREEZE_ID,
        "transport_result_freeze_sha256": EXPECTED_TRANSPORT_RESULT_FREEZE_SHA256,
        "reviewer_model": EXPECTED_MODEL,
        "base_url": EXPECTED_BASE_URL,
        "paper_review_system_prompt_sha256": EXPECTED_PAPER_PROMPT_SHA256,
        "final_adjudicator_system_prompt_sha256": EXPECTED_FINAL_PROMPT_SHA256,
        "paper_review_schema_sha256": EXPECTED_PAPER_SCHEMA_SHA256,
        "final_adjudication_schema_sha256": EXPECTED_FINAL_SCHEMA_SHA256,
        "paper_review_order": list(range(1, 26)),
        "paper_review_calls": 25,
        "final_adjudication_calls": 1,
        "maximum_scientific_llm_calls": 26,
        "maximum_scientific_network_calls": 26,
        "temperature_parameter_sent": False,
        "deterministic_seed": 0,
        "reasoning_effort": "medium",
        "reasoning_exclude": True,
        "provider_only": ["openai"],
        "provider_allow_fallbacks": False,
        "provider_require_parameters": True,
        "provider_data_collection": "deny",
        "paper_review_max_tokens": 3500,
        "final_adjudication_max_tokens": 5000,
        "full_paper_text_truncation_allowed": False,
        "all_25_papers_must_be_processed": True,
        "cherry_pick_allowed": False,
        "external_literature_lookup_allowed": False,
        "count_threshold_novelty_inference_allowed": False,
        "single_paper_negative_absence_inference_allowed": False,
        "hypothesis_rewrite_allowed": False,
        "hypothesis_upgrade_allowed": False,
        "h2_resurrection_allowed": False,
        "repaired_reserve_index": 14,
        "repaired_positive_evidence_allowed": True,
        "repaired_absence_inference_allowed": False,
        "repaired_completeness_claim_allowed": False,
        "scientific_read_start_marker_required": True,
        "same_epoch_rerun_allowed_after_start": False,
        "failure_restores_freshness": False,
        "failure_authorizes_tuning_on_fresh_c": False,
        "operator_confirmation_required": True,
        "automatic_post_c1b2_transition_allowed": False,
        "stop_after_result_freeze": True,
    }
    for key, expected in exact.items():
        if p.get(key) != expected:
            raise ValueError(f"C1B.2 protocol field drifted: {key}")
    return p


def validate_runtime_env() -> dict[str, Any]:
    base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    if base_url != EXPECTED_BASE_URL:
        raise RuntimeError(
            f"OPENAI_BASE_URL must be {EXPECTED_BASE_URL!r}; got {base_url!r}"
        )
    model = os.getenv("FRESH_C_C1B_REVIEWER_MODEL", "").strip()
    if model != EXPECTED_MODEL:
        raise RuntimeError(
            f"FRESH_C_C1B_REVIEWER_MODEL must be {EXPECTED_MODEL!r}; got {model!r}"
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not present")
    return {
        "base_url": base_url,
        "reviewer_model": model,
        "credential_present": True,
    }


def validate_frozen_lineage(root: Path) -> dict[str, Any]:
    r2 = load_object(root / R2_REPORT)
    r2f = load_object(root / R2_FREEZE)
    c1arf = load_object(root / C1AR1_RESULT_FREEZE)
    c1b0 = load_object(root / C1B0_RESULT_FREEZE)
    c1b1 = load_object(root / C1B1_REVIEWER_FREEZE)
    transport = load_object(root / C1B1_R1_TRANSPORT_RESULT_FREEZE)

    expected_r2 = {
        "report_id": EXPECTED_R2_REPORT_ID,
        "report_sha256": EXPECTED_R2_REPORT_SHA256,
    }
    for key, value in expected_r2.items():
        if r2.get(key) != value:
            raise ValueError(f"R2 report drifted: {key}")

    if r2f.get("freeze_id") != EXPECTED_R2_FREEZE_ID:
        raise ValueError("R2 freeze ID drifted")
    if r2f.get("manifest_sha256") != EXPECTED_R2_FREEZE_SHA256:
        raise ValueError("R2 freeze SHA drifted")

    if c1arf.get("freeze_id") != EXPECTED_C1AR1_FREEZE_ID:
        raise ValueError("C1A-R1 freeze ID drifted")
    if c1arf.get("manifest_sha256") != EXPECTED_C1AR1_FREEZE_SHA256:
        raise ValueError("C1A-R1 freeze SHA drifted")
    if c1arf.get("recovered_corpus_sha256") != EXPECTED_C1AR1_CORPUS_SHA256:
        raise ValueError("C1A-R1 corpus binding drifted")

    if c1b0.get("freeze_id") != EXPECTED_C1B0_FREEZE_ID:
        raise ValueError("C1B.0 freeze ID drifted")
    if c1b0.get("manifest_sha256") != EXPECTED_C1B0_FREEZE_SHA256:
        raise ValueError("C1B.0 freeze SHA drifted")

    if c1b1.get("freeze_id") != EXPECTED_C1B1_FREEZE_ID:
        raise ValueError("C1B.1 reviewer freeze ID drifted")
    if c1b1.get("manifest_sha256") != EXPECTED_C1B1_FREEZE_SHA256:
        raise ValueError("C1B.1 reviewer freeze SHA drifted")
    if c1b1.get("reviewer_model") != EXPECTED_MODEL:
        raise ValueError("C1B.1 reviewer model drifted")

    if transport.get("freeze_id") != EXPECTED_TRANSPORT_RESULT_FREEZE_ID:
        raise ValueError("Transport result freeze ID drifted")
    if transport.get("manifest_sha256") != EXPECTED_TRANSPORT_RESULT_FREEZE_SHA256:
        raise ValueError("Transport result freeze SHA drifted")
    if transport.get("requested_model") != EXPECTED_MODEL:
        raise ValueError("Transport requested model drifted")
    if transport.get("served_model") != EXPECTED_MODEL:
        raise ValueError("Transport served model drifted")
    if transport.get("catalog_membership_verified") is not True:
        raise ValueError("Transport catalog membership not verified")
    if transport.get("structured_json_schema_call_passed") is not True:
        raise ValueError("Transport structured qualification not verified")
    if transport.get("fresh_c_scientific_text_semantic_read_performed") is not False:
        raise ValueError("Transport stage unexpectedly read Fresh-C science")

    if sha256_text(PAPER_REVIEW_SYSTEM_PROMPT) != EXPECTED_PAPER_PROMPT_SHA256:
        raise ValueError("Paper-review system prompt drifted")
    if sha256_text(FINAL_ADJUDICATOR_SYSTEM_PROMPT) != EXPECTED_FINAL_PROMPT_SHA256:
        raise ValueError("Final-adjudicator system prompt drifted")
    if canonical_json_sha256(FreshCPaperReview.model_json_schema()) != EXPECTED_PAPER_SCHEMA_SHA256:
        raise ValueError("Paper-review schema drifted")
    if canonical_json_sha256(FreshCFinalAdjudication.model_json_schema()) != EXPECTED_FINAL_SCHEMA_SHA256:
        raise ValueError("Final-adjudication schema drifted")

    return {
        "r2_report": r2,
        "transport_freeze": transport,
    }


def build_target_boundaries(r2_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = r2_report.get("hypothesis_decisions")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("R2 must contain exactly three decisions")

    selected = []
    rejected = []
    for row in rows:
        disposition = row.get("candidate_disposition")
        if disposition == "REJECT_AS_FORMULATED":
            rejected.append(row)
            continue
        if disposition not in {
            "KEEP_BOUNDED_EXTENSION",
            "KEEP_RELATIONAL_GAP_CANDIDATE",
        }:
            raise ValueError("Unexpected nonterminal R2 disposition")
        if row.get("hypothesis_rewrite_performed") is not False:
            raise ValueError("R2 target records a hypothesis rewrite")
        if row.get("residual_question_is_new_hypothesis") is not False:
            raise ValueError("R2 residual boundary became a new hypothesis")
        for key in (
            "hypothesis_id",
            "title",
            "interpretation",
            "r2_classification",
            "scientific_support",
            "residual_question",
        ):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"R2 target boundary missing {key}")
        selected.append({
            "hypothesis_id": row["hypothesis_id"],
            "title": row["title"],
            "pre_c_state": disposition,
            "r2_classification": row["r2_classification"],
            "scientific_support": row["scientific_support"],
            "boundary_interpretation": row["interpretation"],
            "residual_question": row["residual_question"],
            "residual_question_is_new_hypothesis": False,
            "use_as_positive_generation_premise": False,
        })

    if {row["hypothesis_id"] for row in selected} != {H1, H3}:
        raise ValueError("C1B.2 scientific target set is not exact H1/H3")
    if len(rejected) != 1 or rejected[0].get("hypothesis_id") != H2:
        raise ValueError("H2 terminal rejected boundary drifted")
    if rejected[0].get("hypothesis_rewrite_performed") is not False:
        raise ValueError("H2 was rewritten")
    return sorted(selected, key=lambda row: row["hypothesis_id"])


def validate_corpus_metadata(root: Path, *, parse_pages: bool) -> list[dict[str, Any]]:
    corpus = load_object(root / C1AR1_CORPUS)
    if corpus.get("corpus_sha256") != EXPECTED_C1AR1_CORPUS_SHA256:
        raise ValueError("Recovered corpus SHA drifted")
    if corpus.get("materialized_source_count") != 25:
        raise ValueError("Recovered corpus is not 25/25")
    if corpus.get("direct_original_count") != 24:
        raise ValueError("Recovered direct-original count drifted")
    if corpus.get("structurally_repaired_derivative_count") != 1:
        raise ValueError("Recovered repaired-derivative count drifted")

    records = corpus.get("records")
    if not isinstance(records, list) or len(records) != 25:
        raise ValueError("Recovered corpus records drifted")

    out = []
    for expected_index, row in enumerate(
        sorted(records, key=lambda x: x["reserve_index"]), start=1
    ):
        if row.get("reserve_index") != expected_index:
            raise ValueError("Reserve indexes are not exact 1..25")
        canonical_id = row.get("canonical_id")
        if not isinstance(canonical_id, str) or not canonical_id:
            raise ValueError("Canonical ID missing")
        mode = row.get("materialization_mode")
        if expected_index == 14:
            if mode != "STRUCTURALLY_REPAIRED_DERIVATIVE":
                raise ValueError("Reserve #14 repair provenance drifted")
            if row.get("completeness_claim_allowed") is not False:
                raise ValueError("Reserve #14 completeness policy drifted")
        else:
            if mode != "DIRECT_ORIGINAL":
                raise ValueError("Only reserve #14 may be repaired")
        if row.get("negative_absence_inference_allowed") is not False:
            raise ValueError("Single-paper negative absence policy drifted")

        text_path = root / row["materialized_text_path"]
        pages_path = root / row["pages_manifest_path"]
        if not text_path.is_file() or not pages_path.is_file():
            raise FileNotFoundError("Materialized text/pages file missing")
        if sha256_file(text_path) != row["materialized_text_sha256"]:
            raise ValueError(f"Materialized text SHA drifted at reserve {expected_index}")
        if sha256_file(pages_path) != row["pages_manifest_sha256"]:
            raise ValueError(f"Pages manifest SHA drifted at reserve {expected_index}")

        record = {
            "reserve_index": expected_index,
            "canonical_id": canonical_id,
            "materialization_mode": mode,
            "page_count": row["page_count"],
            "materialized_text_path": row["materialized_text_path"],
            "materialized_text_sha256": row["materialized_text_sha256"],
            "pages_manifest_path": row["pages_manifest_path"],
            "pages_manifest_sha256": row["pages_manifest_sha256"],
            "negative_absence_inference_allowed": False,
            "completeness_claim_allowed": row.get("completeness_claim_allowed"),
        }
        if parse_pages:
            record["pages_manifest"] = validate_pages_manifest(root, row)
        out.append(record)
    return out


def _sha256_text_json(text: str) -> str:
    return canonical_json_sha256({"text": text})


def validate_pages_manifest(root: Path, corpus_row: dict[str, Any]) -> dict[str, Any]:
    path = root / corpus_row["pages_manifest_path"]
    if sha256_file(path) != corpus_row["pages_manifest_sha256"]:
        raise ValueError("Pages manifest file SHA drifted")
    pages = load_object(path)
    expected_index = corpus_row["reserve_index"]
    if pages.get("schema_version") != "sers-fresh-c-c1a-r1-pages-v1":
        raise ValueError("Unexpected pages-manifest schema")
    if pages.get("reserve_index") != expected_index:
        raise ValueError("Pages-manifest reserve index drifted")
    if pages.get("canonical_id") != corpus_row["canonical_id"]:
        raise ValueError("Pages-manifest canonical ID drifted")
    if pages.get("materialization_mode") != corpus_row["materialization_mode"]:
        raise ValueError("Pages-manifest materialization mode drifted")
    if pages.get("negative_absence_inference_allowed") is not False:
        raise ValueError("Pages-manifest absence policy drifted")
    rows = pages.get("pages")
    if not isinstance(rows, list) or len(rows) != corpus_row["page_count"]:
        raise ValueError("Pages-manifest page count drifted")
    for number, page in enumerate(rows, start=1):
        if page.get("page_number") != number:
            raise ValueError("Page numbers are not contiguous")
        text = page.get("text")
        if not isinstance(text, str):
            raise ValueError("Page text missing")
        if page.get("text_sha256") != _sha256_text_json(text):
            raise ValueError("Page text SHA drifted")
    return pages


def format_paper_prompt(
    *,
    target_boundaries: list[dict[str, Any]],
    reserve_index: int,
    canonical_id: str,
    materialization_mode: str,
    pages_manifest: dict[str, Any],
) -> str:
    boundary_json = json.dumps(
        target_boundaries,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    page_blocks = []
    for row in pages_manifest["pages"]:
        page_blocks.append(
            f"[PAGE {row['page_number']}]\n{row['text']}\n[/PAGE {row['page_number']}]"
        )
    paper_text = "\n\n".join(page_blocks)
    repair_note = (
        "This paper is reserve #14 and was materialized from a structurally "
        "repaired derivative. Positive evidence may be used; absence and "
        "completeness inference are forbidden."
        if reserve_index == 14
        else
        "This paper was materialized directly from the frozen original PDF. "
        "Failure to find a relation in this single paper is still not negative evidence."
    )
    return (
        "Evaluate this one frozen Fresh-C paper against BOTH frozen R2 scientific "
        "boundaries below. The residual_question field is not a new hypothesis and "
        "must not be rewritten; it records the remaining differentiating boundary "
        "inside the unchanged pre-C hypothesis.\n\n"
        "FROZEN_R2_BOUNDARIES:\n"
        f"{boundary_json}\n\n"
        "PAPER_METADATA:\n"
        f"reserve_index={reserve_index}\n"
        f"canonical_id={canonical_id}\n"
        f"materialization_mode={materialization_mode}\n"
        f"repair_policy={repair_note}\n\n"
        "FROZEN_PAGE_BOUNDED_TEXT:\n"
        f"{paper_text}"
    )


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_review_grounding(
    review: FreshCPaperReview,
    *,
    expected_record: dict[str, Any],
    pages_manifest: dict[str, Any],
) -> None:
    if review.reserve_index != expected_record["reserve_index"]:
        raise ValueError("Paper review reserve index drifted")
    if review.canonical_id != expected_record["canonical_id"]:
        raise ValueError("Paper review canonical ID drifted")
    if review.materialization_mode != expected_record["materialization_mode"]:
        raise ValueError("Paper review materialization mode drifted")
    page_by_number = {
        row["page_number"]: row["text"]
        for row in pages_manifest["pages"]
    }
    for assessment in review.assessments:
        for evidence in assessment.evidence:
            page_text = page_by_number.get(evidence.page_number)
            if page_text is None:
                raise ValueError("Evidence page outside frozen paper")
            if evidence.verbatim_quote:
                quote = _normalize_ws(evidence.verbatim_quote)
                source = _normalize_ws(page_text)
                if quote not in source:
                    raise ValueError(
                        "Verbatim quote is not an exact whitespace-normalized "
                        "substring of the cited frozen page"
                    )
    if review.reserve_index == 14:
        if review.paper_level_negative_absence_inference_used is not False:
            raise ValueError("Reserve #14 used negative absence inference")
        if review.paper_level_completeness_claim_made is not False:
            raise ValueError("Reserve #14 made completeness claim")


def review_payload_sha(review: FreshCPaperReview) -> str:
    return canonical_json_sha256(review.model_dump(mode="json"))


def validate_final_against_reviews(
    final: FreshCFinalAdjudication,
    reviews: list[FreshCPaperReview],
) -> None:
    if len(reviews) != 25:
        raise ValueError("Final adjudication requires all 25 paper reviews")
    by_key = {}
    for review in reviews:
        for assessment in review.assessments:
            by_key[(review.reserve_index, assessment.hypothesis_id)] = assessment

    if final.h2_resurrected is not False:
        raise ValueError("H2 resurrection forbidden")
    if final.hypothesis_rewrite_performed is not False:
        raise ValueError("Hypothesis rewrite forbidden")
    if final.hypothesis_upgrade_performed is not False:
        raise ValueError("Hypothesis upgrade forbidden")
    if final.count_threshold_used is not False:
        raise ValueError("Count threshold forbidden")
    if final.literature_wide_novelty_claim_made is not False:
        raise ValueError("Literature-wide novelty claim forbidden")
    if final.external_literature_used is not False:
        raise ValueError("External literature forbidden")

    for ref in final.supporting_evidence:
        key = (ref.reserve_index, ref.hypothesis_id)
        assessment = by_key.get(key)
        if assessment is None:
            raise ValueError("Final evidence points outside frozen paper reviews")
        if assessment.relation_label != ref.relation_label:
            raise ValueError("Final evidence relation label does not match paper review")

    verdicts = {
        H1: final.h1_fresh_c_verdict,
        H3: final.h3_fresh_c_verdict,
    }
    for hypothesis_id, verdict in verdicts.items():
        if "ERODES_" not in verdict:
            continue
        supports = [
            ref
            for ref in final.supporting_evidence
            if ref.hypothesis_id == hypothesis_id
            and ref.relation_label in EROSION_LABELS
        ]
        if not supports:
            raise ValueError(
                "An erosion verdict requires at least one substantive "
                "Fresh-C evidence reference"
            )


def build_final_prompt(
    *,
    target_boundaries: list[dict[str, Any]],
    reviews: list[FreshCPaperReview],
) -> str:
    structured_reviews = [
        review.model_dump(mode="json")
        for review in sorted(reviews, key=lambda r: r.reserve_index)
    ]
    return (
        "Adjudicate the complete Fresh-C corpus using ONLY the 25 structured "
        "paper reviews below and the frozen R2 boundaries. A PRESERVES verdict "
        "means only that this Fresh-C corpus does not change the pre-C state; it "
        "is NOT evidence of literature-wide absence or novelty. An ERODES verdict "
        "must cite substantive supporting Fresh-C evidence. H2 remains terminally "
        "rejected. Do not rewrite or upgrade any hypothesis.\n\n"
        "FROZEN_R2_BOUNDARIES:\n"
        + json.dumps(target_boundaries, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n\nSTRUCTURED_FRESH_C_PAPER_REVIEWS:\n"
        + json.dumps(structured_reviews, ensure_ascii=False, indent=2, sort_keys=True)
    )


def schema_qualification_valid(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "sers-fresh-c-c1b2-exact-schema-qualification-v1":
        raise ValueError("Exact-schema qualification schema mismatch")
    if payload.get("requested_model") != EXPECTED_MODEL:
        raise ValueError("Qualification requested-model drift")
    if payload.get("served_model_paper") != EXPECTED_MODEL:
        raise ValueError("Paper schema served-model drift")
    if payload.get("served_model_final") != EXPECTED_MODEL:
        raise ValueError("Final schema served-model drift")
    if payload.get("paper_schema_passed") is not True:
        raise ValueError("Paper-review exact schema not qualified")
    if payload.get("final_schema_passed") is not True:
        raise ValueError("Final-adjudication exact schema not qualified")
    if payload.get("network_calls") != 2 or payload.get("llm_calls") != 2:
        raise ValueError("Exact-schema qualification call counts drifted")
    for key in (
        "fresh_c_scientific_text_used",
        "scientific_hypothesis_text_used",
        "scientific_adjudication_performed",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"Qualification safety field drifted: {key}")
