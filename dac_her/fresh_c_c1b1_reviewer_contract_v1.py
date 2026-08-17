import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


STAGE = "C1B.1"
SEMANTICS_ID = "sers_fresh_c_c1b1_scientific_reviewer_contract_v1"
PROTOCOL_PREFIX = "sers_fresh_c_c1b1_reviewer_protocol_v1"

H1 = "direction_aware_trend_hypothesis:ad13dac8334238124899"
H2 = "direction_aware_trend_hypothesis:8507f8cadfc46d8d80de"
H3 = "direction_aware_trend_hypothesis:1cf889e57332402d88c9"

SOURCE_C1B0_FREEZE = Path(
    "evaluation/sers_fresh_c/"
    "c1b0_contract_result_freeze_v1/freeze_manifest.json"
)
EXPECTED_C1B0_FREEZE_ID = (
    "sers_fresh_c_c1b0_contract_result_freeze_v1:"
    "accdc6f461b02c13dfc0"
)
EXPECTED_C1B0_FREEZE_SHA256 = (
    "a89f6149177431a1db00d8798657b2504e455937926e7ed7fc75813d9c07fb40"
)
EXPECTED_C1B0_CONTRACT_ID = (
    "sers_fresh_c_c1b0_input_contract_v1:ed6046eaf585ddd7eacd"
)
EXPECTED_C1B0_CONTRACT_SHA256 = (
    "ed6046eaf585ddd7eacd4742a072ee2fbe2b74f486ce53dfea3f1b208519a67a"
)
EXPECTED_C1AR1_CORPUS_SHA256 = (
    "cffb7eab1465258b61ea28d64b1a703cb5a2b0cb940da0342bc7c1929db89e19"
)

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_c1b1_reviewer_protocol_v1.json"
)
DEFAULT_PROTOCOL_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1b1_reviewer_protocol_freeze_v1"
)

RELATION_LABELS = (
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "COMPONENTS_ONLY",
    "CONTEXT_ONLY",
    "CONTRADICTORY_OR_DISCONFIRMING",
    "IRRELEVANT",
    "UNRESOLVED_INSUFFICIENT_EXTRACTION",
)

POSITIVE_OR_SUBSTANTIVE_LABELS = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "COMPONENTS_ONLY",
    "CONTEXT_ONLY",
    "CONTRADICTORY_OR_DISCONFIRMING",
}

H1_FINAL_VERDICTS = (
    "FRESH_C_PRESERVES_PRE_C_BOUNDED_EXTENSION",
    "FRESH_C_ERODES_PRE_C_BOUNDED_EXTENSION",
    "FRESH_C_INCONCLUSIVE",
)
H3_FINAL_VERDICTS = (
    "FRESH_C_PRESERVES_PRE_C_RELATIONAL_GAP",
    "FRESH_C_ERODES_PRE_C_RELATIONAL_GAP",
    "FRESH_C_INCONCLUSIVE",
)

PAPER_REVIEW_SYSTEM_PROMPT = """You are the Fresh Reserve C scientific prior-art reviewer.
You evaluate ONLY the supplied frozen paper text against the supplied frozen hypotheses.
You must not search external literature, infer literature-wide absence, or rewrite a hypothesis.
A paper containing known components is not direct prior art unless the paper supports the same
scientific relation that defines the hypothesis. Use DIRECT_PRIOR_ART only for a direct match
to the hypothesis relation. Use PARTIAL_PRIOR_ART when only part of the relation is supported.
Use COMPONENTS_ONLY when relevant components are present but the relation is not established.
Use CONTEXT_ONLY for background/context overlap. Use CONTRADICTORY_OR_DISCONFIRMING when the
paper provides evidence against the stated relation. Use IRRELEVANT when it does not bear on
the hypothesis. Use UNRESOLVED_INSUFFICIENT_EXTRACTION when the supplied extraction is not
sufficient for a scientific classification.

Every substantive classification must be grounded to page-numbered evidence. Provide a concise
paraphrase. A verbatim quote is optional and, when used, must be no more than 20 words.
Do not treat failure to find a relation in one paper as evidence that the relation is absent.
Do not use counts or thresholds to establish novelty. Do not propose refinements, rewrites,
upgrades, or new hypotheses. H2 is terminally rejected and is not a review target.
For repaired reserve #14, positive evidence may be used, but absence and completeness claims
are forbidden. Return only the required structured schema."""

FINAL_ADJUDICATOR_SYSTEM_PROMPT = """You are the Fresh Reserve C corpus-level scientific adjudicator.
You receive the frozen R2 states and the complete set of 25 structured paper reviews.
You must evaluate only whether this Fresh-C corpus preserves, erodes, or leaves inconclusive
the pre-C scientific status. You must not claim literature-wide novelty or absence.
You must not use numerical paper-count thresholds. Direct or partial prior-art evidence must
be interpreted scientifically according to the relation actually supported, not by frequency.
H1 may be preserved, eroded, or inconclusive but never upgraded or rewritten.
H2 remains terminally REJECT_AS_FORMULATED and cannot be resurrected.
H3 may preserve its relational-gap candidate status, be eroded, or be inconclusive but never
upgraded or rewritten. Cite supporting reserve indexes and relation labels. For repaired
reserve #14, positive evidence may support erosion, but missing evidence may not support
preservation or completeness. Return only the required structured schema."""


def canonical_json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha_without(payload, field):
    tmp = dict(payload)
    tmp.pop(field, None)
    return canonical_json_sha256(tmp)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceLocator(StrictModel):
    page_number: int = Field(ge=1)
    evidence_paraphrase: str = Field(min_length=1, max_length=700)
    verbatim_quote: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _bounded_quote(self):
        if self.verbatim_quote is not None:
            words = self.verbatim_quote.strip().split()
            if len(words) > 20:
                raise ValueError("verbatim_quote exceeds 20 words")
        return self


class HypothesisPaperAssessment(StrictModel):
    hypothesis_id: Literal[
        "direction_aware_trend_hypothesis:ad13dac8334238124899",
        "direction_aware_trend_hypothesis:1cf889e57332402d88c9",
    ]
    relation_label: Literal[
        "DIRECT_PRIOR_ART",
        "PARTIAL_PRIOR_ART",
        "COMPONENTS_ONLY",
        "CONTEXT_ONLY",
        "CONTRADICTORY_OR_DISCONFIRMING",
        "IRRELEVANT",
        "UNRESOLVED_INSUFFICIENT_EXTRACTION",
    ]
    scientific_rationale: str = Field(min_length=1, max_length=1600)
    evidence: list[EvidenceLocator] = Field(default_factory=list)
    negative_absence_inference_used: Literal[False] = False
    count_threshold_used: Literal[False] = False
    literature_wide_novelty_claim_made: Literal[False] = False
    hypothesis_rewrite_proposed: Literal[False] = False
    hypothesis_upgrade_proposed: Literal[False] = False

    @model_validator(mode="after")
    def _substantive_requires_evidence(self):
        if (
            self.relation_label in POSITIVE_OR_SUBSTANTIVE_LABELS
            and not self.evidence
        ):
            raise ValueError(
                f"{self.relation_label} requires page-grounded evidence"
            )
        return self


class FreshCPaperReview(StrictModel):
    reserve_index: int = Field(ge=1, le=25)
    canonical_id: str = Field(min_length=1)
    materialization_mode: Literal[
        "DIRECT_ORIGINAL",
        "STRUCTURALLY_REPAIRED_DERIVATIVE",
    ]
    assessments: list[HypothesisPaperAssessment] = Field(min_length=2, max_length=2)
    paper_level_negative_absence_inference_used: Literal[False] = False
    paper_level_completeness_claim_made: Literal[False] = False

    @model_validator(mode="after")
    def _exact_targets_and_repair_policy(self):
        ids = [item.hypothesis_id for item in self.assessments]
        if set(ids) != {H1, H3} or len(ids) != len(set(ids)):
            raise ValueError("Each paper must assess H1 and H3 exactly once")
        if self.reserve_index == 14:
            if self.materialization_mode != "STRUCTURALLY_REPAIRED_DERIVATIVE":
                raise ValueError("Reserve #14 must retain repaired provenance")
            if self.paper_level_completeness_claim_made is not False:
                raise ValueError("Reserve #14 completeness claim forbidden")
        elif self.materialization_mode != "DIRECT_ORIGINAL":
            raise ValueError("Only reserve #14 may use repaired derivative")
        return self


class FinalEvidenceReference(StrictModel):
    reserve_index: int = Field(ge=1, le=25)
    hypothesis_id: Literal[
        "direction_aware_trend_hypothesis:ad13dac8334238124899",
        "direction_aware_trend_hypothesis:1cf889e57332402d88c9",
    ]
    relation_label: Literal[
        "DIRECT_PRIOR_ART",
        "PARTIAL_PRIOR_ART",
        "COMPONENTS_ONLY",
        "CONTEXT_ONLY",
        "CONTRADICTORY_OR_DISCONFIRMING",
        "IRRELEVANT",
        "UNRESOLVED_INSUFFICIENT_EXTRACTION",
    ]
    scientific_role: str = Field(min_length=1, max_length=700)


class FreshCFinalAdjudication(StrictModel):
    h1_hypothesis_id: Literal[
        "direction_aware_trend_hypothesis:ad13dac8334238124899"
    ] = H1
    h1_pre_c_state: Literal["KEEP_BOUNDED_EXTENSION"] = "KEEP_BOUNDED_EXTENSION"
    h1_fresh_c_verdict: Literal[
        "FRESH_C_PRESERVES_PRE_C_BOUNDED_EXTENSION",
        "FRESH_C_ERODES_PRE_C_BOUNDED_EXTENSION",
        "FRESH_C_INCONCLUSIVE",
    ]
    h1_rationale: str = Field(min_length=1, max_length=2200)

    h2_hypothesis_id: Literal[
        "direction_aware_trend_hypothesis:8507f8cadfc46d8d80de"
    ] = H2
    h2_terminal_state: Literal["REJECT_AS_FORMULATED"] = "REJECT_AS_FORMULATED"
    h2_resurrected: Literal[False] = False

    h3_hypothesis_id: Literal[
        "direction_aware_trend_hypothesis:1cf889e57332402d88c9"
    ] = H3
    h3_pre_c_state: Literal[
        "KEEP_RELATIONAL_GAP_CANDIDATE"
    ] = "KEEP_RELATIONAL_GAP_CANDIDATE"
    h3_fresh_c_verdict: Literal[
        "FRESH_C_PRESERVES_PRE_C_RELATIONAL_GAP",
        "FRESH_C_ERODES_PRE_C_RELATIONAL_GAP",
        "FRESH_C_INCONCLUSIVE",
    ]
    h3_rationale: str = Field(min_length=1, max_length=2200)

    supporting_evidence: list[FinalEvidenceReference] = Field(default_factory=list)
    all_25_papers_processed: Literal[True] = True
    count_threshold_used: Literal[False] = False
    literature_wide_novelty_claim_made: Literal[False] = False
    hypothesis_rewrite_performed: Literal[False] = False
    hypothesis_upgrade_performed: Literal[False] = False
    external_literature_used: Literal[False] = False


def validate_source_c1b0_freeze(root):
    payload = load_object(Path(root) / SOURCE_C1B0_FREEZE)
    expected = {
        "freeze_id": EXPECTED_C1B0_FREEZE_ID,
        "manifest_sha256": EXPECTED_C1B0_FREEZE_SHA256,
        "contract_id": EXPECTED_C1B0_CONTRACT_ID,
        "contract_sha256": EXPECTED_C1B0_CONTRACT_SHA256,
        "source_identity_count": 25,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "c1b1_authorized": False,
        "stop": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"C1B.0 result freeze drifted: {key}")
    if payload.get("c1ar1_corpus_sha256") != EXPECTED_C1AR1_CORPUS_SHA256:
        raise ValueError("C1B.0 C1A-R1 corpus binding drifted")
    return payload


def protocol_expected_id(payload):
    tmp = dict(payload)
    tmp.pop("protocol_id", None)
    tmp.pop("protocol_sha256", None)
    return PROTOCOL_PREFIX + ":" + canonical_json_sha256(tmp)[:20]


def validate_protocol(path):
    p = load_object(path)
    if p.get("schema_version") != "sers-fresh-c-c1b1-reviewer-protocol-v1":
        raise ValueError("C1B.1 protocol schema mismatch")
    if p.get("stage") != STAGE or p.get("semantics_id") != SEMANTICS_ID:
        raise ValueError("C1B.1 stage/semantics mismatch")
    if p.get("protocol_id") != protocol_expected_id(p):
        raise ValueError("C1B.1 protocol ID mismatch")
    if p.get("protocol_sha256") != sha_without(p, "protocol_sha256"):
        raise ValueError("C1B.1 protocol SHA mismatch")

    exact = {
        "source_c1b0_result_freeze_id": EXPECTED_C1B0_FREEZE_ID,
        "source_c1b0_result_freeze_sha256": EXPECTED_C1B0_FREEZE_SHA256,
        "source_c1b0_contract_id": EXPECTED_C1B0_CONTRACT_ID,
        "source_c1b0_contract_sha256": EXPECTED_C1B0_CONTRACT_SHA256,
        "source_c1ar1_corpus_sha256": EXPECTED_C1AR1_CORPUS_SHA256,
        "source_identity_count": 25,
        "paper_review_order": list(range(1, 26)),
        "paper_review_calls": 25,
        "final_adjudication_calls": 1,
        "maximum_llm_calls": 26,
        "paper_review_targets": [H1, H3],
        "terminal_rejected_hypothesis_ids": [H2],
        "external_literature_lookup_allowed": False,
        "hypothesis_rewrite_allowed": False,
        "hypothesis_upgrade_allowed": False,
        "count_threshold_novelty_inference_allowed": False,
        "negative_absence_inference_from_single_paper_allowed": False,
        "repaired_reserve_index": 14,
        "repaired_positive_evidence_allowed": True,
        "repaired_absence_inference_allowed": False,
        "repaired_completeness_claim_allowed": False,
        "scientific_text_read_during_c1b1": False,
        "network_calls_during_c1b1": 0,
        "llm_calls_during_c1b1": 0,
        "automatic_c1b2_transition_allowed": False,
        "stop_after_freeze": True,
    }
    for key, value in exact.items():
        if p.get(key) != value:
            raise ValueError(f"C1B.1 protocol field drifted: {key}")

    if p.get("paper_review_system_prompt_sha256") != sha256_text(
        PAPER_REVIEW_SYSTEM_PROMPT
    ):
        raise ValueError("Paper reviewer prompt hash drifted")
    if p.get("final_adjudicator_system_prompt_sha256") != sha256_text(
        FINAL_ADJUDICATOR_SYSTEM_PROMPT
    ):
        raise ValueError("Final adjudicator prompt hash drifted")
    if p.get("paper_review_schema_sha256") != canonical_json_sha256(
        FreshCPaperReview.model_json_schema()
    ):
        raise ValueError("Paper reviewer schema hash drifted")
    if p.get("final_adjudication_schema_sha256") != canonical_json_sha256(
        FreshCFinalAdjudication.model_json_schema()
    ):
        raise ValueError("Final adjudicator schema hash drifted")

    if p.get("reviewer_backend") != "openai_chat_completions_json_schema_v1":
        raise ValueError("Reviewer backend drifted")
    if p.get("reviewer_model_env") != "FRESH_C_C1B_REVIEWER_MODEL":
        raise ValueError("Reviewer model env drifted")
    if p.get("api_key_env") != "OPENAI_API_KEY":
        raise ValueError("Reviewer API key env drifted")
    if p.get("temperature") != 0.0:
        raise ValueError("Temperature drifted")
    if p.get("paper_review_max_tokens") != 3500:
        raise ValueError("Paper max token budget drifted")
    if p.get("final_adjudication_max_tokens") != 5000:
        raise ValueError("Final max token budget drifted")
    return p


def validate_model_env():
    model = os.getenv("FRESH_C_C1B_REVIEWER_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "FRESH_C_C1B_REVIEWER_MODEL must be set before C1B.1 protocol freeze"
        )
    if any(ch.isspace() for ch in model):
        raise RuntimeError("Reviewer model must be a single model identifier")
    return model
