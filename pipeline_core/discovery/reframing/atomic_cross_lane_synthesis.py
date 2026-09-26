from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
    NoveltyClaimScientificStructure,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.atomic_scientific_specification import (
    AtomicClaimKind,
    AtomicSelectionRole,
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    LiteratureQueryPlanner,
    _clean_branch_specific_bridge,
    _clean_branch_specific_specification,
)
from pipeline_core.discovery.novelty_structure_validation import (
    compile_claim_scientific_structure,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
    ProductionScientificCandidate,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AtomicSynthesisKind = Literal[
    "complementary_mechanism_integration",
    "conditional_relation_refinement",
    "competing_model_formulation",
    "measurement_model_integration",
    "cross_lane_explanatory_synthesis",
]

class AtomicSpecificationDraft(StrictModel):
    local_id: str = Field(min_length=1)
    kind: AtomicClaimKind
    importance: Literal["core", "supporting"] = "core"
    novelty_selection_role: AtomicSelectionRole
    text: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    source_candidate_ids: list[str] = Field(min_length=1)
    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)

    prior_art_identity_terms: list[str] = Field(min_length=1)
    relation_endpoint_anchors: list[str] = Field(min_length=2)
    scope_qualifier_spans: list[str] = Field(default_factory=list)
    directional_qualifier_spans: list[str] = Field(default_factory=list)
    distinguishing_terms: list[str] = Field(default_factory=list)

    required_bridge: str = Field(min_length=1)
    observable: str = Field(min_length=1)
    predicted_observation: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)

    search_concepts: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(min_length=1, max_length=2)
    scientific_structure: NoveltyClaimScientificStructure = Field(
        default_factory=NoveltyClaimScientificStructure
    )

    @model_validator(mode="after")
    def validate_unique_refs(self) -> "AtomicSpecificationDraft":
        for label, values in (
            ("source_candidate_ids", self.source_candidate_ids),
            ("premise_statement_ids", self.premise_statement_ids),
            ("gap_statement_ids", self.gap_statement_ids),
            ("prior_art_identity_terms", self.prior_art_identity_terms),
            ("relation_endpoint_anchors", self.relation_endpoint_anchors),
            ("search_queries", self.search_queries),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self

    @model_validator(mode="after")
    def validate_text_contract_for_structured_retry(self) -> "AtomicSpecificationDraft":
        # Keep the compiler fail-closed check below, but surface the same
        # deterministic model-authored text contract here so Instructor can
        # repair schema-valid-but-contract-invalid drafts within parse retries.
        _validate_atomic_text_contract(self)
        return self


class AtomicCrossLaneHypothesisDraft(StrictModel):
    local_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=2)
    synthesis_kind: AtomicSynthesisKind
    hypothesis_statement: str = Field(min_length=1)
    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    atomic_specifications: list[AtomicSpecificationDraft] = Field(
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_shape(self) -> "AtomicCrossLaneHypothesisDraft":
        if len(self.source_candidate_ids) != len(set(self.source_candidate_ids)):
            raise ValueError("source_candidate_ids must be unique")
        local_ids = [row.local_id for row in self.atomic_specifications]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("atomic specification local_id values must be unique")
        if not any(
            row.novelty_selection_role == "NOVELTY_BEARING"
            for row in self.atomic_specifications
        ):
            raise ValueError(
                "atomic synthesis requires at least one NOVELTY_BEARING specification"
            )
        if not any(row.importance == "core" for row in self.atomic_specifications):
            raise ValueError("atomic synthesis requires at least one core specification")
        return self


class AtomicCrossLaneSynthesisBatchDraft(StrictModel):
    schema_version: Literal[
        "atomic-cross-lane-scientific-synthesis-batch-draft-v1"
    ] = "atomic-cross-lane-scientific-synthesis-batch-draft-v1"
    hypotheses: list[AtomicCrossLaneHypothesisDraft] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_abstention(self) -> "AtomicCrossLaneSynthesisBatchDraft":
        if not self.hypotheses and not self.abstention_reason:
            raise ValueError("empty atomic synthesis requires abstention_reason")
        if self.hypotheses and self.abstention_reason:
            raise ValueError("non-empty atomic synthesis must not carry abstention_reason")
        return self


class CompiledAtomicCrossLaneHypothesis(StrictModel):
    hypothesis_id: str
    source_candidate_ids: list[str]
    source_lanes: list[Literal["RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"]]
    synthesis_kind: AtomicSynthesisKind
    title: str
    hypothesis_statement: str
    premise_statement_ids: list[str]
    gap_statement_ids: list[str]
    assumptions: list[str]
    atomic_specifications: list[CompiledAtomicSpecification]


class AtomicCrossLaneSynthesisReport(StrictModel):
    schema_version: Literal[
        "atomic-cross-lane-scientific-synthesis-report-v1"
    ] = "atomic-cross-lane-scientific-synthesis-report-v1"

    report_id: str
    source_candidate_portfolio_id: str
    source_task_id: str
    source_context_id: str
    question: str
    backend_name: str
    model_name: str
    hypotheses: list[CompiledAtomicCrossLaneHypothesis]
    hypothesis_count: int = Field(ge=0)
    atomic_specification_count: int = Field(ge=0)
    llm_calls_performed: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    abstention_reason: str | None = None

    source_candidates_preserved: Literal[True] = True
    external_novelty_llm_redecomposition_required: Literal[False] = False
    branch_specific_sanitizers_reused: Literal[True] = True
    relation_nucleus_terms_model_authored: Literal[False] = False
    base_relation_nucleus_derived_from_endpoint_anchors: Literal[True] = True
    required_bridge_materialized_as_exact_hypothesis_source: Literal[True] = True
    prediction_falsifier_materialized_before_n9: Literal[True] = True
    novelty_assessment_performed: Literal[False] = False
    production_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "AtomicCrossLaneSynthesisReport":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        total = sum(len(row.atomic_specifications) for row in self.hypotheses)
        if self.atomic_specification_count != total:
            raise ValueError("atomic_specification_count mismatch")
        return self


@dataclass(frozen=True)
class AtomicSynthesisPrompt:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class AtomicSynthesisGeneration:
    draft: AtomicCrossLaneSynthesisBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None


_HYPOTHESIS_TYPE_MAP = {
    "complementary_mechanism_integration": "cross_evidence_synthesis",
    "conditional_relation_refinement": "context_dependency",
    "competing_model_formulation": "mechanistic_extension",
    "measurement_model_integration": "descriptor_mediation",
    "cross_lane_explanatory_synthesis": "cross_evidence_synthesis",
}


def _sha256_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _normalize(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐-‒–—−-]+", " ", text)
    text = re.sub(r"[^\w\s+*/().,]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _surface_contains(text: str, phrase: str) -> bool:
    needle = _normalize(phrase)
    haystack = _normalize(text)
    return bool(needle and needle in haystack)


def _single_sentence(text: str) -> bool:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned or cleaned[-1] not in ".!?":
        return False
    pieces = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", cleaned)
        if part.strip()
    ]
    return len(pieces) == 1


def _candidate_payload(
    row: ProductionScientificCandidate,
    *,
    candidate_ref: str,
) -> dict[str, object]:
    return {
        "candidate_ref": candidate_ref,
        "source_lane": row.source_lane,
        "source_object_kind": row.source_object_kind,
        "reasoning_label": row.reasoning_label,
        "title": row.title,
        "scientific_proposal": row.scientific_proposal,
        "reasoning_rationale": row.reasoning_rationale,
        "premise_statement_ids": list(row.premise_statement_ids),
        "gap_statement_ids": list(row.gap_statement_ids),
        "assumptions": list(row.assumptions),
        "predictions": [row.model_dump(mode="json") for row in row.predictions],
        "falsifiers": [row.model_dump(mode="json") for row in row.falsifiers],
        "discriminating_test": (
            row.discriminating_test.model_dump(mode="json")
            if row.discriminating_test is not None
            else None
        ),
        "unresolved_questions": list(row.unresolved_questions),
    }


def build_atomic_synthesis_prompt(
    portfolio: ProductionFacingScientificCandidatePortfolio,
    *,
    max_syntheses: int,
) -> AtomicSynthesisPrompt:
    if max_syntheses < 1:
        raise ValueError("max_syntheses must be at least 1")

    system = """You perform source-bounded cross-lane scientific synthesis for downstream strict N9/N10 evaluation.

The existing source candidates remain preserved. Your output is NOT a novelty judgment and NOT production authority.

For each synthesized hypothesis, emit 1-4 ATOMIC scientific specifications at the same time as the hypothesis. Do not leave atomic decomposition for a later model.

Hard constraints:
- Each hypothesis must cite at least one RELATIONAL_DISCOVERY candidate and at least one SCIENTIFIC_REFRAMING candidate.
- Each synthesized hypothesis must include at least one atomic specification with novelty_selection_role exactly "NOVELTY_BEARING".
- Use NOVELTY_BEARING only for a source-bounded differentiating scientific relation that downstream N9/N10 should evaluate for novelty; TESTING_PREDICTION alone is not sufficient.
- If no source-bounded NOVELTY_BEARING relation is justified, abstain rather than emitting a hypothesis whose specifications are all enabling, testing, or auxiliary.
- Use candidate_ref aliases exactly. Do not invent canonical IDs.
- premise_statement_ids and gap_statement_ids must come only from cited candidates.
- Every atomic specification must cite source_candidate_ids that are a subset of its parent synthesis source candidates.
- Every atomic specification must be a self-contained single scientific relation, not an umbrella summary or conjunction of unrelated relations.
- required_bridge must be ONE standalone sentence ending in punctuation. It must explicitly state the scientific inference connecting the atomic relation.
- prior_art_identity_terms must explicitly occur in the atomic text, required_bridge, predicted_observation, and falsification_condition. Prefer one precise branch identity phrase.
- relation_endpoint_anchors must identify TWO OR MORE scientific entities/variables at the ends of the lower-order BASE relation. They must be literal phrases that occur in BOTH atomic text and required_bridge. Do not use generic relation operators such as "associated with lower", "will have lower", "increases", or "decreases" as endpoints.
- relation_nucleus_terms are compiler-owned and MUST NOT be authored by the model. The compiler deterministically derives the BASE retrieval nucleus from relation_endpoint_anchors plus any explicit directional_qualifier_spans.
- prior_art_identity_terms describe the distinguishing branch/factor/context. Do not substitute generic relation operators for either BASE relation endpoint.
- If you list scope_qualifier_spans or directional_qualifier_spans, each listed phrase must occur literally in BOTH atomic text and required_bridge. Otherwise leave that list empty.
- predicted_observation must state a testable expected result for this exact atomic branch.
- falsification_condition must state an outcome that would falsify this exact atomic branch.
- observable names what is measured; do not use it as a substitute for the prediction.
- search_queries are prior-art search phrases only. They provide no positive evidence.
- scientific_structure must be conservative. Any non-default structural label must include an exact source_text basis copied from this synthesized hypothesis's hypothesis_statement, required_bridge, prediction, falsifier, or assumption and must name this atomic branch identity.
- Do not claim novelty, priority, literature absence, truth, superiority, or empirical confirmation.
- Do not invent papers, evidence, measurements, statement IDs, or unsupported mechanisms.
- If genuine synthesis cannot be represented under these atomic constraints, abstain instead of weakening the specification.
"""

    payload = {
        "question": portfolio.question,
        "max_syntheses": max_syntheses,
        "source_candidates": [
            _candidate_payload(row, candidate_ref=f"CANDIDATE_{index:02d}")
            for index, row in enumerate(portfolio.candidates, start=1)
        ],
    }
    user = (
        "Construct up to max_syntheses cross-lane hypotheses under the atomic "
        "specification contract. Return abstention_reason if no valid synthesis "
        "is justified.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return AtomicSynthesisPrompt(system_prompt=system, user_prompt=user)


class InstructorAtomicSynthesisBackend:
    backend_name = "instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key_env = api_key_env
        self.api_key = os.getenv(api_key_env)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env}."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Atomic cross-lane synthesis requires 'openai' and 'instructor'."
            ) from exc

        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            raise ValueError(f"Unknown Instructor mode: {self.instructor_mode}")
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        self._client = instructor.from_openai(OpenAI(**kwargs), mode=mode)
        return self._client

    def generate(self, prompt: AtomicSynthesisPrompt) -> AtomicSynthesisGeneration:
        client = self._get_client()
        draft, event = run_instructor_structured_call(
            client.chat.completions,
            model=self.model_name,
            response_model=AtomicCrossLaneSynthesisBatchDraft,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "atomic_cross_lane_scientific_synthesis",
                "stage": "atomic_synthesis",
                "call_kind": "scientific_hypothesis_synthesis",
            },
            semantic_components={
                "synthesis_contract": "cross_lane_atomic_v1",
            },
        )
        if not isinstance(draft, AtomicCrossLaneSynthesisBatchDraft):
            draft = AtomicCrossLaneSynthesisBatchDraft.model_validate(draft)
        return AtomicSynthesisGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
        )


def _candidate_ref_map(
    portfolio: ProductionFacingScientificCandidatePortfolio,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, row in enumerate(portfolio.candidates, start=1):
        result[f"CANDIDATE_{index:02d}"] = row.candidate_id
        result[row.candidate_id] = row.candidate_id
    return result


def _resolve_refs(
    values: list[str],
    *,
    ref_map: Mapping[str, str],
    label: str,
) -> list[str]:
    unknown = sorted(set(values) - set(ref_map))
    if unknown:
        raise ValueError(f"{label} contains unknown candidate refs: {unknown}")
    return [ref_map[value] for value in values]


def _context_statement_sets(
    context: HypothesisContext,
) -> tuple[dict[str, object], set[str]]:
    statements = {
        row.statement_id: row
        for row in context.evidence_statements
    }
    research_gaps = {
        row.statement_id
        for row in context.research_gaps
    }
    return statements, research_gaps



_RELATION_OPERATOR_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "associated",
        "association",
        "at",
        "be",
        "between",
        "by",
        "conditioned",
        "controls",
        "decrease",
        "decreases",
        "decreased",
        "depend",
        "depends",
        "effect",
        "effects",
        "for",
        "from",
        "greater",
        "has",
        "have",
        "higher",
        "in",
        "increase",
        "increases",
        "increased",
        "is",
        "less",
        "lower",
        "more",
        "of",
        "on",
        "relation",
        "relationship",
        "than",
        "the",
        "to",
        "under",
        "with",
        "will",
    }
)


def _scientific_endpoint_tokens(value: str) -> tuple[str, ...]:
    tokens = tuple(
        token.casefold()
        for token in re.findall(
            r"\w+",
            _normalize(value),
            flags=re.UNICODE,
        )
        if token
    )
    return tuple(
        token
        for token in tokens
        if token not in _RELATION_OPERATOR_TOKENS
    )


def _derive_base_relation_nucleus(
    row: AtomicSpecificationDraft,
) -> list[str]:
    """Derive BASE retrieval terms from validated scientific endpoints.

    relation_nucleus_terms is intentionally compiler-owned. This prevents
    generic relation operators such as "associated with lower" from becoming
    standalone BASE_RELATION queries.

    Directional qualifiers are appended only after the endpoint pair, so every
    BASE query necessarily contains the scientific variables/entities.
    """

    endpoints = list(
        dict.fromkeys(
            " ".join(value.split())
            for value in row.relation_endpoint_anchors
            if value.strip()
        )
    )
    if len(endpoints) < 2:
        raise ValueError(
            f"atomic specification {row.local_id} requires at least two "
            "distinct relation_endpoint_anchors for BASE retrieval"
        )

    normalized_endpoints = [_normalize(value) for value in endpoints]
    if len(normalized_endpoints) != len(set(normalized_endpoints)):
        raise ValueError(
            f"atomic specification {row.local_id} relation endpoints "
            "collapse after conservative normalization"
        )

    for endpoint in endpoints:
        if not _scientific_endpoint_tokens(endpoint):
            raise ValueError(
                f"atomic specification {row.local_id} relation endpoint "
                f"contains no scientific content beyond generic relation "
                f"operators: {endpoint}"
            )

    directional = list(
        dict.fromkeys(
            " ".join(value.split())
            for value in row.directional_qualifier_spans
            if value.strip()
        )
    )
    return [*endpoints, *directional]


def _validate_atomic_text_contract(
    row: AtomicSpecificationDraft,
) -> None:
    if not _single_sentence(row.required_bridge):
        raise ValueError(
            f"atomic specification {row.local_id} required_bridge must be one sentence"
        )

    bridge = _clean_branch_specific_bridge(
        row.required_bridge,
        list(row.prior_art_identity_terms),
        [row.required_bridge],
    )
    if bridge != row.required_bridge:
        raise ValueError(
            f"atomic specification {row.local_id} required_bridge fails existing branch sanitizer"
        )

    prediction = _clean_branch_specific_specification(
        row.predicted_observation,
        list(row.prior_art_identity_terms),
    )
    if prediction != row.predicted_observation:
        raise ValueError(
            f"atomic specification {row.local_id} prediction fails existing branch sanitizer"
        )

    falsifier = _clean_branch_specific_specification(
        row.falsification_condition,
        list(row.prior_art_identity_terms),
    )
    if falsifier != row.falsification_condition:
        raise ValueError(
            f"atomic specification {row.local_id} falsifier fails existing branch sanitizer"
        )

    for label, phrases in (
        ("relation_endpoint_anchors", row.relation_endpoint_anchors),
        ("scope_qualifier_spans", row.scope_qualifier_spans),
        ("directional_qualifier_spans", row.directional_qualifier_spans),
    ):
        for phrase in phrases:
            if not _surface_contains(row.text, phrase):
                raise ValueError(
                    f"atomic specification {row.local_id} {label} phrase absent from claim text: {phrase}"
                )
            if not _surface_contains(row.required_bridge, phrase):
                raise ValueError(
                    f"atomic specification {row.local_id} {label} phrase absent from required_bridge: {phrase}"
                )

    # Fail closed before any query-plan materialization if the BASE
    # relation does not carry at least two scientific endpoints.
    _derive_base_relation_nucleus(row)


def compile_atomic_synthesis(
    *,
    portfolio: ProductionFacingScientificCandidatePortfolio,
    context: HypothesisContext,
    draft: AtomicCrossLaneSynthesisBatchDraft,
    backend_name: str,
    model_name: str,
    max_syntheses: int,
    llm_calls_performed: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> tuple[
    AtomicCrossLaneSynthesisReport,
    HypothesisPortfolio,
    LiteratureQueryPlan,
]:
    if portfolio.source_context_id != context.context_id:
        raise ValueError("candidate portfolio/context ID mismatch")
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("candidate portfolio/context SHA mismatch")
    if len(draft.hypotheses) > max_syntheses:
        raise ValueError("atomic synthesis exceeded max_syntheses")

    ref_map = _candidate_ref_map(portfolio)
    candidate_by_id = {
        row.candidate_id: row
        for row in portfolio.candidates
    }
    statement_by_id, research_gap_ids = _context_statement_sets(context)

    cards: list[HypothesisCard] = []
    decompositions: list[HypothesisNoveltyClaims] = []
    compiled_rows: list[CompiledAtomicCrossLaneHypothesis] = []

    for hypothesis_index, row in enumerate(draft.hypotheses, start=1):
        source_candidate_ids = _resolve_refs(
            row.source_candidate_ids,
            ref_map=ref_map,
            label=f"hypothesis[{hypothesis_index}]",
        )
        selected = [
            candidate_by_id[candidate_id]
            for candidate_id in source_candidate_ids
        ]
        lanes = sorted({candidate.source_lane for candidate in selected})
        if set(lanes) != {"RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"}:
            raise ValueError(
                "atomic synthesis hypothesis must cite both reasoning lanes"
            )

        available_premises = {
            statement_id
            for candidate in selected
            for statement_id in candidate.premise_statement_ids
        }
        available_gaps = {
            statement_id
            for candidate in selected
            for statement_id in candidate.gap_statement_ids
        }
        if not set(row.premise_statement_ids) <= available_premises:
            raise ValueError("atomic synthesis parent cites unsupported premises")
        if not set(row.gap_statement_ids) <= available_gaps:
            raise ValueError("atomic synthesis parent cites unsupported gaps")

        hypothesis_id = _stable_id(
            "hypothesis",
            "atomic_cross_lane_synthesis",
            portfolio.portfolio_id,
            row.local_id,
            row.synthesis_kind,
            row.title,
            row.hypothesis_statement,
        )

        compiled_specs: list[CompiledAtomicSpecification] = []
        claims: list[NoveltyClaim] = []
        observations: list[PredictedObservation] = []
        falsifiers: list[FalsificationCriterion] = []
        inferential_bridge_sentences: list[str] = []

        for atomic_index, atomic in enumerate(
            row.atomic_specifications,
            start=1,
        ):
            atomic_source_ids = _resolve_refs(
                atomic.source_candidate_ids,
                ref_map=ref_map,
                label=f"atomic[{hypothesis_index},{atomic_index}]",
            )
            if not set(atomic_source_ids) <= set(source_candidate_ids):
                raise ValueError("atomic specification cites source outside parent synthesis")

            atomic_selected = [
                candidate_by_id[candidate_id]
                for candidate_id in atomic_source_ids
            ]
            atomic_available_premises = {
                statement_id
                for candidate in atomic_selected
                for statement_id in candidate.premise_statement_ids
            }
            atomic_available_gaps = {
                statement_id
                for candidate in atomic_selected
                for statement_id in candidate.gap_statement_ids
            }
            if not set(atomic.premise_statement_ids) <= atomic_available_premises:
                raise ValueError("atomic specification cites unsupported premise IDs")
            if not set(atomic.gap_statement_ids) <= atomic_available_gaps:
                raise ValueError("atomic specification cites unsupported gap IDs")

            for statement_id in atomic.premise_statement_ids:
                source = statement_by_id.get(statement_id)
                if source is None or not getattr(source, "eligible_as_premise", False):
                    raise ValueError(
                        f"atomic specification cites ineligible premise: {statement_id}"
                    )
            for statement_id in atomic.gap_statement_ids:
                source = statement_by_id.get(statement_id)
                if source is None:
                    raise ValueError(
                        f"atomic specification cites unknown gap: {statement_id}"
                    )
                if not (
                    getattr(source, "eligible_as_gap", False)
                    or statement_id in research_gap_ids
                ):
                    raise ValueError(
                        f"atomic specification cites ineligible gap: {statement_id}"
                    )

            _validate_atomic_text_contract(atomic)
            base_relation_nucleus = _derive_base_relation_nucleus(atomic)

            observation_id = _stable_id(
                "prediction",
                hypothesis_id,
                atomic.local_id,
                atomic.observable,
                atomic.predicted_observation,
            )
            falsifier_id = _stable_id(
                "falsifier",
                hypothesis_id,
                atomic.local_id,
                atomic.observable,
                atomic.falsification_condition,
            )
            claim_id = _stable_id(
                "external_novelty_claim",
                hypothesis_id,
                atomic.local_id,
                atomic.kind,
                atomic.text,
            )

            structure, structure_reasons = compile_claim_scientific_structure(
                atomic.scientific_structure,
                identity_terms=list(atomic.prior_art_identity_terms),
                source_texts=[
                    row.hypothesis_statement,
                    atomic.required_bridge,
                    atomic.predicted_observation,
                    atomic.falsification_condition,
                    *row.assumptions,
                ],
            )

            binding = NoveltyClaimSemanticFidelityBindingDraft(
                proposition_basis=atomic.required_bridge,
                relation_endpoint_anchors=list(atomic.relation_endpoint_anchors),
                scope_qualifier_spans=list(atomic.scope_qualifier_spans),
                directional_qualifier_spans=list(atomic.directional_qualifier_spans),
                prediction_observation_id=observation_id,
                falsification_criterion_id=falsifier_id,
            )

            claim = NoveltyClaim(
                claim_id=claim_id,
                hypothesis_id=hypothesis_id,
                claim_rank=atomic_index,
                kind=atomic.kind,
                importance=atomic.importance,
                novelty_selection_role=atomic.novelty_selection_role,
                text=atomic.text,
                rationale=atomic.rationale,
                search_concepts=list(dict.fromkeys(atomic.search_concepts)),
                search_queries=list(dict.fromkeys(atomic.search_queries)),
                distinguishing_terms=list(dict.fromkeys(atomic.distinguishing_terms)),
                prior_art_identity_terms=list(
                    dict.fromkeys(atomic.prior_art_identity_terms)
                ),
                relation_nucleus_terms=list(base_relation_nucleus),
                required_bridge=atomic.required_bridge,
                predicted_observation=atomic.predicted_observation,
                falsification_condition=atomic.falsification_condition,
                scientific_structure=structure,
                scientific_structure_reason_codes=list(structure_reasons),
            )

            # The semantic-fidelity binding is provenance carried by the
            # synthesis report. NoveltyClaim itself has no canonical binding
            # field, so the directly materialized claim specification above
            # remains the N9 input contract.
            observations.append(
                PredictedObservation(
                    observation_id=observation_id,
                    observable=atomic.observable,
                    expected_direction="unspecified",
                    rationale=atomic.predicted_observation,
                )
            )
            falsifiers.append(
                FalsificationCriterion(
                    criterion_id=falsifier_id,
                    observable=atomic.observable,
                    falsifying_outcome=atomic.falsification_condition,
                )
            )
            inferential_bridge_sentences.append(atomic.required_bridge)
            claims.append(claim)
            compiled_specs.append(
                CompiledAtomicSpecification(
                    local_id=atomic.local_id,
                    claim_id=claim_id,
                    kind=atomic.kind,
                    importance=atomic.importance,
                    novelty_selection_role=atomic.novelty_selection_role,
                    text=atomic.text,
                    rationale=atomic.rationale,
                    source_candidate_ids=list(atomic_source_ids),
                    premise_statement_ids=list(
                        dict.fromkeys(atomic.premise_statement_ids)
                    ),
                    gap_statement_ids=list(
                        dict.fromkeys(atomic.gap_statement_ids)
                    ),
                    prior_art_identity_terms=list(
                        dict.fromkeys(atomic.prior_art_identity_terms)
                    ),
                    relation_endpoint_anchors=list(
                        dict.fromkeys(atomic.relation_endpoint_anchors)
                    ),
                    scope_qualifier_spans=list(
                        dict.fromkeys(atomic.scope_qualifier_spans)
                    ),
                    directional_qualifier_spans=list(
                        dict.fromkeys(atomic.directional_qualifier_spans)
                    ),
                    relation_nucleus_terms=list(base_relation_nucleus),
                    distinguishing_terms=list(
                        dict.fromkeys(atomic.distinguishing_terms)
                    ),
                    required_bridge=atomic.required_bridge,
                    observable=atomic.observable,
                    predicted_observation=atomic.predicted_observation,
                    falsification_condition=atomic.falsification_condition,
                    prediction_observation_id=observation_id,
                    falsification_criterion_id=falsifier_id,
                    search_concepts=list(dict.fromkeys(atomic.search_concepts)),
                    search_queries=list(dict.fromkeys(atomic.search_queries)),
                    scientific_structure=structure,
                    scientific_structure_reason_codes=list(structure_reasons),
                )
            )

            # Make sure the provenance object itself validates while keeping it
            # outside canonical NoveltyClaim authority.
            if binding.proposition_basis != atomic.required_bridge:
                raise RuntimeError("atomic semantic-fidelity binding drift")

        premise_ids = list(dict.fromkeys(row.premise_statement_ids))
        gap_ids = list(dict.fromkeys(row.gap_statement_ids))
        premise_rows = [statement_by_id[sid] for sid in premise_ids]
        gap_rows = [statement_by_id[sid] for sid in gap_ids]

        source_paper_ids = sorted(
            {
                paper_id
                for statement in premise_rows
                for paper_id in getattr(statement, "paper_ids", [])
            }
        )
        gap_paper_ids = sorted(
            {
                paper_id
                for statement in gap_rows
                for paper_id in getattr(statement, "paper_ids", [])
            }
        )
        candidate_premise_count = sum(
            bool(getattr(statement, "requires_verification", False))
            for statement in premise_rows
        )
        candidate_dependency = (
            "none"
            if candidate_premise_count == 0
            else (
                "essential"
                if candidate_premise_count == len(premise_rows)
                else "supporting"
            )
        )

        card = HypothesisCard(
            hypothesis_id=hypothesis_id,
            domain_profile_id=context.domain_profile_id,
            source_context_id=context.context_id,
            source_context_sha256=context.context_sha256,
            source_report_id=context.source_report_id,
            source_report_sha256=context.source_report_sha256,
            title=row.title,
            hypothesis_statement=row.hypothesis_statement,
            hypothesis_type=_HYPOTHESIS_TYPE_MAP[row.synthesis_kind],
            premise_statement_ids=premise_ids,
            gap_statement_ids=gap_ids,
            inferential_bridge=" ".join(
                dict.fromkeys(inferential_bridge_sentences)
            ),
            predicted_observations=observations,
            falsification_criteria=falsifiers,
            assumptions=list(dict.fromkeys(row.assumptions)),
            source_paper_ids=source_paper_ids,
            gap_paper_ids=gap_paper_ids,
            cross_paper_synthesis=len(source_paper_ids) > 1,
            candidate_dependency=candidate_dependency,
            evidence_profile=HypothesisEvidenceProfile(
                premise_count=len(premise_rows),
                gap_count=len(gap_rows),
                source_paper_count=len(source_paper_ids),
                candidate_premise_count=candidate_premise_count,
                reported_premise_count=sum(
                    getattr(statement, "epistemic_role", "") == "reported"
                    for statement in premise_rows
                ),
                synthesis_premise_count=sum(
                    getattr(statement, "epistemic_role", "") == "evidence_synthesis"
                    for statement in premise_rows
                ),
            ),
        )

        # Every required bridge must remain an exact sentence in the canonical
        # projected HypothesisCard.
        for spec in compiled_specs:
            if spec.required_bridge not in card.inferential_bridge:
                raise RuntimeError(
                    "atomic required bridge lost during HypothesisCard projection"
                )

        cards.append(card)
        decompositions.append(
            HypothesisNoveltyClaims(
                hypothesis_id=hypothesis_id,
                title=row.title,
                claims=claims,
                decomposition_notes=(
                    "Atomic claims were generated under cross_lane_atomic_v1 "
                    "and deterministically projected without external-novelty "
                    "LLM re-decomposition."
                ),
            )
        )
        compiled_rows.append(
            CompiledAtomicCrossLaneHypothesis(
                hypothesis_id=hypothesis_id,
                source_candidate_ids=list(source_candidate_ids),
                source_lanes=lanes,  # type: ignore[arg-type]
                synthesis_kind=row.synthesis_kind,
                title=row.title,
                hypothesis_statement=row.hypothesis_statement,
                premise_statement_ids=premise_ids,
                gap_statement_ids=gap_ids,
                assumptions=list(dict.fromkeys(row.assumptions)),
                atomic_specifications=compiled_specs,
            )
        )

    abstention = draft.abstention_reason
    portfolio_id = _stable_id(
        "hypothesis_portfolio",
        "atomic_cross_lane_synthesis",
        portfolio.portfolio_id,
        *[row.hypothesis_id for row in cards],
        abstention or "",
    )
    projected = HypothesisPortfolio(
        portfolio_id=portfolio_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=cards,
        abstention_reason=abstention if not cards else None,
    )
    query_plan = LiteratureQueryPlanner().build(
        projected,
        decompositions,
    )

    report_id = _stable_id(
        "atomic_cross_lane_scientific_synthesis",
        portfolio.portfolio_id,
        projected.portfolio_id,
        query_plan.plan_id,
    )
    report = AtomicCrossLaneSynthesisReport(
        report_id=report_id,
        source_candidate_portfolio_id=portfolio.portfolio_id,
        source_task_id=portfolio.source_task_id,
        source_context_id=portfolio.source_context_id,
        question=portfolio.question,
        backend_name=backend_name,
        model_name=model_name,
        hypotheses=compiled_rows,
        hypothesis_count=len(compiled_rows),
        atomic_specification_count=sum(
            len(row.atomic_specifications)
            for row in compiled_rows
        ),
        llm_calls_performed=llm_calls_performed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        abstention_reason=abstention,
    )
    return report, projected, query_plan


def run_atomic_synthesis(
    *,
    portfolio: ProductionFacingScientificCandidatePortfolio,
    context: HypothesisContext,
    backend: InstructorAtomicSynthesisBackend,
    max_syntheses: int,
) -> tuple[
    AtomicCrossLaneSynthesisReport,
    HypothesisPortfolio,
    LiteratureQueryPlan,
    AtomicSynthesisPrompt | None,
]:
    lanes = {row.source_lane for row in portfolio.candidates}
    if lanes != {"RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"}:
        draft = AtomicCrossLaneSynthesisBatchDraft(
            hypotheses=[],
            abstention_reason=(
                "Atomic cross-lane synthesis requires at least one candidate "
                "from each reasoning lane."
            ),
        )
        report, projected, plan = compile_atomic_synthesis(
            portfolio=portfolio,
            context=context,
            draft=draft,
            backend_name=backend.backend_name,
            model_name=backend.model_name,
            max_syntheses=max_syntheses,
            llm_calls_performed=0,
        )
        return report, projected, plan, None

    prompt = build_atomic_synthesis_prompt(
        portfolio,
        max_syntheses=max_syntheses,
    )
    generation = backend.generate(prompt)
    report, projected, plan = compile_atomic_synthesis(
        portfolio=portfolio,
        context=context,
        draft=generation.draft,
        backend_name=backend.backend_name,
        model_name=backend.model_name,
        max_syntheses=max_syntheses,
        llm_calls_performed=1,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
    )
    return report, projected, plan, prompt


__all__ = [
    "AtomicCrossLaneSynthesisBatchDraft",
    "AtomicCrossLaneSynthesisReport",
    "AtomicSpecificationDraft",
    "InstructorAtomicSynthesisBackend",
    "build_atomic_synthesis_prompt",
    "compile_atomic_synthesis",
    "run_atomic_synthesis",
]
