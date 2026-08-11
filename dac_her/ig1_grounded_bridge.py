from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.hypothesis_contracts import (
    ExpectedDirection,
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from dac_her.hypothesis_llm import (
    HypothesisDraftBackend,
    HypothesisDraftGeneration,
)
from dac_her.hypothesis_prompt import HypothesisPrompt


IG1_BLUEPRINT_VERSION = "ig1-grounded-endpoints-one-bridge-v1"
IG1_PROMPT_VERSION = "hypothesis-maker-ig1-v2.9.1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IG1GroundedEndpoint(StrictModel):
    endpoint_id: Literal["endpoint_a", "endpoint_b"]
    anchor_statement_id: str
    grounded_excerpt: str = Field(min_length=1)
    supporting_statement_ids: list[str] = Field(min_length=1)
    scientific_role: str = Field(min_length=1)

    @model_validator(mode="after")
    def _support_integrity(self) -> "IG1GroundedEndpoint":
        if self.anchor_statement_id not in self.supporting_statement_ids:
            raise ValueError(
                "anchor_statement_id must appear in supporting_statement_ids"
            )
        if len(self.supporting_statement_ids) != len(
            set(self.supporting_statement_ids)
        ):
            raise ValueError("duplicate endpoint supporting_statement_ids")
        return self


class IG1NovelBridge(StrictModel):
    bridge_id: Literal["novel_bridge_1"] = "novel_bridge_1"
    subject_endpoint_id: Literal["endpoint_a", "endpoint_b"]
    relation: str = Field(min_length=1)
    object_endpoint_id: Literal["endpoint_a", "endpoint_b"]
    bridge_kind: Literal[
        "mechanistic_dependency",
        "mediation",
        "moderation",
        "conditional_dependency",
        "descriptor_link",
        "tradeoff",
        "design_rule",
        "other",
    ]
    axis_inspiration_summary: str = Field(min_length=1)

    reported_fact: Literal[False] = False
    evidence_boundary_acknowledged: Literal[True] = True

    @model_validator(mode="after")
    def _distinct_endpoints(self) -> "IG1NovelBridge":
        if self.subject_endpoint_id == self.object_endpoint_id:
            raise ValueError(
                "IG1 novel bridge must connect two distinct endpoints"
            )
        return self


class IG1DiscriminativeTest(StrictModel):
    observable: str = Field(min_length=1)
    expected_direction: ExpectedDirection
    falsifying_outcome: str = Field(min_length=1)


class IG1Blueprint(StrictModel):
    schema_version: Literal[
        "ig1-grounded-endpoints-one-bridge-v1"
    ] = "ig1-grounded-endpoints-one-bridge-v1"

    axis_id: str
    abstain: bool = False
    abstention_reason: str | None = None

    endpoint_a: IG1GroundedEndpoint | None = None
    endpoint_b: IG1GroundedEndpoint | None = None
    novel_bridge: IG1NovelBridge | None = None
    discriminative_test: IG1DiscriminativeTest | None = None

    scope_conditions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _abstention_consistency(self) -> "IG1Blueprint":
        scientific = (
            self.endpoint_a,
            self.endpoint_b,
            self.novel_bridge,
            self.discriminative_test,
        )
        if self.abstain:
            if not (self.abstention_reason or "").strip():
                raise ValueError(
                    "abstention_reason required when IG1 blueprint abstains"
                )
            if any(row is not None for row in scientific):
                raise ValueError(
                    "abstaining IG1 blueprint must not contain scientific plan"
                )
            return self

        if self.abstention_reason is not None:
            raise ValueError(
                "abstention_reason must be null for active IG1 blueprint"
            )
        if any(row is None for row in scientific):
            raise ValueError(
                "active IG1 blueprint requires two endpoints, one bridge, "
                "and one discriminative test"
            )
        assert self.endpoint_a is not None
        assert self.endpoint_b is not None
        if self.endpoint_a.endpoint_id != "endpoint_a":
            raise ValueError("endpoint_a field must use endpoint_id=endpoint_a")
        if self.endpoint_b.endpoint_id != "endpoint_b":
            raise ValueError("endpoint_b field must use endpoint_id=endpoint_b")
        if (
            self.endpoint_a.anchor_statement_id
            == self.endpoint_b.anchor_statement_id
            and self.endpoint_a.grounded_excerpt
            == self.endpoint_b.grounded_excerpt
        ):
            raise ValueError(
                "IG1 endpoints must represent distinct grounded claims"
            )
        return self


class IG1BlueprintValidationIssue(StrictModel):
    code: str
    detail: str


class IG1BlueprintRecord(StrictModel):
    axis_id: str
    source_prompt_sha256: str
    blueprint_sha256: str
    blueprint: IG1Blueprint

    generation_attempts: int
    valid: bool
    validation_issues: list[IG1BlueprintValidationIssue] = Field(
        default_factory=list
    )

    final_generation_count: int = 0
    ig1_conformance_repair_count: int = 0


class IG1BlueprintReport(StrictModel):
    schema_version: Literal["ig1-blueprint-report-v1"] = (
        "ig1-blueprint-report-v1"
    )
    report_id: str
    report_sha256: str

    source_context_id: str
    source_context_sha256: str
    source_axis_plan_id: str

    blueprint_model: str
    blueprint_count: int
    active_blueprint_count: int
    abstained_blueprint_count: int
    invalid_blueprint_count: int
    ig1_conformance_repair_count: int

    records: list[IG1BlueprintRecord] = Field(default_factory=list)


class IG1ConformanceIssue(StrictModel):
    code: str
    detail: str


class IG1HypothesisConformanceCard(StrictModel):
    hypothesis_id: str
    axis_id: str
    blueprint_sha256: str
    passes: bool

    expected_premise_statement_ids: list[str] = Field(default_factory=list)
    actual_premise_statement_ids: list[str] = Field(default_factory=list)

    novel_relation: str
    relation_in_hypothesis_statement: bool
    relation_in_inferential_bridge: bool

    expected_observable: str
    actual_observables: list[str] = Field(default_factory=list)
    predicted_observation_count: int
    falsification_criterion_count: int

    issues: list[IG1ConformanceIssue] = Field(default_factory=list)


class IG1ConformanceReport(StrictModel):
    schema_version: Literal["ig1-conformance-report-v1"] = (
        "ig1-conformance-report-v1"
    )
    report_id: str
    report_sha256: str

    source_portfolio_id: str
    source_axis_report_id: str
    source_blueprint_report_id: str

    hypothesis_count: int
    passing_count: int
    failing_count: int
    issue_counts: dict[str, int] = Field(default_factory=dict)

    cards: list[IG1HypothesisConformanceCard] = Field(
        default_factory=list
    )


@dataclass(frozen=True)
class _StructuredCall:
    value: BaseModel
    elapsed_seconds: float


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _extract_axis_id(prompt: HypothesisPrompt) -> str:
    matches = re.findall(
        r"(?m)^axis_id:\s*(\S+)\s*$",
        prompt.user_prompt,
    )
    if len(matches) != 1:
        raise ValueError(
            "IG1 expected exactly one axis_id line in discovery-axis prompt; "
            f"found {len(matches)}"
        )
    return matches[0]


def _relation_chain_markers(relation: str) -> list[str]:
    text = relation.lower()
    markers = [
        " and then ",
        " followed by ",
        " which then ",
        " subsequently ",
        " thereby causing ",
        " leading in turn to ",
    ]
    return [
        marker.strip()
        for marker in markers
        if marker in text
    ]


def _eligible_statement_map(
    context: HypothesisContext,
) -> dict[str, Any]:
    return {
        row.statement_id: row
        for row in context.evidence_statements
        if row.eligible_as_premise
    }


def validate_blueprint_against_context(
    blueprint: IG1Blueprint,
    *,
    context: HypothesisContext,
    expected_axis_id: str,
) -> list[IG1BlueprintValidationIssue]:
    issues: list[IG1BlueprintValidationIssue] = []

    if blueprint.axis_id != expected_axis_id:
        issues.append(
            IG1BlueprintValidationIssue(
                code="axis_id_mismatch",
                detail=(
                    f"expected {expected_axis_id}, "
                    f"got {blueprint.axis_id}"
                ),
            )
        )

    if blueprint.abstain:
        return issues

    assert blueprint.endpoint_a is not None
    assert blueprint.endpoint_b is not None
    assert blueprint.novel_bridge is not None

    eligible = _eligible_statement_map(context)

    for endpoint in (
        blueprint.endpoint_a,
        blueprint.endpoint_b,
    ):
        anchor = eligible.get(endpoint.anchor_statement_id)
        if anchor is None:
            issues.append(
                IG1BlueprintValidationIssue(
                    code="ineligible_anchor_statement",
                    detail=(
                        f"{endpoint.endpoint_id}: "
                        f"{endpoint.anchor_statement_id}"
                    ),
                )
            )
        else:
            if endpoint.grounded_excerpt not in anchor.text:
                issues.append(
                    IG1BlueprintValidationIssue(
                        code="anchor_excerpt_not_verbatim",
                        detail=(
                            f"{endpoint.endpoint_id}: grounded_excerpt "
                            "is not an exact substring of anchor statement"
                        ),
                    )
                )

        for sid in endpoint.supporting_statement_ids:
            if sid not in eligible:
                issues.append(
                    IG1BlueprintValidationIssue(
                        code="ineligible_supporting_statement",
                        detail=f"{endpoint.endpoint_id}: {sid}",
                    )
                )

    markers = _relation_chain_markers(
        blueprint.novel_bridge.relation
    )
    if markers:
        issues.append(
            IG1BlueprintValidationIssue(
                code="multi_hop_relation_marker",
                detail=(
                    "novel relation contains chain marker(s): "
                    + ", ".join(markers)
                ),
            )
        )

    return issues


def blueprint_premise_ids(
    blueprint: IG1Blueprint,
) -> list[str]:
    if blueprint.abstain:
        return []
    assert blueprint.endpoint_a is not None
    assert blueprint.endpoint_b is not None
    return sorted(
        set(
            blueprint.endpoint_a.supporting_statement_ids
            + blueprint.endpoint_b.supporting_statement_ids
        )
    )


def build_blueprint_messages(
    prompt: HypothesisPrompt,
) -> list[dict[str, str]]:
    system = """You are the IG1 scientific hypothesis planning stage.

Do NOT write the final hypothesis portfolio yet. Produce only an IG1Blueprint.

Goal:
Create a scientifically bounded plan consisting of exactly two already-grounded
endpoints and exactly one explicitly novel/testable relation between them.

GROUNDING RULES
- Each endpoint must be anchored to an eligible positive premise from the
  supplied grounded HypothesisContext.
- grounded_excerpt MUST be copied VERBATIM as one contiguous substring from the
  endpoint's anchor premise statement.
- supporting_statement_ids must be the smallest sufficient set of eligible
  positive premise IDs for that endpoint. Discovery-axis IDs, graph IDs, path
  IDs, inspiration IDs, and candidate-unit IDs are never premises.
- Do not convert a discovery-axis statement into a grounded endpoint.

ONE-BRIDGE RULE
- The blueprint contains exactly one novel_bridge object. That object is the
  entire new scientific dependency allowed for this hypothesis.
- relation must express ONE testable scientific relationship, not a causal
  chain. Do not encode "A changes X, which changes Y, which causes B".
- The discovery axis may inspire the one proposed relation, including an
  unverified mechanism or condition, but it remains explicitly hypothetical.
- Do not add a second mechanism, downstream design recommendation, optimization
  heuristic, universal-rule rejection, or extra consequence to the bridge.
- If a design rule itself is the one bridge, choose bridge_kind=design_rule;
  otherwise do not append a design rule.

TEST RULE
- Return exactly one discriminative_test. It must directly test/falsify the
  one novel bridge rather than merely remeasure a grounded endpoint.

ABSTENTION
- If two grounded endpoints cannot support a scientifically coherent single
  novel bridge using this discovery axis without adding extra unsupported
  relations, set abstain=true.

External literature knowledge is forbidden. External novelty claims are
forbidden.
"""
    user = (
        "SOURCE DISCOVERY-AXIS PROMPT\n"
        "============================\n"
        + prompt.user_prompt
        + "\n\n"
        "Produce the IG1Blueprint only. The final HypothesisPortfolioDraft "
        "will be generated in a separate stage."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def augment_prompt_with_blueprint(
    prompt: HypothesisPrompt,
    blueprint: IG1Blueprint,
) -> HypothesisPrompt:
    if blueprint.abstain:
        raise ValueError(
            "cannot augment final hypothesis prompt from abstaining blueprint"
        )

    assert blueprint.endpoint_a is not None
    assert blueprint.endpoint_b is not None
    assert blueprint.novel_bridge is not None
    assert blueprint.discriminative_test is not None

    exact_premises = blueprint_premise_ids(blueprint)
    bridge = blueprint.novel_bridge
    test = blueprint.discriminative_test

    appendix = f"""
IG1 GROUNDED-ENDPOINT / ONE-NOVEL-BRIDGE CONTRACT
=================================================
The planning stage has already fixed the epistemic structure below. Follow it
exactly. Do not add another scientific edge.

GROUNDed endpoint A
- anchor_statement_id: {blueprint.endpoint_a.anchor_statement_id}
- supporting_statement_ids: {json.dumps(blueprint.endpoint_a.supporting_statement_ids, ensure_ascii=False)}
- verbatim grounded excerpt: {blueprint.endpoint_a.grounded_excerpt}
- role: {blueprint.endpoint_a.scientific_role}

GROUNDED endpoint B
- anchor_statement_id: {blueprint.endpoint_b.anchor_statement_id}
- supporting_statement_ids: {json.dumps(blueprint.endpoint_b.supporting_statement_ids, ensure_ascii=False)}
- verbatim grounded excerpt: {blueprint.endpoint_b.grounded_excerpt}
- role: {blueprint.endpoint_b.scientific_role}

THE ONLY ALLOWED NOVEL SCIENTIFIC EDGE
- bridge_id: {bridge.bridge_id}
- subject: {bridge.subject_endpoint_id}
- relation EXACT PHRASE: {bridge.relation}
- object: {bridge.object_endpoint_id}
- bridge_kind: {bridge.bridge_kind}
- axis inspiration: {bridge.axis_inspiration_summary}
- status: PROPOSED / UNESTABLISHED; never present it as reported fact

SCOPE CONDITIONS
{json.dumps(blueprint.scope_conditions, ensure_ascii=False, indent=2)}

THE ONE DIRECT DISCRIMINATIVE TEST
- observable EXACT TEXT: {test.observable}
- expected_direction: {test.expected_direction}
- falsifying_outcome: {test.falsifying_outcome}

FINAL-DRAFT HARD DISCIPLINE
- Return exactly ONE hypothesis or abstain.
- premise_statement_ids MUST be exactly this set, with no extra "safe" premise:
  {json.dumps(exact_premises, ensure_ascii=False)}
- hypothesis_statement MUST contain the novel relation exact phrase:
  {bridge.relation}
- inferential_bridge MUST contain that same exact relation phrase and explain
  only how the two grounded endpoints motivate testing this ONE proposed edge.
- Do not introduce a second mediator, second causal link, downstream mechanism,
  design recommendation, optimization heuristic, or universal-rule conclusion.
- Exactly ONE predicted_observation is allowed.
- Its observable MUST equal exactly: {test.observable}
- Its expected_direction MUST equal exactly: {test.expected_direction}
- Exactly ONE falsification_criterion is allowed and its observable MUST equal
  the same exact observable.
- The falsifying outcome must directly challenge the one novel relation.
- Discovery-axis content may appear only inside the explicitly proposed bridge
  or its test; it may not be promoted to reported evidence.
- If following this blueprint would require another novel edge, abstain instead.
""".strip()

    system_prompt = (
        prompt.system_prompt.rstrip()
        + "\n\n"
        + "IG1 OVERRIDING GENERATION POLICY\n"
        + "===============================\n"
        + "The IG1 contract in the user prompt is a stricter epistemic "
        + "constraint than ordinary discovery-axis synthesis. Preserve all "
        + "ordinary grounding rules, but never expand beyond the single "
        + "planned novel scientific edge.\n"
    )
    user_prompt = (
        prompt.user_prompt.rstrip()
        + "\n\n"
        + appendix
        + "\n"
    )

    canonical = _canonical_json(
        {
            "prompt_version": IG1_PROMPT_VERSION,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
    )
    return HypothesisPrompt(
        prompt_version=IG1_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_sha256=hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    )


def draft_conformance_issues(
    draft: HypothesisPortfolioDraft,
    blueprint: IG1Blueprint,
) -> list[IG1ConformanceIssue]:
    issues: list[IG1ConformanceIssue] = []

    if blueprint.abstain:
        if draft.hypotheses:
            issues.append(
                IG1ConformanceIssue(
                    code="blueprint_abstention_violated",
                    detail="blueprint abstained but final draft contains hypothesis",
                )
            )
        return issues

    assert blueprint.novel_bridge is not None
    assert blueprint.discriminative_test is not None

    if not draft.hypotheses:
        return issues

    if len(draft.hypotheses) != 1:
        issues.append(
            IG1ConformanceIssue(
                code="hypothesis_cardinality",
                detail=(
                    f"expected exactly 1 hypothesis, got "
                    f"{len(draft.hypotheses)}"
                ),
            )
        )
        return issues

    proposal = draft.hypotheses[0]
    expected_premises = set(blueprint_premise_ids(blueprint))
    actual_premises = set(proposal.premise_statement_ids)

    if actual_premises != expected_premises:
        issues.append(
            IG1ConformanceIssue(
                code="premise_set_mismatch",
                detail=(
                    f"expected={sorted(expected_premises)}; "
                    f"actual={sorted(actual_premises)}"
                ),
            )
        )

    relation = blueprint.novel_bridge.relation
    if relation not in proposal.hypothesis_statement:
        issues.append(
            IG1ConformanceIssue(
                code="novel_relation_missing_from_hypothesis",
                detail="exact IG1 novel relation phrase is absent",
            )
        )
    if relation not in proposal.inferential_bridge:
        issues.append(
            IG1ConformanceIssue(
                code="novel_relation_missing_from_bridge",
                detail="exact IG1 novel relation phrase is absent",
            )
        )

    if len(proposal.predicted_observations) != 1:
        issues.append(
            IG1ConformanceIssue(
                code="prediction_cardinality",
                detail=(
                    "expected exactly 1 predicted observation; "
                    f"got {len(proposal.predicted_observations)}"
                ),
            )
        )
    else:
        prediction = proposal.predicted_observations[0]
        test = blueprint.discriminative_test
        if prediction.observable != test.observable:
            issues.append(
                IG1ConformanceIssue(
                    code="prediction_observable_mismatch",
                    detail=(
                        f"expected={test.observable!r}; "
                        f"actual={prediction.observable!r}"
                    ),
                )
            )
        if prediction.expected_direction != test.expected_direction:
            issues.append(
                IG1ConformanceIssue(
                    code="prediction_direction_mismatch",
                    detail=(
                        f"expected={test.expected_direction}; "
                        f"actual={prediction.expected_direction}"
                    ),
                )
            )

    if len(proposal.falsification_criteria) != 1:
        issues.append(
            IG1ConformanceIssue(
                code="falsifier_cardinality",
                detail=(
                    "expected exactly 1 falsification criterion; "
                    f"got {len(proposal.falsification_criteria)}"
                ),
            )
        )
    else:
        criterion = proposal.falsification_criteria[0]
        if (
            criterion.observable
            != blueprint.discriminative_test.observable
        ):
            issues.append(
                IG1ConformanceIssue(
                    code="falsifier_observable_mismatch",
                    detail=(
                        "falsifier observable must exactly equal "
                        "IG1 discriminative-test observable"
                    ),
                )
            )

    return issues


def _conformance_feedback(
    blueprint: IG1Blueprint,
    issues: list[IG1ConformanceIssue],
) -> str:
    assert not blueprint.abstain
    assert blueprint.novel_bridge is not None
    assert blueprint.discriminative_test is not None

    return "\n".join(
        [
            "IG1 CONFORMANCE REPAIR",
            "======================",
            "The previous draft violates the fixed grounded-endpoint / "
            "one-novel-bridge blueprint.",
            "Do not change the blueprint. Return a complete replacement "
            "HypothesisPortfolioDraft.",
            "",
            "Issues:",
            *[
                f"- {row.code}: {row.detail}"
                for row in issues
            ],
            "",
            "Required exact premise IDs:",
            json.dumps(
                blueprint_premise_ids(blueprint),
                ensure_ascii=False,
            ),
            "Required exact novel relation phrase:",
            blueprint.novel_bridge.relation,
            "Required exact observable:",
            blueprint.discriminative_test.observable,
            "Required direction:",
            blueprint.discriminative_test.expected_direction,
            "",
            "Return exactly one hypothesis, one predicted observation, "
            "and one falsification criterion. Do not add a second novel "
            "scientific relation. If the blueprint cannot be expressed "
            "without another edge, abstain.",
        ]
    )


class IG1BlueprintGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 2,
        timeout: float | None = 180.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(api_key_env)
        )
        self.api_key_env = api_key_env
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or None
        )
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
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
        except ImportError as exc:
            raise RuntimeError(
                "IG1 requires installed 'openai' and 'instructor'."
            ) from exc

        mode = getattr(
            instructor.Mode,
            self.instructor_mode,
            None,
        )
        if mode is None:
            raise ValueError(
                f"Unknown Instructor mode: {self.instructor_mode}"
            )

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers

        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=mode,
        )
        return self._client

    def call(
        self,
        messages: list[dict[str, str]],
    ) -> _StructuredCall:
        started = time.perf_counter()
        value = self._get_client().chat.completions.create(
            model=self.model_name,
            response_model=IG1Blueprint,
            messages=messages,
            temperature=self.temperature,
            max_retries=self.parse_retries,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(value, IG1Blueprint):
            value = IG1Blueprint.model_validate(value)
        return _StructuredCall(
            value=value,
            elapsed_seconds=elapsed,
        )


class IG1HypothesisBackend:
    """Experimental Alpha4 backend wrapper.

    The canonical DiscoveryAxisSynthesisRuntime remains unchanged.
    IG1 adds one structured blueprint call before each initial per-axis
    hypothesis generation, then constrains every generation/repair to that
    immutable blueprint.
    """

    backend_name = "ig1_grounded_endpoints_one_bridge"

    def __init__(
        self,
        base_backend: HypothesisDraftBackend,
        blueprint_generator: IG1BlueprintGenerator,
        *,
        context: HypothesisContext,
        max_blueprint_grounding_repairs: int = 1,
        max_ig1_conformance_repairs: int = 1,
    ) -> None:
        if max_blueprint_grounding_repairs not in {0, 1}:
            raise ValueError(
                "max_blueprint_grounding_repairs must be 0 or 1"
            )
        if max_ig1_conformance_repairs not in {0, 1}:
            raise ValueError(
                "max_ig1_conformance_repairs must be 0 or 1"
            )
        self.base_backend = base_backend
        self.blueprint_generator = blueprint_generator
        self.context = context
        self.model_name = base_backend.model_name

        self.max_blueprint_grounding_repairs = (
            max_blueprint_grounding_repairs
        )
        self.max_ig1_conformance_repairs = (
            max_ig1_conformance_repairs
        )

        self._records_by_prompt_sha: dict[
            str,
            IG1BlueprintRecord,
        ] = {}
        self._augmented_by_prompt_sha: dict[
            str,
            HypothesisPrompt,
        ] = {}

    def _blueprint_for_prompt(
        self,
        prompt: HypothesisPrompt,
    ) -> IG1BlueprintRecord:
        cached = self._records_by_prompt_sha.get(
            prompt.prompt_sha256
        )
        if cached is not None:
            return cached

        axis_id = _extract_axis_id(prompt)
        generation = self.blueprint_generator.call(
            build_blueprint_messages(prompt)
        )
        blueprint = generation.value
        assert isinstance(blueprint, IG1Blueprint)

        attempts = 1
        issues = validate_blueprint_against_context(
            blueprint,
            context=self.context,
            expected_axis_id=axis_id,
        )

        if (
            issues
            and self.max_blueprint_grounding_repairs
        ):
            repair_messages = build_blueprint_messages(prompt)
            repair_messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": blueprint.model_dump_json(
                            indent=2
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                "IG1 BLUEPRINT GROUNDING REPAIR",
                                "The prior blueprint failed deterministic "
                                "grounding/conformance checks.",
                                "Repair only those issues while preserving "
                                "exactly two grounded endpoints and exactly "
                                "one novel bridge.",
                                *[
                                    f"- {row.code}: {row.detail}"
                                    for row in issues
                                ],
                            ]
                        ),
                    },
                ]
            )
            generation = self.blueprint_generator.call(
                repair_messages
            )
            blueprint = generation.value
            assert isinstance(blueprint, IG1Blueprint)
            attempts += 1
            issues = validate_blueprint_against_context(
                blueprint,
                context=self.context,
                expected_axis_id=axis_id,
            )

        valid = not issues

        if not valid:
            blueprint = IG1Blueprint(
                axis_id=axis_id,
                abstain=True,
                abstention_reason=(
                    "IG1 blueprint failed deterministic grounding checks: "
                    + "; ".join(
                        f"{row.code}={row.detail}"
                        for row in issues
                    )
                ),
            )

        blueprint_sha = _sha256_json(blueprint)
        record = IG1BlueprintRecord(
            axis_id=axis_id,
            source_prompt_sha256=prompt.prompt_sha256,
            blueprint_sha256=blueprint_sha,
            blueprint=blueprint,
            generation_attempts=attempts,
            valid=valid,
            validation_issues=list(issues),
        )
        self._records_by_prompt_sha[prompt.prompt_sha256] = record
        return record

    def _finalize_generation(
        self,
        *,
        original_prompt: HypothesisPrompt,
        augmented_prompt: HypothesisPrompt,
        record: IG1BlueprintRecord,
        generation: HypothesisDraftGeneration,
    ) -> HypothesisDraftGeneration:
        blueprint = record.blueprint
        if blueprint.abstain:
            return HypothesisDraftGeneration(
                draft=HypothesisPortfolioDraft(
                    hypotheses=[],
                    abstention_reason=(
                        blueprint.abstention_reason
                        or "IG1 blueprint abstained"
                    ),
                )
            )

        issues = draft_conformance_issues(
            generation.draft,
            blueprint,
        )
        repair_count = 0

        if issues and self.max_ig1_conformance_repairs:
            repaired = self.base_backend.repair(
                augmented_prompt,
                generation.draft,
                _conformance_feedback(
                    blueprint,
                    issues,
                ),
            )
            generation = repaired
            repair_count = 1

        record.final_generation_count += 1
        record.ig1_conformance_repair_count += repair_count
        return generation

    def generate(
        self,
        prompt: HypothesisPrompt,
    ) -> HypothesisDraftGeneration:
        record = self._blueprint_for_prompt(prompt)
        blueprint = record.blueprint

        if blueprint.abstain:
            record.final_generation_count += 1
            return HypothesisDraftGeneration(
                draft=HypothesisPortfolioDraft(
                    hypotheses=[],
                    abstention_reason=(
                        blueprint.abstention_reason
                        or "IG1 blueprint abstained"
                    ),
                )
            )

        augmented = augment_prompt_with_blueprint(
            prompt,
            blueprint,
        )
        self._augmented_by_prompt_sha[
            prompt.prompt_sha256
        ] = augmented

        generation = self.base_backend.generate(
            augmented
        )
        return self._finalize_generation(
            original_prompt=prompt,
            augmented_prompt=augmented,
            record=record,
            generation=generation,
        )

    def repair(
        self,
        prompt: HypothesisPrompt,
        previous_draft: HypothesisPortfolioDraft,
        feedback: str,
    ) -> HypothesisDraftGeneration:
        record = self._blueprint_for_prompt(prompt)
        blueprint = record.blueprint

        if blueprint.abstain:
            return HypothesisDraftGeneration(
                draft=HypothesisPortfolioDraft(
                    hypotheses=[],
                    abstention_reason=(
                        blueprint.abstention_reason
                        or "IG1 blueprint abstained"
                    ),
                )
            )

        augmented = self._augmented_by_prompt_sha.get(
            prompt.prompt_sha256
        )
        if augmented is None:
            augmented = augment_prompt_with_blueprint(
                prompt,
                blueprint,
            )
            self._augmented_by_prompt_sha[
                prompt.prompt_sha256
            ] = augmented

        ig1_feedback = (
            feedback.rstrip()
            + "\n\n"
            + _conformance_feedback(
                blueprint,
                [],
            )
        )
        generation = self.base_backend.repair(
            augmented,
            previous_draft,
            ig1_feedback,
        )
        return self._finalize_generation(
            original_prompt=prompt,
            augmented_prompt=augmented,
            record=record,
            generation=generation,
        )

    def records(self) -> list[IG1BlueprintRecord]:
        return sorted(
            self._records_by_prompt_sha.values(),
            key=lambda row: row.axis_id,
        )

    def augmented_prompt(
        self,
        source_prompt_sha256: str,
    ) -> HypothesisPrompt | None:
        return self._augmented_by_prompt_sha.get(
            source_prompt_sha256
        )


def build_blueprint_report(
    backend: IG1HypothesisBackend,
    *,
    context: HypothesisContext,
    axis_plan_id: str,
) -> IG1BlueprintReport:
    records = backend.records()
    payload = {
        "schema_version": "ig1-blueprint-report-v1",
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "source_axis_plan_id": axis_plan_id,
        "blueprint_model": backend.blueprint_generator.model_name,
        "blueprint_count": len(records),
        "active_blueprint_count": sum(
            not row.blueprint.abstain
            for row in records
        ),
        "abstained_blueprint_count": sum(
            row.blueprint.abstain
            for row in records
        ),
        "invalid_blueprint_count": sum(
            not row.valid
            for row in records
        ),
        "ig1_conformance_repair_count": sum(
            row.ig1_conformance_repair_count
            for row in records
        ),
        "records": [
            row.model_dump(mode="json")
            for row in records
        ],
    }
    payload["report_id"] = _stable_id(
        "ig1_blueprint_report",
        context.context_sha256,
        axis_plan_id,
        backend.blueprint_generator.model_name,
        _sha256_json(payload["records"]),
    )
    return IG1BlueprintReport(
        **payload,
        report_sha256=_sha256_json(payload),
    )


def _portfolio_conformance_card(
    *,
    hypothesis: Any,
    axis_id: str,
    blueprint_record: IG1BlueprintRecord,
) -> IG1HypothesisConformanceCard:
    blueprint = blueprint_record.blueprint
    if blueprint.abstain:
        raise ValueError(
            "accepted hypothesis cannot map to abstaining IG1 blueprint"
        )
    assert blueprint.novel_bridge is not None
    assert blueprint.discriminative_test is not None

    issues: list[IG1ConformanceIssue] = []
    expected_premises = blueprint_premise_ids(
        blueprint
    )
    actual_premises = sorted(
        set(hypothesis.premise_statement_ids)
    )
    if set(expected_premises) != set(actual_premises):
        issues.append(
            IG1ConformanceIssue(
                code="premise_set_mismatch",
                detail=(
                    f"expected={expected_premises}; "
                    f"actual={actual_premises}"
                ),
            )
        )

    relation = blueprint.novel_bridge.relation
    in_statement = (
        relation in hypothesis.hypothesis_statement
    )
    in_bridge = (
        relation in hypothesis.inferential_bridge
    )
    if not in_statement:
        issues.append(
            IG1ConformanceIssue(
                code="novel_relation_missing_from_hypothesis",
                detail="exact relation phrase absent",
            )
        )
    if not in_bridge:
        issues.append(
            IG1ConformanceIssue(
                code="novel_relation_missing_from_bridge",
                detail="exact relation phrase absent",
            )
        )

    actual_observables = [
        row.observable
        for row in hypothesis.predicted_observations
    ]
    if len(hypothesis.predicted_observations) != 1:
        issues.append(
            IG1ConformanceIssue(
                code="prediction_cardinality",
                detail=(
                    f"got {len(hypothesis.predicted_observations)}"
                ),
            )
        )
    else:
        row = hypothesis.predicted_observations[0]
        if (
            row.observable
            != blueprint.discriminative_test.observable
        ):
            issues.append(
                IG1ConformanceIssue(
                    code="prediction_observable_mismatch",
                    detail="observable differs from blueprint",
                )
            )
        if (
            row.expected_direction
            != blueprint.discriminative_test.expected_direction
        ):
            issues.append(
                IG1ConformanceIssue(
                    code="prediction_direction_mismatch",
                    detail="direction differs from blueprint",
                )
            )

    if len(hypothesis.falsification_criteria) != 1:
        issues.append(
            IG1ConformanceIssue(
                code="falsifier_cardinality",
                detail=(
                    f"got {len(hypothesis.falsification_criteria)}"
                ),
            )
        )
    else:
        row = hypothesis.falsification_criteria[0]
        if (
            row.observable
            != blueprint.discriminative_test.observable
        ):
            issues.append(
                IG1ConformanceIssue(
                    code="falsifier_observable_mismatch",
                    detail="observable differs from blueprint",
                )
            )

    return IG1HypothesisConformanceCard(
        hypothesis_id=hypothesis.hypothesis_id,
        axis_id=axis_id,
        blueprint_sha256=blueprint_record.blueprint_sha256,
        passes=not issues,
        expected_premise_statement_ids=expected_premises,
        actual_premise_statement_ids=actual_premises,
        novel_relation=relation,
        relation_in_hypothesis_statement=in_statement,
        relation_in_inferential_bridge=in_bridge,
        expected_observable=(
            blueprint.discriminative_test.observable
        ),
        actual_observables=actual_observables,
        predicted_observation_count=len(
            hypothesis.predicted_observations
        ),
        falsification_criterion_count=len(
            hypothesis.falsification_criteria
        ),
        issues=issues,
    )


def build_conformance_report(
    *,
    portfolio: HypothesisPortfolio,
    axis_report: Any,
    blueprint_report: IG1BlueprintReport,
) -> IG1ConformanceReport:
    blueprint_by_axis = {
        row.axis_id: row
        for row in blueprint_report.records
    }
    lineage_by_hypothesis = {
        row.hypothesis_id: row
        for row in axis_report.lineages
    }

    cards: list[IG1HypothesisConformanceCard] = []
    for hypothesis in portfolio.hypotheses:
        lineage = lineage_by_hypothesis.get(
            hypothesis.hypothesis_id
        )
        if lineage is None:
            raise ValueError(
                "IG1 conformance missing axis lineage for "
                f"{hypothesis.hypothesis_id}"
            )
        record = blueprint_by_axis.get(
            lineage.axis_id
        )
        if record is None:
            raise ValueError(
                "IG1 conformance missing blueprint for axis "
                f"{lineage.axis_id}"
            )
        cards.append(
            _portfolio_conformance_card(
                hypothesis=hypothesis,
                axis_id=lineage.axis_id,
                blueprint_record=record,
            )
        )

    counts = Counter(
        issue.code
        for card in cards
        for issue in card.issues
    )

    payload = {
        "schema_version": "ig1-conformance-report-v1",
        "source_portfolio_id": portfolio.portfolio_id,
        "source_axis_report_id": axis_report.report_id,
        "source_blueprint_report_id": blueprint_report.report_id,
        "hypothesis_count": len(cards),
        "passing_count": sum(
            row.passes
            for row in cards
        ),
        "failing_count": sum(
            not row.passes
            for row in cards
        ),
        "issue_counts": dict(counts),
        "cards": [
            row.model_dump(mode="json")
            for row in cards
        ],
    }
    payload["report_id"] = _stable_id(
        "ig1_conformance_report",
        portfolio.portfolio_id,
        axis_report.report_id,
        blueprint_report.report_id,
    )
    return IG1ConformanceReport(
        **payload,
        report_sha256=_sha256_json(payload),
    )
