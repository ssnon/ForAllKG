from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompileError,
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


IdentificationAssessment = Literal[
    "SUPPORTED",
    "PARTIALLY_IDENTIFIED",
    "NOT_IDENTIFIABLE",
]


class IdentificationContract(StrictModel):
    proposed_relation: str = Field(min_length=1)
    independent_variable: str = Field(min_length=1)
    dependent_observable: str = Field(min_length=1)
    comparison_context: str = Field(min_length=1)
    required_controls: list[str] = Field(default_factory=list)
    measurement_compatibility: str = Field(min_length=1)
    measurement_compatibility_mode: Literal[
        "DIRECTLY_COMPARABLE",
        "CONTEXTUAL_ONLY",
        "PROSPECTIVE_MATCH_REQUIRED",
        "NOT_COMPARABLE",
    ]
    measurement_support_statement_ids: list[str] = Field(
        default_factory=list
    )
    contextual_measurement_statement_ids: list[str] = Field(
        default_factory=list
    )
    directional_evidence_basis: str = Field(min_length=1)
    directionality_mode: Literal[
        "DIRECTLY_SUPPORTED",
        "GROUNDED_PREDICTION",
        "NON_DIRECTIONAL",
        "NOT_IDENTIFIABLE",
    ]
    directional_support_statement_ids: list[str] = Field(
        default_factory=list
    )
    potential_confounders: list[str] = Field(default_factory=list)
    task_alignment_rationale: str = Field(min_length=1)
    premise_statement_ids: list[str] = Field(min_length=1)
    assessment: IdentificationAssessment
    limitation_reason: str | None = None

    current_evidence_status: Literal[
        "DIRECT_RELATION_SUPPORTED",
        "PARTIAL_GROUNDING",
        "CONTEXT_ONLY",
    ]
    prospective_identifiability: Literal[
        "CURRENTLY_IDENTIFIED",
        "PROSPECTIVELY_IDENTIFIABLE",
        "NOT_OPERATIONALIZABLE",
    ]
    current_relation_support_statement_ids: list[str] = Field(
        default_factory=list
    )
    grounded_bridge_statement_ids: list[str] = Field(
        default_factory=list
    )
    prospective_test_design: str | None = None
    prospective_falsifier: str | None = None
    ungrounded_required_concepts: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _directionality_contract(self) -> "IdentificationContract":
        premise_ids = set(map(str, self.premise_statement_ids))
        support_ids = list(
            map(str, self.directional_support_statement_ids)
        )

        outside = sorted(set(support_ids) - premise_ids)
        if outside:
            raise ValueError(
                "directional support statement IDs must be a "
                "subset of premise_statement_ids: "
                + repr(outside)
            )

        if self.assessment == "NOT_IDENTIFIABLE":
            if self.directionality_mode != "NOT_IDENTIFIABLE":
                raise ValueError(
                    "NOT_IDENTIFIABLE assessment requires "
                    "directionality_mode=NOT_IDENTIFIABLE"
                )
            if support_ids:
                raise ValueError(
                    "NOT_IDENTIFIABLE must not claim directional "
                    "support statement IDs"
                )
            return self

        if self.directionality_mode == "NOT_IDENTIFIABLE":
            raise ValueError(
                "directionality_mode=NOT_IDENTIFIABLE requires "
                "assessment=NOT_IDENTIFIABLE"
            )

        if self.directionality_mode == "DIRECTLY_SUPPORTED":
            if not support_ids:
                raise ValueError(
                    "DIRECTLY_SUPPORTED requires at least one "
                    "directional support statement ID"
                )

        elif self.directionality_mode == "GROUNDED_PREDICTION":
            if not support_ids:
                raise ValueError(
                    "GROUNDED_PREDICTION requires at least one "
                    "grounding statement ID"
                )
            if (
                self.prospective_identifiability
                != "PROSPECTIVELY_IDENTIFIABLE"
            ):
                raise ValueError(
                    "GROUNDED_PREDICTION is allowed only for "
                    "PROSPECTIVELY_IDENTIFIABLE hypotheses"
                )

        elif self.directionality_mode == "NON_DIRECTIONAL":
            if support_ids:
                raise ValueError(
                    "NON_DIRECTIONAL must not claim directional "
                    "support statement IDs"
                )

        return self

    @model_validator(mode="after")
    def _measurement_compatibility_contract(
        self,
    ) -> "IdentificationContract":
        premise_ids = set(map(str, self.premise_statement_ids))
        support_ids = list(
            map(str, self.measurement_support_statement_ids)
        )
        contextual_ids = list(
            map(str, self.contextual_measurement_statement_ids)
        )

        outside_support = sorted(set(support_ids) - premise_ids)
        outside_context = sorted(set(contextual_ids) - premise_ids)
        if outside_support:
            raise ValueError(
                "measurement support statement IDs must be a "
                "subset of premise_statement_ids: "
                + repr(outside_support)
            )
        if outside_context:
            raise ValueError(
                "contextual measurement statement IDs must be a "
                "subset of premise_statement_ids: "
                + repr(outside_context)
            )

        overlap = sorted(set(support_ids) & set(contextual_ids))
        if overlap:
            raise ValueError(
                "measurement support and contextual-only statement "
                "IDs must be disjoint: "
                + repr(overlap)
            )

        mode = self.measurement_compatibility_mode

        if mode == "DIRECTLY_COMPARABLE":
            if not support_ids:
                raise ValueError(
                    "DIRECTLY_COMPARABLE requires at least one "
                    "measurement support statement ID"
                )

        elif mode == "CONTEXTUAL_ONLY":
            if not support_ids:
                raise ValueError(
                    "CONTEXTUAL_ONLY requires at least one primary "
                    "measurement support statement ID"
                )
            if not contextual_ids:
                raise ValueError(
                    "CONTEXTUAL_ONLY requires at least one contextual "
                    "measurement statement ID"
                )

        elif mode == "PROSPECTIVE_MATCH_REQUIRED":
            if (
                self.prospective_identifiability
                != "PROSPECTIVELY_IDENTIFIABLE"
            ):
                raise ValueError(
                    "PROSPECTIVE_MATCH_REQUIRED is allowed only for "
                    "PROSPECTIVELY_IDENTIFIABLE hypotheses"
                )
            if not (support_ids or contextual_ids):
                raise ValueError(
                    "PROSPECTIVE_MATCH_REQUIRED requires grounded "
                    "measurement evidence"
                )

        elif mode == "NOT_COMPARABLE":
            if support_ids:
                raise ValueError(
                    "NOT_COMPARABLE must not claim primary measurement "
                    "support statement IDs"
                )
            if (
                self.prospective_identifiability
                != "NOT_OPERATIONALIZABLE"
            ):
                raise ValueError(
                    "NOT_COMPARABLE is terminal only when the relation "
                    "is NOT_OPERATIONALIZABLE"
                )

        return self

    @model_validator(mode="after")
    def _prospective_identification_contract(
        self,
    ) -> "IdentificationContract":
        premise_ids = set(map(str, self.premise_statement_ids))
        current_ids = list(
            map(str, self.current_relation_support_statement_ids)
        )
        bridge_ids = list(
            map(str, self.grounded_bridge_statement_ids)
        )

        outside_current = sorted(set(current_ids) - premise_ids)
        outside_bridge = sorted(set(bridge_ids) - premise_ids)
        if outside_current:
            raise ValueError(
                "current relation support statement IDs must be a "
                "subset of premise_statement_ids: "
                + repr(outside_current)
            )
        if outside_bridge:
            raise ValueError(
                "grounded bridge statement IDs must be a subset of "
                "premise_statement_ids: "
                + repr(outside_bridge)
            )

        status = self.prospective_identifiability

        if status == "CURRENTLY_IDENTIFIED":
            if self.assessment == "NOT_IDENTIFIABLE":
                raise ValueError(
                    "CURRENTLY_IDENTIFIED cannot use "
                    "assessment=NOT_IDENTIFIABLE"
                )
            if (
                self.current_evidence_status
                != "DIRECT_RELATION_SUPPORTED"
            ):
                raise ValueError(
                    "CURRENTLY_IDENTIFIED requires "
                    "current_evidence_status=DIRECT_RELATION_SUPPORTED"
                )
            if not current_ids:
                raise ValueError(
                    "CURRENTLY_IDENTIFIED requires at least one "
                    "current relation support statement ID"
                )

        elif status == "PROSPECTIVELY_IDENTIFIABLE":
            if self.assessment != "PARTIALLY_IDENTIFIED":
                raise ValueError(
                    "PROSPECTIVELY_IDENTIFIABLE requires "
                    "assessment=PARTIALLY_IDENTIFIED"
                )
            if not bridge_ids:
                raise ValueError(
                    "PROSPECTIVELY_IDENTIFIABLE requires at least "
                    "one grounded bridge statement ID"
                )
            if not str(self.prospective_test_design or "").strip():
                raise ValueError(
                    "PROSPECTIVELY_IDENTIFIABLE requires an explicit "
                    "prospective test design"
                )
            if not str(self.prospective_falsifier or "").strip():
                raise ValueError(
                    "PROSPECTIVELY_IDENTIFIABLE requires an explicit "
                    "prospective falsifier"
                )
            if self.ungrounded_required_concepts:
                raise ValueError(
                    "PROSPECTIVELY_IDENTIFIABLE cannot depend on "
                    "ungrounded required concepts"
                )

        elif status == "NOT_OPERATIONALIZABLE":
            if self.assessment != "NOT_IDENTIFIABLE":
                raise ValueError(
                    "NOT_OPERATIONALIZABLE requires "
                    "assessment=NOT_IDENTIFIABLE"
                )

        return self


class RelationValidityAwareGenerationResponse(StrictModel):
    identification_contract: IdentificationContract
    draft: HypothesisPortfolioDraft

    @model_validator(mode="after")
    def _treatment_internal_contract(
        self,
    ) -> "RelationValidityAwareGenerationResponse":
        assessment = self.identification_contract.assessment

        if assessment == "NOT_IDENTIFIABLE":
            if self.draft.hypotheses:
                raise ValueError(
                    "NOT_IDENTIFIABLE requires an empty treatment draft"
                )
            return self

        if len(self.draft.hypotheses) != 1:
            raise ValueError(
                "SUPPORTED/PARTIALLY_IDENTIFIED requires exactly one "
                "treatment hypothesis"
            )

        row = self.draft.hypotheses[0]

        if (
            self.identification_contract.directionality_mode
            == "NON_DIRECTIONAL"
        ):
            directional_predictions = [
                str(item.expected_direction)
                for item in row.predicted_observations
                if str(item.expected_direction).strip().lower()
                not in {"", "unspecified", "none", "unknown"}
            ]
            if directional_predictions:
                raise ValueError(
                    "NON_DIRECTIONAL treatment requires all "
                    "predicted_observations.expected_direction to be "
                    "unspecified/none/unknown; got "
                    + repr(directional_predictions)
                )

        predicted = {
            str(item.observable).strip()
            for item in row.predicted_observations
        }
        for item in row.falsification_criteria:
            observable = str(item.observable).strip()
            if observable not in predicted:
                raise ValueError(
                    "every treatment falsifier.observable must exactly "
                    "reuse one predicted_observation.observable; "
                    f"unmatched={observable!r}"
                )
        return self


SYSTEM_PROMPT = """You are the relation-validity-aware treatment generator in a
shadow scientific-discovery ablation.

Your purpose is NOT to find a more novel hypothesis. Your purpose is to test
whether the same grounded evidence can support a more scientifically
identifiable formulation.

The supplied CONTROL hypothesis and the supplied grounded HypothesisContext are
the only positive scientific basis. External literature is not a positive
premise and is not supplied here.

You MUST preserve exactly:
- the CONTROL premise_statement_ids,
- the CONTROL gap_statement_ids,
- the CONTROL hypothesis_type,
- the research question and source context.

You MAY change:
- title,
- hypothesis_statement,
- inferential_bridge,
- qualitative predictions,
- falsifiers,
- assumptions.

Prefer a narrower conditional or associative claim over an unsupported causal
or directional claim. Explicitly account for comparison context, required
controls, measurement compatibility, and confounders.

DIRECTIONALITY MUST DISTINGUISH OBSERVATION FROM PREDICTION.

Use directionality_mode=DIRECTLY_SUPPORTED only when supplied grounded premises
explicitly support the same direction in the same relevant comparison context.

Use directionality_mode=GROUNDED_PREDICTION when the direction is not already
observed, but the supplied grounded premises support the variables or mechanistic
bridge strongly enough to motivate a falsifiable directional prediction.
In that case:
- prospective_identifiability MUST be PROSPECTIVELY_IDENTIFIABLE;
- directional_support_statement_ids are grounding/bridge IDs, NOT evidence that
  the predicted direction has already been observed;
- write the hypothesis as a prediction (for example: may, is predicted to, or
  we hypothesize), never as an established finding;
- the prospective test design must state how the direction would be tested.

Use NON_DIRECTIONAL when neither an observed nor a grounded predicted direction
is warranted. Use NOT_IDENTIFIABLE only when the relation is not operationalizable.

Do not infer direction merely because a variable was varied, a response depends
on a parameter, two systems each show an effect, a mechanism is merely plausible,
a resonance is involved, or the literature leaves a relational gap.

MEASUREMENT COMPATIBILITY MUST DISTINGUISH CURRENT COMPARABILITY FROM A
PROSPECTIVE MATCHED EXPERIMENT.

DIRECTLY_COMPARABLE:
The supplied grounded evidence already establishes compatible observable
definitions, measurement level, normalization, aggregation, and replicate
semantics for the proposed relation.

CONTEXTUAL_ONLY:
One grounded measurement context directly supports the scoped current-context
claim; other records are only contextual motivation. Do not use contextual-only
records as positive support for a current cross-context relation.

PROSPECTIVE_MATCH_REQUIRED:
The current records are not directly comparable, but they ground the relevant
variables or observables well enough that an explicit future matched experiment
can identify the proposed relation. This is allowed for a novel hypothesis.
Required future matching does NOT retroactively become current positive evidence.
The hypothesis must be written prospectively and the contract must specify the
matched measurement plan, controls, and falsifier.

NOT_COMPARABLE:
Use only when no coherent prospective matched design can operationalize the
relation without inventing ungrounded variables, observables, mechanisms, or
scientific concepts. This must map to NOT_OPERATIONALIZABLE.

PROSPECTIVE IDENTIFICATION IS DISTINCT FROM CURRENT EVIDENCE SUPPORT.

Set current_evidence_status:
- DIRECT_RELATION_SUPPORTED when current grounded evidence itself supports the
  proposed relation;
- PARTIAL_GROUNDING when grounded evidence supports the variables, mechanism,
  bridge, or component relations but not the full proposed relation;
- CONTEXT_ONLY when supplied evidence merely motivates the topic and cannot
  ground the variables or bridge needed by the proposed relation.

Set prospective_identifiability:
- CURRENTLY_IDENTIFIED when the current grounded evidence directly supports the
  relation;
- PROSPECTIVELY_IDENTIFIABLE when the relation is not currently established but
  its variables or bridge are grounded and an explicit feasible matched
  experiment or observation can identify it;
- NOT_OPERATIONALIZABLE when testing the relation would require ungrounded
  concepts or no coherent falsifiable design can be specified.

For CURRENTLY_IDENTIFIED, list the direct positive-support premise IDs in
current_relation_support_statement_ids.

For PROSPECTIVELY_IDENTIFIABLE:
- list premise IDs grounding the variables or mechanistic bridge in
  grounded_bridge_statement_ids;
- give an explicit prospective_test_design;
- give an explicit prospective_falsifier;
- ungrounded_required_concepts MUST be empty;
- do not describe the prospective relation as something the existing evidence
  already demonstrates.

If the hypothesis requires a reporter-specific chemical affinity, analyte
property, mechanism, observable, structural distinction, or comparison variable
that is absent from the supplied grounded premises, record it in
ungrounded_required_concepts and choose NOT_OPERATIONALIZABLE rather than
inventing support.

Future controls, future matched protocols, and future measurements establish
TESTABILITY, not current positive evidence.

The positive scientific premise invariant remains absolute:
only the supplied grounded KG or corpus evidence may ground the hypothesis.
External literature, prior-art search, and prospective experimental design are
never new positive premises.

Do not invent matched controls that are absent from the grounded premises.
Do not claim novelty.
Do not add numerical values absent from the grounded premises.
Do not add positive scientific facts from general knowledge.

If the same grounded premises cannot support a scientifically identifiable
relation without inventing controls or collapsing incompatible contexts, set
identification_contract.assessment=NOT_IDENTIFIABLE and return an empty draft
with a concise abstention_reason.

If you return a hypothesis, return exactly one.

For every falsification criterion, falsifier.observable MUST exactly reuse the
observable string from one of that same hypothesis's predicted_observations.
Do not create a new, broader, narrower, or differently worded observable only
for falsification. Put the falsifying condition in falsifying_outcome.

If assessment=NOT_IDENTIFIABLE, return hypotheses=[].
If assessment is SUPPORTED or PARTIALLY_IDENTIFIED, return exactly one
hypothesis.

The identification contract is diagnostic only. It creates no scientific,
novelty, generation, or production-selection authority."""


def _statement_index(context: HypothesisContext) -> dict[str, Any]:
    return {str(row.statement_id): row for row in context.evidence_statements}


def build_user_payload(*, context: HypothesisContext, control: Any) -> dict[str, Any]:
    statements = _statement_index(context)
    premise_ids = list(map(str, control.premise_statement_ids))
    gap_ids = list(map(str, control.gap_statement_ids))
    return {
        "task": {
            "question": context.question,
            "context_id": context.context_id,
            "domain_profile_id": context.domain_profile_id,
        },
        "control_hypothesis": {
            "hypothesis_id": str(control.hypothesis_id),
            "title": str(control.title),
            "hypothesis_statement": str(control.hypothesis_statement),
            "hypothesis_type": str(control.hypothesis_type),
            "premise_statement_ids": premise_ids,
            "gap_statement_ids": gap_ids,
            "inferential_bridge": str(control.inferential_bridge),
            "predicted_observations": [
                row.model_dump(mode="json") for row in control.predicted_observations
            ],
            "falsification_criteria": [
                row.model_dump(mode="json") for row in control.falsification_criteria
            ],
            "assumptions": list(control.assumptions),
            "cross_paper_synthesis": bool(control.cross_paper_synthesis),
            "source_paper_ids": list(control.source_paper_ids),
        },
        "locked_grounded_premises": [
            {
                "statement_id": sid,
                "text": str(statements[sid].text),
                "epistemic_role": str(statements[sid].epistemic_role),
                "paper_ids": list(statements[sid].paper_ids),
            }
            for sid in premise_ids
            if sid in statements
        ],
        "locked_gap_statements": [
            {
                "statement_id": sid,
                "text": str(statements[sid].text),
                "paper_ids": list(statements[sid].paper_ids),
            }
            for sid in gap_ids
            if sid in statements
        ],
        "treatment_constraints": {
            "exact_premise_identity_required": True,
            "exact_gap_identity_required": True,
            "exact_hypothesis_type_required": True,
            "new_positive_premises_allowed": False,
            "external_prior_art_as_positive_premise": False,
            "novelty_optimization_allowed": False,
            "abstention_preferred_over_unidentified_relation": True,
        },
    }


def validate_response_against_control(
    *, control: Any, response: RelationValidityAwareGenerationResponse
) -> list[str]:
    failures: list[str] = []
    expected_premises = sorted(map(str, control.premise_statement_ids))
    expected_gaps = sorted(map(str, control.gap_statement_ids))
    contract_premises = sorted(
        map(str, response.identification_contract.premise_statement_ids)
    )
    if contract_premises != expected_premises:
        failures.append("IDENTIFICATION_CONTRACT_PREMISE_MISMATCH")

    draft = response.draft
    if len(draft.hypotheses) > 1:
        failures.append("MULTIPLE_TREATMENT_HYPOTHESES")
        return failures
    if not draft.hypotheses:
        if response.identification_contract.assessment != "NOT_IDENTIFIABLE":
            failures.append("TREATMENT_ABSTENTION_REQUIRES_NOT_IDENTIFIABLE")
        return failures

    row = draft.hypotheses[0]
    if sorted(map(str, row.premise_statement_ids)) != expected_premises:
        failures.append("TREATMENT_PREMISE_IDENTITY_VIOLATION")
    if sorted(map(str, row.gap_statement_ids)) != expected_gaps:
        failures.append("TREATMENT_GAP_IDENTITY_VIOLATION")
    if str(row.hypothesis_type) != str(control.hypothesis_type):
        failures.append("TREATMENT_HYPOTHESIS_TYPE_VIOLATION")
    if response.identification_contract.assessment == "NOT_IDENTIFIABLE":
        failures.append("NOT_IDENTIFIABLE_RESPONSE_MUST_ABSTAIN")
    return failures


def compile_treatment(
    *,
    context: HypothesisContext,
    control: Any,
    response: RelationValidityAwareGenerationResponse,
) -> tuple[HypothesisPortfolio | None, list[str]]:
    failures = validate_response_against_control(control=control, response=response)
    if failures:
        return None, failures
    if not response.draft.hypotheses:
        return None, []
    try:
        portfolio = HypothesisCompiler().compile(context, response.draft)
    except HypothesisCompileError as exc:
        return None, [
            f"COMPILE:{row.code}:{row.location}:{row.message}" for row in exc.issues
        ]
    except Exception as exc:
        return None, [f"COMPILE:{type(exc).__name__}:{exc}"]

    validation = HypothesisValidator().validate(context, portfolio)
    if not validation.passes:
        return None, [
            f"VALIDATION:{row.code}:{row.location}:{row.message}"
            for row in validation.issues
            if row.severity == "error"
        ]
    if len(portfolio.hypotheses) != 1:
        return None, ["COMPILED_TREATMENT_NOT_SINGLE_HYPOTHESIS"]
    card = portfolio.hypotheses[0]
    if sorted(map(str, card.premise_statement_ids)) != sorted(
        map(str, control.premise_statement_ids)
    ):
        return None, ["COMPILED_TREATMENT_PREMISE_IDENTITY_VIOLATION"]
    return portfolio, []


class RelationValidityAwareGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str | None = None,
        parse_retries: int = 2,
        max_tokens: int = 8192,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = str(model)
        self.api_key_env = str(api_key_env)
        self.base_url = base_url
        self.parse_retries = int(parse_retries)
        self.max_tokens = max(1024, int(max_tokens))
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"No API key available in {self.api_key_env}")
        import instructor
        from openai import OpenAI
        kwargs: dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        raw = OpenAI(**kwargs)
        self._client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
        return self._client

    def generate(
        self, *, context: HypothesisContext, control: Any
    ) -> RelationValidityAwareGenerationResponse:
        payload = build_user_payload(context=context, control=control)
        response, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model,
            response_model=RelationValidityAwareGenerationResponse,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
            temperature=0.0,
            max_retries=self.parse_retries,
            request_kwargs={"max_tokens": self.max_tokens},
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "relation_validity_aware_generation_shadow_v1",
                "stage": "paired_reformulation",
            },
        )
        if not isinstance(response, RelationValidityAwareGenerationResponse):
            response = RelationValidityAwareGenerationResponse.model_validate(response)
        return response
