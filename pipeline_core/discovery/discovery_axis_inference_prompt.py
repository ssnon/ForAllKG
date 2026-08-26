from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxis
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
)


INFERENCE_PROMPT_VERSION = (
    "axis-inference-critic-prompt-v1.1-basis-references"
)


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def axis_basis_reference_map(
    axis: DiscoveryAxis,
) -> dict[str, str]:
    """Stable LLM-facing references to authoritative axis semantics.

    The model selects short immutable reference IDs rather than
    reproducing long axis strings. The compiler later resolves those
    IDs back to the exact authoritative strings.
    """

    rows: dict[str, str] = {}

    label = str(
        axis.label
    ).strip()

    if label:
        rows[
            "axis_basis:label"
        ] = label

    pieces = [
        str(
            axis.proposed_subject
        ).strip(),
        str(
            axis.proposed_relation
        ).strip(),
        str(
            axis.proposed_object
        ).strip(),
    ]

    if all(pieces):
        triple = (
            " | ".join(
                pieces
            )
        )

        # Avoid exposing two different reference IDs for identical
        # semantic text.
        if triple not in rows.values():
            rows[
                "axis_basis:triple"
            ] = triple

    return rows


def allowed_axis_basis(
    axis: DiscoveryAxis,
) -> tuple[str, ...]:
    """Authoritative human-readable axis basis strings."""

    return tuple(
        axis_basis_reference_map(
            axis
        ).values()
    )


def resolve_axis_basis_reference(
    axis: DiscoveryAxis,
    value: str,
) -> str | None:
    """Resolve a stable basis ID or a legacy exact-string reference.

    Exact historical strings remain accepted for backward compatibility.
    Abbreviations, paraphrases, translations, and fuzzy matches remain
    invalid.
    """

    reference_map = (
        axis_basis_reference_map(
            axis
        )
    )

    value = str(
        value
    ).strip()

    if value in reference_map:
        return reference_map[
            value
        ]

    if value in reference_map.values():
        return value

    return None


def expected_assertions(
    card: HypothesisCard,
) -> tuple[dict[str, str], ...]:
    rows = [
        {
            "assertion_id":
                f"central:{card.hypothesis_id}",
            "assertion_kind":
                "central_hypothesis",
            "assertion_text":
                card.hypothesis_statement,
        }
    ]

    for prediction in card.predicted_observations:
        rows.append({
            "assertion_id":
                prediction.observation_id,
            "assertion_kind":
                "prediction",
            "assertion_text":
                prediction.observable,
        })

    return tuple(rows)


@dataclass(frozen=True)
class AxisInferencePrompt:
    prompt_version: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> "AxisInferencePrompt":
        canonical = _compact_json({
            "prompt_version":
                INFERENCE_PROMPT_VERSION,
            "system_prompt":
                system_prompt,
            "user_prompt":
                user_prompt,
        })

        return cls(
            prompt_version=
                INFERENCE_PROMPT_VERSION,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_sha256=_sha256(canonical),
        )


SYSTEM_PROMPT = """
You are the discovery-axis inference-strength critic for an
evidence-grounded scientific hypothesis system.

Your task is narrow:

Determine whether a hypothesis uses its assigned inspiration-only
discovery axis at an epistemically appropriate level of specificity.

You MUST use only:
1. the selected positive premises supplied for this hypothesis;
2. the explicit discovery-axis semantics;
3. the generated hypothesis, bridge, and predictions.

Do not use outside scientific knowledge.
Do not assess external novelty.
Do not assess experimental feasibility.
Do not rewrite or repair the hypothesis.

The discovery axis is inspiration-only. It is NOT positive evidence.

Classify the central hypothesis and EVERY prediction using exactly one
source class:

G_GROUNDED
    The assertion or the scientific specificity being evaluated is
    directly supported by selected positive premises.

A_AXIS
    The assertion is explicitly supplied by the discovery axis itself.
    The axis remains inspiration-only rather than established evidence.

S_BOUNDED_SYNTHESIS
    A genuinely new, testable relation needed to connect selected
    grounded premises with the assigned axis. This is permitted
    scientific hypothesis formation.

    S must remain open-direction unless its sign, order, response shape,
    optimum, threshold, or other directional specificity is itself
    supplied by G or A.

X_UNSUPPORTED_SPECIFICITY
    Additional specificity not supplied by G or A and not needed merely
    to connect them.

    Typical examples:
    - invented increase or decrease;
    - unsupported monotonic or non-monotonic shape;
    - unsupported optimum or turning point;
    - unsupported ordering;
    - unsupported threshold;
    - unsupported specific spectral shift;
    - unsupported mechanistic intermediate.

Important distinctions:

- A new moderator or interaction linking G and A can be
  S_BOUNDED_SYNTHESIS.
- Do not classify every new relation as X.
- An open qualitative difference can be valid S.
- S_BOUNDED_SYNTHESIS is limited to the minimum new relation needed to
  connect G and A, or a prediction that directly tests that central
  synthesized relation. It is not a license to add a second downstream
  mediator, descriptor shift, optimum, response shape, or mechanistic
  consequence merely because the individual variables occur in G or A.
- A secondary mechanistic or descriptor consequence that is not itself
  supplied by G or A and is not required to state or directly test the
  central synthesized relation is X_UNSUPPORTED_SPECIFICITY.
- A grounded variable does not automatically ground a newly proposed
  interaction involving that variable.
- A grounded statement that LSPR placement matters does not by itself
  ground a new moderator-dependent shift in optimum LSPR.
- Axis vocabulary appearing in a prediction does not make all
  downstream consequences A_AXIS.
- Treat expected_direction as part of the scientific assertion.
  increase, decrease, shift, and non_monotonic each assert additional
  directional or response-form content. Do not call such a prediction
  open-direction merely because a sign or magnitude is unspecified.
  If that direction or response form is not supplied by G or A, require
  OPEN_DIRECTION, REFRAME, or REMOVE as appropriate.

Action semantics are fixed:

G_GROUNDED
    KEEP

A_AXIS
    KEEP_HYPOTHETICAL

S_BOUNDED_SYNTHESIS
    KEEP when already open-direction;
    OPEN_DIRECTION when unnecessary sign/shape/direction is embedded.

X_UNSUPPORTED_SPECIFICITY
    REFRAME or REMOVE.

Reference discipline:

- grounded_statement_ids may contain ONLY IDs from the supplied
  allowed_grounded_statement_ids list.
- axis_basis may contain ONLY basis_id values supplied in
  axis_basis_references.
- Do NOT copy, abbreviate, paraphrase, or regenerate the human-readable
  axis text into axis_basis. Select its stable basis_id instead.
- Copy assertion_id and assertion_text exactly from expected_assertions.
- Review every expected assertion exactly once.
- Never invent, abbreviate, translate, or infer an identifier.
- If no permitted reference applies, leave that list empty.

Return only the requested structured inference review draft.
""".strip()


class DiscoveryAxisInferencePromptAssembler:
    def build(
        self,
        context: HypothesisContext,
        axis: DiscoveryAxis,
        card: HypothesisCard,
    ) -> AxisInferencePrompt:
        if card.source_context_id != context.context_id:
            raise ValueError(
                "hypothesis/context ID mismatch"
            )

        if (
            card.source_context_sha256
            != context.context_sha256
        ):
            raise ValueError(
                "hypothesis/context SHA mismatch"
            )

        statement_by_id = {
            row.statement_id: row
            for row in context.evidence_statements
        }

        premises = []

        for statement_id in card.premise_statement_ids:
            statement = statement_by_id.get(
                statement_id
            )

            if statement is None:
                raise ValueError(
                    "unknown hypothesis premise statement: "
                    f"{statement_id}"
                )

            if not statement.eligible_as_premise:
                raise ValueError(
                    "inference critic received an ineligible "
                    "positive premise: "
                    f"{statement_id}"
                )

            premises.append(
                statement.model_dump(
                    mode="json"
                )
            )

        axis_basis_map = (
            axis_basis_reference_map(
                axis
            )
        )

        axis_basis = list(
            axis_basis_map.values()
        )

        if not axis_basis_map:
            raise ValueError(
                "discovery axis exposes no usable basis"
            )

        predictions = [
            {
                "observation_id":
                    row.observation_id,
                "observable":
                    row.observable,
                "expected_direction":
                    row.expected_direction,
                "rationale":
                    row.rationale,
            }
            for row in card.predicted_observations
        ]

        payload = {
            "source_lineage": {
                "context_id":
                    context.context_id,
                "context_sha256":
                    context.context_sha256,
                "axis_id":
                    axis.axis_id,
                "hypothesis_id":
                    card.hypothesis_id,
            },

            "selected_positive_premises":
                premises,

            "allowed_grounded_statement_ids":
                list(card.premise_statement_ids),

            "discovery_axis": {
                "axis_id":
                    axis.axis_id,
                "inspiration_id":
                    axis.inspiration_id,
                "candidate_unit_id":
                    axis.candidate_unit_id,
                "label":
                    axis.label,
                "proposed_subject":
                    axis.proposed_subject,
                "proposed_relation":
                    axis.proposed_relation,
                "proposed_object":
                    axis.proposed_object,
                "requires_verification":
                    axis.requires_verification,
            },

            # Human-readable authoritative semantics.
            "allowed_axis_basis":
                axis_basis,

            # LLM-facing reference surface. axis_basis in the response
            # should contain basis_id values from this table.
            "axis_basis_references": [
                {
                    "basis_id":
                        basis_id,
                    "text":
                        basis_text,
                }
                for basis_id, basis_text
                in axis_basis_map.items()
            ],

            "hypothesis": {
                "hypothesis_id":
                    card.hypothesis_id,
                "title":
                    card.title,
                "hypothesis_statement":
                    card.hypothesis_statement,
                "inferential_bridge":
                    card.inferential_bridge,
                "predicted_observations":
                    predictions,
                "assumptions":
                    list(card.assumptions),
            },

            "expected_assertions":
                list(
                    expected_assertions(card)
                ),
        }

        user_prompt = (
            "DISCOVERY-AXIS INFERENCE REVIEW INPUT\n"
            "====================================\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n\nOUTPUT REQUIREMENTS\n"
            "===================\n"
            "- Review every expected_assertion exactly once.\n"
            "- Use assertion_id, assertion_kind, and assertion_text exactly as supplied.\n"
            "- grounded_statement_ids may use only allowed_grounded_statement_ids.\n"
            "- axis_basis may use only basis_id values from axis_basis_references.\n"
            "- G_GROUNDED requires direct selected-premise support.\n"
            "- A_AXIS requires explicit discovery-axis support.\n"
            "- S_BOUNDED_SYNTHESIS is allowed only for a bounded G+A connection.\n"
            "- X_UNSUPPORTED_SPECIFICITY marks unnecessary added specificity.\n"
            "- Do not rewrite the hypothesis in this review.\n"
        )

        return AxisInferencePrompt.create(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
