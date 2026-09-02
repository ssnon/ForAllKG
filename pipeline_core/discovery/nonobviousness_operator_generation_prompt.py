from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismOperatorPolicyResult,
    MechanismSearchOperator,
    MechanismSemanticDraft,
)
from pipeline_core.discovery.nonobviousness_operator_generation_validation import (
    N11OperatorGenerationAuthority,
)


N11_OPERATOR_GENERATION_PROMPT_VERSION = (
    "n11-operator-generation-prompt-v1"
)


def _canonical_json(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _stable_component_id(
    kind: str,
    text: str,
) -> str:
    normalized = " ".join(
        str(text).split()
    )

    digest = hashlib.sha256(
        (
            f"{kind}|{normalized}"
        ).encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"n11_component:{kind}:{digest}"
    )


@dataclass(frozen=True)
class N11OperatorGenerationPrompt:
    hypothesis_id: str
    requested_operator: MechanismSearchOperator

    prompt_version: str
    prompt_sha256: str

    system_prompt: str
    user_prompt: str

    authority: N11OperatorGenerationAuthority

    shared_component_text_by_id: Mapping[
        str,
        str,
    ]

    supplemental_only_component_text_by_id: Mapping[
        str,
        str,
    ]


SYSTEM_PROMPT = """
You are the N11 operator-conditioned scientific hypothesis generator
for an evidence-grounded discovery system.

Your role is narrow.

You receive:
1. positive baseline scientific evidence already eligible as premises;
2. a separately grounded supplemental mechanism component;
3. unresolved research-gap statements;
4. a semantic decomposition comparing the baseline and supplemental
   mechanisms;
5. exactly one deterministically authorized scientific search operator.

You may generate AT MOST ONE candidate hypothesis.

A generated hypothesis is an inference, not reported evidence.

EPISTEMIC LANES
===============

BASELINE POSITIVE PREMISES

These are existing HypothesisContext statements that are positively
grounded and eligible as scientific premises.

They may be cited only through baseline_premise_statement_ids.

SUPPLEMENTAL GROUNDED MECHANISM NODES

These are scientifically grounded mechanism nodes recovered through a
separate strict retrieval lane.

They are NOT HypothesisContext premise statements.

They may be referenced only through supplemental_mechanism_node_ids.

Their existence does NOT establish that the task variable controls,
changes, activates, suppresses, or causes the supplemental mechanism.

RESEARCH GAPS

These are unresolved statements.

They may motivate the generated inference through gap_statement_ids.

They are NOT positive evidence.

SEMANTIC COMPONENTS

The B1 semantic reviewer has decomposed the mechanisms into:
- shared components;
- baseline-only components;
- supplemental-only components.

These semantic component labels organize the authorized inference.
They are not independent literature claims.

AUTHORIZED OPERATOR
===================

For C1, the only implemented generation operator is:

RELATIVE_CONTRIBUTION_SHIFT

Its meaning is:

Given separately grounded mechanistic components, propose a falsifiable
possibility that the task variable changes their RELATIVE contribution,
balance, weighting, or mechanistic importance to the observed response.

This is NOT permission to claim:
- pathway competition;
- a mechanism switch;
- a threshold;
- a reversal;
- non-monotonic behavior;
- a specific quantitative ratio;
- a known direction of increase/decrease.

The relative-contribution relation itself is the NEW INFERENCE.

Do not write it as though it were already reported.

GENERATION REQUIREMENTS
=======================

If generating a candidate:

1. Use exactly the requested operator.

2. Select only allowed baseline_premise_statement_ids.

3. Select only allowed supplemental_mechanism_node_ids.

4. Select at least one allowed unresolved gap_statement_id.

5. Use at least one shared_component_id.

6. Use at least one supplemental_only_component_id.

7. relative_contribution_claim must explicitly state a proposed
   relative contribution/balance/weighting relation.

8. inferential_bridge must clearly distinguish:
   - what is grounded;
   - what relation is being inferred.

9. Provide at least one qualitative predicted observation.

10. One predicted observation must discriminate the operator-level
    hypothesis from a simple "total SERS magnitude changes" hypothesis.

    In other words, the discriminating observable must concern evidence
    for a CHANGE IN RELATIVE MECHANISTIC CONTRIBUTION or balance, not
    merely stronger/weaker total SERS.

11. Provide at least one falsification condition tied to a predicted
    observable. Set prediction_local_id to the exact local_id of the
    prediction being falsified. Do not restate or paraphrase the
    observable inside the falsifier.

12. generated_relation_status must be INFERENCE_NOT_REPORTED.

13. task_to_supplemental_relation_grounded must be false.

14. Keep predictions qualitative.

15. If the supplied evidence does not support a scientifically
    interpretable discriminating prediction without inventing stronger
    structure, abstain.

PROHIBITED
==========

Do NOT:
- claim external novelty;
- say novel, unprecedented, or first;
- invent papers or IDs;
- promote supplemental nodes into baseline premise IDs;
- use gaps as positive evidence;
- claim competition;
- claim mechanism switching;
- invent thresholds;
- invent reversals;
- invent non-monotonic behavior;
- invent numeric values;
- write an experimental protocol;
- claim that spacing is already known to modulate the supplemental
  chemical mechanism;
- merely restate that total SERS changes with spacing.

Return only the requested structured N11OperatorGenerationDraft.
""".strip()


class N11OperatorGenerationPromptAssembler:
    def build(
        self,
        *,
        hypothesis_id: str,
        scientific_task: str,
        requested_operator: MechanismSearchOperator,
        policy: MechanismOperatorPolicyResult,
        semantic_draft: MechanismSemanticDraft,
        baseline_mechanism_statements: Sequence[Mapping[str, Any]],
        supplemental_mechanism_nodes: Sequence[Mapping[str, Any]],
        gap_statements: Sequence[Mapping[str, Any]],
        task_feature: Mapping[str, Any],
    ) -> N11OperatorGenerationPrompt:
        hypothesis_id = str(
            hypothesis_id
        ).strip()

        scientific_task = str(
            scientific_task
        ).strip()

        if not hypothesis_id:
            raise ValueError(
                "hypothesis_id is required"
            )

        if not scientific_task:
            raise ValueError(
                "scientific_task is required"
            )

        if (
            requested_operator
            not in policy.eligible_operators
        ):
            raise ValueError(
                "requested operator is not "
                "deterministically authorized"
            )

        if (
            requested_operator
            != "RELATIVE_CONTRIBUTION_SHIFT"
        ):
            raise ValueError(
                "C1 currently supports only "
                "RELATIVE_CONTRIBUTION_SHIFT"
            )

        if semantic_draft.task_relation_grounded:
            raise ValueError(
                "operator generation requires an "
                "unresolved task-to-supplemental relation"
            )

        if (
            semantic_draft.classification
            != (
                "PARTIAL_OVERLAP_"
                "WITH_DISTINCT_COMPONENT"
            )
        ):
            raise ValueError(
                "RELATIVE_CONTRIBUTION_SHIFT requires "
                "partial overlap with a distinct component"
            )

        if not (
            semantic_draft
            .shared_mechanistic_components
        ):
            raise ValueError(
                "shared mechanistic components are required"
            )

        if not (
            semantic_draft
            .supplemental_only_components
        ):
            raise ValueError(
                "supplemental-only mechanistic "
                "components are required"
            )

        baseline_rows = []
        baseline_ids = []

        for row in (
            baseline_mechanism_statements
        ):
            statement_id = str(
                row.get(
                    "statement_id",
                    "",
                )
            ).strip()

            text = str(
                row.get(
                    "text",
                    "",
                )
            ).strip()

            if (
                not statement_id
                or not text
            ):
                raise ValueError(
                    "baseline statement requires "
                    "statement_id and text"
                )

            baseline_ids.append(
                statement_id
            )

            baseline_rows.append(
                {
                    "statement_id":
                        statement_id,
                    "text":
                        text,
                    "paper_ids":
                        list(
                            row.get(
                                "paper_ids",
                                [],
                            )
                            or []
                        ),
                    "claim_kind":
                        row.get(
                            "claim_kind"
                        ),
                    "epistemic_role":
                        row.get(
                            "epistemic_role"
                        ),
                }
            )

        if not baseline_rows:
            raise ValueError(
                "baseline mechanism evidence is required"
            )

        supplemental_rows = []
        supplemental_ids = []

        for row in (
            supplemental_mechanism_nodes
        ):
            node_id = str(
                row.get(
                    "node_id",
                    "",
                )
            ).strip()

            node_text = str(
                row.get(
                    "node_text",
                    "",
                )
            ).strip()

            if (
                not node_id
                or not node_text
            ):
                raise ValueError(
                    "supplemental mechanism requires "
                    "node_id and node_text"
                )

            supplemental_ids.append(
                node_id
            )

            supplemental_rows.append(
                {
                    "node_id":
                        node_id,
                    "label":
                        row.get(
                            "label"
                        ),
                    "node_text":
                        node_text,
                    "source_paper_id":
                        row.get(
                            "source_paper_id"
                        ),
                }
            )

        if not supplemental_rows:
            raise ValueError(
                "supplemental mechanism evidence is required"
            )

        gap_rows = []
        gap_ids = []

        for row in gap_statements:
            statement_id = str(
                row.get(
                    "statement_id",
                    "",
                )
            ).strip()

            text = str(
                row.get(
                    "text",
                    "",
                )
            ).strip()

            if (
                not statement_id
                or not text
            ):
                raise ValueError(
                    "gap statement requires "
                    "statement_id and text"
                )

            gap_ids.append(
                statement_id
            )

            gap_rows.append(
                {
                    "statement_id":
                        statement_id,
                    "text":
                        text,
                    "paper_ids":
                        list(
                            row.get(
                                "paper_ids",
                                [],
                            )
                            or []
                        ),
                    "claim_kind":
                        row.get(
                            "claim_kind"
                        ),
                    "epistemic_role":
                        row.get(
                            "epistemic_role"
                        ),
                }
            )

        if not gap_rows:
            raise ValueError(
                "at least one unresolved gap is required"
            )

        shared_component_text_by_id = {}

        for text in (
            semantic_draft
            .shared_mechanistic_components
        ):
            component_id = (
                _stable_component_id(
                    "shared",
                    text,
                )
            )

            shared_component_text_by_id[
                component_id
            ] = text

        supplemental_component_text_by_id = {}

        for text in (
            semantic_draft
            .supplemental_only_components
        ):
            component_id = (
                _stable_component_id(
                    "supplemental",
                    text,
                )
            )

            supplemental_component_text_by_id[
                component_id
            ] = text

        authority = (
            N11OperatorGenerationAuthority(
                requested_operator=
                    requested_operator,

                eligible_operators=tuple(
                    policy.eligible_operators
                ),

                allowed_baseline_statement_ids=tuple(
                    baseline_ids
                ),

                allowed_supplemental_node_ids=tuple(
                    supplemental_ids
                ),

                allowed_gap_statement_ids=tuple(
                    gap_ids
                ),

                allowed_shared_component_ids=tuple(
                    shared_component_text_by_id
                ),

                allowed_supplemental_only_component_ids=tuple(
                    supplemental_component_text_by_id
                ),
            )
        )

        payload = {
            "hypothesis_id":
                hypothesis_id,

            "scientific_task":
                scientific_task,

            "requested_operator":
                requested_operator,

            "operator_authority": {
                "eligible_operators":
                    list(
                        policy
                        .eligible_operators
                    ),

                "requested_operator":
                    requested_operator,

                "blocked_operators":
                    dict(
                        policy
                        .blocked_operators
                    ),
            },

            "task_feature":
                dict(
                    task_feature
                ),

            "baseline_positive_mechanism_evidence":
                baseline_rows,

            "supplemental_grounded_mechanism_nodes":
                supplemental_rows,

            "unresolved_gap_evidence":
                gap_rows,

            "semantic_decomposition": {
                "classification":
                    semantic_draft
                    .classification,

                "shared_components": [
                    {
                        "component_id":
                            component_id,
                        "text":
                            text,
                    }
                    for (
                        component_id,
                        text,
                    )
                    in (
                        shared_component_text_by_id
                        .items()
                    )
                ],

                "baseline_only_components":
                    list(
                        semantic_draft
                        .baseline_only_components
                    ),

                "supplemental_only_components": [
                    {
                        "component_id":
                            component_id,
                        "text":
                            text,
                    }
                    for (
                        component_id,
                        text,
                    )
                    in (
                        supplemental_component_text_by_id
                        .items()
                    )
                ],

                "task_relation_grounded":
                    semantic_draft
                    .task_relation_grounded,

                "reason_summary":
                    semantic_draft
                    .reason_summary,

                "epistemic_cautions":
                    list(
                        semantic_draft
                        .epistemic_cautions
                    ),
            },

            "allowed_ids": {
                "baseline_premise_statement_ids":
                    baseline_ids,

                "supplemental_mechanism_node_ids":
                    supplemental_ids,

                "gap_statement_ids":
                    gap_ids,

                "shared_component_ids":
                    list(
                        shared_component_text_by_id
                    ),

                "supplemental_only_component_ids":
                    list(
                        supplemental_component_text_by_id
                    ),
            },

            "epistemic_boundary": {
                "supplemental_nodes_are_baseline_premises":
                    False,

                "gaps_are_positive_evidence":
                    False,

                "task_to_supplemental_relation_grounded":
                    False,

                "generated_relation_is_reported":
                    False,

                "llm_has_operator_authority":
                    False,
            },
        }

        user_prompt = (
            "N11 OPERATOR-CONDITIONED GENERATION INPUT\n"
            "========================================\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n\nOUTPUT DISCIPLINE\n"
            "=================\n"
            "- Return N11OperatorGenerationDraft only.\n"
            "- Generate at most one candidate.\n"
            "- Use exactly the requested operator.\n"
            "- Use only IDs listed in allowed_ids.\n"
            "- Keep supplemental mechanism node IDs separate "
            "from baseline premise statement IDs.\n"
            "- The proposed spacing-to-relative-contribution "
            "relation must remain explicitly inferential.\n"
            "- Include a discriminating observation that tests "
            "relative mechanistic contribution, not merely total "
            "SERS magnitude.\n"
            "- Do not introduce competition, switching, threshold, "
            "reversal, non-monotonicity, numeric predictions, "
            "experimental protocols, or external novelty claims.\n"
            "- Abstain if a useful discriminating prediction cannot "
            "be formed without violating these boundaries.\n"
        )

        hash_body = {
            "prompt_version":
                N11_OPERATOR_GENERATION_PROMPT_VERSION,

            "hypothesis_id":
                hypothesis_id,

            "requested_operator":
                requested_operator,

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                user_prompt,

            "authority": {
                "requested_operator":
                    authority
                    .requested_operator,

                "eligible_operators":
                    list(
                        authority
                        .eligible_operators
                    ),

                "allowed_baseline_statement_ids":
                    list(
                        authority
                        .allowed_baseline_statement_ids
                    ),

                "allowed_supplemental_node_ids":
                    list(
                        authority
                        .allowed_supplemental_node_ids
                    ),

                "allowed_gap_statement_ids":
                    list(
                        authority
                        .allowed_gap_statement_ids
                    ),

                "allowed_shared_component_ids":
                    list(
                        authority
                        .allowed_shared_component_ids
                    ),

                "allowed_supplemental_only_component_ids":
                    list(
                        authority
                        .allowed_supplemental_only_component_ids
                    ),
            },
        }

        return (
            N11OperatorGenerationPrompt(
                hypothesis_id=
                    hypothesis_id,

                requested_operator=
                    requested_operator,

                prompt_version=
                    N11_OPERATOR_GENERATION_PROMPT_VERSION,

                prompt_sha256=_sha256(
                    _canonical_json(
                        hash_body
                    )
                ),

                system_prompt=
                    SYSTEM_PROMPT,

                user_prompt=
                    user_prompt,

                authority=
                    authority,

                shared_component_text_by_id=
                    dict(
                        shared_component_text_by_id
                    ),

                supplemental_only_component_text_by_id=
                    dict(
                        supplemental_component_text_by_id
                    ),
            )
        )
