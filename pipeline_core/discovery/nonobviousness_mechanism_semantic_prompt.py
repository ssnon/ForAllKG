from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismSupplyGeometry,
)


MECHANISM_SEMANTIC_PROMPT_VERSION = (
    "n11-mechanism-semantic-review-prompt-v1"
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


@dataclass(frozen=True)
class MechanismSemanticPrompt:
    hypothesis_id: str

    prompt_version: str
    prompt_sha256: str

    system_prompt: str
    user_prompt: str

    baseline_support_statement_ids: tuple[
        str,
        ...,
    ]

    supplemental_mechanism_node_ids: tuple[
        str,
        ...,
    ]

    supply_geometry: MechanismSupplyGeometry


SYSTEM_PROMPT = """
You are the mechanism-semantic reviewer for an
evidence-grounded scientific hypothesis discovery system.

Your task is narrow.

Compare two separately grounded mechanistic evidence components:

1. a baseline mechanism branch already used by the hypothesis;
2. a supplemental mechanism component recovered from a separate
   grounded scientific structure.

You are NOT:
- generating a hypothesis;
- assessing literature-wide novelty;
- accepting or rejecting a hypothesis;
- deciding which N11 search operator is allowed;
- deciding production selection.

Your output contains semantic interpretation only.

SEMANTIC CLASSIFICATIONS
========================

SAME_MECHANISM

    The two components express substantially the same governing
    mechanistic content. Differences are mainly wording, granularity,
    material context, or representation.

SUPPLEMENTAL_SUBSUMES_BASELINE

    The supplemental mechanism explicitly contains the baseline
    mechanism as a mechanistic component and adds broader structure.

BASELINE_SUBSUMES_SUPPLEMENTAL

    The baseline mechanism explicitly contains the supplemental
    mechanism as a mechanistic component.

PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT

    The two components share one or more substantive mechanistic
    components, but each comparison also reveals mechanistic content
    not reducible to the shared component.

DISTINCT_MECHANISMS

    The supplied evidence supports genuinely different mechanistic
    explanations with no substantive governing mechanism overlap
    established by the supplied evidence.

INSUFFICIENT_FOR_JUDGMENT

    The supplied evidence is insufficient to classify the semantic
    relation conservatively.

EPISTEMIC RULES
===============

1. Different wording does not imply different mechanisms.

2. Multiple papers do not imply multiple mechanisms.

3. A composite mechanism containing electromagnetic/plasmonic
   enhancement plus an additional chemical, charge-transfer,
   molecular-resonance, catalytic, or other component must not be
   called simply DISTINCT if the baseline itself is electromagnetic
   or plasmonic.

4. Co-occurrence does not imply causation.

5. Synergy does not imply competition.

6. Two mechanisms do not imply a mechanism switch.

7. A COMMON_ANCHOR_CONTEXT establishes only that the task-related
   feature and supplemental mechanism are separately grounded under
   the same scientific anchor.

   It does NOT establish that changing the task variable changes,
   activates, suppresses, switches, or controls that mechanism.

8. Set task_relation_grounded=true only if the supplied scientific
   relations themselves explicitly support a relation from the
   task variable/feature to the supplemental mechanism.

9. Do not use ordinary scientific background knowledge to repair a
   missing relation.

10. Do not invent papers, mechanisms, causal links, moderators,
    thresholds, competition, switches, or relative-contribution
    effects.

COMPONENT DECOMPOSITION
=======================

shared_mechanistic_components:
    Mechanistic content explicitly supported in both components.

baseline_only_components:
    Mechanistic content supported by the baseline but not established
    as part of the supplemental component.

supplemental_only_components:
    Mechanistic content supported by the supplemental component but
    not established as part of the baseline.

Use concise scientific descriptions.

Do not output operator eligibility, search recommendations, acceptance
decisions, or novelty scores.

If uncertain, fail closed with INSUFFICIENT_FOR_JUDGMENT.

Return only the requested structured mechanism-semantic draft.
""".strip()


class MechanismSemanticPromptAssembler:
    def build(
        self,
        *,
        hypothesis_id: str,
        scientific_task: str,
        supply_geometry: MechanismSupplyGeometry,
        baseline_mechanism_statements: Sequence[
            Mapping[str, Any]
        ],
        task_feature: Mapping[str, Any],
        supplemental_mechanism_nodes: Sequence[
            Mapping[str, Any]
        ],
        scientific_steps: Sequence[
            Mapping[str, Any]
        ],
    ) -> MechanismSemanticPrompt:
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

        if not baseline_mechanism_statements:
            raise ValueError(
                "baseline mechanism evidence is required"
            )

        if not supplemental_mechanism_nodes:
            raise ValueError(
                "supplemental mechanism evidence is required"
            )

        baseline_ids = []

        baseline_payload = []

        for row in baseline_mechanism_statements:
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
                    "baseline mechanism statement "
                    "requires statement_id and text"
                )

            baseline_ids.append(
                statement_id
            )

            baseline_payload.append(
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

        supplemental_ids = []

        supplemental_payload = []

        for row in supplemental_mechanism_nodes:
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
                    "supplemental mechanism node "
                    "requires node_id and node_text"
                )

            supplemental_ids.append(
                node_id
            )

            supplemental_payload.append(
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

        payload = {
            "hypothesis_id":
                hypothesis_id,

            "scientific_task":
                scientific_task,

            "supply_geometry":
                supply_geometry,

            "baseline_grounded_mechanism":
                baseline_payload,

            "supplemental_grounded_component": {
                "task_feature":
                    dict(
                        task_feature
                    ),

                "mechanism_nodes":
                    supplemental_payload,

                "scientific_steps": [
                    dict(row)
                    for row
                    in scientific_steps
                ],
            },

            "epistemic_boundary": {
                "grounded_components_do_not_imply_grounded_relation":
                    True,

                "common_anchor_context_is_not_causal_relation":
                    (
                        supply_geometry
                        == "COMMON_ANCHOR_CONTEXT"
                    ),

                "reviewer_has_operator_authority":
                    False,
            },
        }

        user_prompt = (
            "N11 MECHANISM SEMANTIC REVIEW INPUT\n"
            "===================================\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n\nOUTPUT REQUIREMENTS\n"
            "===================\n"
            "- Return only MechanismSemanticDraft.\n"
            "- Classify semantic mechanism relation only.\n"
            "- Identify shared, baseline-only, and supplemental-only "
            "mechanistic components.\n"
            "- Assess task_relation_grounded only from supplied "
            "scientific relations.\n"
            "- Do not decide operator eligibility.\n"
            "- Do not generate a hypothesis.\n"
        )

        hash_body = {
            "prompt_version":
                MECHANISM_SEMANTIC_PROMPT_VERSION,

            "hypothesis_id":
                hypothesis_id,

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                user_prompt,

            "baseline_support_statement_ids":
                baseline_ids,

            "supplemental_mechanism_node_ids":
                supplemental_ids,

            "supply_geometry":
                supply_geometry,
        }

        return MechanismSemanticPrompt(
            hypothesis_id=
                hypothesis_id,

            prompt_version=
                MECHANISM_SEMANTIC_PROMPT_VERSION,

            prompt_sha256=_sha256(
                _canonical_json(
                    hash_body
                )
            ),

            system_prompt=
                SYSTEM_PROMPT,

            user_prompt=
                user_prompt,

            baseline_support_statement_ids=tuple(
                baseline_ids
            ),

            supplemental_mechanism_node_ids=tuple(
                supplemental_ids
            ),

            supply_geometry=
                supply_geometry,
        )
