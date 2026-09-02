from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismSupplyGeometry,
)


MECHANISM_SEMANTIC_PROMPT_VERSION = (
    "n11-mechanism-semantic-review-prompt-v2-governing-mechanism"
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

GOVERNING-MECHANISM IDENTITY
============================

Classify semantic mechanism identity at the level of the governing
physical or chemical process, not at the level of its downstream
observable, measurement, consequence, geometry, or application.

11. A different downstream consequence does NOT by itself establish
    a distinct mechanism.

    Examples of downstream consequences include:
    - plasmon spectral red shift;
    - local electric-field magnitude;
    - hotspot formation;
    - Raman or SERS intensity;
    - enhancement factor;
    - detection sensitivity.

12. When both supplied components describe the same plasmonic or
    electromagnetic near-field process, differences such as

        coupling -> spectral shift

    versus

        coupling / localized field -> hotspot / SERS enhancement

    must not by themselves be counted as a distinct governing
    mechanism.

13. Near-field coupling, plasmonic coupling, plasmon hybridization,
    electromagnetic-field localization, and plasmonic hotspot
    behavior may describe different levels or consequences of one
    electromagnetic/plasmonic mechanism family.

    Do not call them distinct merely because one component emphasizes
    coupling, another emphasizes local field enhancement, and another
    emphasizes the resulting SERS response.

14. Material identity, particle architecture, geometry, hotspot
    location, polarization, fabrication route, or measurement
    configuration are not separate governing mechanisms merely because
    they differ between the supplied components.

    They count as distinct mechanistic content only when the supplied
    evidence explicitly introduces a different physical or chemical
    pathway.

15. Chemical enhancement, charge transfer, molecular resonance,
    catalytic/electronic-state effects, or another explicitly reported
    non-electromagnetic pathway may constitute a distinct mechanism
    when that pathway is actually present in the supplied supplemental
    evidence.

16. Experimental uncertainty, laser positioning, dose stability,
    fabrication history, or structural formation alone are not
    governing SERS mechanisms.

    If a supplemental component reports only such a factor and does
    not establish a governing physical or chemical mechanism, prefer
    INSUFFICIENT_FOR_JUDGMENT rather than DISTINCT_MECHANISMS.

17. Use PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT only when the
    supplemental evidence contains an additional GOVERNING mechanism,
    not merely an additional consequence, readout, architecture,
    localization pattern, or experimental condition.

TASK-RELATION GROUNDING
=======================

18. task_relation_grounded concerns whether the supplied factor-local
    scientific relation itself connects the task factor/feature to the
    supplemental governing mechanism.

19. Use the supplied grounded_factor_nodes as the authoritative
    upstream resolution of task-factor identity.

    Do not require the exact lexical phrase "interparticle spacing"
    when an upstream-grounded factor node identifies an interparticle
    gap, gap width, nanogap, or distance as the task-related factor.

20. task_relation_grounded=true is allowed when the factor-local
    relation explicitly states, for example:

    - smaller/larger gaps change the mechanism;
    - a distance or gap-size regime changes the mechanism;
    - nanogaps create or enable the mechanism;
    - the mechanism depends on interparticle distance or gap width.

21. task_relation_grounded=false when the task factor is only a
    location, anchor, or contextual object.

    Examples:
    - laser positioning on a nanogap changes the measurement;
    - material identity changes a field located in a nanogap;
    - polarization changes a field at a nanogap;
    - a mechanism happens to be measured in a nanogap structure.

22. Grounded factor identity does not repair relation direction.
    The factor itself must participate in the reported scientific
    relation.

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
