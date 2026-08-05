from __future__ import annotations

import json
from typing import Any


BRIDGE_PROMPT_VERSION = "dac-her-bridge-v2.3-calibration"


BRIDGE_SYSTEM_PROMPT = r"""
You extract a small, source-grounded bridge layer from scientific literature
about dual-atom catalysts and hydrogen evolution. The strict evidence graph has
already extracted catalysts, models, experiments, calculations, measurements,
and claims. Do not repeat that graph.

The bridge layer exists for GraphAgents-style discovery. Its main content is a
small number of GENERALIZABLE RELATION PATTERNS. A rare, mechanistically useful
concept may be retained separately as a paper-local frontier concept.

OUTPUT MODES

A. accepted_pattern
- concept_type must be RelationPattern.
- Provide pattern_subject, pattern_relation, pattern_object, and
  relation_strength.
- Link every accepted pattern to a strict node with EXPRESSES_PATTERN.
- Use a concise label such as "coordination-geometry–binding-strength
  correlation" or "adsorption-regime-dependent site selection".

B. paper_local_frontier
- Use only for a rare, specific, source-explicit concept that may enable future
  cross-domain discovery, for example local electric field, proton relay,
  interfacial water-network reorganization, dynamic reconstruction, or
  strain-induced orbital shift.
- It must remain scientifically meaningful after removing catalyst names and
  numeric values.
- Set every pattern field, relation_strength, and pattern_support_mode to null.
- Set supporting_phrases, comparison_items, and qualifiers to [].
- Set subject_evidence_phrase, relation_evidence_phrase, and
  object_evidence_phrase to null.
- Use a non-pattern anchor relation appropriate to the concept type.

PATTERN SUPPORT MODES

1. explicit_single_span
- Use when one sentence or contiguous source span explicitly states the entire
  subject–relation–object pattern.
- source_phrase and supporting_phrases[0] must be the same complete verbatim
  source span.
- subject_evidence_phrase, relation_evidence_phrase, and
  object_evidence_phrase must each be verbatim substrings of that span.
- The relation evidence must actually express the selected relation. A noun
  phrase such as "the most stable adsorption site" does not by itself support
  a VARIES_WITH relation.
- comparison_items must be [].

2. derived_multi_span
- Use only when a reusable relation is derived by comparing at least two
  explicit rows/items in a table, list, or parallel statements.
- Provide at least two supporting_phrases and at least two comparison_items.
- Each comparison item records one explicit subject_value, object_value, and
  verbatim source_phrase. subject_value must instantiate pattern_subject and
  object_value must instantiate pattern_object.
- Set subject_evidence_phrase, relation_evidence_phrase, and
  object_evidence_phrase to null because the relation is derived across items.
- Use only CORRELATES_WITH, VARIES_WITH, or CONTRASTS_WITH.
- All grounding links for this pattern must use evidence_strength=indirect.
- Do not use this mode to invent a trend from one row or one generic heading.

ALLOWED PATTERN RELATIONS
- CORRELATES_WITH
- VARIES_WITH
- COMPETES_WITH
- COMPETES_FOR
- SELECTS
- CONTRASTS_WITH
- MODULATES
- MEDIATES
- PROMOTES
- SUPPRESSES
- SUGGESTS_DESIGN_RULE
- IMPOSES_TRADEOFF
- IDENTIFIES_FAILURE_MODE

RELATION ARGUMENT SEMANTICS

- X VARIES_WITH Y means X is the observed outcome/property that changes as Y
  changes. Correct: "preferred adsorption site VARIES_WITH metal identity".
  Incorrect: "metal identity VARIES_WITH preferred adsorption site".

- X SELECTS Y means X is the selecting condition/regime and Y is the selected
  site, state, route, or class.

- X COMPETES_WITH Y means X and Y are peer competitors. Add qualifier
  competition_target=<what they compete for>.
  Example:
    pattern_subject = "graphene bond site"
    pattern_relation = "COMPETES_WITH"
    pattern_object = "nitrogen-vacancy site"
    qualifier competition_target = "single-atom adsorption"

- X COMPETES_FOR Y means X is a collective competitor class/set and Y is the
  contested target/resource/process. Add qualifier competitor_members with at
  least two members separated by semicolons.
  Example:
    pattern_subject = "support adsorption-site classes"
    pattern_relation = "COMPETES_FOR"
    pattern_object = "single-atom adsorption"
    qualifier competitor_members = "graphene bond site; nitrogen-vacancy site"

- Never use COMPETES_WITH to connect a set of competitors directly to the
  process/resource they compete for.

CORE RULES

1. Extract only information explicitly supported by CORE_TEXT.
2. Every source_phrase and every supporting/comparison phrase must occur
   verbatim in CORE_TEXT.
3. Returning zero concepts is valid. Prefer precision over quota filling.
4. Create at most 8 concepts and 16 links.
5. Every concept must connect to at least one supplied STRICT_GRAPH_NODE.
6. anchor_id must exactly match one supplied strict node ID.
7. Do not create catalysts, models, experiments, calculations, measurements,
   claims, or numerical results.
8. Do not extract a table column, axis title, scalar metric, geometric variable,
   best-site entry, or parameter by itself. Examples to omit include M-N
   distance, metal height, bond angle, adsorption energy, Tafel slope, and
   "most stable site" when they are merely fields or instance results.
9. A metric may participate in an accepted pattern only when the evidence
   explicitly states the relation or multiple explicit items support a clearly
   labeled derived_multi_span comparison.
10. Do not repeat information already fully represented by strict Measurement
    nodes.
11. Prefer correlation, competition, selectivity, contrast, mediation,
    transition, trade-off, design-rule, and failure-mode patterns.
12. Remove specific metal names, catalyst IDs, numbers, units, and figure/table
    labels mentally. If the remaining statement has no reusable scientific
    meaning, omit it.
13. Do not infer causal relations. MODULATES, MEDIATES, PROMOTES, and SUPPRESSES
    require explicit author interpretation in CORE_TEXT and
    relation_strength=causal_interpretive.
14. CORRELATES_WITH, VARIES_WITH, COMPETES_WITH, COMPETES_FOR, and
    CONTRASTS_WITH must not be labeled causal_interpretive.
15. Preserve evidence_scope:
    - paper_result for results directly reported by this paper;
    - author_interpretation for the authors' mechanistic interpretation;
    - background for contextual statements in introduction/background text.
16. Use structured qualifiers only to preserve scientifically important context
    such as electrolyte, support family, defect class, adsorption regime,
    reaction step, competition_target, or competitor_members.
17. Use the chemistry-aware fields in STRICT_GRAPH_NODES to choose an anchor
    with compatible metal composition, nuclearity, support, and model/physical
    context. Do not anchor N-coordination information to an explicitly pure
    graphene model.
18. Every link requires at least one EvidencePointer using only supplied pages
    and assets. Use asset_ids=[] for text-only evidence.
19. Figure labels, filenames, authors, references, affiliations, and publisher
    metadata are not bridge concepts.
20. The bridge layer is not a hypothesis graph. Never add a relation merely
    because it seems plausible.
""".strip()


def build_bridge_prompt(
    *,
    paper_id: str,
    chunk_id: str,
    document_id: str,
    document_role: str,
    section: str,
    page_ids: list[int],
    asset_ids: list[str],
    strict_nodes: list[dict[str, Any]],
    core_text: str,
    validation_feedback: str | None = None,
) -> str:
    prompt = f"""
PAPER_ID:
{paper_id}

CHUNK_ID:
{chunk_id}

DOCUMENT_ID:
{document_id}

DOCUMENT_ROLE:
{document_role}

SECTION:
{section}

PAGE_IDS:
{page_ids}

ASSET_IDS:
{asset_ids}

STRICT_GRAPH_NODES:
{json.dumps(strict_nodes, ensure_ascii=False, indent=2)}

CORE_TEXT:
{core_text}
""".strip()

    if validation_feedback:
        prompt += f"""

PREVIOUS VALIDATION ERROR:
{validation_feedback}

Return a complete corrected BridgeChunkGraph, not a patch. Remove metric-only,
table-field, strict-duplicate, generic, unsupported, or anchor-incompatible
candidates. Ensure every accepted relation has auditable grounding:
- explicit_single_span: one complete verbatim span plus verbatim subject,
  relation, and object evidence phrases;
- derived_multi_span: at least two explicit comparison items and indirect
  evidence strength.
Use COMPETES_WITH only between peer competitors and COMPETES_FOR only between a
collective competitor class and its contested target. Keep every required JSON
field. For paper_local_frontier, use null/empty values exactly as instructed.
""".rstrip()

    return prompt


def build_bridge_repair_prompt(
    *,
    original_rejections: list[
        dict[str, Any]
    ],
    strict_nodes: list[
        dict[str, Any]
    ],
    core_text: str,
    source_metadata: dict[
        str,
        Any,
    ],
) -> str:
    return f"""
Repair only the rejected Bridge candidates
listed below.

Do not regenerate accepted candidates.
Do not introduce new scientific relations.
Correct only evidence phrase alignment or
relation-cue mismatch.

SOURCE METADATA:
{json.dumps(source_metadata, ensure_ascii=False)}

STRICT NODE CATALOG:
{json.dumps(strict_nodes, ensure_ascii=False)}

REJECTED CANDIDATES:
{json.dumps(original_rejections, ensure_ascii=False)}

CORE_TEXT:
{core_text}
"""