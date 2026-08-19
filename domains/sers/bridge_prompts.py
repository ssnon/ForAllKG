from __future__ import annotations

import json
from typing import Any


SERS_BRIDGE_PROMPT_VERSION = 'sers-au-ag-bridge-v1-alpha4b2b'

SERS_BRIDGE_SYSTEM_PROMPT = r"""
You extract a small, source-grounded Bridge layer from scientific literature
about Au-Ag plasmonic substrates and surface-enhanced Raman scattering (SERS).
The strict evidence graph has already extracted substrates, nanostructures,
experiments, calculations, measurements, observation claims, and mechanism
claims. Do not repeat that graph.

The Bridge layer exists for GraphAgents-style discovery. Prefer a small number
of GENERALIZABLE RELATION PATTERNS and rare source-explicit frontier concepts.
Returning zero concepts is valid and preferred to quota filling.

OUTPUT MODES

A. accepted_pattern
- concept_type must be RelationPattern.
- Provide pattern_subject, pattern_relation, pattern_object, and
  relation_strength.
- Link every accepted pattern to a supplied strict node with EXPRESSES_PATTERN.
- Good reusable examples include:
  * SERS enhancement VARIES_WITH nanogap size
  * plasmon resonance VARIES_WITH shell thickness
  * local-field localization VARIES_WITH nanostructure geometry
  * composition MODULATES interfacial charge transfer
  * protective-shell architecture SUPPRESSES silver oxidation
  * enhancement optimization IMPOSES_TRADEOFF structural stability
- Do not copy paper-specific numeric instances as Bridge concepts.

B. paper_local_frontier
- Use only for a rare, source-explicit concept with discovery value after
  removing sample IDs and numeric values, for example plasmon hybridization,
  hotspot localization, interfacial charge-transfer state, dynamic surface
  segregation, or morphology reconstruction.
- Set all pattern fields/relation_strength/pattern_support_mode to null.
- Set supporting_phrases, comparison_items, and qualifiers to [].
- Set subject/relation/object evidence phrases to null.
- Use a non-pattern anchor relation appropriate to the concept type.

PATTERN SUPPORT

1. explicit_single_span
- One sentence/contiguous span explicitly states the whole relation.
- source_phrase == supporting_phrases[0].
- subject_evidence_phrase, relation_evidence_phrase, and
  object_evidence_phrase must be verbatim substrings of that span.
- comparison_items must be [].

2. derived_multi_span
- Only for a relation derived from at least two explicit table/list/parallel
  items.
- At least two supporting_phrases and comparison_items are required.
- Use only CORRELATES_WITH, VARIES_WITH, or CONTRASTS_WITH.
- Grounding links must use evidence_strength=indirect.
- Never infer a trend from one table row or a bare column heading.

ALLOWED PATTERN RELATIONS
CORRELATES_WITH, VARIES_WITH, COMPETES_WITH, COMPETES_FOR, SELECTS,
CONTRASTS_WITH, MODULATES, MEDIATES, PROMOTES, SUPPRESSES,
SUGGESTS_DESIGN_RULE, IMPOSES_TRADEOFF, IDENTIFIES_FAILURE_MODE.

RELATION SEMANTICS
- X VARIES_WITH Y: X is the observed outcome/property and Y is the changing
  condition/axis. Example: SERS intensity VARIES_WITH nanogap size.
- CONTRASTS_WITH is for explicit peer alternatives, not merely two values on a
  continuous axis.
- MODULATES/MEDIATES/PROMOTES/SUPPRESSES require explicit author
  interpretation and relation_strength=causal_interpretive.
- CORRELATES_WITH/VARIES_WITH/CONTRASTS_WITH/COMPETES_* are non-causal.
- Do not convert association, co-occurrence, or a higher numeric value into
  causation.

SERS-SPECIFIC PRECISION RULES
1. Extract only relations explicit in CORE_TEXT.
2. Every grounding/support phrase must occur verbatim in CORE_TEXT.
3. At most 8 concepts and 16 links; zero is valid.
4. anchor_id must exactly match a supplied STRICT_GRAPH_NODE.
5. Use the supplied metal/architecture/morphology/motif/context signatures only
   to choose compatible anchors. Missing detail is not a contradiction.
6. Do not create PlasmonicSubstrate, Nanostructure, Experiment, Calculation,
   Measurement, ObservationClaim, or MechanismClaim objects in the Bridge layer.
7. Do not emit a bare enhancement factor, AEF, LOD, Raman/SERS intensity,
   LSPR wavelength, Raman peak, nanogap width, shell thickness, particle size,
   Au:Ag ratio, concentration, or other scalar/axis field as a frontier concept.
8. Such observables may participate in an accepted pattern only when the source
   explicitly supports the relation or a valid derived_multi_span comparison
   supports it.
9. Never rank two substrates merely because their EF/LOD/intensity values differ
   under different reporters, excitation wavelengths, concentrations, powers,
   acquisition times, Raman peaks, media, or substrate states.
10. Preserve those conditions as qualifiers only when they materially narrow a
    source-supported relation. Cross-paper quantitative compatibility is handled
    later by deterministic comparison-context logic, not by this extractor.
11. Do not infer electromagnetic enhancement, chemical enhancement, charge
    transfer, hotspot formation, plasmon coupling, or oxidation mechanisms unless
    the authors explicitly state/interpet them.
12. Remove sample IDs, exact metal ratios, numeric values, units, figure/table
    labels mentally. If no reusable scientific content remains, omit it.
13. Analyte, RamanReporter, and OpticalCondition are context roles, not generic
    material/mechanism subjects merely because they occur in a SERS experiment.
14. Do not import DAC-HER concepts such as hydrogen adsorption, Tafel slope,
    proton competition, N-coordination, or d-band tuning unless the actual SERS
    source independently and explicitly discusses such a concept.
15. The Bridge layer is not a hypothesis graph. Plausibility is not evidence.
""".strip()


def build_sers_bridge_prompt(
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

Return a complete corrected BridgeChunkGraph, not a patch. Preserve the SERS
scientific meaning. Remove scalar-only, strict-duplicate, generic,
source-unsupported, or anchor-incompatible candidates. Do not strengthen a
correlation into a causal relation. Keep every required JSON field.
""".rstrip()
    return prompt
