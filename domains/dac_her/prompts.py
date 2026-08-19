from __future__ import annotations


PROMPT_VERSION = "dac-her-extraction-v5-semantic-si-assets"


SYSTEM_PROMPT = '\nYou extract a provenance-preserving knowledge\ngraph from scientific literature about dual-atom\ncatalysts for the hydrogen evolution reaction.\n\nRules:\n\n1. Extract only information explicitly supported\n   by the supplied source chunk.\n\n2. Every edge source and target must also appear\n   in the nodes list.\n\n3. Distinguish physical catalysts from\n   computational catalyst models.\n\n4. Distinguish:\n   - experimental observations,\n   - structural characterization,\n   - computational results,\n   - author mechanism interpretations.\n\n5. Represent measurements with their conditions,\n   including electrolyte, overpotential, current\n   density, scan rate, duration, or cycle count\n   whenever those conditions are supplied.\n\n6. Do not extract references, author names,\n   affiliations, publisher notes, licenses,\n   figure filenames, or image descriptions as\n   scientific entities.\n\n7. Do not infer causal mechanisms that are not\n   explicitly stated in the source.\n\n8. Use concise canonical node labels.\n\n9. Preserve the supplied paper_id, chunk_id,\n   and section exactly.\n\n10. Put catalysts, supports, metals, reaction\n    steps, intermediates, and synthesis methods\n    in entities.\n\n11. Put each experimental or characterization\n    setup in experiments. Do not put numerical\n    results in experiment names.\n\n12. Put DFT, adsorption-energy, PDOS, charge,\n    FPMD, and spectral simulations in\n    calculations.\n\n13. Put every catalyst-specific numerical result\n    in a separate measurement object.\n\n14. Never create one shared measurement node such\n    as "HER Tafel slopes" for multiple catalysts.\n    Create one measurement per catalyst and value.\n\n15. Represent experimental conditions as\n    structured Condition objects.\n\n16. Put causal, mechanistic, active-site, and\n    stability explanations in mechanism_claims,\n    not in ordinary entities.\n\n17. Connect measurements to experiments or\n    calculations using HAS_MEASUREMENT.\n\n18. Connect evidence to mechanism claims using\n    SUPPORTS_CLAIM.\n\n19. Use author_interpretation only for claims\n    made by the paper authors.\n\n20. Use medium confidence for mechanism claims\n    that are inferential rather than directly\n    demonstrated.\n\n21. Use the following exact edge directions:\n\n    Catalyst or CatalystModel\n        --EVALUATED_IN-->\n    Experiment\n\n    Catalyst, Support, Material, or CoordinationMotif\n        --CHARACTERIZED_BY-->\n    Experiment\n\n    CatalystModel\n        --MODELED_BY-->\n    Calculation\n\n    Catalyst\n        --SYNTHESIZED_BY-->\n    SynthesisMethod\n\n    SynthesisMethod\n        --USES_PRECURSOR-->\n    Precursor\n\n    Experiment or Calculation\n        --HAS_MEASUREMENT-->\n    Measurement\n\n    Measurement, Experiment, or Calculation\n        --SUPPORTS_CLAIM-->\n    ObservationClaim or MechanismClaim\n\n    ObservationClaim\n        --INTERPRETED_AS-->\n    MechanismClaim\n\n    ObservationClaim or MechanismClaim\n        --APPLIES_TO-->\n    scientific Entity\n\n22. Never reverse any of the edge directions above.\n\n23. A catalyst does not USES_PRECURSOR. The synthesis\n    method uses the precursor. Represent this as:\n\n    Catalyst --SYNTHESIZED_BY--> SynthesisMethod\n    SynthesisMethod --USES_PRECURSOR--> Precursor\n\n24. Put directly supported numerical comparisons,\n    performance conclusions, stability observations,\n    structural observations, and adsorption-energy\n    comparisons in observation_claims.\n\n25. Put causal explanations, formation mechanisms,\n    stability mechanisms, active-site interpretations,\n    reaction pathways, and electronic-structure\n    explanations in mechanism_claims.\n\n26. Do not combine an observation and its proposed\n    cause in one claim.\n\n27. Connect evidence to an ObservationClaim with\n    SUPPORTS_CLAIM. When authors interpret that\n    observation mechanistically, connect:\n\n    ObservationClaim\n        --INTERPRETED_AS-->\n    MechanismClaim\n\n28. Every ObservationClaim and MechanismClaim must\n    have at least one APPLIES_TO edge.\n\n29. Before returning the result, verify all edge\n    directions against rule 21.\n\n30. Before returning the graph, verify:\n    - no isolated nodes exist;\n    - every measurement is linked;\n    - every claim has supporting evidence;\n    - every claim has an application target.\n    \n31. The input contains LEFT_CONTEXT, CORE_TEXT,\n    and RIGHT_CONTEXT.\n\n32. Extract nodes, measurements, claims, and edges\n    only from CORE_TEXT.\n\n33. LEFT_CONTEXT and RIGHT_CONTEXT may be used only\n    to resolve references such as "this catalyst",\n    "these results", "the former", or abbreviations.\n\n34. Do not extract a fact solely because it appears\n    in LEFT_CONTEXT or RIGHT_CONTEXT.\n\n35. Keep the graph concise:\n    - at most 20 entities;\n    - at most 10 experiments;\n    - at most 5 calculations;\n    - at most 25 measurements;\n    - at most 10 observation claims;\n    - at most 8 mechanism claims;\n    - at most 80 edges.\n\n36. Keep descriptions under 160 characters and\n    evidence_text under 200 characters.\n\n37. Prefer a complete valid graph over exhaustive\n    extraction of low-value details.\n\n38. Every Measurement must be the target of at least\n    one HAS_MEASUREMENT edge from an Experiment or\n    Calculation.\n\n39. Every Calculation must participate in at least\n    one edge. A catalyst model should normally connect\n    to a Calculation through MODELED_BY.\n\n40. Every Intermediate must participate in at least\n    one INVOLVES_INTERMEDIATE, ADSORBS, APPLIES_TO,\n    or other semantically valid edge.\n\n41. Never represent a scientific conclusion, activity\n    comparison, stability conclusion, or mechanistic\n    interpretation as a Material entity.\n\n42. IDs beginning with claim_, obs_, oc_, or mech_\n    must appear only in observation_claims or\n    mechanism_claims, never in entities.\n\n43. If a Measurement, Calculation, Intermediate, or\n    claim cannot be connected using information from\n    CORE_TEXT, omit that node rather than returning it\n    as an isolated node.\n\n44. Before returning the graph, verify that every node\n    has at least one incoming or outgoing edge.\n\n45. HAS_MEASUREMENT may target only a Measurement.\n    Never target an ObservationClaim or MechanismClaim.\n\n46. SUPPORTS_CLAIM may originate only from a\n    Measurement, Experiment, or Calculation.\n    Claims must never be the source of SUPPORTS_CLAIM.\n\n47. When an experiment explicitly compares multiple\n    catalysts, create a separate EVALUATED_IN edge from\n    every compared catalyst to that Experiment.\n\n48. Do not create comparator catalysts as nodes unless\n    they are connected to the comparison experiment or\n    another explicitly supported relationship.\n\n49. Preserve document_id, document_role, page_ids, and\n    asset_ids exactly as supplied.\n\n50. Every edge must include at least one EvidencePointer.\n    Its document_id and document_role must match the chunk.\n\n51. An EvidencePointer may use only PAGE_IDS and ASSET_IDS\n    supplied in the prompt. Use page_id=null when no reliable\n    page is available. Use asset_ids=[] for text-only evidence.\n\n52. Figure files, captions, and Marker alt text are evidence\n    provenance, not scientific entities. Never create nodes\n    whose identity is only a filename or figure number.\n\n53. Official captions may support extraction. Marker alt text\n    is potentially noisy and must never be the sole evidence\n    for a claim or numerical measurement.\n\n54. A cached VISION_SUMMARY may be used only together with\n    its listed limitations and the supplied figure/caption.\n    Do not infer unreadable curve values.\n\n55. Supporting-information chunks are valid evidence, but\n    preserve document_role=supporting_information and do not\n    present SI evidence as if it came from the main article.\n\n56. For each edge, choose only the asset IDs that directly\n    support that relation; do not attach every chunk asset to\n    every edge.\n'



SYSTEM_PROMPT += r"""

57. Every Measurement is scalar and subject-specific:
    - exactly one subject_id;
    - exactly one metric_id;
    - exactly one numeric or textual result;
    - exactly one coherent condition set.

58. Add Measurement --MEASURED_FOR--> Entity, and make the edge
    target exactly equal to Measurement.subject_id.

59. When one source sentence reports multiple catalysts, conditions,
    time points, or values, create separate Measurement nodes. Preserve
    their joint comparison using a MeasurementGroup and one
    IN_MEASUREMENT_GROUP edge per member.

60. Never place strings such as "1 ohm in acid; 2 ohms in base" in one
    value_text. Split them into two scalar measurements with explicit
    electrolyte conditions.

61. Use metric IDs and experiment method IDs from VOCABULARY_CONTEXT.
    If no entry fits, use unregistered_<concise_slug>; do not force an
    incorrect registered category.

62. Every Experiment must include experiment_family, method_label, and
    raw_method_name. raw_method_name is null when unnecessary.

63. Physical Catalyst and computational CatalystModel are distinct.
    When the source explicitly states that a model represents a catalyst,
    connect CatalystModel --MODEL_OF--> Catalyst rather than merging them.


64. Use generic metric IDs plus structured parameter conditions. Do not
    create element-specific metric IDs when the quantity is generic. Examples:
    - atomic_fraction + Condition(analyte=W);
    - element_loading + Condition(analyte=Mo);
    - xps_binding_energy + Conditions(analyte=W, orbital=4f7/2).

65. A Material or Support that is explicitly evaluated as a comparator in an
    electrochemical experiment and receives catalyst-specific measurements
    should be typed as Catalyst for that catalytic role. Preserve a separate
    Support node only when the source also uses it as a support substrate.
"""

def build_extraction_prompt(
    *,
    paper_id: str,
    chunk_id: str,
    document_id: str,
    document_role: str,
    section: str,
    page_ids: tuple[int, ...] | list[int],
    asset_ids: tuple[str, ...] | list[str],
    asset_context: str,
    vocabulary_context: str,
    left_context: str,
    core_text: str,
    right_context: str,
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
{list(page_ids)}

ASSET_IDS:
{list(asset_ids)}

ASSET_CONTEXT:
{asset_context or 'No linked assets.'}

VOCABULARY_CONTEXT:
{vocabulary_context or 'No registry context supplied.'}

LEFT_CONTEXT:
{left_context}

CORE_TEXT:
{core_text}

RIGHT_CONTEXT:
{right_context}
""".strip()

    if validation_feedback:
        prompt += f"""

PREVIOUS VALIDATION ERROR:
{validation_feedback}

The previous graph failed validation. Return a complete
new KnowledgeGraph, not a partial patch.

STRICT REPAIR RULES:

1. HAS_MEASUREMENT must be Experiment/Calculation -> Measurement.
2. SUPPORTS_CLAIM must start from Measurement, Experiment, or Calculation.
3. ObservationClaim -> INTERPRETED_AS -> MechanismClaim.
4. Every claim requires APPLIES_TO.
5. Every node must participate in at least one edge.
6. Rebuild the entire edge list, not only the first error.
7. Preserve PAPER_ID, CHUNK_ID, DOCUMENT_ID, DOCUMENT_ROLE,
   PAGE_IDS, and ASSET_IDS exactly.
8. Every edge requires evidence_pointers. Each pointer must use
   the supplied DOCUMENT_ID/ROLE and only supplied PAGE_IDS/ASSET_IDS.
9. Use asset_ids=[] for text-only evidence. Marker alt text alone
   is never sufficient evidence.
10. Re-check scalar measurements: one subject, one result, one condition
    set, one MEASURED_FOR edge, and matching MeasurementGroups.
11. Use only valid registry IDs from VOCABULARY_CONTEXT, or an explicit
    unregistered_<slug> when no method/metric fits.

Do not invent evidence, experiments, measurements, assets, pages,
or relationships unsupported by CORE_TEXT and ASSET_CONTEXT.
""".strip()

    return prompt


SYSTEM_PROMPT += r"""

66. A CatalystModel --MODEL_OF--> Catalyst edge must preserve catalyst
    composition identity. W1Mo1 models map to W1Mo1 catalysts, Mo2 models to
    Mo2 catalysts, and W2 models to W2 catalysts. Never point every comparison
    model to the primary catalyst merely because they appear in one figure.

67. Distinguish oxidation state from coordination number. Statements such as
    "average oxidation state close to 6" use metric_id=oxidation_state, while
    EXAFS coordination-number results use metric_id=coordination_number.

68. Use pcohp_antibonding_state_energy for reported pCOHP antibonding-state
    energies and epr_g_factor for EPR g values.

69. When ASSET_CONTEXT supplies a figure/table asset that directly supports an
    edge and the edge locator names that figure/table, include that asset ID in
    the EvidencePointer. Do not leave asset_ids empty for explicit figure-based
    evidence when a matching supplied asset exists.
"""
