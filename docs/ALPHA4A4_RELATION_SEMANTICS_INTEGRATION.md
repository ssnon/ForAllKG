# v2.9.0 alpha4a.4 — relation semantics & graph integration

alpha4a.4 follows the SERS_1 / SERS_5 / SERS_8 graph audit after alpha4a.3.

The extraction/recovery layer is now sufficiently stable. The remaining graph
warnings fall into distinct semantic classes rather than one generic validation
problem:

A. definite direction/role errors:
   - Experiment --TESTED_IN--> subject instead of subject --TESTED_IN--> Experiment;
   - Calculation --SIMULATED_BY--> subject instead of subject --SIMULATED_BY--> Calculation;
   - reporter --TESTED_IN--> Experiment instead of Experiment --USES_REPORTER--> reporter;
   - Experiment --CHARACTERIZED_IN--> Paper;
   - StructuralMotif --HAS_COMPONENT--> Metal;
   - SynthesisMethod --PROPOSES_CLAIM--> claim;
   - abstract Metal --PREPARED_BY--> method.

B. legitimate scientific content for which the current relation contract is
   too narrow or lacks a dedicated relation:
   - Calculation --USES_MATERIAL--> water/model medium;
   - SynthesisMethod --USES_MATERIAL--> Nanostructure/Metal feedstock.

C. ontology typing gaps:
   - biological samples such as cells represented as Material but connected as
     if they were Analyte.

alpha4a.4 does not collapse these categories. It records them separately.

## Safe contract calibration

Two relations are widened because the relation meaning itself is already
appropriate:

- Calculation --USES_REPORTER--> RamanReporter is allowed when a reporter
  explicitly parameterizes a calculation, such as an EF calculation.
- SynthesisMethod --USES_OPTICAL_CONDITION--> OpticalCondition is allowed when
  illumination is part of synthesis/growth.

`USES_MATERIAL` remains synthesis-input-only and its target remains Material.
No Calculation source, Nanostructure target, or Metal target is admitted merely
to silence diagnostics.

## Relation triage

`graph_semantics` now emits:

- relation_contract_triage.json/csv
- relation_direction_issues.json/csv

Each invalid edge is grouped once even if both source and target violate a
contract. Categories include:

- likely_reversed_relation
- wrong_relation_for_role
- wrong_direction_or_scope
- owner_attachment_required
- paper_claim_scope_required
- scope_typing_mismatch
- ontology_relation_gap
- ontology_typing_gap
- unclassified_contract_issue

All suggestions are review-only. No edge is reversed, replaced, deleted, or
created automatically.

## Integration diagnostics

Disconnected SERS components are no longer treated as harmless merely because
they contain a primary scientific subject. alpha4a.4 emits:

- integration_components.json/csv
- component_bridge_candidates.json/csv

A disconnected component containing a scientific subject or evidence chain is
flagged for review. If a Paper node exists, up to three candidate
`Paper --STUDIES--> subject` bridges are listed for human review. They are never
applied automatically.

This is designed for the SERS_1 pattern where internally coherent dimer,
SiO2@Au, or reporter/experiment islands may be scientifically valid but lack a
paper-level bridge after cross-chunk merging.

## Non-goals

- no recovery-budget change;
- no automatic edge repair;
- no automatic component bridging;
- no new entity type;
- no new relation type;
- no BiologicalSample/Target ontology yet;
- no general widening of USES_MATERIAL;
- no Bridge hypothesis-generation stage yet.
