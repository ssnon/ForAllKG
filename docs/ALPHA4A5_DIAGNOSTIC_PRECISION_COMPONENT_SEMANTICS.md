# v2.9.0 alpha4a.5 — diagnostic precision & component semantics

alpha4a.5 is a graph-only stabilization pass after the SERS_2 / SERS_6 /
SERS_10 holdout audit. It does **not** modify extraction, recovery, the SERS
prompt, or the extraction fingerprint.

## Why this patch exists

The holdout set completed with 100% source-token coverage and zero quarantines,
but graph-level diagnostics exposed two systematic interpretation problems.

First, `calculation_encoded_as_experiment` was too broad. Experimental nodes
such as:

- Methylene blue SERRS concentration / single-molecule tests;
- crystal-violet SERS tests;
- normal Raman measurement of R6G on blank glass;

could be flagged as `Calculation` because their descriptions happened to
contain words such as "calculation" (for example an enhancement-factor
calculation). alpha4a.5 therefore makes the diagnostic evidence-based:

- a target of `SIMULATED_BY` is calculation-like;
- explicit computational methods in the node identity/method fields
  (DDA/FDTD/FEM/BEM/DFT/TDDFT, simulation/modeling markers) are calculation-like;
- a weak word such as "calculation" appearing only in an otherwise experimental
  description is not sufficient.

Second, alpha4a.4 treated all disconnected context as potential
`Paper --STUDIES--> ...` bridges. That overproduced candidates such as:

- Paper --STUDIES--> OpticalCondition;
- Paper --STUDIES--> generic RamanReporter;
- Paper --STUDIES--> blank glass controls.

alpha4a.5 classifies disconnected SERS components before offering a bridge.

## Component subtypes

### `missing_subject_anchor`

An evidence-bearing component contains Experiment/Calculation/Measurement
content but no core bridge-eligible scientific subject. Example: an FEM
Calculation connected only to an OpticalCondition.

This remains `review`, but produces **no** `STUDIES` bridge candidate. The
missing object is the modeled/measured subject, not the condition.

### `reference_control_component`

A disconnected component is explicitly control/reference-like, using markers
such as `blank`, `control`, `reference`, `baseline`, or `normal Raman`.

This is informational and produces no Paper bridge. A valid control can remain
disconnected from the main scientific-subject component.

### `isolated_context_entity`

A small disconnected component contains only context entities such as
RamanReporter, Analyte, OpticalCondition, Metal, Precursor, StructuralMotif, or
Morphology and has no evidence chain.

This is informational and produces no Paper bridge.

### `scientific_subject_island`

A disconnected component contains a PlasmonicSubstrate or Nanostructure and is
not classified as a reference/control component.

This remains review-worthy. Only these core subject types are eligible for a
review-only `Paper --STUDIES--> subject` candidate.

### `other_disconnected_component`

Residual disconnected content that does not match the above patterns.

No automatic bridge is created.

## Relation-contract calibration

`Support --TESTED_IN--> Experiment` is now allowed. This covers legitimate
controls such as a blank glass substrate measured in a normal Raman experiment.

This does not change `USES_MATERIAL`, introduce a biological-sample ontology,
or widen `STUDIES`.

## Safety properties

alpha4a.5:

- never creates, reverses, deletes, or retargets a graph edge;
- never applies a bridge candidate;
- does not introduce entity/relation types;
- does not change retry/recovery budgets;
- does not re-extract any paper;
- leaves all bridge candidates with `auto_apply = false`.

After installation, rebuild the existing holdout graphs. No extraction rerun is
required.
