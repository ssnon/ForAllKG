# v2.9.0 alpha4a.2-fix1 — SERS scientific-role calibration

This patch calibrates the graph semantics exposed by the first alpha4a.2 audit.

## Scientific role rules

- `SynthesisMethod` is the protocol used to make, grow, reduce, coat,
  functionalize, or reporter-load a material/nanostructure.
- `Experiment` is a measurement/characterization/performance procedure that
  generates experimental observations or measurements.
- `Calculation` is a computational procedure. DDA, FDTD, FEM, BEM, DFT, TDDFT,
  field simulations, and charge calculations must not be encoded as Experiment.
- `SynthesisMethod --USES_MATERIAL--> Material` represents non-precursor
  synthesis inputs such as reducing agents, stabilizers, and
  structure-directing agents.
- `USES_PRECURSOR` remains reserved for actual precursors/feedstocks.
- `USES_REPORTER` may originate from either Experiment or SynthesisMethod.

## Diagnostics

The graph-semantics report now also emits:

- `node_role_issues.json`
- `node_role_issues.csv`

for SERS-specific collection-role mismatches such as a DDA calculation encoded
as `Experiment` or a synthesis procedure encoded as `Experiment`.

These remain diagnostics rather than destructive auto-repairs.
