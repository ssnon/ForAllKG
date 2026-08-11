# v2.9.0 alpha4a.2-fix2 — SERS support/component semantics

This patch calibrates the final four relation-contract warnings observed after
`alpha4a.2-fix1`.

## Contract changes

- `Support --PREPARED_BY--> SynthesisMethod` is valid.
- `Support --HAS_COMPONENT--> Material` is valid when the source explicitly
  describes the support as consisting of / being made from that material.
- `Support --USES_MATERIAL--> Material` remains invalid. `USES_MATERIAL` is a
  synthesis-input relation whose source must be `SynthesisMethod`.
- `HAS_COMPONENT` must not target `Analyte`, `RamanReporter`, or
  `OpticalCondition`.
- Reporter loading remains represented as
  `Experiment|SynthesisMethod --USES_REPORTER--> RamanReporter`.

No new relation type is introduced.
