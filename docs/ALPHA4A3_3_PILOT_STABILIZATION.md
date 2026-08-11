# alpha4a.3.3 — pilot stabilization

This hotfix closes two issues revealed by the alpha4a.3 three-paper rerun.

## 1. DLS family enum mismatch

alpha4a.3 registered Dynamic light scattering with:

    family: particle_characterization

but the current `Experiment.experiment_family` schema accepts only:

    electrochemistry
    microscopy
    spectroscopy
    diffraction
    composition_analysis
    surface_area_analysis
    thermal_analysis
    synthesis
    stability_test
    other

Therefore one SERS_8 leaf was quarantined during vocabulary normalization even
though the scientific extraction was otherwise valid.

alpha4a.3.3 changes DLS to `family: other`. This is deliberately compatibility
first: it avoids widening the shared experiment-family ontology merely for one
characterization technique.

## 2. Measurement scalar XOR generation guard

SERS_1 produced a local draft-schema failure because one Measurement did not
satisfy the invariant that exactly one of `value_numeric` or `value_text` is
populated.

The SERS generation prompt now states the serialized contract explicitly:

- numeric result -> `value_numeric` non-null, `value_text` null;
- qualitative/text result -> `value_text` non-null, `value_numeric` null;
- source wording remains in `source_expression`;
- never populate both and never leave both null.

This does not relax the schema.

## 3. Remaining pilot vocabulary

The following general characterization terms observed after alpha4a.3 are
registered:

- Energy-dispersive X-ray spectroscopy
- Absorption-band wavelength
- Absorption shoulder wavelength
- Concentration-SERS intensity correlation R²

No entity/relation ontology or recovery budget is changed.
