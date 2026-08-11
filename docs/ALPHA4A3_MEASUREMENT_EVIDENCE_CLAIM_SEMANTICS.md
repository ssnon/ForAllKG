# v2.9.0 alpha4a.3 — measurement / evidence / claim semantics

This patch is based on the SERS_1 / SERS_5 / SERS_8 pilot.

The pilot showed that the remaining failures are no longer primarily vocabulary drift or generic recovery plumbing. The repeated pattern is incomplete or misdirected evidence topology:

- scientific subjects or synthesis methods used directly as SUPPORTS_CLAIM evidence sources;
- Measurements emitted without a source-grounded Experiment/Calculation producer;
- ObservationClaims missing SUPPORTS_CLAIM and/or APPLIES_TO;
- MechanismClaims missing direct evidence or an ObservationClaim INTERPRETED_AS link;
- SynthesisMethod IDs referenced by PREPARED_BY / USES_PRECURSOR / USES_MATERIAL without a complete typed SynthesisMethod node.

alpha4a.3 therefore changes extraction semantics rather than adding another generic recovery exception.

Canonical topology:

    scientific subject
          ^
          | MEASURED_FOR / APPLIES_TO
          |
    Measurement <--- HAS_MEASUREMENT --- Experiment / Calculation
          |
          +--- SUPPORTS_CLAIM ---> ObservationClaim
                                      |
                                      +--- APPLIES_TO ---> subject
                                      |
                                      +--- INTERPRETED_AS ---> MechanismClaim
                                                                 |
                                                                 +--- APPLIES_TO ---> subject

A scientific subject is not an evidence producer. A Measurement may be emitted only if the current source provides enough information to identify a source-grounded Experiment or Calculation producer. Do not invent a generic Experiment merely to satisfy validation.

Any node ID used as a PREPARED_BY target or USES_PRECURSOR / USES_MATERIAL source must exist exactly once as a typed SynthesisMethod node.

Pilot vocabulary calibration registers: particle yield, sandwich-hybridization complex formation, log concentration-log SERS-intensity correlation, XRD peak position, Raman peak position, aspect ratio, lattice-plane spacing, crystal lattice constant, and dynamic light scattering.

Post-build graph semantics now emits evidence_topology_issues.json/csv. This is diagnostic only and does not mutate the graph.

Non-goals: no new entity/relation type, no relation-contract widening, no recovery-budget increase, no deterministic scientific-node invention, no Bridge/feasibility changes.
