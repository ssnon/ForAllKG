# alpha4a.5.2 — FEM token-boundary hotfix

SERS_6 `exp_sers_mb` and `exp_sers_cv` are genuine spectroscopy experiments.
Neither has an incoming `SIMULATED_BY` edge. Their descriptions contain the
word `femtomolar`.

alpha4a.5 used substring matching for the computational acronym `FEM`, causing
`fem` in `femtomolar` to trigger `calculation_encoded_as_experiment`.

alpha4a.5.2 changes DDA/FDTD/FEM/BEM/DFT/TDDFT detection to regex token-boundary
matching while preserving long-form computational phrase matching.

No extraction, recovery, ontology, relation, or graph mutation behavior changes.
Only graph rebuilding is required.
