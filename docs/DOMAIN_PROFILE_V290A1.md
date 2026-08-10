# Scientific Domain Profile v2.9.0-alpha1

This refactor separates scientific-domain semantics from reusable graph,
discovery, and external-novelty infrastructure while preserving DAC-HER as the
default behavior.

## Extracted in alpha1

- entity-resolution node/alias/text semantics
- candidate-unit discovery semantic markers and context node types
- external prior-art domain/scope matching semantics
- CLI profile selection for candidate-unit traversal and external novelty

## Intentionally not generalized yet

The feasibility stack remains a dedicated DAC-HER adapter in alpha1:

- `scope_compiler.py`
- `validation_specification.py`
- `physics_rules.py` / `physics_runtime.py`
- `experimental_rules.py` / `experimental_runtime.py`

These files contain deeply coupled HER concepts such as DAC/SAC scope, hydrogen
adsorption, water dissociation, electrochemical performance, and pair stability.
They should be extracted behind a `FeasibilityDomainAdapter` in alpha2 rather
than flattened into a single giant configuration object.

## Compatibility

Existing commands still default to `dac_her`. The new explicit form is:

```bash
python -m scripts.run_candidate_unit_traversal --domain-profile dac_her ...
python -m scripts.run_external_novelty --domain-profile dac_her ...
```

## Why SERS is not registered yet

A built-in `sers_au_ag` profile would make the CLI look end-to-end ready while
SERS extraction schema and feasibility adapters are not implemented. The alpha1
test suite instead creates a synthetic SERS-like novelty profile to prove that
the abstraction itself is no longer HER-specific.
