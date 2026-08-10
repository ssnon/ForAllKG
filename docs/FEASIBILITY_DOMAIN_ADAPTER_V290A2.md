# Feasibility Domain Adapter v2.9.0-alpha2

## Purpose

Alpha1 extracted graph/discovery/external-novelty scientific semantics into
`ScientificDomainProfile`. Alpha2 creates the corresponding boundary for the
feasibility stack.

This is intentionally a compatibility refactor, not a SERS implementation.

## Architecture

```text
full E2E / run_feasibility_e2e
              |
              v
      ScientificDomainProfile
              |
    feasibility_adapter_id
              |
              v
   FeasibilityDomainAdapter
              |
      +-------+-------+
      |               |
 DAC-HER adapter   future SERS adapter
      |
      +-- scope compiler
      +-- validation specification
      +-- physics runtime
      +-- experimental runtime
```

## Safety invariant

A scientific domain profile without a registered feasibility adapter fails
closed. A profile also cannot point at an adapter owned by another domain.

This prevents a future `sers_au_ag` profile from accidentally being evaluated by
HER-specific rules such as hydrogen adsorption, water dissociation, or HER
electrochemical testing.

## What remains HER-specific after alpha2

The existing v0.2 contracts and implementations are deliberately preserved:

- `scope_contracts.py` still has DAC/SAC catalyst classes and HER reaction class.
- `physics_contracts.py` still enumerates HER-oriented check types.
- `scope_compiler.py`, `validation_specification.py`, `physics_rules.py`,
  `physics_runtime.py`, `experimental_rules.py`, and `experimental_runtime.py`
  still implement the validated DAC-HER feasibility semantics.

They are now *owned through* `DacHerFeasibilityAdapter` rather than assumed to be
universal.

## Next step

Alpha3 should generalize the feasibility contracts into domain-neutral
`system_class`, `process/domain`, and extensible check identifiers, while
providing v0.2 compatibility aliases for existing HER artifacts. Only after that
should `SersAuAgFeasibilityAdapter` be added.
