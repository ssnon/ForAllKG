# Domain-neutral feasibility contracts v2.9.0-alpha3

## Objective

Make the feasibility artifact contracts capable of representing a second
scientific domain without breaking existing DAC-HER v0.2 artifacts.

Alpha3 is additive and compatibility-first. It does **not** implement SERS
scientific rules yet.

## ScientificScope

Legacy fields remain:

- `catalyst_class`
- `reaction`
- `metals`
- `coordination_variables`

Generic mirrors are added:

- `system_class`
- `scientific_domain`
- `process`
- `components`
- `structural_variables`

Old HER inputs automatically populate safe generic mirrors. Generic inputs only
populate legacy fields when the mapping is semantically unambiguous.

## ValidationSpecification

`ValidationStrategy` is extensible and the contract adds:

- `required_scientific_checks`
- `not_applicable_scientific_checks`
- `required_experimental_capabilities`

The old `required_physics_checks` names remain synchronized for v0.2 runtimes.

## Physics

`PhysicsCheckType` is now an extensible string. Therefore checks such as
`local_field_enhancement`, `lspr_alignment`, or `nanogap_stability` can exist in
contracts without pretending they are HER checks.

Requests/results/reports also carry `scientific_domain` and optional
`backend_class` metadata.

## Experimental contracts

Experimental check identifiers and requirement categories are extensible.
`required_performance_tests` is introduced as the generic performance-test
field. Legacy HER `required_electrochemical_tests` is retained and is safely
mirrored into the generic field when old reports are loaded.

The reverse mapping is intentionally not automatic: a SERS Raman measurement
must never be mislabeled as electrochemistry.

## Candidate decisions

Decision cards now carry `system_class`, `scientific_domain`, `process`, and
`required_performance_tests` while retaining the legacy HER fields.

## Schema versions

Each expanded artifact accepts both `v02` and `v03`. Existing HER constructors
continue to default to `v02`, preserving regression behavior. A future SERS
adapter can explicitly emit `v03` artifacts.

## Safety boundary

Alpha3 only generalizes *representation*. Domain-specific interpretation remains
inside the adapter established by alpha2. The DAC-HER adapter still owns HER
scope, hydrogen adsorption/water dissociation checks, and electrochemical test
planning. A future SERS adapter must supply its own scientific rules.
