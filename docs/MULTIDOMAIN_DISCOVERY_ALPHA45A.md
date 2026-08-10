# Multi-domain discovery alpha4a.5-A

This patch completes the structural half of the v2.9 multi-domain discovery
refactor. It does not weaken Explorer validation and it does not implement the
LLM-boundary normalizer/reference sanitizer planned for alpha4a.5-B.

## Source of truth

Scientific mechanism/scaffold/shared-entity classification is owned by
`ScientificDomainProfile.discovery` and consumed through
`dac_her.discovery_semantics`.

The following subsystems become domain-aware:

- PathQualityScorer
- DirectConceptHitSelector
- Graph Explorer compiler
- Graph Explorer validator
- candidate-unit continuity/path-quality scoring

## Domain lineage

`domain_profile_id` is carried through:

CLI -> traversal JSON -> GraphExplorerPacket -> Explorer compiler/validator

Explicit traversal/corpus-manifest profile conflicts fail closed. Legacy
traversals without a profile are interpreted as `dac_her`.

## Data roots

General traversal, candidate-unit traversal, and Explorer packet construction
resolve their corpus root through the selected extraction adapter, so HER uses
`data_dac` and SERS uses `data_sers` unless `--data-root` is supplied.

## Deferred to alpha4a.5-B

- ExplorerDraft deterministic normalization
- strong-causal-language normalization policy
- semantic-critic reference sanitizer/audit
- Explorer prompt reinforcement
