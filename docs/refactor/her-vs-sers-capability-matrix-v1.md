# HER vs SERS Capability Matrix v1

## Scope

This checkpoint compares the current DAC-HER and SERS implementations inside
`dac_her/` to distinguish:

- shared domain-agnostic infrastructure
- mature HER-only capabilities
- newer SERS-only capabilities
- SERS implementations that are likely the newer common-model direction rather
  than permanent domain-specific special cases

It does not change runtime behavior, imports, or protected scientific state.

Baseline HEAD during characterization:

```text
fa5bd2b65e74f1d91181e5dd19110d377651847b
```

## High-level conclusion

SERS is not just a narrow special case layered on top of the old HER pipeline.

Instead, the repository appears to have evolved as follows:

1. HER established the original runtime and several foundational contracts.
2. Shared adapter boundaries were introduced.
3. SERS, being implemented later, filled more of those newer adapter surfaces.

Therefore the right generalization strategy is:

- do not treat HER as the sole canonical model
- do not assume SERS-only means permanently domain-specific
- evaluate many SERS implementations as candidate next-generation core/domain
  patterns

In practice, HER currently owns more of the older feasibility stack, while
SERS currently owns more of the newer comparison/reproducibility/metric-
definition/trend stack.

## Capability matrix

| Capability | Shared interface exists | HER implementation | SERS implementation | Current interpretation |
|---|---|---|---|---|
| Domain profile | Yes | [dac_her/domains/dac_her.py](../../dac_her/domains/dac_her.py) | [dac_her/domains/sers_au_ag.py](../../dac_her/domains/sers_au_ag.py) | Symmetric domain declarations |
| Extraction adapter | Yes | [dac_her/domains/dac_her_extraction.py](../../dac_her/domains/dac_her_extraction.py) | [dac_her/domains/sers_au_ag_extraction.py](../../dac_her/domains/sers_au_ag_extraction.py) | Symmetric; good core/domain split already |
| Graph adapter | Yes | [dac_her/domains/dac_her_graph.py](../../dac_her/domains/dac_her_graph.py) | [dac_her/domains/sers_au_ag_graph.py](../../dac_her/domains/sers_au_ag_graph.py) | Symmetric adapter surface; domain semantics differ |
| Bridge adapter | Yes | [dac_her/domains/dac_her_bridge.py](../../dac_her/domains/dac_her_bridge.py) | [dac_her/domains/sers_au_ag_bridge.py](../../dac_her/domains/sers_au_ag_bridge.py) | Symmetric adapter surface; strong candidate for core + domain packages |
| Feasibility adapter | Yes | [dac_her/domains/dac_her_feasibility.py](../../domains/dac_her/feasibility.py) | None registered | HER owns older validated feasibility stack |
| Comparison adapter | Yes | None registered | [dac_her/domains/sers_au_ag_comparison.py](../../dac_her/domains/sers_au_ag_comparison.py) | SERS owns newer comparison layer; likely future general pattern |
| Reproducibility adapter | Yes | None registered | [dac_her/domains/sers_au_ag_reproducibility.py](../../dac_her/domains/sers_au_ag_reproducibility.py) | SERS owns newer reproducibility layer; likely generalizable |
| Metric-definition adapter | Yes | None registered | [dac_her/domains/sers_au_ag_metric_definition.py](../../dac_her/domains/sers_au_ag_metric_definition.py) | SERS owns newer metric-definition layer; likely generalizable |
| Trend adapter | Yes | None registered | [dac_her/domains/sers_au_ag_trend_alpha4c2121.py](../../dac_her/domains/sers_au_ag_trend_alpha4c2121.py) via [dac_her/domains/trend_registry.py](../../dac_her/domains/trend_registry.py) | SERS owns newer trend stack; adapter surface already generalized |
| Trend precision adapter | Yes | None registered | [dac_her/domains/sers_au_ag_trend_precision_alpha4c21211.py](../../dac_her/domains/sers_au_ag_trend_precision_alpha4c21211.py) | SERS owns precision refinement of the trend stack |
| Cross-context trend adapter | Yes | None registered | [dac_her/domains/sers_au_ag_cross_context_trend.py](../../dac_her/domains/sers_au_ag_cross_context_trend.py) | SERS owns newer cross-context assessment layer |
| Fresh-C / reserve / holdout campaign logic | No domain-neutral surface yet | No comparable HER campaign layer in current package layout | `fresh_c_*`, `alpha4c5*`, `sers_fresh_c_*` | Campaign-specific; not a first extraction target |

## Registry evidence

The current registries make the asymmetry explicit.

Shared and populated by both HER and SERS:

- [dac_her/domains/registry.py](../../dac_her/domains/registry.py)
- [dac_her/domains/extraction_registry.py](../../dac_her/domains/extraction_registry.py)
- [dac_her/domains/graph_registry.py](../../domains/graph_registry.py)
- [dac_her/domains/bridge_registry.py](../../dac_her/domains/bridge_registry.py)

HER-populated, SERS absent:

- [dac_her/domains/feasibility_registry.py](../../domains/feasibility_registry.py)

SERS-populated, HER absent:

- [dac_her/domains/comparison_registry.py](../../dac_her/domains/comparison_registry.py)
- [dac_her/domains/reproducibility_registry.py](../../dac_her/domains/reproducibility_registry.py)
- [dac_her/domains/metric_definition_registry.py](../../dac_her/domains/metric_definition_registry.py)
- [dac_her/domains/trend_registry.py](../../dac_her/domains/trend_registry.py)
- [dac_her/domains/trend_precision_registry.py](../../dac_her/domains/trend_precision_registry.py)
- [dac_her/domains/cross_context_trend_registry.py](../../dac_her/domains/cross_context_trend_registry.py)

Interpretation:

- adapter interfaces are already generalized
- implementation coverage is staggered by historical development order
- missing HER adapters should not be read as proof that those capabilities are
  intrinsically SERS-only

## What looks truly shared already

The following files are strong core candidates because they define adapter or
contract boundaries without embedding one domain's scientific semantics:

- [dac_her/domain_profile.py](../../dac_her/domain_profile.py)
- [dac_her/extraction_domain.py](../../dac_her/extraction_domain.py)
- [dac_her/bridge_domain.py](../../dac_her/bridge_domain.py)
- [dac_her/graph_domain.py](../../dac_her/graph_domain.py)
- [dac_her/trend_domain.py](../../dac_her/trend_domain.py)
- [dac_her/feasibility_domain.py](../../dac_her/feasibility_domain.py)
- [dac_her/comparison_domain.py](../../dac_her/comparison_domain.py)
- [dac_her/reproducibility_domain.py](../../dac_her/reproducibility_domain.py)
- [dac_her/metric_definition_domain.py](../../dac_her/metric_definition_domain.py)
- [dac_her/cross_context_trend.py](../../dac_her/cross_context_trend.py)

These are among the safest first-step extraction candidates for a future
`pipeline_core` package.

## What looks SERS-only but is probably “new common model”

The following SERS layers should be evaluated as generalizable infrastructure
rather than dismissed as one-off domain code:

- comparison contexts and compatibility assessment
- reproducibility evidence extraction
- metric-definition contexts
- trend evidence and trend precision
- cross-context trend assessment

Rationale:

- each already sits behind a generic adapter contract
- each is registered via a domain registry
- each uses domain-profile IDs rather than hard-coding SERS as a global mode

This suggests the correct refactor is:

```text
extract common contract/runtime boundary
    ->
keep SERS implementation as first concrete provider
    ->
later add HER implementation only where scientifically meaningful
```

not:

```text
strip SERS features out as permanent special cases
```

## What still looks genuinely campaign-specific

The following groups currently look too tied to frozen evaluation history to be
good first-step extraction targets:

- `fresh_c_*`
- `alpha4c5*`
- `alpha4c4*`
- `sers_fresh_c_*`

These should be treated as campaign/runtime bindings layered on top of the
shared domain/runtime stack, not as early core-refactor targets.

## Refactor implication

The repository should likely be re-centered around:

1. domain-agnostic contracts and adapter interfaces
2. shared runtimes for extraction / bridge / corpus / hypothesis / novelty
3. domain packages for HER and SERS
4. campaign packages for SERS Fresh-C and similar frozen evaluation flows

The safest near-term strategy is:

1. extract the pure interface/contract layers first
2. keep SERS as the first complete implementation of the newer adapter stack
3. treat HER as a legacy-but-important domain that may need backfilling onto
   the newer shared surfaces
4. defer campaign-bound SERS Fresh-C logic until after the core/domain split is
   stable

## Recommended next checkpoint

The next useful checkpoint is:

```text
define the first extraction slice for shared interface/contract modules
```

Candidate first slice:

- `domain_profile.py`
- `extraction_domain.py`
- `bridge_domain.py`
- `graph_domain.py`
- `comparison_domain.py`
- `reproducibility_domain.py`
- `metric_definition_domain.py`
- `trend_domain.py`
- `feasibility_domain.py`
- `evaluation_runtime/artifacts.py`

Reason:

- deterministic
- adapter-oriented
- minimal campaign coupling
- strong foundation for later domain-package separation
