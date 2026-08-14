# alpha4c.1 — Generic TrendEvidence Contract

## Scope

alpha4c.1 adds the generic evidence contract for paper-local scientific trends.
It does **not** yet add SERS-specific trend extraction rules and does not modify
HypothesisContext, CrossContextTrendAssessment, or the frozen SERS comparison
semantics.

```text
canonical scientific graph
        |
        v
TrendEvidence                 <- alpha4c.1 contract
        |
        v
CrossContextTrendAssessment   <- later alpha4c.3
        |
        v
HypothesisContext             <- later alpha4c.5
```

Comparison still answers whether absolute measurements may be compared.
TrendEvidence represents a directional relationship supported inside one
paper/context.

## Contract semantics

`trend_evidence_contract_v1_alpha4c1`

TrendEvidence separates independent variable, dependent observable, direction,
shape, evidence basis, causal status, context anchors, and provenance.

### Directions

- `positive`
- `negative`
- `non_monotonic`
- `unchanged`
- `unspecified`

Direction is defined with respect to an **increase of the independent
variable**. “Smaller nanogap gives higher signal” therefore becomes a negative
relation between `nanogap_size` and the dependent signal.

### Shapes

- `monotonic`
- `saturating`
- `single_optimum`
- `threshold`
- `u_shaped`
- `inverted_u`
- `unspecified`

Direction and shape are separate. Increasing shell thickness followed by a
plateau may be `direction=positive`, `shape=saturating`.

### Evidence bases

- `controlled_numeric_series`
- `controlled_numeric_pair`
- `reported_directional_claim`
- `reported_correlation`

Numeric and claim lanes remain explicit. Claim evidence cannot carry numeric
series points, and controlled numeric evidence cannot establish causation.

## Fail-closed invariants

1. Numeric trends are paper-local.
2. Numeric trends require explicit MeasurementGroup or Experiment lineage.
3. Numeric pair requires exactly two points; numeric series requires at least three.
4. Controlled numeric points require numeric x/y values and consistent unit representations.
5. Reported claims require Claim-node and source-text provenance.
6. Reported correlation cannot be upgraded to causation.
7. Source nodes must be grounded and of supported evidence types.
8. Sidecar references, when used, must resolve against supplied sidecar rows.
9. Cross-paper absolute values are never combined by the generic builder.
10. KG adjacency alone cannot instantiate TrendEvidence without an explicit evidence basis.

This phase intentionally does not perform majority voting, cross-context
aggregation, causal synthesis, or universalization.

## Domain adapter hook

`ScientificDomainProfile` gains:

```python
trend_adapter_id: str | None = None
```

alpha4c.1 deliberately leaves the SERS profile at `None`. The generic registry
therefore fails closed for SERS until alpha4c.2 registers and activates a
SERS-specific adapter.

## Generic builder

A generic builder is installed:

```bash
python -m scripts.build_trend_evidence ...
```

It requires an activated domain Trend adapter. Running it against SERS
immediately after alpha4c.1 is expected to fail with
`TrendAdapterUnavailableError`.

alpha4c.2 should add the SERS adapter and wire MeasurementResultIdentity /
MethodContext sidecars into `TrendEvidenceSource` without changing this
contract.
