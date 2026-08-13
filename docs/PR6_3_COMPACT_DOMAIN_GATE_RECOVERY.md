# PR6.3 — Adapter-owned compact domain-gate recovery

## Goal

Close the remaining Broad compact-schema inconsistency:

```text
compact initial generation
    -> domain-gate violation
    -> full KnowledgeGraphDraft recovery
```

The recovery response model is now an explicit `ExtractionDomainAdapter`
capability. Generic strict recovery remains domain-neutral.

## Experimental flag

```bash
--broad-compact-domain-recovery
```

It requires:

```bash
--broad-compact-schema
```

For the PR6.3 A/B keep `--broad-prune-metric-vocabulary` OFF.

Control A:

```text
compact initial + full domain-gate recovery
```

Treatment B:

```text
compact initial + compact domain-gate recovery
```

## Adapter capability

`ExtractionDomainAdapter` has two separate optional compact capabilities:

```text
compact_generation_response_model
compact_domain_gate_recovery_response_model
```

`catalysis_mechanism` supplies `BroadMechanismGraphDraft` for both.
DAC-HER and SERS do not acquire either compact capability implicitly.

## Expected effect

Previous telemetry measured approximately:

```text
KnowledgeGraphDraft schema       3071 tokens
BroadMechanismGraphDraft schema  2250 tokens
difference                        821 tokens/recovery call
```

Provider usage remains authoritative.

The scientific guardrail is more important than the raw token reduction:
Broad compact recovery should not reintroduce measurement structures forbidden
by the Broad abstract contract.

Measure:
- provider input per `domain_gate_recovery` call;
- recovery success/rejection;
- measurement-related validation issue incidence;
- usable fraction;
- mechanism-bearing/usable;
- direct mechanism edge distribution.
