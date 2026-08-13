# PR6.1 — Broad vocabulary serialization audit/pruning

This experiment removes the measurement-metric registry only from the LLM
prompt surface for `catalysis_mechanism` Broad abstract extraction.

The metric registry remains loaded for normal finalization and validation.
Experiment-method vocabulary remains serialized because Broad abstracts can
still emit explicit Experiment objects.

Opt-in flag:

```bash
--broad-prune-metric-vocabulary
```

First run the no-API inspector, then the fixed six-paper matched A/B.
