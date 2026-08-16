# alpha4c.5h.1 — Guarded Reserve-B Readiness and One-Shot Confirmation

alpha4c.5h froze:

- Trend semantics: `sers_au_ag_trend_v6r2_alpha4c5g2r2`
- Freeze ID: `sers_alpha4c5h_v6r2_freeze:710786859181e21535f2`
- Reserve-B confirmation protocol:
  `sers_alpha4c5h_reserve_b_confirmation:7e71422f94caadf9161a`

This epoch does not change scientific Trend semantics.

## Runtime Trend binding

The production SERS Trend registry remains at the historical v5 adapter.
Reserve-B execution explicitly binds the already-frozen v6r2 adapter only
inside dedicated wrapper processes. Global registry files are not mutated.

## Precision transport compatibility

v6r2 changes TrendEvidence recall but not the scientific meaning of the
existing precision annotation/consolidation rules.

The runtime precision binding therefore delegates annotation and consolidation
to the frozen v5 precision adapter and preserves its
`precision_semantics_id`. Only `PaperLocalTrendResult.trend_semantics_id` is
rebound to the actual frozen v6r2 parent Trend source.

This binding is tested on the already-open 53-paper Development partition and
its exact wrapper SHA is frozen into the Reserve-B execution protocol before
consumption.

## Four gates

1. Development downstream compatibility:
   v6r2 Trend -> unchanged Precision -> CrossContext -> 5a grounding.

2. Reserve-B canonical readiness:
   attempt-layout-aware Strict source resolution, deterministic canonical
   migration only when eligible, no extraction LLM.

3. Execution protocol freeze:
   binds 5h freeze, Reserve-B IDs, canonical readiness lock, 5e evaluation
   protocol, newly registered Reserve-B 5e manifest, inherited Explorer/Maker
   settings, and all new execution-wrapper hashes.

4. One-shot execution:
   revalidates every binding and canonical readiness immediately before
   guarded irreversible consumption. Only after the marker is written may
   canonical scientific graphs be parsed/copied and the scientific pipeline
   execute.

Reserve-B failure never authorizes semantic tuning or rerun.
