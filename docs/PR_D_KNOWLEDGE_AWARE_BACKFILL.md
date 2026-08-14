# PR D — Knowledge-aware Strict/Bridge backfill

## Purpose

M1–M4.5 keep their existing acquisition semantics.  This stage does **not**
change scientific eligibility, quality gating, OA policy, or the Strict/Bridge
validators.  It changes only the production stopping criterion:

> keep drawing from the already-ranked, quality-pass M3.2 reserve until a
> requested number of knowledge-layer papers is reached.

The default production target is `BRIDGE_USEFUL`.  A paper counts only when its
Strict extraction is usable, its Bridge status is useful, its mechanism
projection is usable, and the Strict/Bridge runner marked it corpus-eligible.

## Feedback loop

```text
paper_outcomes.jsonl
        |
        v
knowledge target deficit
        |
        v
M3.2 quality-pass ranked reserve (+K slots)
        |
        v
optional M3.1 refresh -> M4 resume/materialize -> M4.5
        |
        v
Strict/Bridge runner (old papers resume, new papers run)
        |
        +-------------------- repeat if deficit remains
```

Each backfill round gets an immutable round directory.  M4 and M4.5 keep their
canonical output directories so existing document package paths remain stable;
this is what allows PR C.1 to reuse prior Strict runs safely.

## Safety bounds

- `--max-rounds`
- `--max-extra-candidates`
- `--oversample-factor >= 1.0`

No new discovery is performed.  Candidate choice remains delegated to the
existing M3.2 engine, which only uses the quality-pass/eligible reserve and
preserves its scientific ranking / axis-aware policy.

## Supplementary information

If both `--m3-1-dir` and `--supplementary-policy` are supplied, M3.1 is rerun
against the expanded selection and its existing per-work state is resumed.
If only `--m3-1-dir` is supplied, existing SI is preserved for already-selected
papers but newly added reserve papers are materialized from their main artifact
only.
