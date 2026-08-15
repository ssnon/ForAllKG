# alpha4c.4b.1 — Pre-existing Resolution Provenance Holdout Fix

## Why alpha4c.4b preflight stopped

The original alpha4c.4b runner contained this operational rule:

```text
if resolution/decisions.jsonl contains any row:
    fail holdout preflight
```

That rule was too strong.

alpha4c.4a selected its 10-paper holdout from a
`pre_existing_curated_domain_corpus`. A pre-existing resolution decision file
can therefore be part of the curated canonical substrate. Its mere presence
does not prove adaptive Trend holdout tuning.

Deleting the file would be the wrong repair because it would destroy
provenance.

## Correct invariant

alpha4c.4b.1 replaces "file must not exist" with:

```text
before Trend scientific outputs:
    snapshot canonical GraphML SHA256
    snapshot decisions.jsonl presence
    if present, snapshot decisions.jsonl SHA256
    persist canonical_input_lock.json

after that lock:
    canonical GraphML may not change
    decisions.jsonl may not be created
    decisions.jsonl may not be deleted
    decisions.jsonl may not change
```

The runner only counts non-empty decision rows for diagnostics; it does not
interpret decision contents during preflight.

## Why the same 10 papers remain valid

The failed alpha4c.4b command stopped in `snapshot_canonical_inputs()` before
projection/corpus/Trend stages and before a holdout manifest/report was
created.

No TrendEvidence, TrendPrecision, cross-context contrast, or assessment was
generated or inspected.

Therefore this is an operational preflight correction, not a scientific
semantic repair, and it does not consume the 10-paper blind Trend holdout.

The frozen split SHA remains:

```text
46d3e42e721ad9e8f92dc41508c3916b3766e8f0bd5ca01617f115d80d55988f
```

and the frozen semantic IDs are unchanged.

## New commands

Install:

```bash
python apply_alpha4c4b1_resolution_provenance_fix.py
```

Then run the new preflight:

```bash
python -m scripts.run_sers_alpha4c4b1_trend_holdout --preflight-only
```

A successful preflight creates:

```text
evaluation/sers_alpha4c4b1/holdout_v1/canonical_input_lock.json
```

This is not a scientific Trend output. It is the persistent input freeze.

After the lock exists, do not edit/rebuild those ten canonical graphs or
their `resolution/decisions.jsonl` files.

Then run:

```bash
python -m scripts.run_sers_alpha4c4b1_trend_holdout
```

Scientific output goes to campaign-specific alpha4c.4b.1 IDs and:

```text
evaluation/sers_alpha4c4b1/holdout_v1/holdout_report.json
```

## No scientific semantics changed

This patch does not change:

- SERS Trend extraction semantics;
- Trend precision semantics;
- Trend context semantics;
- CrossContextTrend assessment semantics;
- Comparison/Method/MeasurementResultIdentity semantics;
- the 10 holdout papers;
- the 22 future reserve papers;
- count-free acceptance policy.

It changes only the operational interpretation of pre-existing resolution
provenance.
