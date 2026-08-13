# alpha4b.2b.2 — SERS Bridge semantic precision

Calibration on SERS_1/5/8 showed that the Bridge architecture was stable, but
most semantic candidates were false candidates caused by an overly narrow
lexical relation-cue detector.

The candidate CSVs showed explicit constructions such as:

- `can be reliably controlled`
- `became larger` under an increased precursor condition
- `was proportional to`
- `linear correlation/dependence`
- `red-shifted as`
- `has a marked effect on`
- `leading`
- `to protect`
- `because of`
- `can be improved by`

all being held only as `RELATION_CUE_MISMATCH`.

This policy-only calibration:

1. expands high-confidence SERS relation cues;
2. keeps ambiguous captions such as `with different particle sizes` conservative;
3. keeps `MEDIATES` stricter than the weaker source verb `facilitates`;
4. rejects direct analyte/reporter concentration calibration patterns from the
   discovery Bridge (strict measurements still retain them);
5. demotes property-vs-axis `CONTRASTS_WITH` shapes to semantic review;
6. detects high-confidence passive causal argument reversal such as
   `effect ... restricted/generated ... by cause`;
7. rejects obvious relation sentences encoded as frontier labels.

No extraction prompt/signature/validation files are changed, so the frozen raw
Bridge extraction identity should remain unchanged. Only the SERS policy
version/fingerprint changes.

Paper-level near-duplicate frontier consolidation is intentionally deferred.
It requires evidence-preserving cross-chunk canonicalization rather than a
string-only deletion rule.
