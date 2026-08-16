# alpha4c.5g.2r2 — Candidate-local Trend Method Compatibility

v6r1 passed Development regression but its method-locality implementation
rewrote 36 singleton MethodContexts before Trend extraction. The scientific
idea was valid, but the scope was broader than necessary.

v6r2 removes every global MethodContext rewrite.

## Frozen base

The current registered v5 adapter still runs on the original, unmodified
TrendEvidenceSource.

The v6r1 claim recovery is retained as a supplemental claim lane. No claim
semantics are widened beyond the Development-audited nanogap recovery.

## Candidate-local numeric override

The supplemental numeric lane reproduces frozen v3 numeric grouping. It is
entered only when frozen `_methods_compatible()` rejects a candidate.

An override is permitted only if all of the following hold:

1. the remaining ambiguous method dimension is excitation wavelength;
2. every Measurement in that exact numeric Trend candidate directly contains
   exactly one structured excitation-wavelength condition;
3. all those Measurement-local wavelengths normalize to the same value;
4. any already-known excitation MethodContext value agrees with that local
   value;
5. all other method dimensions pass the frozen compatibility contract.

No MethodContext row is changed. The compatibility exception exists only for
that candidate group.

Examples:

- 1.2-nm gap @ 532 nm vs 15.6-nm gap @ 532 nm: eligible;
- 532 nm vs 633 nm: blocked;
- 532 nm vs missing local wavelength: blocked;
- 2-nm/8-nm simulated pair with no Measurement-local excitation: blocked.

## Development regression scope

The Development fixture now requires:

- all frozen v5 evidence preserved;
- exactly the five audited/sibling nanogap claim IDs recovered;
- the only added numeric measurement set is the audited 1.2/15.6-nm 532-nm
  pair;
- the only allowed/emitted candidate-local override is that same pair;
- zero global MethodContext mutations;
- unresolved 2-nm/8-nm pair remains blocked;
- candidate structural gate passes;
- Reserve A unused;
- Reserve B unused and sealed;
- zero LLM calls.

These are regression-scope invariants, not scientific count thresholds.

A PASS still does not activate v6r2 or execute Reserve B.
