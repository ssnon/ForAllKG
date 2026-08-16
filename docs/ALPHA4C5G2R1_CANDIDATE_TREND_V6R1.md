# alpha4c.5g.2r1 — Revised Candidate Trend v6r1

The first alpha4c.5g.2 candidate failed its Development regression:

- current v5 evidence: 9
- candidate v6 evidence: 10
- added numeric evidence: 1
- removed evidence: 0
- four audited claim regressions recovered: 0/4
- structural gate: pass

A read-only helper-stage diagnostic localized the remaining failures.

## Claim regression localization

### 1. `claim_gap_dependent_ef`

v1:
- control: nanogap_size
- response: SERS enhancement factor
- direction: negative

v3:
- control: nanogap_size
- response: SERS enhancement factor
- direction: unresolved

Therefore this is a v3 direction-parser regression.

### 2. `claim_gap_size_sers_intensity`

The phrase uses plural `gap sizes`.

v1 parses it as nanogap_size. v3 misses the control entirely.
The failed v6 recovered control/response but still lacked direction.

### 3. `claim_gap_enhancement_trend`

The grounded statement says a large gap transitions to stronger enhancement
as the gap decreases. v1 recognizes this quantitative size relation; v3 and
the failed v6 do not select a control.

### 4. `obs_gap_dependent_enhancement`

The failed v6 comparative helper correctly derives:

`2-nm gap > 8-nm gap in EF` -> increasing gap gives decreasing EF

but the supplemental path never reaches it because control remains unresolved.

## Revised rules

v6r1 keeps the failed-v6 measurement-local method resolution unchanged and
adds only narrow nanogap-size recovery:

1. quantitative size cue:
   - gap size(s), width(s), distance(s)
   - numeric length + gap
   - smaller/larger/narrower/wider gap
   - gap increases/decreases/narrows/widens

2. when v3 direction parsing fails for `nanogap_size`, use the already
   established historical v1 nanogap-size direction parser as a fallback.

3. if that also fails, use the explicit numeric comparative-pair parser from
   the failed v6 candidate.

Presence-only nanogap claims are not promoted to size trends.

## Development-only regression invariants

PASS requires all of:

- no frozen v5 evidence removed;
- candidate structural gate passes;
- all four audited claim regressions recovered;
- the explicit local 532-nm 1.2/15.6-nm numeric pair remains recovered;
- the 2-nm/8-nm simulated pair without measurement-local excitation remains
  blocked;
- Reserve A unused;
- Reserve B unused and sealed;
- zero LLM calls.

PASS still does not activate the candidate in the production registry.
