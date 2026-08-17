# SERS R0 Manual Scientific Adjudication v1

This sidecar freezes an LLM-assisted scientific adjudication performed after the frozen T1 retrieval.
It does **not** modify or backfill T1. Reviewer-found literature is explicitly separated from frozen T1 provenance.

## Reviewer mode

- Model: GPT-5.6 Sol
- Human scientist reviewer present: no
- Primary-source verification: yes
- Deterministic R0 router LLM calls: 0
- Hypothesis rewrite: no
- Fresh Reserve C: untouched

## H1 — Nanogap-mediated hotspot coupling in Au/Ag architectures

### Claim `external_novelty_claim:99ed0af7161d694818f6`

**DIRECT_PRIOR_ART.**

Jiang et al., *Scientific Reports* (2017), DOI `10.1038/s41598-017-10262-9`,
describes PSPAA-separated inner/outer Ag layers, an Ag bridge crossing the polymer shell,
and electromagnetic hotspot formation in the narrow polymer gap.

### Claim `external_novelty_claim:a735f559d97b4208dca3`

**DIRECT_PRIOR_ART at the frozen claim level.**

Rastogi et al., *ACS Applied Materials & Interfaces* (2021), DOI `10.1021/acsami.0c17929`,
systematically varies sub-10-nm gap distances and directly connects gap geometry,
analyte access/dimensions, hotspot leverage, and SERS sensitivity.

Ma et al., *J. Phys. Chem. C* (2020), DOI `10.1021/acs.jpcc.0c07701`,
independently shows gap-width-dependent SERS in Au/AgAu hybrids, with decreasing signal
as a one-gap width increases from about 0.8 to 3.8 nm.

The narrower architecture-specific controlled gap-size sweep is **not** silently substituted
for the frozen target claim. That narrower question is deferred to R2.

**R0 outcome:** `directly_covered → pass_original_to_r2`; R1 not authorized.

## H2 — Excitation-mode matching

Frozen `keep` is preserved without new targeted review.

**R0 outcome:** `pass_through_frozen`; R1 not authorized.

## H3 — Measurement-context dependence

Wu et al., *Scientific Reports* (2017), DOI `10.1038/s41598-017-13577-9`,
uses Ag-shell-coated core-satellite nanostructures and compares dispersed-in-solution
measurement with a drying-accumulated substrate. The accumulated substrate gives much
stronger and more reproducible SERS. The comparison is not the exact same analyte-mixed
preparation before/after drying, so claim `4579...` is **PARTIAL_PRIOR_ART**.

The paper does not establish a qualitative peak-profile transformation for the exact frozen
claim `778e...`; that claim is conservatively **COMPONENTS_ONLY**.

**R0 outcome:** `relational_gap_remains → pass_original_to_r2`; R1 is mechanically prohibited
because the frozen action is `targeted_search_only`.

## Stage boundary

After freeze:

- R1: not authorized for any hypothesis
- R2: not started
- T1: unchanged
- automatic next stage: disabled
- Fresh Reserve C: untouched
- STOP
