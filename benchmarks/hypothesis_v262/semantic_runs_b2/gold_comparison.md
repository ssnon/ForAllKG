# Semantic Gold Comparison hypothesis-semantic-gold-v262-b1

- Overall: **FAIL**
- Cases: 3 passed / 5 failed
- Critical mismatches: 9
- Noncritical mismatches: 0
- Missing reviews: 0

| Case | Result | Agreements | Mismatches |
|---|---:|---:|---|
| `canonical_valid` | FAIL | 2 | falsifier_informativeness:fail |
| `canonical_candidate` | PASS | 1 | - |
| `canonical_abstention` | PASS | 2 | - |
| `adv_candidate_overclaim` | FAIL | 1 | premise_fidelity:fail, inferential_proportionality:fail, causal_strengthening:fail |
| `adv_alignment_causalization` | FAIL | 2 | inferential_proportionality:fail, falsifier_informativeness:fail |
| `adv_directional_specificity` | PASS | 1 | - |
| `adv_causal_strengthening` | FAIL | 1 | falsifier_informativeness:fail |
| `adv_redundancy` | FAIL | 1 | prediction_linkage:fail, falsifier_informativeness:fail |
