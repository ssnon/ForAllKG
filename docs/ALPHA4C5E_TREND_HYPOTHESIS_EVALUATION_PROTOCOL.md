# alpha4c.5e — Trend-aware Hypothesis Evaluation & Reserve Protocol

## Purpose

alpha4c.5e freezes the acceptance/failure rules **before** an unseen
Trend-to-Hypothesis reserve is registered or consumed.

It adds no scientific extraction rule and does not change alpha4c.5a–5d.1.
The reserve is evaluated against the exact frozen implementation and rule set.

```text
5c.1 / 5d.1 seen smoke PASS
        |
        v
alpha4c.5e protocol freeze
        |
        v
unseen reserve registration
        |
        v
reserve Trend / CrossContext / 5a / 5b
        |
        v
5d.1 LLM run
        |
        v
5e exact recompile + revalidation + scope audit
        |
        v
PASS / FAIL
```

## Acceptance rule

Acceptance is **not** based on hypothesis count.

A run passes iff there are zero fatal 5e issues. In particular:

- zero hypotheses is allowed when the portfolio is a valid abstention;
- insufficient/context-specific/reversed evidence is allowed when the frozen
  limitation companions are preserved;
- verification-required hypotheses are allowed;
- no cross-paper synthesis is allowed;
- non-monotonic/unspecified directions are allowed;
- one bounded repair followed by a valid final draft is allowed.

## Fatal rules

The frozen fatal categories are:

- fabricated Trend view;
- Trend sign inversion;
- missing direction binding;
- positive/negative direction mismatch;
- missing replication-gap companion;
- missing context-qualification companion;
- missing reversal boundary;
- cross-paper overclaim;
- Trend-evidence causal escalation;
- Trend-evidence universal escalation;
- unsupported numeric prediction;
- external novelty claim;
- experiment-protocol leakage;
- Explorer/Trend namespace collision;
- provenance/corpus binding failure;
- frozen implementation drift;
- run not accepted by the runtime;
- Maker setting drift;
- final-draft/portfolio deterministic recompile mismatch;
- independent revalidation failure;
- reserve binding failure.

## Direction invariant

The sign frame is frozen:

```text
independent_change = increase

positive  -> dependent increase
negative  -> dependent decrease
```

LLM sign transformation to a decrease/smaller/lower frame is not allowed.
This is intentionally conservative.

## Independent evaluation

5e does not trust `Accepted: True` from the original run by itself. It:

1. reloads the frozen source input;
2. reconstructs the 5d.1 exposure and prompt;
3. verifies the frozen model/runtime settings;
4. recompiles the saved final LLM draft with the 5c.1 compiler;
5. requires byte-equivalent structured portfolio content;
6. reruns the 5c.1 validator, which also reruns frozen 5c validation;
7. audits textual cross-paper, causal-evidence, universal, and sign-scope
   overclaims;
8. checks the reserve manifest in reserve mode.

## Reserve registration

The reserve paper IDs are registered only after the 5e protocol exists.
Registration performs no extraction and no LLM calls.

The registration command requires explicit confirmation that the paper set was
not used to tune alpha4c.5e and was not inspected for the Trend/Hypothesis
acceptance rules before registration.

A reserve run must use a Trend corpus whose paper set exactly matches the
registered reserve paper set.

## Protocol immutability

The protocol hash-locks the exact 5b/5c/5d/5c.1/5d.1/5e implementation files.
If any of those files change, the frozen protocol fails verification.

A semantic patch after reserve execution requires a **new protocol epoch**.
A failed reserve campaign may not be resumed after such a patch.

Reserve results may not be used to change the acceptance rules of the campaign
that produced them.


## 5e v2 evaluator precision correction

The first installer attempt did **not** freeze a valid 5e protocol and was
automatically rolled back before reserve registration or consumption.

That attempt exposed an evaluator-only false positive. 5e had added a second
lexical sign scan over the entire compiled card, including falsification text.
For a correct positive Trend hypothesis, the falsifier can legitimately say:

`the dependent observable decreases as the independent variable increases`

because that sentence describes the **falsifying outcome**, not the supported
Trend direction. The broad 5e proximity scan misread the dependent-variable
word `decreases` as a decrease-frame transformation of the independent
variable.

The corrected 5e evaluator therefore does **not** invent a broader sign-text
scope than alpha4c.5c.1. Sign consistency is checked by the frozen
`DirectionAwareTrendHypothesisValidator`, which already checks the supported
hypothesis/bridge/prediction scope and emits
`NONCANONICAL_TREND_DIRECTION_FRAME`. 5e independently re-runs that validator
and maps that exact error to fatal `TREND_SIGN_INVERSION`.

Falsification criteria remain required and may describe the opposite
dependent-variable outcome; they are not positive Trend claims.
