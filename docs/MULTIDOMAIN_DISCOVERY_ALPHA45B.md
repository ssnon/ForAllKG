# Multi-domain discovery alpha4a.5-B

## Scope

B hardens the two LLM boundaries without weakening deterministic scientific
validation.

### Graph Explorer

Normal execution remains:

generation
-> compile / validate
-> one bounded LLM repair
-> compile / validate

Only if that still fails, a deterministic one-way normalizer may run:

- an unsupported `mechanism` statement may be downgraded to `association`
  only when its text does **not** contain profile-defined strong causal language;
- an unsupported recurring mechanistic motif may be dropped;
- scientific text and evidence references are never rewritten;
- strong-causal unsupported claims are deliberately left unchanged so the strict
  validator can continue to reject them.

A normalization audit is always available and the CLI writes it beside the
Explorer output.

### Hypothesis semantic critic

Unknown references are no longer all-or-nothing:

- mixed exact-valid/exact-invalid ID lists: invalid IDs are safely dropped;
- no fuzzy matching is allowed;
- if a non-empty reference list loses every supplied ID, sanitization is fatal
  and the semantic review is rejected;
- a reference audit is persisted by the CLI.

## Domain semantics

Strong causal language is owned by `DiscoverySemantics`, not by the normalizer.
The existing generic causal vocabulary becomes the default profile policy, and
the SERS profile extends it with SERS-relevant verbs such as focusing/coupling.

## Non-goals

- no validator rule is downgraded to warning;
- no unknown reference is guessed or fuzzy-matched;
- no scientific sentence is rewritten;
- no novelty or feasibility behavior changes.
