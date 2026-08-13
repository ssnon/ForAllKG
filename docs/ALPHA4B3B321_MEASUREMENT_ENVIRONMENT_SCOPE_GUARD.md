# alpha4b.3b.3.2.1 — Measurement Environment Scope Guard

This is a narrow precision hotfix on top of the frozen role semantics from
alpha4b.3b.3.2.

## Problem

A producer `Experiment.description` can summarize multiple tasks in one
umbrella sentence, for example:

`Experimental evaluation of DIP SERS enhancement, polarization dependence,
DNA detection, and cell imaging performance.`

The previous environment harvester saw both `SERS` and `cell imaging` and
propagated `measurement_environment=cellular` to every Measurement produced by
that Experiment, including EF and DNA-range measurements whose local evidence
did not establish a cellular environment.

## Guard

Free-text producer (`experiment_method_text`) environment evidence is now
accepted only when the environment is syntactically tied to the physical
measurement:

- `SERS maps of U87MG cells`
- `Raman profiles ... recorded ... in a cell`
- concise direct method label `SERS-based target-specific cell imaging`
- direct solution/air measurement language

Measurement-local `source_expression`, explicit protocol attributes, and
structured conditions remain admissible under the existing provenance rules.

Multi-task umbrella prose alone no longer assigns an environment.

## Unchanged

- role split from alpha4b.3b.3.2
- concentration locality and leak guard
- protocol classification
- observable applicability
- numeric ranking gate
- Bridge / projection / corpus
- holdout papers
