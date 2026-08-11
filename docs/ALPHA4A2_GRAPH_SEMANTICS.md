# v2.9.0 alpha4a.2 — graph-stage semantic diagnostics

This patch adds graph-stage relation semantics after the first complete
`Kiwook_SERS_1` strict extraction.

## What it adds

- Declarative relation endpoint constraints for `sers_au_ag`.
- Graph-stage semantic diagnostics written to:
  - `runs/<run_id>/graph_semantics/relation_contract_issues.{json,csv}`
  - `runs/<run_id>/graph_semantics/duplicate_label_groups.{json,csv}`
  - `runs/<run_id>/graph_semantics/components.{json,csv}`
  - `runs/<run_id>/graph_semantics/summary.json`
- Deterministic same-paper-node canonicalization within one paper graph.
- A standalone audit command:
  `python -m scripts.audit_paper_graph_semantics --graphml ... --domain-profile sers_au_ag`

## Intentional behavior

Relation endpoint problems are diagnostics, not hard failures. The first SERS
corpus pass should surface graph semantics issues without blocking provenance
preservation.

The patch deliberately does not merge same-label PlasmonicSubstrate or
Nanostructure nodes. Those can denote distinct specimens with different
shell thickness, nanogap, precursor concentration, or measurement context.
