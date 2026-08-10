# SERS alpha4a — local ingestion and strict extraction

Alpha4a introduces the first real second scientific domain, `sers_au_ag`, at
the local PDF/SI ingestion and strict-extraction boundary.

It intentionally does not add a SERS feasibility adapter yet. Feasibility
remains fail-closed for `sers_au_ag`.

## Acceptance path

```text
data_sers/inbox/*.pdf
  -> scripts.ingest_local_corpus
  -> Marker markdown + configs/papers_sers_au_ag.yaml
  -> scripts.extract_paper --domain-profile sers_au_ag
  -> strict SERS graph chunks
  -> scripts.build_paper_graph --domain-profile sers_au_ag
  -> scripts.build_graphagents_projection --mode evidence
```

Mechanism/exploratory projections still require Bridge extraction. Bridge
domain separation and the SERS feasibility adapter are deferred to alpha4b.

## Scientific safeguards

- SERS substrates are not coerced into Catalyst.
- SERS is not represented as Reaction.
- analyte, concentration, excitation wavelength, laser power, acquisition time,
  Raman peak, and medium are preserved as measurement context when reported.
- observations remain distinct from author mechanism interpretations.
- LSPR/hotspot/local-field/charge-transfer mechanisms require source support.
- duplicate main-PDF content fails closed at local ingestion.
- SERS feasibility cannot silently reuse the HER feasibility adapter.

HER remains under `data_dac`; SERS defaults to `data_sers`.
