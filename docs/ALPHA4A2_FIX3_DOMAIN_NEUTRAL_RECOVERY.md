# v2.9.0 alpha4a.2-fix3 — domain-neutral recovery hardening

This patch is intentionally domain-neutral. It does **not** change the SERS
ontology or its relation contracts.

It addresses two failure modes observed in the strict SERS extraction:

1. A structurally parseable `KnowledgeGraphDraft` is rejected only because a
   reserved structured node type (for example `Experiment`) was emitted in
   `entities[]`. After ordinary generation retries are exhausted, strict
   recovery now performs one targeted full-draft collection-placement recovery
   using the active domain adapter's micro-reextract system prompt. The
   previously rejected draft and exact domain-gate error are supplied as
   diagnostics. The recovered draft must pass the same fail-closed domain gate.

2. A small unsplittable leaf contains multiple undefined edges sharing one
   missing endpoint ID. The micro-reextract prompt now explicitly summarizes
   that common endpoint cluster and tells the model to either emit the
   source-grounded missing node in the correct collection or remove unsupported
   dangling relations. If the first micro-reextract leaves *only* such a common
   endpoint cluster, recovery policy permits exactly one additional targeted
   micro-reextract before the existing post-micro patch phase.

Safety properties:

- no ontology widening;
- no deterministic invention of missing scientific nodes;
- no `add_node` semantic-patch operation;
- every recovered graph still passes the active extraction-domain gate and
  strict validation;
- the extra micro-reextract is bounded to one retry and only for a homogeneous
  common undefined-endpoint residual.
