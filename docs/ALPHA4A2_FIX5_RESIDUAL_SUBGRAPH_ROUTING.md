# v2.9.0 alpha4a.2-fix5 — residual subgraph recovery routing

This is the final single-paper recovery calibration before the multi-paper SERS
pilot. It is domain-neutral and does not change the SERS ontology.

It addresses two residual patterns observed after fix4:

1. A scientifically coherent missing node is referenced by several dangling
   edges, but other validation errors coexist. The previous strict
   `has_common_undefined_endpoint_cluster()` path intentionally required every
   error to be an undefined-endpoint error, so a cluster such as

       substrate --PREPARED_BY--> [missing synthesis method]
       [missing synthesis method] --USES_PRECURSOR--> precursor
       [missing synthesis method] --USES_MATERIAL--> reagent

   could fail to receive the bounded second micro-reextract when unrelated
   measurement-producer errors were also present.

   fix5 adds a conservative *dominant missing-node cluster* detector. It
   requires at least three incident undefined-endpoint errors sharing one node
   ID and at least half of all current errors to belong to that cluster. It
   never creates the node deterministically; it only routes one additional
   complete source-grounded re-extraction.

2. Multiple claim validation failures can form one malformed claim-relation
   subgraph rather than independent edge defects. fix5 recognizes a coupled
   claim residual only when a claim-specific error coexists with at least one
   claim-relation problem involving SUPPORTS_CLAIM, INTERPRETED_AS, or
   APPLIES_TO. Such a residual receives one bounded second complete
   micro-reextract so the model can rebuild the claim subgraph as a unit.

The micro-reextract prompt now surfaces an explicit claim-subgraph rebuilding
hint while preserving the validator as the source of truth for allowed
directions and endpoint types.

Safety:
- no ontology widening;
- no relation-contract change;
- no automatic node invention;
- no add-node semantic-patch operation;
- no infinite retry loop;
- the extra complete re-extract remains bounded to one additional attempt;
- strict validation and the active domain gate remain mandatory.
