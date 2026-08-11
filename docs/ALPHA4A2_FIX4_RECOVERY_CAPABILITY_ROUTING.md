# v2.9.0 alpha4a.2-fix4 — recovery capability routing

This patch is domain-neutral and leaves the SERS ontology unchanged.

It is targeted at the two residual Q2 failure modes observed after fix3:

1. One missing node ID is referenced as both an undefined source and undefined
   target. The recovery hint now aggregates undefined endpoints by node ID
   regardless of edge direction and prints all invalid incident edges.

2. `CLAIM_LIKE_ENTITY` cannot be repaired by the current semantic-patch
   operation set because the patch schema cannot migrate an object from
   `entities[]` into `observation_claims[]` or `mechanism_claims[]`.
   Recovery policy now recognizes this capability boundary and permits one
   bounded second complete micro-reextract before post-micro patching.

The patch also hardens semantic-patch retries. When a provider selects an
operation such as `replace_edge` but omits required fields, the next retry gets
operation-specific shape feedback instead of a raw Pydantic error only.

Safety:
- semantic-patch schema remains strict;
- no `add_node` operation is introduced;
- no SERS relation or entity vocabulary is widened;
- no graph node is deterministically invented from dangling edges;
- all recovered drafts still pass ordinary domain and strict validation.
