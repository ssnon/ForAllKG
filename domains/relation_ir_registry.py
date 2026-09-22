from __future__ import annotations

from pipeline_core.discovery.scientific_relation_ir import (
    NullRelationTypingAdapter,
    RelationTypingAdapter,
)
from domains.sers.relation_ir_semantics import (
    SERS_RELATION_TYPING_ADAPTER,
)


_ADAPTERS: dict[str, RelationTypingAdapter] = {
    "sers_au_ag": SERS_RELATION_TYPING_ADAPTER,
}


def get_relation_typing_adapter(
    domain_profile_id: str,
) -> RelationTypingAdapter:
    key = str(domain_profile_id or "").strip().lower()
    return _ADAPTERS.get(
        key,
        NullRelationTypingAdapter(
            adapter_id=f"null_relation_typing_v1:{key or 'unknown'}"
        ),
    )


__all__ = ["get_relation_typing_adapter"]
