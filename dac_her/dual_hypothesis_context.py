from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dac_her.discovery_contracts import DiscoveryBundle
from dac_her.hypothesis_contracts import HypothesisContext


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class DualHypothesisContext(StrictModel):
    """Epistemically separated hypothesis input.

    `grounded_context` remains the *only* source of positive premise IDs.
    `discovery_bundle` is inspiration-only and cannot become evidence merely by
    being present in this envelope.
    """

    schema_version: Literal["dual-hypothesis-context-v1"] = "dual-hypothesis-context-v1"
    dual_context_id: str
    dual_context_sha256: str
    domain_profile_id: str = "dac_her"
    grounded_context: HypothesisContext
    discovery_bundle: DiscoveryBundle

    @model_validator(mode="after")
    def _lineage_consistency(self) -> "DualHypothesisContext":
        context_domain = self.grounded_context.domain_profile_id
        bundle_domain = self.discovery_bundle.domain_profile_id
        if self.domain_profile_id != context_domain or self.domain_profile_id != bundle_domain:
            raise ValueError(
                "DualHypothesisContext domain profile mismatch: "
                f"dual={self.domain_profile_id!r}, "
                f"context={context_domain!r}, "
                f"bundle={bundle_domain!r}"
            )
        if self.grounded_context.corpus_id != self.discovery_bundle.corpus_id:
            raise ValueError(
                "grounded HypothesisContext and DiscoveryBundle must use the same corpus_id: "
                f"{self.grounded_context.corpus_id!r} != "
                f"{self.discovery_bundle.corpus_id!r}"
            )
        return self

    @classmethod
    def build(
        cls,
        grounded_context: HypothesisContext,
        discovery_bundle: DiscoveryBundle,
    ) -> "DualHypothesisContext":
        if grounded_context.corpus_id != discovery_bundle.corpus_id:
            raise ValueError(
                "grounded HypothesisContext and DiscoveryBundle must use the same corpus_id: "
                f"{grounded_context.corpus_id!r} != {discovery_bundle.corpus_id!r}"
            )
        if grounded_context.domain_profile_id != discovery_bundle.domain_profile_id:
            raise ValueError(
                "grounded HypothesisContext and DiscoveryBundle must use the same "
                "domain_profile_id: "
                f"{grounded_context.domain_profile_id!r} != "
                f"{discovery_bundle.domain_profile_id!r}"
            )
        domain_profile_id = grounded_context.domain_profile_id
        dual_id = _stable_id(
            "dual_hypothesis_context",
            domain_profile_id,
            grounded_context.context_sha256,
            discovery_bundle.bundle_sha256,
        )
        payload = {
            "schema_version": "dual-hypothesis-context-v1",
            "dual_context_id": dual_id,
            "domain_profile_id": domain_profile_id,
            "grounded_context": grounded_context.model_dump(mode="json"),
            "discovery_bundle": discovery_bundle.model_dump(mode="json"),
        }
        return cls(**payload, dual_context_sha256=_sha256_json(payload))
