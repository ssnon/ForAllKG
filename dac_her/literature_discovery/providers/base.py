from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..contracts import LiteratureRecord


@dataclass(frozen=True)
class LiteratureSearchRequest:
    query: str
    mechanism_bucket: str
    limit: int

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.mechanism_bucket.strip():
            raise ValueError("mechanism_bucket must not be empty")
        if self.limit <= 0:
            raise ValueError("limit must be positive")


@runtime_checkable
class LiteratureProvider(Protocol):
    """Provider-independent scholarly search contract.

    Concrete network providers are intentionally deferred to the next
    implementation slice. Implementations must return normalized
    LiteratureRecord values and preserve provider-specific IDs in each record.
    """

    provider_name: str

    def search(self, request: LiteratureSearchRequest) -> list[LiteratureRecord]: ...
