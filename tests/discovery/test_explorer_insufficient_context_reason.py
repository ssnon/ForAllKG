from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.explorer_contracts import (
    UnresolvedConnection,
)
from pipeline_core.discovery.explorer_draft import (
    UnresolvedConnectionDraft,
)
from pipeline_core.discovery.explorer_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
)


def test_unresolved_draft_accepts_insufficient_context() -> None:
    row = UnresolvedConnectionDraft(
        local_id="u1",
        statement_local_id="s1",
        related_path_ids=["path:1"],
        reason="insufficient_context",
    )

    assert row.reason == "insufficient_context"


def test_final_unresolved_contract_accepts_insufficient_context() -> None:
    row = UnresolvedConnection(
        gap_id="gap:1",
        statement_id="stmt:1",
        related_path_ids=["path:1"],
        reason="insufficient_context",
    )

    assert row.reason == "insufficient_context"


def test_unresolved_reason_remains_fail_closed_for_unknown_value() -> None:
    with pytest.raises(ValidationError):
        UnresolvedConnectionDraft(
            local_id="u1",
            statement_local_id="s1",
            reason="random_unknown_reason",
        )


def test_prompt_version_bumped_for_reason_contract() -> None:
    assert (
        PROMPT_VERSION
        == "graph-explorer-prompt-v2.5.1.3"
    )


def test_system_prompt_still_forbids_hypothesis_generation() -> None:
    # Scientific role boundary must remain unchanged by this schema fix.
    assert "You are not the Hypothesis Maker." in SYSTEM_PROMPT
