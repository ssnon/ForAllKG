from types import SimpleNamespace

from pipeline_core.discovery.explorer_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
)
from pipeline_core.discovery.explorer_validation import (
    _validate_unresolved_connection_roles,
)


def _statement(statement_id, role):
    return SimpleNamespace(
        statement_id=statement_id,
        epistemic_role=role,
    )


def test_unresolved_connection_rejects_non_unresolved_statement():
    report = SimpleNamespace(
        statements=[
            _statement("stmt:reported", "reported"),
            _statement("stmt:gap", "unresolved"),
        ],
        unresolved_connections=[
            SimpleNamespace(
                statement_id="stmt:reported",
            ),
            SimpleNamespace(
                statement_id="stmt:gap",
            ),
        ],
    )

    issues = []

    def error(code, location, message):
        issues.append(
            (code, location, message)
        )

    _validate_unresolved_connection_roles(
        report,
        error,
    )

    assert len(issues) == 1
    assert issues[0][0] == (
        "UNRESOLVED_CONNECTION_REQUIRES_"
        "UNRESOLVED_STATEMENT"
    )
    assert issues[0][1] == (
        "unresolved_connections[0].statement_id"
    )
    assert "stmt:reported" in issues[0][2]


def test_unresolved_connection_accepts_unresolved_statement():
    report = SimpleNamespace(
        statements=[
            _statement("stmt:gap", "unresolved"),
        ],
        unresolved_connections=[
            SimpleNamespace(
                statement_id="stmt:gap",
            ),
        ],
    )

    issues = []

    _validate_unresolved_connection_roles(
        report,
        lambda *args: issues.append(args),
    )

    assert issues == []


def test_unknown_statement_is_left_to_reference_validator():
    report = SimpleNamespace(
        statements=[],
        unresolved_connections=[
            SimpleNamespace(
                statement_id="stmt:missing",
            ),
        ],
    )

    issues = []

    _validate_unresolved_connection_roles(
        report,
        lambda *args: issues.append(args),
    )

    assert issues == []


def test_prompt_exposes_unresolved_gap_contract():
    assert (
        PROMPT_VERSION
        == "graph-explorer-prompt-v2.5.1.4"
    )

    assert (
        "Every unresolved_connections entry MUST reference "
        "a dedicated statement whose epistemic_role is exactly "
        "'unresolved'."
        in SYSTEM_PROMPT
    )
