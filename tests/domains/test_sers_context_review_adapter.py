from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from domains.context_review_registry import (
    available_context_review_adapters,
    available_context_review_profiles,
    resolve_context_review_adapter,
)
from domains.registry import (
    get_domain_profile,
)
from domains.sers.context_review_adapter import (
    SERSDiscoveryAxisContextReviewer,
)


@dataclass(frozen=True)
class _Signature:
    signature_id: str


class _Compiler:
    def __init__(self) -> None:
        self.grounded = []
        self.axes = []

    def compile_grounded_statement(
        self,
        statement,
    ):
        self.grounded.append(
            statement.statement_id
        )

        return _Signature(
            "grounded:"
            + statement.statement_id
        )

    def compile_axis_inspiration(
        self,
        inspiration,
    ):
        self.axes.append(
            inspiration.inspiration_id
        )

        return _Signature(
            "axis:"
            + inspiration.inspiration_id
        )


class _Interpreter:
    def __init__(self) -> None:
        self.calls = []

    def interpret(
        self,
        *,
        card,
        source_signatures,
    ):
        self.calls.append(
            (
                card.hypothesis_id,
                [
                    row.signature_id
                    for row
                    in source_signatures
                ],
            )
        )

        return SimpleNamespace(
            interpretation=(
                "validated interpretation"
            )
        )


class _Comparator:
    def __init__(self) -> None:
        self.calls = []

    def compare(
        self,
        *,
        interpretation,
        source_signatures,
        domain_profile_id,
    ):
        self.calls.append(
            (
                interpretation,
                [
                    row.signature_id
                    for row
                    in source_signatures
                ],
                domain_profile_id,
            )
        )

        return SimpleNamespace(
            review_id="review:test",
            hypothesis_id="hypothesis:h1",
            status="reframe_required",
        )


def _dual():
    statements = [
        SimpleNamespace(
            statement_id="stmt:a"
        ),
        SimpleNamespace(
            statement_id="stmt:b"
        ),
        SimpleNamespace(
            statement_id="stmt:unused"
        ),
    ]

    inspirations = [
        SimpleNamespace(
            inspiration_id="insp:selected"
        ),
        SimpleNamespace(
            inspiration_id="insp:unused"
        ),
    ]

    return SimpleNamespace(
        grounded_context=SimpleNamespace(
            domain_profile_id="sers_au_ag",
            evidence_statements=statements,
        ),
        discovery_bundle=SimpleNamespace(
            inspirations=inspirations,
        ),
    )


def test_sers_context_review_capability_is_external_to_profile_identity():
    profile = get_domain_profile(
        "sers_au_ag"
    )

    # Context-review capability must not mutate the historical
    # ScientificDomainProfile schema/fingerprint contract.
    assert not hasattr(
        profile,
        "context_review_adapter_id",
    )

    adapter = (
        resolve_context_review_adapter(
            profile
        )
    )

    assert (
        adapter.adapter_id
        == "sers_au_ag"
    )

    assert (
        adapter.domain_profile_id
        == "sers_au_ag"
    )

    assert (
        "sers_au_ag"
        in available_context_review_adapters()
    )

    assert (
        "sers_au_ag"
        in available_context_review_profiles()
    )


def test_other_domain_profiles_do_not_inherit_sers_context_rules():
    for profile_id in (
        "dac_her",
        "catalysis_mechanism",
    ):
        profile = get_domain_profile(
            profile_id
        )

        try:
            resolve_context_review_adapter(
                profile
            )

        except ValueError as exc:
            assert (
                "has no context-review capability"
                in str(exc)
            )

        else:
            raise AssertionError(
                f"{profile_id} must not resolve "
                "the SERS context reviewer"
            )


def test_reviewer_uses_only_selected_premises_and_assigned_axis():
    compiler = _Compiler()
    interpreter = _Interpreter()
    comparator = _Comparator()

    reviewer = (
        SERSDiscoveryAxisContextReviewer(
            compiler=compiler,
            interpreter=interpreter,
            comparator=comparator,
        )
    )

    card = SimpleNamespace(
        hypothesis_id="hypothesis:h1",
        premise_statement_ids=[
            "stmt:a",
            "stmt:b",
        ],
    )

    axis = SimpleNamespace(
        axis_id="axis:1",
        inspiration_id="insp:selected",
    )

    review = reviewer.review(
        dual=_dual(),
        axis=axis,
        card=card,
    )

    assert (
        compiler.grounded
        == [
            "stmt:a",
            "stmt:b",
        ]
    )

    assert (
        compiler.axes
        == [
            "insp:selected",
        ]
    )

    supplied = (
        interpreter.calls[0][1]
    )

    assert supplied == [
        "grounded:stmt:a",
        "grounded:stmt:b",
        "axis:insp:selected",
    ]

    assert (
        "stmt:unused"
        not in repr(
            interpreter.calls
        )
    )

    assert (
        "insp:unused"
        not in repr(
            interpreter.calls
        )
    )

    assert (
        comparator.calls[0][1]
        == supplied
    )

    assert (
        review.status
        == "reframe_required"
    )


def test_reviewer_fails_when_selected_premise_is_not_grounded():
    reviewer = (
        SERSDiscoveryAxisContextReviewer(
            compiler=_Compiler(),
            interpreter=_Interpreter(),
            comparator=_Comparator(),
        )
    )

    card = SimpleNamespace(
        hypothesis_id="hypothesis:h1",
        premise_statement_ids=[
            "stmt:missing",
        ],
    )

    axis = SimpleNamespace(
        axis_id="axis:1",
        inspiration_id="insp:selected",
    )

    try:
        reviewer.review(
            dual=_dual(),
            axis=axis,
            card=card,
        )

    except RuntimeError as exc:
        assert (
            "absent from grounded context"
            in str(exc)
        )

    else:
        raise AssertionError(
            "missing grounded premise must fail"
        )


def test_reviewer_fails_when_assigned_axis_is_not_in_bundle():
    reviewer = (
        SERSDiscoveryAxisContextReviewer(
            compiler=_Compiler(),
            interpreter=_Interpreter(),
            comparator=_Comparator(),
        )
    )

    card = SimpleNamespace(
        hypothesis_id="hypothesis:h1",
        premise_statement_ids=[
            "stmt:a",
        ],
    )

    axis = SimpleNamespace(
        axis_id="axis:1",
        inspiration_id="insp:missing",
    )

    try:
        reviewer.review(
            dual=_dual(),
            axis=axis,
            card=card,
        )

    except RuntimeError as exc:
        assert (
            "assigned discovery inspiration"
            in str(exc)
        )

    else:
        raise AssertionError(
            "missing assigned inspiration must fail"
        )


def test_reviewer_keeps_grounded_and_axis_compilation_on_separate_lanes():
    grounded_compiler = _Compiler()
    axis_compiler = _Compiler()

    reviewer = (
        SERSDiscoveryAxisContextReviewer(
            grounded_compiler=(
                grounded_compiler
            ),
            axis_compiler=(
                axis_compiler
            ),
            interpreter=_Interpreter(),
            comparator=_Comparator(),
        )
    )

    card = SimpleNamespace(
        hypothesis_id="hypothesis:h1",
        premise_statement_ids=[
            "stmt:a",
            "stmt:b",
        ],
    )

    axis = SimpleNamespace(
        axis_id="axis:1",
        inspiration_id="insp:selected",
    )

    reviewer.review(
        dual=_dual(),
        axis=axis,
        card=card,
    )

    assert grounded_compiler.grounded == [
        "stmt:a",
        "stmt:b",
    ]

    assert grounded_compiler.axes == []

    assert axis_compiler.grounded == []

    assert axis_compiler.axes == [
        "insp:selected",
    ]
