from __future__ import annotations

import hashlib
from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import NoveltyClaim
from pipeline_core.discovery.reframing.atomic_specification_bridge_shadow import (
    _binding_from_record,
    _source_sentence_spans,
)


def _claim() -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Fe coordination controls H adsorption.",
        rationale="test",
        prior_art_identity_terms=["Fe coordination"],
    )


def _authorized_record() -> dict:
    sentence = "Fe coordination controls H adsorption."
    return {
        "exact_source_recompile_shadow": {
            "relation_endpoint_anchors": [
                "Fe coordination",
                "H adsorption",
            ],
            "scope_qualifier_spans": [],
            "directional_qualifier_spans": [],
        },
        "exact_source_recompile_authority_shadow": {
            "diagnostic_only": True,
            "production_authority": False,
            "production_recompile_enabled": False,
            "recompile_performed": False,
            "authority_status": "AUTHORIZED_SHADOW",
            "bounded_recompile_contract_satisfied": True,
            "candidate_source_path": "inferential_bridge.unit[0]",
            "exact_source_text": sentence,
            "authority_reason_codes": [],
        },
    }


def test_sentence_spans_preserve_exact_source_offsets():
    source = "First relation. Fe coordination controls H adsorption."
    rows = _source_sentence_spans(source)
    assert rows[0][2] == "First relation."
    assert rows[1][2] == "Fe coordination controls H adsorption."
    for start, end, quote in rows:
        assert source[start:end] == quote


def test_authorized_exact_source_builds_typed_binding():
    hypothesis = SimpleNamespace(
        hypothesis_id="h1",
        inferential_bridge="Fe coordination controls H adsorption.",
    )
    binding, contract, decision = _binding_from_record(
        hypothesis=hypothesis,
        claim=_claim(),
        record=_authorized_record(),
    )
    assert decision.status == "BOUND_EXACT_SOURCE"
    assert binding is not None
    assert contract is not None
    assert binding["quote"] == hypothesis.inferential_bridge
    assert binding["free_text_bridge_generated"] is False
    assert binding["source_sha256"] == hashlib.sha256(
        hypothesis.inferential_bridge.encode("utf-8")
    ).hexdigest()


def test_denied_shadow_never_builds_binding():
    record = _authorized_record()
    record["exact_source_recompile_authority_shadow"] = dict(
        record["exact_source_recompile_authority_shadow"]
    )
    record["exact_source_recompile_authority_shadow"][
        "authority_status"
    ] = "DENIED_SHADOW"
    record["exact_source_recompile_authority_shadow"][
        "bounded_recompile_contract_satisfied"
    ] = False
    record["exact_source_recompile_authority_shadow"][
        "authority_reason_codes"
    ] = ["not_safe"]

    hypothesis = SimpleNamespace(
        hypothesis_id="h1",
        inferential_bridge="Fe coordination controls H adsorption.",
    )
    binding, contract, decision = _binding_from_record(
        hypothesis=hypothesis,
        claim=_claim(),
        record=record,
    )
    assert binding is None
    assert contract is None
    assert decision.status == "SKIPPED_EXACT_SOURCE_NOT_AUTHORIZED"


def test_assumption_source_is_not_promoted_to_typed_bridge():
    record = _authorized_record()
    record["exact_source_recompile_authority_shadow"] = dict(
        record["exact_source_recompile_authority_shadow"]
    )
    record["exact_source_recompile_authority_shadow"][
        "candidate_source_path"
    ] = "assumptions[0].unit[0]"

    hypothesis = SimpleNamespace(
        hypothesis_id="h1",
        inferential_bridge="Fe coordination controls H adsorption.",
    )
    binding, contract, decision = _binding_from_record(
        hypothesis=hypothesis,
        claim=_claim(),
        record=record,
    )
    assert binding is None
    assert contract is None
    assert decision.status == "SKIPPED_NON_INFERENTIAL_BRIDGE_SOURCE"
