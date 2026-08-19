from pathlib import Path
import socket

import pytest

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1a_materialization_v1 import (
    load_and_validate_protocol,
    network_disabled,
    render_page_bounded_text,
)


def test_c1a_protocol_marks_irreversible_consumption_boundary():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_c1a_materialization_v1_protocol.json")
    )
    assert p.selected_pdf_count == 25
    assert p.consumption_marker_before_first_text_extraction is True
    assert p.fresh_reserve_c_consumed_at_marker_write is True
    assert p.consumption_irreversible is True
    assert p.same_epoch_rerun_after_marker_allowed is False
    assert p.failure_restores_freshness is False


def test_c1a_materializer_is_local_and_pinned():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_c1a_materialization_v1_protocol.json")
    )
    assert p.materializer == "pdftext_plain_text_v0_6_3"
    assert p.pdftext_version == "0.6.3"
    assert p.pypdfium2_version == "4.30.0"
    assert p.network_allowed_during_materialization is False
    assert p.socket_network_guard_required is True
    assert p.llm_calls == 0


def test_c1a_does_not_mutate_science():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_c1a_materialization_v1_protocol.json")
    )
    assert p.scientific_reviewer_read_performed is False
    assert p.scientific_adjudication_performed is False
    assert p.hypothesis_state_mutation_allowed is False
    assert p.positive_evidence_promotion_allowed is False
    assert p.automatic_c1b_transition_allowed is False


def test_network_guard_blocks_socket_connect():
    with network_disabled():
        s = socket.socket()
        try:
            with pytest.raises(RuntimeError, match="C1A_NETWORK_DISABLED"):
                s.connect(("127.0.0.1", 9))
        finally:
            s.close()


def test_page_bounded_rendering_is_stable():
    text = render_page_bounded_text(["alpha", "beta"])
    assert text == "[[PAGE 1]]\nalpha\n\n[[PAGE 2]]\nbeta\n"
