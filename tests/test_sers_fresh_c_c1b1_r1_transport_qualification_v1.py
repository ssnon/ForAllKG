import pytest
from pydantic import ValidationError

from dac_her.fresh_c_c1b1_r1_transport_qualification_v1 import (
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_BASE_URL,
    EXPECTED_MODEL,
    SyntheticTransportProbe,
    normalized_base_url,
    validate_protocol,
)

def test_protocol_repairs_transport_only():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["parent_model_binding_preserved"] is True
    assert p["parent_prompt_and_schema_semantics_preserved"] is True
    assert p["corrected_transport_semantics"] == (
        "openrouter_openai_compatible_chat_completions_json_schema_v1"
    )
    assert p["base_url"] == EXPECTED_BASE_URL
    assert p["reviewer_model"] == EXPECTED_MODEL
    assert p["temperature_parameter_policy"] == "OMIT_UNSUPPORTED_PARAMETER"
    assert p["deterministic_seed"] == 0
    assert p["reasoning_effort"] == "medium"
    assert p["reasoning_exclude"] is True
    assert p["provider_only"] == ["openai"]
    assert p["provider_allow_fallbacks"] is False
    assert p["provider_require_parameters"] is True

def test_qualification_is_synthetic_only():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["fresh_c_text_allowed_in_qualification"] is False
    assert p["scientific_hypothesis_text_allowed_in_qualification"] is False
    assert p["external_literature_allowed_in_qualification"] is False
    assert p["scientific_adjudication_allowed_in_qualification"] is False
    assert p["synthetic_structured_output_qualification_calls"] == 1

def test_no_automatic_c1b2_transition():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["automatic_c1b2_transition_allowed"] is False
    assert p["stop_after_result_freeze"] is True

def test_base_url_normalization():
    assert normalized_base_url(EXPECTED_BASE_URL + "/") == EXPECTED_BASE_URL

def test_synthetic_probe_schema_is_fail_closed():
    ok = SyntheticTransportProbe(
        status="PASS",
        scientific_content_used=False,
        fresh_c_content_used=False,
    )
    assert ok.status == "PASS"
    with pytest.raises(ValidationError):
        SyntheticTransportProbe(
            status="PASS",
            scientific_content_used=True,
            fresh_c_content_used=False,
        )
