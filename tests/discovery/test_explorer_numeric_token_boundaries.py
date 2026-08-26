from __future__ import annotations

from pipeline_core.discovery.explorer_validation import (
    _numbers,
)


def test_versus_is_not_parsed_as_voltage_unit() -> None:
    assert _numbers(
        "The maximum field was 78 versus 7."
    ) == {
        "78",
        "7",
    }


def test_and_is_not_parsed_as_ampere_unit() -> None:
    assert _numbers(
        "The values were 78 and 7."
    ) == {
        "78",
        "7",
    }


def test_hyphenated_entity_numbers_are_stable() -> None:
    assert _numbers(
        "GNS-2 and GNS-5 were compared."
    ) == {
        "2",
        "5",
    }


def test_real_voltage_unit_still_parses() -> None:
    assert _numbers(
        "The applied potential was 5 V."
    ) == {
        "5 v",
    }


def test_real_current_unit_still_parses() -> None:
    assert _numbers(
        "The current was 2 mA."
    ) == {
        "2 ma",
    }


def test_attached_nm_unit_still_parses() -> None:
    assert _numbers(
        "An 80nm Au sphere was used."
    ) == {
        "80nm",
    }


def test_spaced_nm_unit_still_parses() -> None:
    assert _numbers(
        "An 80 nm Au sphere was used."
    ) == {
        "80 nm",
    }


def test_percent_and_energy_units_still_parse() -> None:
    assert _numbers(
        "RSD was 9.94% and energy was 0.25 eV."
    ) == {
        "9.94%",
        "0.25 ev",
    }
