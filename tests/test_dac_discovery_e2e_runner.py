from scripts.run_dac_discovery_e2e import _returned_path_count


def test_returned_path_count_supports_current_and_fallback_shapes():
    assert _returned_path_count({"returned_path_count": 4}) == 4
    assert _returned_path_count({"paths": [{}, {}]}) == 2
    assert _returned_path_count({"summary": {"returned_path_count": 3}}) == 3
    assert _returned_path_count({}) == 0
