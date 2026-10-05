from scripts.discovery.discover_e1_cohort_cases import resolve_roles


def row(case_id, path, mtime):
    return {
        "case_id": case_id,
        "path": path,
        "candidate_count": 5,
        "mtime_ns": mtime,
    }


def test_resolve_roles_can_match_path_when_case_id_lacks_q2():
    rows = [
        row("sers_q1_dev", "/eval/q1_hotspot_geometry/x.json", 1),
        row("sers_tradeoff_dev", "/eval/q2_reproducibility/x.json", 2),
        row("sers_q3_dev", "/eval/q3_excitation_reporter_matching/x.json", 3),
    ]
    resolved = resolve_roles(
        rows,
        {
            "q1": ["q1", "hotspot_geometry"],
            "q2": ["q2", "tradeoff", "reproducibility"],
            "q3": ["q3", "excitation_reporter_matching"],
        },
    )
    assert resolved["q2"]["case_id"] == "sers_tradeoff_dev"


def test_resolve_roles_prefers_latest_match():
    rows = [
        row("sers_q1_old", "/eval/q1/x.json", 1),
        row("sers_q1_new", "/eval/q1/new/x.json", 5),
    ]
    resolved = resolve_roles(rows, {"q1": ["q1"]})
    assert resolved["q1"]["case_id"] == "sers_q1_new"
