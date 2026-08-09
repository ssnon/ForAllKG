from __future__ import annotations

from dac_her.lab_profile import load_lab_profile


def test_load_lab_profile(tmp_path):
    path = tmp_path / "lab.yaml"
    path.write_text(
        """schema_version: lab-profile-v0
lab_id: test_lab
resources:
  - resource_id: potentiostat
    category: electrochemistry
    access: internal
precursor_ids: []
""",
        encoding="utf-8",
    )
    profile = load_lab_profile(path)
    assert profile.lab_id == "test_lab"
    assert profile.resources[0].resource_id == "potentiostat"
