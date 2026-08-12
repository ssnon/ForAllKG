from pathlib import Path

import yaml

from dac_her.literature_discovery.abstract_packages import build_abstract_packages
from dac_her.literature_discovery.contracts import LiteratureRecord
from dac_her.literature_discovery.relevance import CandidateAssessment
from dac_her.literature_discovery.selection import SelectedLiterature


def test_abstract_package_is_graphagents_config_compatible_shape(tmp_path: Path):
    record = LiteratureRecord.from_provider_result(
        provider="openalex",
        provider_id="W123",
        title="Dynamic active sites in electrocatalysis",
        abstract="A sufficiently long abstract about catalyst reconstruction and mechanism.",
        doi="10.1000/demo",
        mechanism_bucket="working_state_reconstruction",
        metadata={"language": "en"},
    )
    assessment = CandidateAssessment(
        paper_id=record.paper_id,
        eligible=True,
        total_score=10.0,
        bucket_scores={"working_state_reconstruction": 8.0},
        best_bucket="working_state_reconstruction",
        exclusion_reasons=(),
        context_hits=("catalyst",),
        mechanism_hits=("reconstruction",),
        bucket_hits={"working_state_reconstruction": ("reconstruction",)},
    )
    selected = SelectedLiterature(
        record=record,
        assessment=assessment,
        assigned_bucket="working_state_reconstruction",
        selection_mode="bucket_quota",
    )
    papers_yaml, manifest = build_abstract_packages(
        [selected], output_dir=tmp_path / "out", project_root=tmp_path
    )
    config = yaml.safe_load(papers_yaml.read_text(encoding="utf-8"))
    assert config["version"] == 3
    assert len(config["papers"]) == 1
    paper_id, paper = next(iter(config["papers"].items()))
    assert paper_id.startswith("broad_")
    document = paper["documents"][0]
    assert document["selection"] == {"mode": "whole_document"}
    assert document["figure_processing"]["mode"] == "none"
    package_dir = tmp_path / document["package_dir"]
    markdown = (package_dir / "main.md").read_text(encoding="utf-8")
    assert "# Dynamic active sites in electrocatalysis" in markdown
    assert "## Abstract" in markdown
    assert manifest.exists()
