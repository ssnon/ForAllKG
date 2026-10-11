import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/discovery/run_m15_r1_reliability_feasibility.py"
spec = importlib.util.spec_from_file_location("m15_r1", SOURCE)
m15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m15)


def make_fixture(root: Path):
    geom = {
        "geometry_family": "core_shell_nanorod", "core_material": "Au", "shell_material": "Ag",
        "nanorod_length_nm": 120, "nanorod_diameter_nm": 40, "shell_thickness_nm": 20,
        "surrounding_medium": "water", "background_index": 1.33,
        "material_model": "meep_materials_library", "wavelength_min_nm": 500,
        "wavelength_max_nm": 850, "spectrum_frequency_points": 3,
        "pml_thickness_um": .15, "padding_um": .2,
        "cell_x_um": .56, "cell_y_um": .48, "cell_z_um": .48, "courant": .25,
    }
    # Four spectra on same geometry. Lower r60 and r90 intentionally have
    # different spectra; these synthetic values are only tests of provenance.
    spectra = {
        "ex_t8_r60": [0.01,0.05,0.09],
        "ex_t16_r60": [0.0101,0.0501,0.0901],
        "ey_t8_r60": [0.05,0.06,0.02],
        "ey_t16_r60": [0.0501,0.0601,0.0201],
        "ey_t16_r90": [0.045,0.043,0.015],
    }
    frozen = []
    for key, dirname in m15.CASES.items():
        d = root / dirname
        d.mkdir(parents=True)
        pol = "Ex" if key.startswith("ex") else "Ey"
        res = 90 if "r90" in key else 60
        time = 8 if "t8" in key else 16
        request = {**geom, "request_id":key, "field_component":pol,
                   "resolution_px_per_um":res, "run_until_after_sources":time,
                   "spectral_observable":"scattering_cross_section_um2",
                   "evidence_eligibility":"execution_sanity_only"}
        result = {"request_id":key,"field_component":pol,
                  "worker_status":"completed", "resolution_px_per_um":res,
                  "run_until_after_sources":time,"wall_seconds":3 if res==60 else 4,
                  "peak_rss_mb":99,
                  "scientific_interpretation_permitted":False,
                  "spectrum":[{"wavelength_nm":w,"scattering_cross_section_um2":s}
                              for w,s in zip([500,633,850],spectra[key])]}
        for name, payload in [("request.json",request),("result.json",result)]:
            p = d/name
            raw = m15.canonical_bytes(payload)
            p.write_bytes(raw)
            frozen.append({"path":str(p.resolve()),"sha256":hashlib.sha256(raw).hexdigest()})
    gate = {"schema_version":"m15b-evidence-gate-v1",
            "quantitative_physics_feedback":"BLOCKED",
            "spatial_stability":"FAIL", "scientific_novelty":"NOT_TESTED",
            "permitted_use":"numerical_diagnostic_feedback_only",
            "ex_peak_location":"UPPER_BOUNDARY_CENSORED",
            "ey_spatial_relative_l1_change": m15._relative_l1(
                {"wavelength":[500,633,850],"scattering":spectra["ey_t16_r60"]},
                {"wavelength":[500,633,850],"scattering":spectra["ey_t16_r90"]}),
            "sources": frozen}
    gate_path = root / "M15B_EVIDENCE_GATE.json"
    gate_path.write_bytes(m15.canonical_bytes(gate))
    return gate_path


class M15R1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.gate_path=make_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_assess_blocked_and_check(self):
        out=self.root/"r1"
        status=m15.run(self.gate_path,out,self.root,5.0)
        self.assertEqual(status["science_backend"],"NO_QUALIFIED_BACKEND")
        verified=m15.check(out)
        self.assertEqual(verified["status"],"M15R_R1_CHECK_OK")
        evidence=m15.read_json(out/"M15R_QUALIFIED_EVIDENCE.json")
        self.assertFalse(evidence["scientific_interpretation_permitted"])
        self.assertNotIn("Ex peak",m15.read_json(out/"M15R_FROZEN_SEED.json")["research_question"])
        self.assertEqual(len(list(out.iterdir())),6)

    def test_hash_tampering_fails_closed(self):
        p=self.root/"diagnostic_short_v1/result.json"
        p.write_text(p.read_text().replace("0.01", "0.02"))
        with self.assertRaisesRegex(m15.FeasibilityError,"SOURCE_SHA256_MISMATCH"):
            m15.create_assessment(self.gate_path,self.root,5)

    def test_not_allowed_unblocked_gate(self):
        data=m15.read_json(self.gate_path)
        data["quantitative_physics_feedback"]="QUALIFIED"
        self.gate_path.write_bytes(m15.canonical_bytes(data))
        with self.assertRaisesRegex(m15.FeasibilityError,"expected M15B BLOCKED"):
            m15.create_assessment(self.gate_path,self.root,5)

    def test_sers_band_ratio_fails(self):
        p=self.root/"diagnostic_short_v1/request.json"
        req=m15.read_json(p)
        req["spectral_observable"]="SERS_band_ratio"
        p.write_bytes(m15.canonical_bytes(req))
        data=m15.read_json(self.gate_path)
        for x in data["sources"]:
            if x["path"]==str(p):
                x["sha256"]=m15.sha(p.read_bytes())
        self.gate_path.write_bytes(m15.canonical_bytes(data))
        with self.assertRaisesRegex(m15.FeasibilityError,"expected scattering"):
            m15.create_assessment(self.gate_path,self.root,5)

    def test_earlier_scientific_permission_fails(self):
        p=self.root/"diagnostic_time16_ey_v1/result.json"
        result=m15.read_json(p)
        result["scientific_interpretation_permitted"]=True
        p.write_bytes(m15.canonical_bytes(result))
        data=m15.read_json(self.gate_path)
        for x in data["sources"]:
            if x["path"]==str(p):
                x["sha256"]=m15.sha(p.read_bytes())
        self.gate_path.write_bytes(m15.canonical_bytes(data))
        with self.assertRaisesRegex(m15.FeasibilityError,"unexpected scientific authority"):
            m15.create_assessment(self.gate_path,self.root,5)

    def test_nonmatched_physical_configuration_rejected(self):
        p=self.root/"diagnostic_resolution90_ey_v1/request.json"
        req=m15.read_json(p);req["shell_thickness_nm"]=25
        p.write_bytes(m15.canonical_bytes(req))
        data=m15.read_json(self.gate_path)
        for x in data["sources"]:
            if x["path"]==str(p):x["sha256"]=m15.sha(p.read_bytes())
        self.gate_path.write_bytes(m15.canonical_bytes(data))
        with self.assertRaisesRegex(m15.FeasibilityError,"physical configuration drift"):
            m15.create_assessment(self.gate_path,self.root,5)

    def test_x1_not_acquired_and_no_claims(self):
        out=self.root/"r1"
        m15.run(self.gate_path,out,self.root,5.0)
        seed=m15.read_json(out/"M15R_FROZEN_SEED.json")
        self.assertEqual(seed["x1_held_out"]["evidence"],"NOT_ACQUIRED")
        self.assertIsNone(seed["numeric_effect_prediction"])
        self.assertFalse(seed["prior_art"]["claim_novelty_permitted"])

    def test_output_no_overwrite(self):
        out=self.root/"r1"
        m15.run(self.gate_path,out,self.root,5.0)
        with self.assertRaisesRegex(m15.FeasibilityError,"not empty"):
            m15.run(self.gate_path,out,self.root,5.0)

    def test_target_resolution_scaling(self):
        out=m15.create_assessment(self.gate_path,self.root,5.0)
        cost=out["backend"]["candidate_backends"][0]["cost"]
        self.assertAlmostEqual(cost["estimated_target_resolution_px_per_um"],250.0)
        self.assertAlmostEqual(cost["wall_scaling_ratio_r4_heuristic"],(250/90)**4)

    def test_no_literal_850nm_peak_claim(self):
        out=self.root/"r1"
        m15.run(self.gate_path,out,self.root,5.0)
        metric=m15.read_json(out/"M15R_QUALIFIED_EVIDENCE.json")["metrics"]
        self.assertEqual(metric["ex_peak_status"],"RIGHT_CENSORED_NOT_LOCATED")
        self.assertNotIn("ex_peak_nm",metric)

    def test_invalid_resolution_target_rejected(self):
        with self.assertRaisesRegex(m15.FeasibilityError,"target-shell-cells"):
            m15.create_assessment(self.gate_path,self.root,100)

    def test_output_hash_modified(self):
        out=self.root/"r1"
        m15.run(self.gate_path,out,self.root,5.0)
        (out/"M15R_R1_REPORT.md").write_text("corrupted")
        with self.assertRaisesRegex(m15.FeasibilityError,"ARTIFACT_SHA256_MISMATCH"):
            m15.check(out)

if __name__=="__main__": unittest.main()
