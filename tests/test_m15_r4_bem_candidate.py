import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.discovery.run_m15_r4_bem_candidate import create, check, _validate


class TestM15R4(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.request = {
            "schema_version": "sers-meep-execution-request-v0",
            "geometry_family": "core_shell_nanorod", "core_material": "Au", "shell_material": "Ag",
            "nanorod_length_nm": 110., "nanorod_diameter_nm": 40.,
            "shell_thickness_nm": 20., "field_component": "Ex",
            "coordinate_convention": "nanorod_x_incident_z", "incident_propagation": "normal_to_nanorod_axis",
            "surrounding_medium": "water", "wavelength_min_nm": 500., "wavelength_max_nm": 850.,
            "request_id": "r", "hypothesis_id": "h", "geometry_candidate_id": "g"
        }
        self.path = self.root / "request.json"
        self.path.write_text(json.dumps(self.request))
        self.out = self.root / "out"

    def test_complete_preflight_no_authority(self):
        row = create(self.path, self.out, 8., 633.)
        self.assertEqual(row["science_gate"], "BLOCKED")
        row = check(self.out)
        self.assertEqual(row["science_gate"], "BLOCKED")
        plan = json.loads((self.out / "M15R4_BEM_CANDIDATE_PLAN.json").read_text())
        self.assertEqual(plan["geometry_um"]["outer_length"], .15)
        self.assertEqual(plan["geometry_um"]["outer_diameter"], .08)
        self.assertIn("MATERIAL FILE_Ag_eps.dat", (self.out / "candidate.scuffgeo.template").read_text())
        self.assertIn("MATERIAL FILE_Au_eps.dat", (self.out / "candidate.scuffgeo.template").read_text())
        self.assertIn("pwPolarization 1 0 0", (self.out / "scuff-scatter.args.template").read_text())
        self.assertEqual(len(plan["generated_files"]), 4)

    def test_no_overwrite(self):
        create(self.path, self.out, 8., 633.)
        with self.assertRaises(FileExistsError):
            create(self.path, self.out, 8., 633.)

    def test_hash_change(self):
        create(self.path, self.out, 8., 633.)
        self.path.write_text("{}")
        with self.assertRaises(AssertionError):
            check(self.out)

    def test_file_change(self):
        create(self.path, self.out, 8., 633.)
        (self.out / "inner_core.geo").write_text("bad")
        with self.assertRaises(AssertionError):
            check(self.out)

    def test_outside_spectrum(self):
        with self.assertRaises(ValueError):
            create(self.path, self.out, 8., 900.)

    def test_invalid_geom(self):
        self.request["nanorod_length_nm"] = 25.
        self.path.write_text(json.dumps(self.request))
        with self.assertRaises(ValueError):
            create(self.path, self.out, 8., 633.)

    def test_mismatched_material(self):
        self.request["shell_material"] = "Au"
        with self.assertRaises(ValueError):
            _validate(self.request)

    def test_y_pol(self):
        self.request["field_component"] = "Ey"
        self.path.write_text(json.dumps(self.request))
        create(self.path, self.out, 8., 633.)
        self.assertIn("pwPolarization 0 1 0", (self.out / "scuff-scatter.args.template").read_text())

    def test_unknown_medium_fails(self):
        self.request["surrounding_medium"] = "oil"
        with self.assertRaises(ValueError):
            _validate(self.request)

    def test_positive_mesh_edge_guard(self):
        with self.assertRaises(ValueError):
            create(self.path, self.out, 0.1, 633.)

if __name__ == "__main__": unittest.main()
