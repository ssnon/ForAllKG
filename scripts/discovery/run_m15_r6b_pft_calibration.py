"""M15-R6B: auditable single-mesh SCUFF sphere PFT vs analytic Mie comparison.

Reads frozen R6 outputs. Never runs solvers, changes original data, or qualifies
Au/Ag/SERS. The PFT fourth column is scattered *power*, not cross section.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'm15r6b-scuff-pft-mie-audit-v1'
Z0_OHM = 376.730313668   # reference vacuum impedance
E0_V_PER_UM = 1.0      # from fixed R6 --pwPolarization 1 0 0
WAVELENGTH_UM = 0.633


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_pft(text: str, expected_label: str='TestSphere', expected_omega: float | None = None) -> dict:
    """Fail closed unless the expected documented PFT header and single row exist."""
    lines = text.splitlines()
    headings = '\n'.join(s.strip() for s in lines if s.lstrip().startswith('#'))
    # Explicitly verify data-column semantics; do NOT blindly assume column 4.
    if not re.search(r'(?im)^#\s*3\s+absorbed\s+power\s*\(watts\)',headings):
        raise ValueError('PFT_ABSORBED_COLUMN_HEADER_UNVERIFIED')
    if not re.search(r'(?im)^#\s*4\s+scattered\s+power\s*\(watts\)',headings):
        raise ValueError('PFT_SCATTERED_COLUMN_HEADER_UNVERIFIED')
    rows = [s.split() for s in lines if s.strip() and not s.lstrip().startswith('#')]
    if len(rows) != 1:
        raise ValueError(f'PFT_EXPECTED_ONE_DATA_ROW_GOT_{len(rows)}')
    row = rows[0]
    if len(row)!=10:
        raise ValueError(f'PFT_EXPECTED_TEN_COLUMNS_GOT_{len(row)}')
    if row[1]!=expected_label:
        raise ValueError('PFT_WRONG_OBJECT_LABEL_'+row[1])
    try:
        omega=float(row[0]); absorbed=float(row[2]); scattered=float(row[3])
        rest=[float(v) for v in row[4:]]
    except ValueError as exc:
        raise ValueError('PFT_NONNUMERIC_DATA') from exc
    if not all(math.isfinite(v) for v in [omega,absorbed,scattered,*rest]):
        raise ValueError('PFT_NONFINITE_DATA')
    if scattered<=0:
        raise ValueError('PFT_SCATTERED_POWER_NONPOSITIVE')
    if expected_omega is not None and not math.isclose(omega,expected_omega,rel_tol=0.001,abs_tol=1e-8):
        raise ValueError('PFT_FREQUENCY_NOT_MATCHED')
    return {'angular_frequency_scuff':omega,'surface_label':row[1],
            'absorbed_power_w':absorbed,'scattered_power_w':scattered,
            'force_torque_components':rest}


def compare(pft: dict, mie_um2: float) -> dict:
    if not math.isfinite(mie_um2) or mie_um2<=0:
        raise ValueError('INVALID_MIE_REFERENCE')
    # A 1 V/um vacuum plane wave: I=(1e6 V/m)^2/(2 Z0) W/m²,
    # converted to W/um² by multiplying 1e-12.
    intensity_w_um2 = E0_V_PER_UM**2/(2*Z0_OHM)
    sigma = pft['scattered_power_w']/intensity_w_um2
    diff = sigma-mie_um2
    error = abs(diff)/mie_um2
    abs_ratio=abs(pft['absorbed_power_w'])/pft['scattered_power_w']
    return {'incident_field_v_per_um':E0_V_PER_UM,'vacuum_impedance_ohm':Z0_OHM,
            'incident_intensity_w_per_um2':intensity_w_um2,'scuff_sigma_sca_um2':sigma,
            'mie_sigma_sca_um2':mie_um2,'signed_difference_um2':diff,
            'relative_error':error,'relative_error_pct':error*100,
            'pabs_over_psca_absolute':abs_ratio,
            'single_mesh_reference_within_10pct':error<=0.1,
            'lossless_sphere_absorption_small_10pct_of_scatter':abs_ratio<=0.1}


def audit(r6_dir:Path,out_dir:Path)->dict:
    from scripts.discovery.run_m15_r6_scuff_control import check as r6_check
    r6_dir=r6_dir.resolve();out_dir=out_dir.resolve()
    state=r6_check(r6_dir)
    if state['scuff_run'] != 'SCUFF_PFT_PRODUCED_UNCALIBRATED':
        raise ValueError('R6_SCATTER_RESULT_NOT_READY')
    plan_file=r6_dir/'M15R6_CONTROL_PLAN.json'
    plan=json.loads(plan_file.read_text())
    receipt_file=r6_dir/'M15R6_SCUFF_RUN.json'
    receipt=json.loads(receipt_file.read_text())
    if receipt.get('exit_code')!=0 or receipt.get('timed_out') is not False:
        raise ValueError('R6_SOLVER_NOT_SUCCESSFUL')
    if receipt.get('nanorod_science_gate') != 'BLOCKED' or receipt.get('science_backend_qualified') is not False:
        raise ValueError('R6_SCIENCE_GATE_CHANGED')
    params=plan['sphere_parameters']
    if not all(math.isclose(float(params[name]),val,rel_tol=0,abs_tol=1e-12)
               for name,val in [('radius_um',.05),('lambda_vacuum_um',.633),('epsilon_relative',2.25)]):
        raise ValueError('R6_SPHERE_PARAMETERS_NOT_MATCHED')
    # Immutable R6 receipt and hashes are checked by r6_check above.
    cmd=receipt['command']
    if not (['--pwPolarization','1','0','0']==cmd[cmd.index('--pwPolarization'):cmd.index('--pwPolarization')+4]
            if '--pwPolarization' in cmd else False):
        raise ValueError('R6_FIELD_AMPLITUDE_NOT_CONFIRMED')
    if not (['--pwDirection','0','0','1']==cmd[cmd.index('--pwDirection'):cmd.index('--pwDirection')+4]
            if '--pwDirection' in cmd else False):
        raise ValueError('R6_DIRECTION_NOT_CONFIRMED')
    pft_file=r6_dir/'sphere.PFT'
    pft=parse_pft(pft_file.read_text(),expected_omega=2*math.pi/WAVELENGTH_UM)
    comparison=compare(pft,float(plan['sphere_analytic_mie_scattering_um2']))
    if out_dir.exists():
        raise FileExistsError('R6B_OUT_DIR_EXISTS_NO_OVERWRITE')
    out_dir.mkdir(parents=True)
    gate={'schema_version':SCHEMA,'status':'SINGLE_MESH_COMPARISON_RECORDED',
          'r6_dir':str(r6_dir),'r6_input_plan_sha256':sha256(plan_file),
          'r6_scuff_run_sha256':sha256(receipt_file),'r6_pft_sha256':sha256(pft_file),
          'sphere_triangle_count':plan['sphere_mesh_audit']['faces'],
          'mesh_edge_um':plan['sphere_parameters']['mesh_edge_um'],
          'observable':'single_dielectric_sphere_scattering_cross_section',
          'pft_extracted':pft,'comparison':comparison,
          'single_mesh_validated':False,'mesh_convergence_demonstrated':False,
          'au_ag_materials_validated':False,'nanorod_science_gate':'BLOCKED',
          'science_backend_qualified':False,'scientific_novelty_certified':False,
          'new_scuff_runs':0,'new_paid_calls':0,
          'next_action':'independent_sphere_mesh_refinement_and_energy_balance_review',
          'limitations':['single mesh only; cannot infer convergence',
                         'PFT-to-cross-section uses documented 1 V/um source convention',
                         'Mie benchmark is a homogeneous lossless dielectric sphere, not Au/Ag or SERS',
                         'no authority to claim material accuracy or scientific novelty']}
    gate_path=out_dir/'M15R6B_SINGLE_MESH_GATE.json'
    gate_path.write_text(json.dumps(gate,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    summary=(f'# M15-R6B SCUFF vs Mie: single-mesh audit\n\n'
             f'- SCUFF scattered power: {pft["scattered_power_w"]:.12g} W\n'
             f'- SCUFF scattering cross section: {comparison["scuff_sigma_sca_um2"]:.12g} um^2\n'
             f'- Mie analytical scattering cross section: {comparison["mie_sigma_sca_um2"]:.12g} um^2\n'
             f'- Relative error: {comparison["relative_error_pct"]:.6g}%\n'
             f'- |P_abs| / P_scat: {comparison["pabs_over_psca_absolute"]:.6g}\n'
             f'- Single-mesh error <=10%: {comparison["single_mesh_reference_within_10pct"]}\n'
             f'- R6 source PFT SHA256: `{gate["r6_pft_sha256"]}`\n\n'
             f'**Not converged or science qualified.** A second independent mesh is required.\n'
             f'Au/Ag nanorod science gate remains **BLOCKED**.\n')
    (out_dir/'M15R6B_SINGLE_MESH_REPORT.md').write_text(summary)
    return {'status':gate['status'],'scuff_cross_section_um2':comparison['scuff_sigma_sca_um2'],
            'mie_cross_section_um2':comparison['mie_sigma_sca_um2'],
            'relative_error_pct':comparison['relative_error_pct'],
            'absorbed_over_scattered_abs':comparison['pabs_over_psca_absolute'],
            'single_mesh_within_10pct':comparison['single_mesh_reference_within_10pct'],
            'nanorod_science_gate':'BLOCKED','new_scuff_runs':0,'report':str(gate_path)}


def check(out_dir:Path)->dict:
    from scripts.discovery.run_m15_r6_scuff_control import check as r6_check
    out_dir=out_dir.resolve();gate_file=out_dir/'M15R6B_SINGLE_MESH_GATE.json'
    gate=json.loads(gate_file.read_text())
    if gate.get('schema_version')!=SCHEMA or gate.get('nanorod_science_gate')!='BLOCKED' or gate.get('science_backend_qualified') is not False:
        raise ValueError('R6B_GATE_INVALID')
    r6=Path(gate['r6_dir']);r6_check(r6)
    for name,expected in [('M15R6_CONTROL_PLAN.json','r6_input_plan_sha256'),
                          ('M15R6_SCUFF_RUN.json','r6_scuff_run_sha256'),('sphere.PFT','r6_pft_sha256')]:
        if sha256(r6/name)!=gate[expected]:raise ValueError('R6B_SOURCE_CHANGED_'+name)
    actual=parse_pft((r6/'sphere.PFT').read_text(),expected_omega=2*math.pi/WAVELENGTH_UM)
    mie=float(json.loads((r6/'M15R6_CONTROL_PLAN.json').read_text())['sphere_analytic_mie_scattering_um2'])
    if actual!=gate['pft_extracted'] or compare(actual,mie)!=gate['comparison']:
        raise ValueError('R6B_RESULT_NOT_REPRODUCIBLE')
    return {'status':'M15R6B_CHECK_OK','science_gate':'BLOCKED',
            'relative_error_pct':gate['comparison']['relative_error_pct'],
            'mesh_convergence_demonstrated':False}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('audit');p.add_argument('--r6-dir',required=True,type=Path);p.add_argument('--out-dir',required=True,type=Path)
    p=sub.add_parser('check');p.add_argument('--out-dir',required=True,type=Path)
    args=parser.parse_args()
    try:
        result=audit(args.r6_dir,args.out_dir) if args.op=='audit' else check(args.out_dir)
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (ValueError,KeyError,FileNotFoundError,FileExistsError,OSError,ImportError,IndexError) as exc:
        parser.exit(2,f'M15R6B_ERROR: {exc}\n')

if __name__=='__main__':main()
