"""M15-R8B: strictly offline physical-interpretation gate for R8A SCUFF PFT.

Deliberately makes NO scattering-cross-section or novelty claims. Nested surface
PFT rows are *not* interpreted as additive independent-particle observables.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA='m15-r8b-minimal-physics-evidence-v1'
OBSERVABLE='R8A_SC UFF_UNCALIBRATED'.replace(' ','')


def digest(p: Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_pft(path: Path, expected_nm:float, expected_labels:set[str])->dict:
    txt=path.read_text(encoding='utf-8')
    if not re.search(r'(?m)^#\s*3\s+absorbed power\s*\(watts\)',txt):
        raise ValueError('PFT_HEADER_ABSORBED_UNVERIFIED')
    if not re.search(r'(?m)^#\s*4\s+scattered power\s*\(watts\)',txt):
        raise ValueError('PFT_HEADER_SCATTERED_UNVERIFIED')
    rows=[x.split() for x in txt.splitlines() if x.strip() and not x.lstrip().startswith('#')]
    if len(rows)!=len(expected_labels): raise ValueError('UNEXPECTED_PFT_ROW_COUNT')
    result={}
    for row in rows:
        if len(row)!=10: raise ValueError('PFT_EXPECTED_10_COLUMNS')
        label=row[1]
        if label not in expected_labels or label in result:
            raise ValueError('UNEXPECTED_OR_DUPLICATE_SURFACE_'+label)
        try: nums=[float(row[0])]+[float(s) for s in row[2:]]
        except ValueError as exc: raise ValueError('NONNUMERIC_PFT') from exc
        if not all(math.isfinite(x) for x in nums):raise ValueError('NONFINITE_PFT')
        if not math.isclose(nums[0],2*math.pi/(expected_nm/1000),rel_tol=1e-4):
            raise ValueError('PFT_WAVELENGTH_MISMATCH')
        result[label]={'omega_scuff':nums[0],'absorbed_w':nums[1],
                       'scattered_w':nums[2], 'force_nN':nums[3:6],
                       'torque_nN_um':nums[6:9]}
    return result


def parse_geometry(txt:str)->dict:
    """Only claim what the scuff-analyze text explicitly contains."""
    label_material={}
    for row in txt.splitlines():
        m=re.match(r'^\s*[012]\s+\|\s+(EXTERIOR|OuterShell|InnerCore)\s+\|\s+(\S+)',row)
        if m:label_material[m.group(1)]=m.group(2)
    surfaces=re.findall(r'^\s*\*?\s*SURFACE\s+\d+: Label = (\S+)',txt,re.M)
    panels=[int(v) for v in re.findall(r'^\s*(\d+) panels\s*$',txt,re.M)]
    contours=[int(v) for v in re.findall(r'^\s*(\d+) boundary contours\s*$',txt,re.M)]
    if label_material!={'EXTERIOR':'CONST_EPS_1.7689','OuterShell':'FILE_Ag_eps.dat','InnerCore':'FILE_Au_eps.dat'}:
        raise ValueError('GEOMETRY_REGIONS_NOT_EXPECTED')
    if surfaces!=['OuterShell','InnerCore'] or panels!=[2280,994] or contours!=[0,0]:
        raise ValueError('GEOMETRY_BOUNDARIES_NOT_EXPECTED')
    odd=re.search(r'\(after eliminating (\d+) redundant vertices\)',txt)
    redundant=[int(n) for n in re.findall(r'after eliminating (\d+) redundant vertices',txt)]
    return {'regions':label_material,'surfaces':surfaces,'triangles':dict(zip(surfaces,panels)),
            'boundary_contours':contours,'redundant_vertices_reported':redundant,
            'implausible_redundant_vertex_count':any(n>2000 for n in redundant),
            'note':'Topology and region mapping only; no independent triangle-triangle clearance.'}


def derive_state(plan:dict,run:dict,pft:dict,geom:dict|None)->dict:
    if plan.get('schema_version')!='m15r8a-auag-scuff-pilot-v1' or run.get('schema_version')!='m15r8a-auag-scuff-pilot-v1':
        raise ValueError('R8A_SCHEMA_MISMATCH')
    if (plan.get('au_ag_science_gate')!='BLOCKED'
        or run.get('au_ag_science_gate')!='BLOCKED'
        or plan.get('science_backend_qualified') is not False
        or run.get('science_backend_qualified') is not False):
        raise ValueError('R8A_CLAIM_AUTHORITY_INVALID')
    if (run.get('status')!='AU_AG_PFT_GENERATED_UNCALIBRATED' or run.get('exit_code')!=0
        or run.get('timed_out') is not False):
        raise ValueError('R8A_RUN_NOT_SUCCESSFUL')
    if not math.isclose(plan['wavelength_nm'],633.0,abs_tol=1e-9):
        raise ValueError('UNEXPECTED_WAVELENGTH')
    if plan['polarization']!='Ex' or plan['material_assignment']!={'exterior':'CONST_EPS_1.7689','inner':'Au','outer':'Ag'}:
        raise ValueError('R8A_MODEL_NOT_EXPECTED')
    if run.get('command',[])[-6:]!=['--pwPolarization','1','0','0','--PFTFile','nanorod.PFT']:
        raise ValueError('R8A_ILLUMINATION_COMMAND_CHANGED')
    o=pft['OuterShell'];i=pft['InnerCore']
    issues=[]
    if i['absorbed_w'] < 0:
        issues.append('INNER_INTERFACE_NEGATIVE_ABSORBED_POWER_NEEDS_PFT_CONVENTION_REVIEW')
    if o['scattered_w']<=0 or i['scattered_w']<=0:
        issues.append('NONPOSITIVE_SCATTERED_POWER')
    if geom is None:
        issues.append('GEOMETRY_ANALYZE_LOG_NOT_SUPPLIED')
    elif geom['implausible_redundant_vertex_count']:
        issues.append('SCUFF_ANALYZE_IMPLAUSIBLE_REDUNDANT_VERTEX_COUNTER')
    issues.extend(['METAL_MESH_CONVERGENCE_NOT_SHOWN',
                   'NESTED_PFT_ROW_NORMALIZATION_NOT_VALIDATED',
                   'SOURCE_MODEL_HAS_ONLY_SINGLE_POLARIZATION_AND_WAVELENGTH',
                   'SERS_MOLECULAR_OBSERVABLE_NOT_MEASURED'])
    return {
        'r8a_run_status':'SUCCESS_UNCALIBRATED',
        'surface_pft_rows':pft,
        'geometry_verified_if_supplied':geom,
        'observed_issue_codes':issues,
        'numerical_science_feedback_permitted':False,
        'diagnostic_feedback_permitted':True,
        'physical_claims_qualified':False,
        'scientific_novelty_certified':False,
        'science_gate':'BLOCKED',
        'interpretation':'Surface PFT powers are not independent whole-nanorod cross sections; an inner negative absorption value requires investigation and is not proof of negative physical dissipation.',
        'next_minimal_physics_action':'Compare a model-matched second Au/Ag mesh or independent surface PFT method, and confirm the nested surface PFT energy convention before any quantitative hypothesis evidence.',
    }


def audit(r8a_dir:Path,out_dir:Path,geometry_log:Path|None=None)->dict:
    r8a_dir=r8a_dir.resolve();out_dir=out_dir.resolve()
    if out_dir.exists():raise FileExistsError('R8B_OUTPUT_EXISTS_NO_OVERWRITE')
    paths={k:r8a_dir/v for k,v in {'plan':'M15R8A_PLAN.json','run':'M15R8A_RUN.json',
                                    'pft':'nanorod.PFT'}.items()}
    plan=json.loads(paths['plan'].read_text());run=json.loads(paths['run'].read_text())
    if digest(paths['plan'])!=run['input_plan_sha256'] or digest(paths['pft'])!=run['pft_sha256']:
        raise ValueError('R8A_PLAN_OR_PFT_HASH_MISMATCH')
    for name,expected in plan['input_hashes'].items():
        p=r8a_dir/name
        if digest(p)!=expected: raise ValueError('R8A_INPUT_HASH_MISMATCH_'+name)
    if geometry_log is None:
        auto=r8a_dir/'M15R8A_GEOMETRY_AUDIT.log'
        geometry_log=auto if auto.is_file() else None
    geom=parse_geometry(geometry_log.read_text()) if geometry_log is not None else None
    pft=parse_pft(paths['pft'],plan['wavelength_nm'],{'OuterShell','InnerCore'})
    state=derive_state(plan,run,pft,geom)
    src={k:{'path':str(p.resolve()),'sha256':digest(p)} for k,p in paths.items()}
    for name in plan['input_hashes']:
        src['input/'+name]={'path':str((r8a_dir/name).resolve()),'sha256':digest(r8a_dir/name)}
    if geometry_log is not None:
        src['geometry_log']={'path':str(geometry_log.resolve()),'sha256':digest(geometry_log)}
    bundle={'schema_version':SCHEMA,'status':'R8B_DIAGNOSTIC_ONLY','source_files':src,
            'frozen_science_problem':'Can Ex/Ey polarization and Ag shell thickness separate electrodynamic field-selection from changes in molecular orientation when interpreting relative SERS band intensity?',
            'evidence':state}
    out_dir.mkdir(parents=True)
    (out_dir/'M15R8B_EVIDENCE_GATE.json').write_text(json.dumps(bundle,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    report=(f'# M15-R8B — Minimal physical-interpretation gate\n\n'
            f'- Au/Ag 633nm Ex SCUFF execution: SUCCESS, not calibrated\n'
            f'- Exterior medium: relative epsilon 1.7689 (water-like background)\n'
            f'- OuterShell power absorbed: {pft["OuterShell"]["absorbed_w"]:.9g} W\n'
            f'- InnerCore interface power absorbed: {pft["InnerCore"]["absorbed_w"]:.9g} W (negative reported; interpretation unresolved)\n'
            f'- Geometry regions verified: {bool(geom)}\n'
            f'- Numerical hypothesis feedback: **BLOCKED**\n'
            f'- Diagnostic methodological feedback: **ALLOWED WITH UNVALIDATED EVIDENCE TAG**\n\n'
            '**Next: paired G2 feedback responsiveness pilot; novelty and correctness cannot be inferred.**\n')
    (out_dir/'M15R8B_REPORT.md').write_text(report)
    return {'status':bundle['status'],'numerical_science_feedback_permitted':False,
            'diagnostic_feedback_permitted':True,'issue_codes':state['observed_issue_codes'],
            'out_dir':str(out_dir)}


def check(out_dir:Path)->dict:
    out_dir=out_dir.resolve();bundle=json.loads((out_dir/'M15R8B_EVIDENCE_GATE.json').read_text())
    if bundle.get('schema_version')!=SCHEMA or bundle.get('status')!='R8B_DIAGNOSTIC_ONLY':raise ValueError('R8B_SCHEMA_CHANGED')
    for name,obj in bundle['source_files'].items():
        if digest(Path(obj['path']))!=obj['sha256']:
            raise ValueError('R8B_SOURCE_CHANGED_'+name)
    p=json.loads(Path(bundle['source_files']['plan']['path']).read_text())
    r=json.loads(Path(bundle['source_files']['run']['path']).read_text())
    t=parse_pft(Path(bundle['source_files']['pft']['path']),p['wavelength_nm'],{'OuterShell','InnerCore'})
    geo=parse_geometry(Path(bundle['source_files']['geometry_log']['path']).read_text()) if 'geometry_log' in bundle['source_files'] else None
    if derive_state(p,r,t,geo)!=bundle['evidence']:
        raise ValueError('R8B_EVIDENCE_CHANGED')
    if bundle['evidence']['science_gate']!='BLOCKED' or bundle['evidence']['numerical_science_feedback_permitted'] is not False:
        raise ValueError('R8B_AUTHORITY_ELEVATED')
    return {'status':'R8B_CHECK_OK','science_gate':'BLOCKED','diagnostic_only':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('audit');a.add_argument('--r8a-dir',type=Path,required=True);a.add_argument('--out-dir',type=Path,required=True);a.add_argument('--geometry-log',type=Path)
    c=sub.add_parser('check');c.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    try:
        result=audit(args.r8a_dir,args.out_dir,args.geometry_log) if args.cmd=='audit' else check(args.out_dir)
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (OSError,ValueError,KeyError,TypeError,IndexError,json.JSONDecodeError) as exc:
        ap.exit(2,'M15R8B_ERROR: '+str(exc)+'\n')

if __name__=='__main__':main()
