#!/usr/bin/env python3
"""M15 pilot minimal Meep *engineering sanity* job generator (no solver invocation).

Creates two independent, prespecified physical comparisons (Ex vs Ey for two
Au/Ag core-shell choices) from the *existing* FDTD contracts. This only makes
plans; not calculations, scientific novelty, physical convergence, or SERS data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

VERSION='m15-meep-execution-sanity-bootstrap-v1'
JOBS='M15_SOLVER_JOBS.json'
NUM='M15_NUMERICAL_PLANS.json'
BOOK='M15_PREDECLARED_CASEBOOK.json'
MAN='M15_BOOTSTRAP_MANIFEST.json'
PAIR_CONFIG=[
    {'tag':'aucore_agshell','core_material':'Au','shell_material':'Ag','hypothesis_id':'m15_fixed_problem_aucore_agshell',
     'research_question':'For an isolated Au-core Ag-shell nanorod in water, how does polarization change its classical electromagnetic scattering spectrum, and which independent optical measurement can discriminate competing geometric explanations?',
     'initial_hypothesis':'For a fixed Au-core Ag-shell nanorod geometry, longitudinal and transverse illumination yield different wavelength-resolved scattering cross sections and resonance features, conditional on the model assumptions.',
     'strong_null':'The observed spectral differences, if any, follow from ordinary Maxwell scattering of the same isolated Au-core Ag-shell geometry; they do not establish new molecular Raman physics.'},
    {'tag':'agcore_aushell','core_material':'Ag','shell_material':'Au','hypothesis_id':'m15_fixed_problem_agcore_aushell',
     'research_question':'For an isolated Ag-core Au-shell nanorod in water, how does polarization change its classical electromagnetic scattering spectrum, and which independent optical measurement can discriminate competing material-loss explanations?',
     'initial_hypothesis':'For a fixed Ag-core Au-shell nanorod geometry, longitudinal and transverse illumination yield different wavelength-resolved scattering cross sections and resonance features, conditional on the model assumptions.',
     'strong_null':'The calculated polarization response is explained by established Maxwell electrodynamics and the assumed dispersive material models rather than a new chemical or SERS-specific mechanism.'},
]
POLARIZATIONS=(('parallel_to_nanorod_long_axis','Ex'),('perpendicular_to_nanorod_long_axis','Ey'))

def canon(obj):return (json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
def sha(b):return hashlib.sha256(b).hexdigest()
def freeze(path,obj):
    path=Path(path);payload=canon(obj)
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('DIFFERENT_FROZEN_FILE '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(payload)

def plan_data(resolution=120.0, pml=0.14, padding=0.17, length=160.0, diameter=50.0, shell=20.0, max_voxels=1_500_000):
    for k,v in locals().copy().items():
        if k=='max_voxels':continue
        if not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError('invalid '+k)
    if length<=diameter:raise ValueError('nanorod length must exceed diameter')
    if max_voxels<1000:raise ValueError('invalid voxel guard')
    cell_x=(length+2*shell)/1000+2*(pml+padding)
    cell_y=(diameter+2*shell)/1000+2*(pml+padding)
    voxels=math.ceil(cell_x*resolution)*math.ceil(cell_y*resolution)**2
    if voxels>max_voxels:raise ValueError(f'VOXEL_GUARD {voxels} > {max_voxels}; use smaller sanity resolution or review local resources')
    grid_nm=1000/resolution
    return {'resolution_px_per_um':resolution,'pml_thickness_um':pml,'padding_um':padding,
            'nanorod_length_nm':length,'nanorod_diameter_nm':diameter,'shell_thickness_nm':shell,
            'cell_x_um':cell_x,'cell_y_um':cell_y,'cell_z_um':cell_y,'grid_spacing_nm':grid_nm,
            'estimated_grid_voxels':voxels,'max_voxels':max_voxels,
            'minimum_feature_cells':shell/grid_nm,'target_cells_per_feature':8.0,
            'wavelength_min_nm':500.0,'wavelength_max_nm':850.0,'spectrum_frequency_points':41,
            'sanity_only':True,'numerical_convergence_certified':False,'positive_control_certified':False,
            'new_scientific_claim_certified':False,'resource_usage_not_benchmarked':True}

def create(out, resolution, max_voxels):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Output must be new for preregistration: '+str(out))
    config=plan_data(resolution=resolution,max_voxels=max_voxels)
    # Import existing typed contracts only when generating real job files.
    from domains.sers.solver_execution_contracts import SERSFDTDSolverJob, SERSFDTDSolverJobBundle
    from domains.sers.numerical_policy_contracts import SERSFDTDNumericalJobPlan, SERSFDTDNumericalPlanBundle
    from domains.sers.meep_execution import SERSMeepExecutionPlanner
    jobs=[];nplans=[];cases=[]
    bundle_id='m15_sanity_solver_jobs_v1';policy_id='m15_sanity_numerical_policy_v1'
    for pair in PAIR_CONFIG:
        req=[]
        for polarization,component in POLARIZATIONS:
            suffix='long' if component=='Ex' else 'trans'
            job_id='m15_job_'+pair['tag']+'_'+suffix
            bindings=[{'wavelength_nm':v,'source_case_id':job_id+'_wave_'+str(int(v))} for v in (500.0,650.0,850.0)]
            j=SERSFDTDSolverJob(job_id=job_id,design_id='m15_predeclared_'+pair['tag'],
              hypothesis_id=pair['hypothesis_id'],source_spec_id='m15_sanity_spec_'+pair['tag'],
              geometry_candidate_id='m15_geometry_'+pair['tag'],geometry_family='core_shell_nanorod',
              core_material=pair['core_material'],shell_material=pair['shell_material'],
              nanorod_length_nm=config['nanorod_length_nm'],nanorod_diameter_nm=config['nanorod_diameter_nm'],
              shell_thickness_nm=config['shell_thickness_nm'],polarization=polarization,
              surrounding_medium='water',wavelength_case_bindings=bindings)
            jobs.append(j)
            nplans.append(SERSFDTDNumericalJobPlan(
                numerical_plan_id='m15_num_'+pair['tag']+'_'+suffix,
                policy_id=policy_id,source_solver_job_id=job_id,source_solver_job_bundle_id=bundle_id,
                hypothesis_id=pair['hypothesis_id'],geometry_candidate_id='m15_geometry_'+pair['tag'],
                intended_use='execution_sanity',resolution_px_per_um=config['resolution_px_per_um'],
                outer_length_nm=config['nanorod_length_nm']+2*config['shell_thickness_nm'],
                outer_diameter_nm=config['nanorod_diameter_nm']+2*config['shell_thickness_nm'],
                minimum_geometric_feature_nm=config['shell_thickness_nm'],
                grid_spacing_nm=config['grid_spacing_nm'],minimum_feature_cells=config['minimum_feature_cells'],
                cells_per_min_feature_target=8.0,target_resolution_px_per_um=8*1000/config['shell_thickness_nm'],
                resolution_status='underresolved',evidence_eligibility='execution_sanity_only',
                pml_thickness_um=config['pml_thickness_um'],padding_um=config['padding_um'],
                cell_x_um=config['cell_x_um'],cell_y_um=config['cell_y_um'],cell_z_um=config['cell_z_um'],
                wavelength_min_nm=config['wavelength_min_nm'],wavelength_max_nm=config['wavelength_max_nm'],
                scientific_wavelength_min_nm=config['wavelength_min_nm'],scientific_wavelength_max_nm=config['wavelength_max_nm'],
                spectrum_frequency_points=config['spectrum_frequency_points'],
                resource_estimate=None))
        cases.append({'case_id':'m15_case_'+pair['tag'],'research_question':pair['research_question'],
          'initial_hypothesis':pair['initial_hypothesis'],'strong_null':pair['strong_null'],
          'control_request_id':None,'intervention_request_id':None,'variable_under_test':'polarization'})
    job_bundle=SERSFDTDSolverJobBundle(bundle_id=bundle_id,source_simulation_case_bundle_id='m15_engineering_predeclared_cases',
        jobs=jobs,solver_job_count=len(jobs),scientific_case_count=sum(len(j.wavelength_case_bindings) for j in jobs),
        coalesced_scientific_case_count=sum(len(j.wavelength_case_bindings) for j in jobs)-len(jobs))
    num_bundle=SERSFDTDNumericalPlanBundle(bundle_id='m15_sanity_num_bundle_v1',source_solver_job_bundle_id=bundle_id,
        policy_id=policy_id,plans=nplans,solver_job_count=len(nplans))
    reqs=SERSMeepExecutionPlanner().build_requests(job_bundle,num_bundle,max_requests=10)
    if reqs.request_count!=4:raise ValueError('expected four Meep execution requests')
    by_job={r.source_solver_job_id:r for r in reqs.requests}
    for c,pair in zip(cases,PAIR_CONFIG):
        c['control_request_id']=by_job['m15_job_'+pair['tag']+'_long'].request_id
        c['intervention_request_id']=by_job['m15_job_'+pair['tag']+'_trans'].request_id
    # The prospective casebook is fixed BEFORE any Meep results are obtained.
    out.mkdir(parents=True)
    freeze(out/JOBS,job_bundle.model_dump(mode='json'))
    freeze(out/NUM,num_bundle.model_dump(mode='json'))
    freeze(out/BOOK,{'cases':cases})
    freeze(out/MAN,{'schema':VERSION,'status':'M15_PREDECLARED_SANITY_JOBS_UNEXECUTED',
        'source_is_existing_typed_solver_contracts':True,'numerics':config,
        'jobs_sha256':sha((out/JOBS).read_bytes()),'numerical_sha256':sha((out/NUM).read_bytes()),
        'casebook_sha256':sha((out/BOOK).read_bytes()),'request_ids':[r.request_id for r in reqs.requests],
        'job_ids':[j.job_id for j in jobs],
        'reference_result_convergence_certified':False,
        'actual_meep_executed':False,'model_calls':0,'new_science_certified':False})
    return {'status':'M15_PREDECLARED_UNEXECUTED_MEEP_SANITY_PILOT',
            'pilot_jobs':len(jobs),'independent_case_pairs':len(cases),
            'grid_voxels_estimate':config['estimated_grid_voxels'],
            'shell_cells':config['minimum_feature_cells'],
            'plan_dir':str(out),'run_order_job_ids':[j.job_id for j in jobs],
            'science_status':'ENGINEERING_SANITY_ONLY_NOT_CONVERGED'}

def check(out):
    root=Path(out);m=json.loads((root/MAN).read_bytes())
    for name,key in ((JOBS,'jobs_sha256'),(NUM,'numerical_sha256'),(BOOK,'casebook_sha256')):
        if sha((root/name).read_bytes())!=m[key]:raise ValueError('BOOTSTRAP_SOURCE_MUTATED '+name)
    from domains.sers.solver_execution_contracts import SERSFDTDSolverJobBundle
    from domains.sers.numerical_policy_contracts import SERSFDTDNumericalPlanBundle
    from domains.sers.meep_execution import SERSMeepExecutionPlanner
    jobs=SERSFDTDSolverJobBundle.model_validate_json((root/JOBS).read_text())
    numeric=SERSFDTDNumericalPlanBundle.model_validate_json((root/NUM).read_text())
    bundle=SERSMeepExecutionPlanner().build_requests(jobs,numeric,max_requests=10)
    if [r.request_id for r in bundle.requests]!=m['request_ids']:raise ValueError('REQUESTS_CHANGED')
    if any(r.intended_use!='execution_sanity' or r.evidence_eligibility!='execution_sanity_only' for r in bundle.requests):raise ValueError('NON_SANITY_INPUT')
    return {'status':'M15_BOOTSTRAP_TYPED_INPUTS_VERIFIED','requests':bundle.request_count,
            'original_inputs_sha_valid':True,'new_model_calls':0,'meep_executed':False}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('create');p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--resolution',type=float,default=120.0)
    p.add_argument('--max-voxels',type=int,default=1_500_000)
    p=sub.add_parser('check');p.add_argument('--out-dir',type=Path,required=True)
    a=parser.parse_args(argv)
    res=create(a.out_dir,a.resolution,a.max_voxels) if a.cmd=='create' else check(a.out_dir)
    print(json.dumps(res,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
