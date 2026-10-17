"""Synthetic fixtures are for runner validation ONLY; never supplied as scientific targets."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[2]))
from scripts.discovery import run_m26_cross_intensity_holdout as m


def fixture_csv(tmp_path):
    p=tmp_path/'SYNTHETIC_NOT_PUBLISHED.csv'
    with p.open('w') as f:
        wr=csv.writer(f);wr.writerow(['P','time','response'])
        import math
        for P in [100,300,500,700,1100]:
            for i in range(41):
                t=i*2.5
                A=1.1+0.0018*P;tau=15+0.009*P
                wr.writerow([P,t,A*(1-math.exp(-t/tau))])
    return p


def make(tmp_path):
    data=fixture_csv(tmp_path);out=tmp_path/'RUN'
    m.prepare(out,data,'CSV','time','P','response',model='TEST-MODEL')
    return out,data


def good_response():
    return {'research_question':'Can a saturating kinetic process predict a held-out laser-power response from neighboring powers?',
      'model_family':'EXP_SAT','parameters':{'y0':0.0,'A':2.0,'tau_min':20.0},
      'mechanistic_rationale':'A first-order charging and relaxation process can create an exponential saturation of the optical response.',
      'training_evidence_used':'Different illumination powers show monotone rise and saturation, with both amplitude and rate changing across powers.',
      'cross_intensity_inference':'Interpolate fit parameters between the neighboring illumination powers; do not assume constant absorption.',
      'falsification_conditions':'Late-time curvature and early-time slope should be consistent with the same parameterization, not merely a final value.',
      'limitations_and_alternatives':'Optical index drift, depletion, structural motion and temperature-dependent leakage can imitate saturation curves.',
      'source_transfer_warning':'No measurement in solvated gold nanorods establishes a quantitative charging or Raman-blinking effect in solid PVA.'}


def test_prepare_sealed_holdout_no_leak(tmp_path):
    out,data=make(tmp_path)
    ins=m.inspect(data)
    assert ins['tables'][0]['header']==['P','time','response']
    pr=m.preview(out)
    assert pr['private_target_not_in_request']
    public=m.read(out/'public_input.json');private=m.read(out/'private_holdout.json')
    assert str(private['holdout_intensity_mW']) not in public['training_measured_traces']
    assert len(private['holdout_trace'])==41
    assert len(public['training_intensities_mW'])==4
    assert pr['unseen_power']>min(public['training_intensities_mW'])
    assert pr['unseen_power']<max(public['training_intensities_mW'])
    assert m.verify(out)['prediction']=='NOT_STARTED'
    assert 'holdout_trace' not in json.dumps(m.request(out))


def test_freeze_score_and_baselines(tmp_path,monkeypatch):
    out,data=make(tmp_path)
    # backend simulation, no external charge
    result=m.propose(out,backend=lambda req:(json.dumps(good_response()),{'total_tokens':0},'mock'))
    assert result['verification']['prediction']=='READY'
    assert m.verify(out)['evaluation']=='SEALED'
    with pytest.raises(ValueError):m.preview(out)
    assert m.reveal(out)['status']=='M26_HOLDOUT_REVEALED_NO_PAID_CALL'
    e=m.read(out/'evaluation.json')
    assert e['metrics']['FAKG']['rmse']>=0
    assert len(e['forecast_time_grid_min'])==12
    assert len(e['baseline_model_forecasts'])==3
    assert m.verify(out)['evaluation']=='REVEALED'
    with pytest.raises(ValueError):m.reveal(out)


def test_data_immutable(tmp_path):
    out,data=make(tmp_path)
    data.write_text(data.read_text()+'\n')
    with pytest.raises(ValueError,match='SOURCE_FILE_MUTATED'):m.verify(out)


def test_reject_bad_response(tmp_path):
    bad=good_response();bad['parameters']['tau_min']=-5
    with pytest.raises(ValueError,match='TAU'):m._safe_model(bad)
    bad=good_response();bad['model_family']='BAD'
    with pytest.raises(ValueError):m._safe_model(bad)


def test_not_enough_powers(tmp_path):
    p=tmp_path/'less.csv'
    with p.open('w') as f:
        wr=csv.writer(f);wr.writerow(['P','t','y'])
        for power in (200,400,600):
            for t in range(10):wr.writerow([power,t,t/(1+t)])
    with pytest.raises(ValueError,match='FOUR_DISTINCT'):m.prepare(tmp_path/'NEW',p,'CSV','t','P','y',model='MOCK')


def test_reject_duplicate_source(tmp_path):
    p=fixture_csv(tmp_path)
    with p.open('a') as f:f.write('100,0,5\n')
    with pytest.raises(ValueError,match='DUPLICATE_POWER_TIME'):m.prepare(tmp_path/'NEW',p,'CSV','time','P','response',model='MOCK')


def test_wide_csv(tmp_path):
    p=tmp_path/'wide.csv'
    with p.open('w') as f:
        wr=csv.writer(f);wr.writerow(['t','P100','P300','P500','P700'])
        for t in range(12):wr.writerow([t,t/10,t/8,t/6,t/4])
    out=tmp_path/'WIDE'
    m.prepare(out,p,'CSV','t',wide_spec='100:P100,300:P300,500:P500,700:P700',model='MOCK')
    assert m.verify(out)['prediction']=='NOT_STARTED'


def test_xlsx_reader(tmp_path):
    # Create a minimal valid spreadsheet package without relying on spreadsheet APIs.
    import zipfile
    p=tmp_path/'simple.xlsx'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Fig4" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>t</t></is></c><c r="B1" t="inlineStr"><is><t>P100</t></is></c></row><row r="2"><c r="A2"><v>0</v></c><c r="B2"><v>1.5</v></c></row></sheetData></worksheet>')
    d=m.inspect(p)
    assert d['tables'][0]['header']==['t','P100']
    assert d['tables'][0]['preview'][0]==['0','1.5']


def test_pairs_csv(tmp_path):
    p=tmp_path/'pairs.csv'
    with p.open('w') as f:
        wr=csv.writer(f);wr.writerow(['tA','yA','tB','yB','tC','yC','tD','yD'])
        for t in range(12):wr.writerow([t,t/10,t+.1,t/8,t+.2,t/6,t+.3,t/4])
    out=tmp_path/'PAIRS'
    m.prepare(out,p,'CSV',pairs_spec='100:tA:yA,300:tB:yB,500:tC:yC,700:tD:yD',model='MOCK')
    assert m.verify(out)['prediction']=='NOT_STARTED'
