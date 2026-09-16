"""A single JSON result is the source of every readable status/metric report."""
from __future__ import annotations
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from . import LANGUAGES, VERSION
from .core import ContractError

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def render_json_report(json_path: Path, markdown_path: Path) -> None:
    # No recomputation, re-ranking or manually entered second Spearman value here.
    raw=json_path.read_bytes(); data=json.loads(raw)
    text=['# 실행 결과 (JSON에서 자동 생성)', '',
          f'- 결과 파일: `{json_path.name}`',
          f'- 결과 SHA-256: `{hashlib.sha256(raw).hexdigest()}`',
          f'- 상태: **{data.get("status","NOT_RUN")}**', '',
          '아래 수치는 해당 JSON을 그대로 표시한다. 서로 다른 cell의 수치를 같은 값으로 요약하지 않는다.',
          '', '```json',json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),'```','']
    markdown_path.parent.mkdir(parents=True,exist_ok=True)
    markdown_path.write_text('\n'.join(text),encoding='utf-8')

def ranks(xs):
    order=sorted(range(len(xs)),key=lambda i:xs[i]); out=[0.]*len(xs); i=0
    while i<len(xs):
        j=i+1
        while j<len(xs) and xs[order[j]]==xs[order[i]]: j+=1
        avg=(i+1+j)/2
        for k in range(i,j): out[order[k]]=avg
        i=j
    return out

def spearman(xs,ys):
    if len(xs)!=len(ys) or len(xs)<2: raise ContractError('INVALID_CORRELATION_LENGTH')
    if any(not math.isfinite(v) for v in list(xs)+list(ys)): raise ContractError('NONFINITE_CORRELATION_INPUT')
    x,y=ranks(xs),ranks(ys); mx,my=statistics.mean(x),statistics.mean(y)
    cov=math.fsum((a-mx)*(b-my) for a,b in zip(x,y))
    den=math.sqrt(math.fsum((a-mx)**2 for a in x)*math.fsum((b-my)**2 for b in y))
    return cov/den if den>0 else None

def measurement(rows: list[dict], expected_concepts: list[str], metadata: dict, gates: dict) -> dict:
    """One root/T3/KO/RD/ANY/dev, ONE identifiable cohort, two dev wrappers.

    Caller must pre-select this documented cell. Mixed cells and duplicate IDs fail.
    expected_concepts comes from the pre-treatment data freeze, not model outputs.
    """
    if len(expected_concepts)<2 or len(set(expected_concepts))!=len(expected_concepts):
        raise ContractError('INVALID_EXPECTED_COHORT')
    fixed={'stage':'T3','input_language':'ko','format':'RD','mode':'ANY','split':'dev'}
    maps={'dev1':{},'dev2':{}}
    for r in rows:
        if any(r.get(k)!=v for k,v in fixed.items()): raise ContractError('MIXED_MEASUREMENT_CELL')
        if str(r.get('root_id'))!=str(metadata['root_id']): raise ContractError('MIXED_ROOTS')
        if r.get('checkpoint_sha256')!=metadata.get('checkpoint_sha256'): raise ContractError('MIXED_CHECKPOINTS')
        if r.get('format_version')!=VERSION: raise ContractError('ROW_VERSION_MISMATCH')
        w=r.get('wrapper'); c=r.get('concept_id')
        if w not in maps or c not in expected_concepts: raise ContractError('UNEXPECTED_MEASUREMENT_ROW')
        if c in maps[w]: raise ContractError('DUPLICATE_MEASUREMENT_ROW')
        q=r.get('Q_language')
        if q is None or set(q)!=set(LANGUAGES): raise ContractError('NONIDENTIFIABLE_ROW_IN_PRIMARY_COHORT')
        if any(not math.isfinite(v) or not 0<=v<=1 for v in q.values()) or not math.isclose(sum(q.values()),1.,abs_tol=1e-8):
            raise ContractError('INVALID_Q')
        if not math.isfinite(r['Z']) or not 0<=r['Z']<=1+1e-10: raise ContractError('INVALID_Z')
        maps[w][c]=r
    for w in maps:
        if set(maps[w])!=set(expected_concepts): raise ContractError('MISSING_MEASUREMENT_ROWS')
    ids=sorted(expected_concepts); cells={}; reasons=[]
    for l in LANGUAGES:
        x=[maps['dev1'][c]['Q_language'][l] for c in ids]
        y=[maps['dev2'][c]['Q_language'][l] for c in ids]
        rho=spearman(x,y); ad=statistics.mean(abs(a-b) for a,b in zip(x,y))
        cells[l]={'spearman':rho,'mean_abs_wrapper_difference':ad,
                  'sd_dev1':statistics.stdev(x),'sd_dev2':statistics.stdev(y),
                  'range_dev1':[min(x),max(x)],'range_dev2':[min(y),max(y)],
                  'unique_values_dev1':len(set(x)),'unique_values_dev2':len(set(y))}
        if rho is None: reasons.append(l+':CORRELATION_UNDEFINED')
        elif rho<gates['rho_min']: reasons.append(l+':RANK_WARNING')
        if ad>gates['wrapper_abs_max']: reasons.append(l+':ABSOLUTE_DIFFERENCE_WARNING')
    z={}
    for w in maps:
        vals=sorted(maps[w][c]['Z'] for c in ids)
        fraction=sum(v<gates['z_threshold'] for v in vals)/len(vals)
        z[w]={'min':vals[0],'median':statistics.median(vals),'max':vals[-1],
              'low_Z_fraction':fraction}
        if fraction>=gates['low_z_fraction_max']: reasons.append(w+':LOW_Z_WARNING')
    return {'version':VERSION,'status':'PASS' if not reasons else 'BLOCKED_MEASUREMENT',
            'scope':'one root, T3, KO input, RD, ANY, dev1 vs dev2, pre-fixed identifiable cohort',
            'metadata':metadata,'n_concepts':len(ids),'metrics_by_language':cells,
            'Z_by_wrapper':z,'gates':gates,'reasons':reasons,
            'interpretation':'Operational warnings, not universal validity cutoffs. Passing is NOT main-study authorization.'}

def readiness(language_accuracy: dict, active: list[str], minimum: float=.90) -> dict:
    if not active or active!=list(LANGUAGES[:len(active)]) or set(language_accuracy)!=set(active):
        raise ContractError('READINESS_REQUIRES_ACTIVE_LANGUAGE_CELLS')
    if any(not math.isfinite(v) or not 0<=v<=1 for v in language_accuracy.values()): raise ContractError('INVALID_ACCURACY')
    failed=[l for l,v in language_accuracy.items() if v<minimum]
    return {'status':'PASS' if not failed else 'BLOCKED_READINESS','cells':language_accuracy,
            'minimum':minimum,'failed_languages':failed,
            'scope':'KO-input RD dev; per requested language; no diagonal/overall averaging'}
