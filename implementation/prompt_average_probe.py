"""Frozen eight-template, high-Z split-half probability exploration; no training."""
import argparse
import itertools
import math
import statistics
from pathlib import Path
from . import model40_remeasurement as native
from .qwen_robustness import PARENT, PARENT_SHA, LANGS
from .src.artifacts import read_verified_json, publish_json_once, sha256_file
from .src.contracts import WORK_ROOT, PROJECT_ROOT, ContractViolation
from .src.semantic_experiment_freeze import _hash
from .src.report import spearman

NAME='prompt_average_probe_v1'
ROOT=WORK_ROOT/'runs'/NAME
CONTRACT=WORK_ROOT/'freeze'/NAME/'contract.json'
MANIFEST=WORK_ROOT/'integrity'/NAME/'implementation_manifest.json'
CPU=WORK_ROOT/'reports'/('cpu_'+NAME)/'cpu_checks.json'
PLAN=PROJECT_ROOT/'implementation/PROMPT_AVERAGE_PROBE_V1.md'
TEMPLATES={f'w{1+4*i+j}':header+': {gloss}\n'+answer+':'
    for i,header in enumerate(('뜻풀이','정의'))
    for j,answer in enumerate(('표현','한 줄 답','답','응답'))}
PRIMARY=('w1','w2','w7','w8')


def records():
    parent=read_verified_json(PARENT,expected_sha256=PARENT_SHA)
    return [{'record_id':c['concept_id']+'|ko|'+w+'|ANY','concept_id':c['concept_id'],
        'input_language':'ko','wrapper':w,'mode':'ANY','answers':c['answers'],
        'prefix':template.format(gloss=c['glosses']['ko'])}
        for w,template in TEMPLATES.items() for c in sorted(parent['concepts'],key=lambda c:c['concept_id'])]


def split_metrics(group,ids,left):
    right=tuple(w for w in TEMPLATES if w not in left)
    means={side:{i:{l:math.fsum(group[w][i]['Q_membership'][l] for w in wrappers)/len(wrappers)
        for l in LANGS} for i in ids} for side,wrappers in [('left',left),('right',right)]}
    languages={}
    for l in LANGS:
        a=[means['left'][i][l] for i in ids];b=[means['right'][i][l] for i in ids]
        rho=spearman(a,b);mae=math.fsum(abs(x-y) for x,y in zip(a,b))/len(ids)
        languages[l]={'spearman':rho,'mean_abs_difference':mae,
            'left_range':[min(a),max(a)],'right_range':[min(b),max(b)],
            'stability_thresholds_pass':rho is not None and rho>=.6 and mae<=.15}
    tv=[math.fsum(abs(means['left'][i][l]-means['right'][i][l]) for l in LANGS)/2 for i in ids]
    return {'left':list(left),'right':list(right),'languages':languages,
        'mean_TV':statistics.mean(tv),'median_TV':statistics.median(tv),'TV_gt_015_count':sum(v>.15 for v in tv),
        'all_languages_stability_pass':all(x['stability_thresholds_pass'] for x in languages.values())}


def summarize(rows):
    group={w:{} for w in TEMPLATES}
    for r in rows:
        w=r['wrapper'];i=r['concept_id'];p=r['probability']
        if r['input_language']!='ko' or r.get('mode')!='ANY' or w not in group or i in group[w]:
            raise ContractViolation('INVALID_OR_DUPLICATE_ROW')
        if p['Q_membership'] is None or set(p['Q_membership'])!=set(LANGS):
            raise ContractViolation('UNDEFINED_OR_SHARED_Q')
        vals=list(p['Q_membership'].values())
        if not all(math.isfinite(v) and 0<=v<=1 for v in vals) or not math.isclose(sum(vals),1,abs_tol=1e-12):
            raise ContractViolation('INVALID_Q')
        if not math.isfinite(p['Z']) or not 0<=p['Z']<=1+1e-6:raise ContractViolation('INVALID_Z')
        group[w][i]=p
    ids=sorted(group['w1'])
    if len(ids)!=60 or any(set(c)!=set(ids) for c in group.values()):
        raise ContractViolation('INCOMPLETE_CELLS')
    z={}
    for w,cell in group.items():
        vals=[cell[i]['Z'] for i in ids];median=statistics.median(vals);low=sum(v<.05 for v in vals)/60
        z[w]={'median':median,'mean':statistics.mean(vals),'minimum':min(vals),
            'fraction_ge_09':sum(v>=.9 for v in vals)/60,'low_Z_fraction':low,
            'high_Z_requirement_pass':median>=.9 and low<.1}
    all_splits=[split_metrics(group,ids,('w1',)+rest) for rest in itertools.combinations(tuple(TEMPLATES)[1:],3)]
    primary=next(s for s in all_splits if set(s['left'])==set(PRIMARY))
    aggregates={}
    for l in LANGS:
        rhos=[s['languages'][l]['spearman'] for s in all_splits]
        finite=[r for r in rhos if r is not None]
        aggregates[l]={'undefined_count':len(rhos)-len(finite),
            'rho_min':min(finite) if finite else None,'rho_median':statistics.median(finite) if finite else None,
            'rho_max':max(finite) if finite else None,'rho_ge_07_count':sum(r is not None and r>=.7 for r in rhos),
            'stability_pass_count':sum(s['languages'][l]['stability_thresholds_pass'] for s in all_splits)}
    high_z=all(c['high_Z_requirement_pass'] for c in z.values())
    return {'n_concepts':60,'templates':TEMPLATES,'Z_by_wrapper':z,'all_wrappers_high_Z':high_z,
        'primary_split':primary,'all_35_splits':all_splits,'split_summary_by_language':aggregates,
        'all_languages_pass_split_count':sum(s['all_languages_stability_pass'] for s in all_splits),
        'interpretation_status':('Z_NOT_MAINTAINED' if not high_z else
            'PRIMARY_MEAN_STABILITY_SUPPORTED' if primary['all_languages_stability_pass'] else 'PRIMARY_MEAN_STABILITY_NOT_SUPPORTED'),
        'all_splits_are_dependent':True,'existing_pilot_gate_replaced':False,
        'Q_average_definition':'equal-weight mean of per-wrapper normalized Q; not pooled p/Z',
        'all_concept_mean_Q':{i:{l:math.fsum(group[w][i]['Q_membership'][l] for w in TEMPLATES)/8 for l in LANGS} for i in ids}}


def prepare():
    from .src.build_tokenizer import load_verified_tokenizer
    from .src.score import build_continuation_events
    source=read_verified_json(native.SOURCE,expected_sha256=native.SOURCE_SHA)
    checkpoint=[e['data']['checkpoint'] for e in source['events'] if e['kind']=='CHECKPOINT' and e['data']['reason']=='FIXED_ENDPOINT'][-1]
    parent=read_verified_json(PARENT,expected_sha256=PARENT_SHA)
    ref=parent['artifacts']['evaluation_plan'];plan=read_verified_json(Path(ref['path']),expected_sha256=ref['sha256'])['prompt_plan']
    if TEMPLATES['w1']!=plan['templates']['dev1']['ko']['ANY'] or TEMPLATES['w6']!=plan['templates']['dev2']['ko']['ANY']:
        raise ContractViolation('ORIGINAL_TEMPLATES_CHANGED')
    tokenizer,_=load_verified_tokenizer(Path(parent['artifacts']['tokenizer']['path']))
    rows=records();lengths=[]
    for r in rows:
        events=build_continuation_events(tokenizer,r['prefix'],r['answers'],context_length=256)
        if len(events)!=4:raise ContractViolation('NONIDENTIFIABLE_COHORT')
        lengths.extend(e['full_token_count'] for e in events)
    return publish_json_once(CONTRACT,{'schema':NAME,'checkpoint':checkpoint,'source_sha256':native.SOURCE_SHA,
        'parent_sha256':PARENT_SHA,'records':rows,'records_sha256':_hash(rows),'templates':TEMPLATES,
        'primary_left':list(PRIMARY),'plan_sha256':sha256_file(PLAN),'maximum_sequence_tokens':max(lengths),
        'context_length':256,'updates':0,'main_enabled':False})


def run():
    return native.run(root=ROOT,contract_path=CONTRACT,manifest_path=MANIFEST,cpu_path=CPU,
        plan_path=PLAN,record_builder=records,summarizer=summarize)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
    if args.action=='prepare':print(prepare())
    else:raise SystemExit(run())
