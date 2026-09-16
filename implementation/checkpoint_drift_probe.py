"""Frozen read-only checkpoint trajectory and worst-split Q measurement audit."""
import argparse
import itertools
import math
import statistics
from pathlib import Path
from . import prompt_average_probe as prompts
from .qwen_robustness import PARENT, PARENT_SHA, LANGS
from .src.artifacts import read_verified_json, publish_json_once, sha256_file
from .src.contracts import WORK_ROOT, PROJECT_ROOT, ContractViolation
from .src.semantic_experiment_freeze import _hash
from .src.report import spearman

NAME='checkpoint_drift_probe_v1'
RUN=WORK_ROOT/'runs'/NAME
CONTRACT=WORK_ROOT/'freeze'/NAME/'contract.json'
MANIFEST=WORK_ROOT/'integrity'/NAME/'implementation_manifest.json'
CPU=WORK_ROOT/'reports'/('cpu_'+NAME)/'cpu_checks.json'
PLAN=PROJECT_ROOT/'implementation/protocol/STRATIFIED_Q_MEASUREMENT_V2_KO.md'
CONFIG=PROJECT_ROOT/'implementation/config/stratified_q_measurement_v2.json'
SOURCES=[('T0','pilot_jm02_4e30ec2','T0'),('T1','pilot_jm02_4e30ec2','T1'),
 ('T2','pilot_jm02_4e30ec2','T2'),('T3','pilot_jm02_4e30ec2','T3'),
 ('T3_3360','accuracy_probe_47dc5c6','prompt_mix'),('T3_3720','accuracy_refinement_fdfa878','lower_lr'),
 ('T3_4080','accuracy_selected_9e209c3','lower_lr')]


def worst_gate(summary):
    cfg=read_verified_json(CONFIG);splits=summary['all_35_splits']
    expected_templates={f'w{1+4*i+j}':h+': {gloss}\n'+a+':'
        for i,h in enumerate(cfg['factors']['header']) for j,a in enumerate(cfg['factors']['answer_label'])}
    expected_splits={frozenset(('w1',)+s) for s in itertools.combinations(tuple(prompts.TEMPLATES)[1:],3)}
    if expected_templates!=prompts.TEMPLATES:raise ContractViolation('UNBALANCED_FACTOR_DESIGN')
    if len(splits)!=35 or {frozenset(s['left']) for s in splits}!=expected_splits or any(
            set(s['right'])!=set(prompts.TEMPLATES)-set(s['left']) for s in splits):
        raise ContractViolation('WORST_GATE_REQUIRES_ALL_35_SPLITS')
    language={};reasons=[]
    z_ok=all(v['median']>=cfg['median_Z_min'] and v['low_Z_fraction']<cfg['low_Z_fraction_strict_max']
             for v in summary['Z_by_wrapper'].values())
    if not z_ok:reasons.append('HIGH_Z_REQUIREMENT_FAILED')
    for l in LANGS:
        vals=[s['languages'][l] for s in splits];rhos=[v['spearman'] for v in vals];defined=[v for v in rhos if v is not None]
        minimum=min(defined) if defined else None;maximum=max(v['mean_abs_difference'] for v in vals)
        good=len(defined)==35 and minimum>=cfg['rho_min'] and maximum<=cfg['mean_absolute_difference_max']
        if not good:reasons.append(l+':WORST_SPLIT_FAILED')
        language[l]={'min_rho':minimum,'undefined_splits':35-len(defined),'max_mean_abs_difference':maximum,
            'min_rho_left': [s['left'] for s in splits if s['languages'][l]['spearman']==minimum],
            'max_abs_left':[s['left'] for s in splits if s['languages'][l]['mean_abs_difference']==maximum],
            'pass':good}
    return {'status':'PASS' if not reasons else 'BLOCKED_HIGH_Z' if not z_ok else 'BLOCKED_WORST_SPLIT_MEASUREMENT',
        'reasons':reasons,'languages':language,'all_wrappers_high_Z':z_ok,
        'rule_sha256':sha256_file(CONFIG),'all_35_required':True,'future_unseen_prompt_guarantee':False}


def indexed(rows):
    # Reuse full-cohort validation, including exact IDs per wrapper and Q normalization.
    prompts.summarize(rows)
    return {w:{r['concept_id']:r['probability'] for r in rows if r['wrapper']==w} for w in prompts.TEMPLATES}


def pair_drift(before_rows,after_rows,before_summary,after_summary):
    before=indexed(before_rows);after=indexed(after_rows);ids=sorted(before['w1'])
    if ids!=sorted(after['w1']):raise ContractViolation('DRIFT_CONCEPT_MISMATCH')
    delta={w:{i:{l:after[w][i]['Q_membership'][l]-before[w][i]['Q_membership'][l] for l in LANGS}
              for i in ids} for w in before}
    def avg(ws,i,l):return math.fsum(delta[w][i][l] for w in ws)/len(ws)
    mean_delta={i:{l:avg(tuple(before),i,l) for l in LANGS} for i in ids}
    bg=worst_gate(before_summary);ag=worst_gate(after_summary)
    language={}
    for l in LANGS:
        vals=[mean_delta[i][l] for i in ids];magnitude=statistics.mean(abs(v) for v in vals)
        noise=max(bg['languages'][l]['max_mean_abs_difference'],ag['languages'][l]['max_mean_abs_difference'])
        language[l]={'mean_signed_delta':statistics.mean(vals),'mean_absolute_delta':magnitude,
            'median_absolute_delta':statistics.median(abs(v) for v in vals),
            'endpoint_worst_wrapper_difference_reference':noise,
            'magnitude_over_endpoint_reference':magnitude/noise if noise else None}
    partitions=[]
    for split in before_summary['all_35_splits']:
        a,b=split['left'],split['right'];metrics={}
        for l in LANGS:
            x=[avg(a,i,l) for i in ids];y=[avg(b,i,l) for i in ids]
            metrics[l]={'mean_absolute_delta_disagreement':statistics.mean(abs(u-v) for u,v in zip(x,y)),
                'delta_spearman':spearman(x,y),
                'delta_sign_agreement_count':sum((u>0)-(u<0)==(v>0)-(v<0) for u,v in zip(x,y))}
        partitions.append({'left':a,'right':b,'languages':metrics})
    for l in LANGS:
        language[l]['worst_split_delta_disagreement']=max(s['languages'][l]['mean_absolute_delta_disagreement'] for s in partitions)
    tv=[math.fsum(abs(v) for v in mean_delta[i].values())/2 for i in ids]
    joint=sum(all(before[w][i]['Z']>=.05 and after[w][i]['Z']>=.05 for w in before) for i in ids)
    return {'languages':language,'mean_vector_MAE':statistics.mean(v['mean_absolute_delta'] for v in language.values()),
        'mean_TV':statistics.mean(tv),'median_TV':statistics.median(tv),'TV_gt_015_count':sum(v>.15 for v in tv),
        'all_35_delta_sensitivity':partitions,'mean_delta_by_concept':mean_delta,
        'all_16_events_Z_ge_005_concepts':joint,'no_concepts_excluded':True,
        'is_H_effect':False,'is_H_lower_bound':False,'power_or_detectability_established':False}


def inventory():
    entries=[]
    for label,run,key in SOURCES:
        path=WORK_ROOT/'runs'/run/('run_summary_000263.json' if run.startswith('pilot_') else 'run_summary.json')
        source=read_verified_json(path)
        candidates=[e['data']['checkpoint'] for e in source['events'] if e['kind']=='CHECKPOINT'
            and e['data']['reason']=='FIXED_ENDPOINT' and e['data'].get('phase',e['data'].get('arm'))==key]
        if len(candidates)!=1:raise ContractViolation('AMBIGUOUS_SOURCE_ENDPOINT')
        ref=candidates[0];manifest=read_verified_json(Path(ref['path'])/'manifest.json',expected_sha256=ref['manifest_sha256'])
        l=manifest['lineage'];p=manifest['progress']
        if manifest['state_sha256']!=ref['state_sha256'] or manifest['model_fingerprint']!=ref['model_fingerprint']:
            raise ContractViolation('CHECKPOINT_SOURCE_MISMATCH')
        entries.append({'label':label,'checkpoint':ref,'source':{'path':str(path),'sha256':sha256_file(path)},
            'global_step':p['global_step'],'phase_step':p['phase_step'],'phase':l['phase'],
            'root_id':l['root_id'],'freeze_sha256':l['freeze_sha256'],'tokenizer_sha256':l['tokenizer_sha256'],
            'initial_model_fingerprint':l['initial_model_fingerprint'],'phase_parent_sha256':l['phase_parent_sha256'],
            'source_model_fingerprint':l.get('source_lineage',{}).get('model_fingerprint')})
    for a,b in zip(entries,entries[1:]):
        if b['global_step']<=a['global_step'] or b['root_id']!=a['root_id'] or b['tokenizer_sha256']!=a['tokenizer_sha256'] or b['initial_model_fingerprint']!=a['initial_model_fingerprint']:
            raise ContractViolation('TRAJECTORY_LINEAGE_MISMATCH')
        if b['label'] in ('T1','T2','T3'):
            if b['phase_parent_sha256']!=a['checkpoint']['state_sha256']:raise ContractViolation('PHASE_PARENT_MISMATCH')
        elif b['source_model_fingerprint']!=a['checkpoint']['model_fingerprint']:
            raise ContractViolation('CONTINUATION_PARENT_MISMATCH')
    return entries


def prepare():
    from .src.build_tokenizer import load_verified_tokenizer
    from .src.score import build_continuation_events
    entries=inventory();parent=read_verified_json(PARENT,expected_sha256=PARENT_SHA)
    tokenizer,_=load_verified_tokenizer(Path(parent['artifacts']['tokenizer']['path']))
    rows=prompts.records()
    for r in rows:build_continuation_events(tokenizer,r['prefix'],r['answers'],context_length=256)
    for e in entries:
        if sha256_file(Path(e['checkpoint']['path'])/'state.pt')!=e['checkpoint']['state_sha256']:
            raise ContractViolation('CHECKPOINT_FILE_CHANGED')
    reuse=read_verified_json(WORK_ROOT/'reports/prompt_average_probe_v1/verified_report.json')
    if reuse['result']['checkpoint']!=entries[-1]['checkpoint'] or not reuse['result']['weights_unchanged']:
        raise ContractViolation('REUSE_CHECKPOINT_MISMATCH')
    doc={'schema':NAME,'entries':entries,'records':rows,'records_sha256':_hash(rows),'plan_sha256':sha256_file(PLAN),
        'rule_sha256':sha256_file(CONFIG),'reuse_report':{'path':str(WORK_ROOT/'reports/prompt_average_probe_v1/verified_report.json'),
        'sha256':sha256_file(WORK_ROOT/'reports/prompt_average_probe_v1/verified_report.json')},
        'reuse_chunks':reuse['raw_chunks'],'parent_sha256':PARENT_SHA,'updates':0,'main_enabled':False,'reserved_seconds':1200}
    return publish_json_once(CONTRACT,doc)


def run():
    from .src.gpu_guard import require_literal_gpu2_mask
    require_literal_gpu2_mask()
    from .src.integrity import verify_implementation_manifest,verify_cpu_check_evidence
    from .src.budget import BudgetCaps,GpuBudgetLedger,require_disk_reservation
    from .src.train import gpu2_training_session,BoundaryStopController
    from .src.checkpoint import load_checkpoint_payload
    from .src.build_tokenizer import load_verified_tokenizer
    from .src.model import build_random_model
    from .src.score import score_probability_partition,semantic_model_fingerprint
    import torch
    contract=read_verified_json(CONTRACT);code=verify_implementation_manifest(MANIFEST)['implementation_manifest_sha256']
    cpu=verify_cpu_check_evidence(CPU)
    if cpu['implementation']['implementation_manifest_sha256']!=code:raise ContractViolation('CPU_CODE_MISMATCH')
    if contract['entries']!=inventory() or contract['records']!=prompts.records() or contract['records_sha256']!=_hash(contract['records']):raise ContractViolation('CONTRACT_CHANGED')
    if contract['plan_sha256']!=sha256_file(PLAN) or contract['rule_sha256']!=sha256_file(CONFIG):raise ContractViolation('RULE_CHANGED')
    reuse=read_verified_json(Path(contract['reuse_report']['path']),expected_sha256=contract['reuse_report']['sha256'])
    reused=[]
    for ref in contract['reuse_chunks']:reused.extend(read_verified_json(Path(ref['path']),expected_sha256=ref['sha256'])['rows'])
    if [{k:v for k,v in r.items() if k!='probability'} for r in reused]!=contract['records']:
        raise ContractViolation('REUSED_RECORDS_CHANGED')
    if prompts.summarize(reused)!=reuse['result']['metrics']:raise ContractViolation('REUSE_SUMMARY_MISMATCH')
    if RUN.exists():raise ContractViolation('NEW_RUN_REQUIRED')
    policy=read_verified_json(PROJECT_ROOT/'implementation/config/campaign_policy.json')['budget']
    require_disk_reservation(WORK_ROOT,planned_bytes=100*1024**2,emergency_free_bytes=policy['minimum_emergency_free_bytes'])
    ledger=GpuBudgetLedger(WORK_ROOT/'ledger/gpu_budget.json',BudgetCaps(policy['campaign_gpu_hours_cap'],policy['prior_gpu_hours_user_reported'],policy['per_root_gpu_hours_cap']))
    RUN.mkdir(parents=True);results={};all_rows={};raw_refs=[];status='NOT_RUN';smokes=[];pairs=[]
    try:
        with gpu2_training_session(ledger=ledger,lock_path=WORK_ROOT/'locks/gpu2.lock',run_id=NAME,root_id=4101,phase='T3',
                reserved_seconds=1200,checkpoint_grace_seconds=120,experiment_freeze_path=PARENT) as session,BoundaryStopController() as stop:
            tokenizer,meta=load_verified_tokenizer(Path(session.experiment_freeze['resolved_artifacts']['tokenizer']))
            if meta['tokenizer_file_sha256']!=contract['entries'][0]['tokenizer_sha256']:raise ContractViolation('TOKENIZER_CHANGED')
            for entry in contract['entries'][:-1]:
                label=entry['label'];ref=entry['checkpoint']
                state,_=load_checkpoint_payload(Path(ref['path']),expected_state_sha256=ref['state_sha256'],expected_manifest_sha256=ref['manifest_sha256'])
                if state['lineage']['freeze_sha256']!=entry['freeze_sha256']:raise ContractViolation('STATE_FREEZE_MISMATCH')
                model=build_random_model(seed=4101).to('cuda:0');model.load_state_dict(state['model'],strict=True);del state
                model.eval();model.requires_grad_(False)
                if semantic_model_fingerprint(model)!=ref['model_fingerprint']:raise ContractViolation('MODEL_MISMATCH')
                rows=[]
                def score(r):
                    ledger.require_time_remaining(session.lease,checkpoint_grace_seconds=120)
                    if stop.requested:raise ContractViolation('INTERRUPTED')
                    return {**r,'probability':score_probability_partition(model,tokenizer,r['prefix'],r['answers'],context_length=256)}
                a=[score(r) for r in contract['records'][:2]];b=[score(r) for r in contract['records'][:2]]
                if a!=b:raise ContractViolation('GPU_REPEAT_MISMATCH')
                smokes.append({'label':label,'first':_hash(a),'second':_hash(b)})
                for row in contract['records']:
                    rows.append(score(row))
                    if len(rows)%60==0:
                        raw_refs.append(publish_json_once(RUN/f'{label}_{len(rows):04d}.json',{'label':label,'checkpoint_state_sha256':ref['state_sha256'],'completed':len(rows),'rows':rows[-60:]}))
                        ledger.heartbeat(session.lease);print(label,len(rows),'/ 480',flush=True)
                if semantic_model_fingerprint(model)!=ref['model_fingerprint']:raise ContractViolation('WEIGHTS_CHANGED')
                m=prompts.summarize(rows);results[label]={'entry':entry,'measurement':m,'worst_gate':worst_gate(m),'weights_unchanged':True,'reused':False}
                all_rows[label]=rows;del model;torch.cuda.empty_cache()
            publish_json_once(RUN/'gpu_smoke.json',{'status':'PASS','checkpoints':smokes,'gpu':session.binding.as_dict()})
            verify_implementation_manifest(MANIFEST)
        last=contract['entries'][-1];m=prompts.summarize(reused)
        results[last['label']]={'entry':last,'measurement':m,'worst_gate':worst_gate(m),'weights_unchanged':True,'reused':True}
        all_rows[last['label']]=reused
        for before,after in zip(contract['entries'],contract['entries'][1:]):
            a,b=before['label'],after['label']
            pairs.append({'before':a,'after':b,'updates_between':after['global_step']-before['global_step'],
                'scope':'original_sequential_stages' if b in ('T1','T2','T3') else 'dev_informed_exploratory_continuation',
                **pair_drift(all_rows[a],all_rows[b],results[a]['measurement'],results[b]['measurement'])})
        status='COMPLETE'
    except Exception as exc:status=f'{type(exc).__name__}: {exc}'
    result={'status':status,'contract_sha256':sha256_file(CONTRACT),'code_sha256':code,'cpu_sha256':sha256_file(CPU),
        'rule_sha256':sha256_file(CONFIG),'checkpoints':results,'pairs':pairs,'new_raw_chunks':raw_refs,
        'reuse_report':contract['reuse_report'],'budget':ledger.snapshot(),'updates':0,'main_enabled':False,
        'historical_source_results_mutated':False,'gate_scope':'v2 exploratory measurement only; does not authorize H or main'}
    print(publish_json_once(RUN/'run_summary.json',result),flush=True)
    return 0 if status=='COMPLETE' else 20

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
    if args.action=='prepare':print(prepare())
    else:raise SystemExit(run())
