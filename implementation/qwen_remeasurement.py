"""Prospectively frozen 4-shot generation and zero/4-shot ANY probability evaluation."""
import argparse
import math
from pathlib import Path
from . import qwen_robustness as base
from .src.artifacts import read_verified_json, publish_json_once, sha256_file
from .src.contracts import WORK_ROOT, PROJECT_ROOT, ContractViolation
from .src.semantic_experiment_freeze import _hash
from .src.score import (build_continuation_events, score_probability_partition,
                        classify_first_line, semantic_model_fingerprint)
from .src.report import spearman

ROOT = WORK_ROOT/'runs/qwen_remeasurement_v1'
CONTRACT = WORK_ROOT/'freeze/qwen_remeasurement_v1_r2/contract.json'
MANIFEST = WORK_ROOT/'integrity/qwen_remeasurement_v1_r2/implementation_manifest.json'
CPU = WORK_ROOT/'reports/cpu_qwen_remeasurement_v1_r2/cpu_checks.json'


def make_records():
    parent = read_verified_json(base.PARENT, expected_sha256=base.PARENT_SHA)
    ref = parent['artifacts']['evaluation_plan']
    plan = read_verified_json(Path(ref['path']), expected_sha256=ref['sha256'])['prompt_plan']
    concepts = sorted(parent['concepts'], key=lambda c:c['concept_id'])
    result = {'probability':[], 'generation':[]}
    for c in concepts:
        # Outcome-independent leave-target-out examples from reviewed material.
        demos = [d for d in concepts if d['concept_id']!=c['concept_id']
                 and not set(d['answers'].values()) & set(c['answers'].values())][:4]
        if len(demos)!=4: raise ContractViolation('INSUFFICIENT_DISJOINT_DEMOS')
        for inp in base.LANGS:
            for wrapper in ('dev1','dev2'):
                template = plan['templates'][wrapper][inp]
                def prefix(item, mode, target=None):
                    return template[mode].format(gloss=item['glosses'][inp],
                        target_language_name=plan['target_language_names'][inp].get(target,''))+'\n'
                common = {'concept_id':c['concept_id'], 'input_language':inp,
                          'wrapper':wrapper, 'answers':c['answers']}
                for condition in ('zero_shot','four_shot'):
                    examples = '' if condition=='zero_shot' else '\n'.join(
                        prefix(d,'ANY')+d['answers'][lang]+'\n' for d,lang in zip(demos,base.LANGS))
                    result['probability'].append({**common,'condition':condition,
                        'record_id':'|'.join((c['concept_id'],inp,wrapper,condition,'ANY')),
                        'prefix':examples+prefix(c,'ANY'),
                        'demo_ids':[] if condition=='zero_shot' else [d['concept_id'] for d in demos]})
                for target in base.LANGS:
                    examples = '\n'.join(prefix(d,'REQUESTED',target)+d['answers'][target]+'\n' for d in demos)
                    result['generation'].append({**common,'requested_language':target,
                        'record_id':'|'.join((c['concept_id'],inp,wrapper,target)),
                        'prefix':examples+prefix(c,'REQUESTED',target),
                        'demo_ids':[d['concept_id'] for d in demos]})
    return result


def probability_summary(rows):
    groups = {}
    for row in rows:
        key = row['condition']+'|'+row['input_language']
        cell = groups.setdefault(key, {'dev1':{},'dev2':{}})[row['wrapper']]
        if row['concept_id'] in cell: raise ContractViolation('DUPLICATE_PROBABILITY_ROW')
        cell[row['concept_id']] = row['probability']
    result = {}
    for key,group in groups.items():
        if set(group['dev1'])!=set(group['dev2']) or len(group['dev1'])!=60:
            raise ContractViolation('INCOMPLETE_PROBABILITY_CELLS')
        ids=sorted(group['dev1']); metrics={}
        for lang in base.LANGS:
            a=[group['dev1'][i]['Q_membership'][lang] for i in ids]
            b=[group['dev2'][i]['Q_membership'][lang] for i in ids]
            metrics[lang]={'spearman':spearman(a,b),'mean_abs_difference':sum(abs(x-y) for x,y in zip(a,b))/60,
                'dev1_range':[min(a),max(a)], 'dev2_range':[min(b),max(b)]}
        z={w:{'low_Z_fraction':sum(p['Z']<.05 for p in cell.values())/60,
              'median_Z':sum(sorted(p['Z'] for p in cell.values())[29:31])/2,
              'min_logZ':min(p['logZ'] for p in cell.values())} for w,cell in group.items()}
        result[key]={'n_concepts':60,'languages':metrics,'Z':z,
            'original_measurement_thresholds_pass':all(m['spearman'] is not None and m['spearman']>=.6
                and m['mean_abs_difference']<=.15 for m in metrics.values())
                and all(v['low_Z_fraction']<.1 for v in z.values())}
    return result


def cached_generation(model, tokenizer, row, max_new_tokens=64):
    import torch
    ids=tokenizer.encode(row['prefix'],add_special_tokens=False)
    current=torch.tensor([ids],device=next(model.parameters()).device)
    generated=[]; cache=None; termination='MAX_NEW_TOKENS'
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output=model(input_ids=current,past_key_values=cache,use_cache=True,logits_to_keep=1)
            token=int(output.logits[0,-1].argmax());generated.append(token)
            cache=output.past_key_values
            decoded=tokenizer.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if '\n' in decoded.replace('\r\n','\n').replace('\r','\n'):
                termination='NEWLINE';break
            current=torch.tensor([[token]],device=current.device)
    classified=classify_first_line(decoded,row['answers'],row['requested_language'])
    if termination!='NEWLINE':
        classified.update({'class':'UNREGISTERED_UNTERMINATED','registered_match':False,'membership':[],
            'request_compatible':False,'registered_other_language':False,'empty':False,'unregistered':True})
    return {**classified,'generated_token_ids':generated,'generated_token_count':len(generated),
            'decoded_continuation':decoded,'termination':termination}


def prepare():
    from transformers import AutoTokenizer
    old=read_verified_json(base.CONTRACT,expected_sha256=base.CONTRACT_SHA)
    tokenizer=AutoTokenizer.from_pretrained(base.DIRECTORY,local_files_only=True)
    records=make_records()
    for r in records['probability']:
        events=build_continuation_events(tokenizer,r['prefix'],r['answers'],context_length=32768)
        if len(events)!=4: raise ContractViolation('NONIDENTIFIABLE_COHORT')
    for r in records['generation']:
        if len(tokenizer.encode(r['prefix']))+64>32768: raise ContractViolation('CONTEXT_OVERFLOW')
    doc={'schema':'qwen-remeasurement-v1','model':base.MODEL,'revision':base.REVISION,
        'files':old['files'],'parent_sha256':base.PARENT_SHA,'records':records,'records_sha256':_hash(records),
        'plan_sha256':sha256_file(PROJECT_ROOT/'implementation/QWEN_REMEASUREMENT_V1.md'),
        'primary':'zero_shot|ko ANY; per-output-language dev1/dev2 Spearman; Q and Z jointly',
        'training':False,'main_enabled':False,'reserved_seconds':7200}
    print(publish_json_once(CONTRACT,doc))


def run():
    from .src.gpu_guard import require_literal_gpu2_mask, require_physical_gpu2, assert_model_on_bound_device
    require_literal_gpu2_mask()
    from .src.integrity import verify_implementation_manifest, verify_cpu_check_evidence
    from .src.budget import BudgetCaps, GpuBudgetLedger, ExclusiveGpu2Lock, require_disk_reservation
    from .src.train import BoundaryStopController
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    contract=read_verified_json(CONTRACT,expected_sha256="f80e1fee4848d165c7b20951346ccce0b3b50a0d7e970cad8d18a3014237e12f")
    code=verify_implementation_manifest(MANIFEST)['implementation_manifest_sha256']
    cpu=verify_cpu_check_evidence(CPU)
    if cpu['implementation']['implementation_manifest_sha256']!=code: raise ContractViolation('CPU_CODE_MISMATCH')
    if contract['records']!=make_records() or contract['records_sha256']!=_hash(contract['records']):
        raise ContractViolation('RECORDS_CHANGED')
    if contract['plan_sha256']!=sha256_file(PROJECT_ROOT/'implementation/QWEN_REMEASUREMENT_V1.md'):
        raise ContractViolation('PLAN_CHANGED')
    for ref in contract['files']:
        if sha256_file(Path(ref['path']))!=ref['sha256']: raise ContractViolation('MODEL_FILE_MISMATCH')
    if ROOT.exists(): raise ContractViolation('NEW_RUN_REQUIRED')
    require_disk_reservation(WORK_ROOT,planned_bytes=200*1024**2,emergency_free_bytes=1024**3)
    policy=read_verified_json(PROJECT_ROOT/'implementation/config/campaign_policy.json')['budget']
    ledger=GpuBudgetLedger(WORK_ROOT/'ledger/gpu_budget.json',BudgetCaps(policy['campaign_gpu_hours_cap'],
        policy['prior_gpu_hours_user_reported'],policy['per_root_gpu_hours_cap']))
    ROOT.mkdir(parents=True); results={'probability':[],'generation':[]}; status='NOT_RUN'; unchanged=False; binding=None
    with ExclusiveGpu2Lock(WORK_ROOT/'locks/gpu2.lock'):
        lease=ledger.reserve(run_id=ROOT.name,root_id=None,phase='QWEN_REMEASUREMENT_READ_ONLY',reserved_seconds=7200)
        try:
            binding=require_physical_gpu2();torch.set_num_threads(4);torch.manual_seed(4101)
            torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
            tokenizer=AutoTokenizer.from_pretrained(base.DIRECTORY,local_files_only=True)
            model=AutoModelForCausalLM.from_pretrained(base.DIRECTORY,local_files_only=True,
                torch_dtype=torch.float32,attn_implementation='eager').to('cuda:0').eval()
            model.requires_grad_(False);assert_model_on_bound_device(model,binding)
            before=semantic_model_fingerprint(model)
            with BoundaryStopController() as stop:
                def score(kind,row):
                    ledger.require_time_remaining(lease,checkpoint_grace_seconds=120)
                    if stop.requested: raise ContractViolation('INTERRUPTED')
                    value=score_probability_partition(model,tokenizer,row['prefix'],row['answers'],context_length=32768) if kind=='probability' else cached_generation(model,tokenizer,row)
                    return {**row,kind:value}
                smoke=[]
                for kind in results:
                    r=contract['records'][kind][0];a=score(kind,r);b=score(kind,r)
                    if a!=b: raise ContractViolation('GPU_REPEAT_MISMATCH')
                    smoke.append({'kind':kind,'sha256':_hash(a)})
                publish_json_once(ROOT/'gpu_smoke.json',{'status':'PASS','checks':smoke})
                for kind in results:
                    for row in contract['records'][kind]:
                        results[kind].append(score(kind,row))
                        n=len(results[kind])
                        if n%60==0:
                            require_disk_reservation(WORK_ROOT,planned_bytes=200*1024**2,emergency_free_bytes=1024**3)
                            publish_json_once(ROOT/f'{kind}_{n:04d}.json',{'completed':n,'total':len(contract['records'][kind]),'rows':results[kind][-60:]})
                            ledger.heartbeat(lease);print(kind,n,'/',len(contract['records'][kind]),flush=True)
                    if kind=='probability':
                        publish_json_once(ROOT/'probability_summary.json',{'status':'COMPLETE','metrics':probability_summary(results[kind]),'contract_sha256':sha256_file(CONTRACT)})
            unchanged=semantic_model_fingerprint(model)==before
            if not unchanged: raise ContractViolation('WEIGHTS_CHANGED')
            verify_implementation_manifest(MANIFEST);status='COMPLETE'
        except Exception as exc:
            status=f'{type(exc).__name__}: {exc}'
        finally:
            ledger.finish(lease,status)
    summary={'status':status,'contract_sha256':sha256_file(CONTRACT),'code_sha256':code,'cpu_sha256':sha256_file(CPU),
        'model':base.MODEL,'revision':base.REVISION,'updates':0,'weights_unchanged':unchanged,
        'gpu':binding.as_dict() if binding else None,'budget':ledger.snapshot(),
        'completed':{k:len(v) for k,v in results.items()},'main_enabled':False,'main_budget_status':'UNSECURED',
        'probability':probability_summary(results['probability']) if len(results['probability'])==960 else None,
        'generation':base.summarize(results['generation'],complete=len(results['generation'])==1920)}
    print(publish_json_once(ROOT/'run_summary.json',summary),flush=True)
    return 0 if status=='COMPLETE' else 20

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args();raise SystemExit({'prepare':prepare,'run':run}[args.action]())
