"""Read-only exact ANY likelihood replication and Qwen-matched zero-shot prompts."""
import argparse
from pathlib import Path
from .qwen_remeasurement import make_records as qwen_records, probability_summary
from .qwen_robustness import PARENT, PARENT_SHA
from .src.artifacts import read_verified_json, publish_json_once, sha256_file
from .src.contracts import WORK_ROOT, PROJECT_ROOT, ContractViolation
from .src.semantic_experiment_freeze import _hash

ROOT=WORK_ROOT/'runs/model40_remeasurement_v1'
CONTRACT=WORK_ROOT/'freeze/model40_remeasurement_v1/contract.json'
MANIFEST=WORK_ROOT/'integrity/model40_remeasurement_v1/implementation_manifest.json'
CPU=WORK_ROOT/'reports/cpu_model40_remeasurement_v1/cpu_checks.json'
SOURCE=WORK_ROOT/'runs/accuracy_selected_9e209c3/run_summary.json'
SOURCE_SHA='bf7a8c230414636f132d4cc8d0d44e2a19c6eaf8d7d0e741f5fc724eefc3eb6a'


def records():
    rows=[]
    for r in qwen_records()['probability']:
        if r['condition']!='zero_shot': continue
        rows.append(r)
        if not r['prefix'].endswith('\n'): raise ContractViolation('BOUNDARY_SUFFIX_MISSING')
        rows.append({**r,'condition':'original','prefix':r['prefix'][:-1],
                     'record_id':r['record_id'].replace('|zero_shot|','|original|')})
    return rows


def prepare():
    from .src.build_tokenizer import load_verified_tokenizer
    from .src.score import build_continuation_events
    source=read_verified_json(SOURCE,expected_sha256=SOURCE_SHA)
    checkpoint=[e['data']['checkpoint'] for e in source['events'] if e['kind']=='CHECKPOINT' and e['data']['reason']=='FIXED_ENDPOINT'][-1]
    parent=read_verified_json(PARENT,expected_sha256=PARENT_SHA)
    ref=parent['artifacts']['tokenizer'];tokenizer,_=load_verified_tokenizer(Path(ref['path']))
    rows=records();lengths=[]
    for row in rows:
        events=build_continuation_events(tokenizer,row['prefix'],row['answers'],context_length=256)
        lengths.extend(e['full_token_count'] for e in events)
    return publish_json_once(CONTRACT,{'schema':'model40-remeasurement-v1','checkpoint':checkpoint,
        'parent_sha256':PARENT_SHA,'source_sha256':SOURCE_SHA,'records':rows,'records_sha256':_hash(rows),
        'plan_sha256':sha256_file(PROJECT_ROOT/'implementation/MODEL40_REMEASUREMENT_V1.md'),
        'maximum_sequence_tokens':max(lengths),'context_length':256,'training':False,'main_enabled':False})


def run(*, root=ROOT, contract_path=CONTRACT, manifest_path=MANIFEST, cpu_path=CPU,
        plan_path=PROJECT_ROOT/'implementation/MODEL40_REMEASUREMENT_V1.md',
        record_builder=records, summarizer=probability_summary):
    from .src.gpu_guard import require_literal_gpu2_mask
    require_literal_gpu2_mask()
    from .src.integrity import verify_implementation_manifest,verify_cpu_check_evidence
    from .src.budget import BudgetCaps,GpuBudgetLedger,require_disk_reservation
    from .src.train import gpu2_training_session,BoundaryStopController
    from .src.checkpoint import load_checkpoint_payload
    from .src.build_tokenizer import load_verified_tokenizer
    from .src.model import build_random_model
    from .src.score import score_probability_partition,semantic_model_fingerprint
    contract=read_verified_json(contract_path)
    expected_records=record_builder()
    code=verify_implementation_manifest(manifest_path)['implementation_manifest_sha256']
    cpu=verify_cpu_check_evidence(cpu_path)
    if cpu['implementation']['implementation_manifest_sha256']!=code: raise ContractViolation('CPU_CODE_MISMATCH')
    source=read_verified_json(SOURCE,expected_sha256=SOURCE_SHA)
    expected=[e['data']['checkpoint'] for e in source['events'] if e['kind']=='CHECKPOINT' and e['data']['reason']=='FIXED_ENDPOINT'][-1]
    if contract['checkpoint']!=expected or contract['records']!=expected_records or contract['records_sha256']!=_hash(expected_records):
        raise ContractViolation('CONTRACT_MISMATCH')
    if contract['plan_sha256']!=sha256_file(plan_path):
        raise ContractViolation('PLAN_CHANGED')
    if root.exists(): raise ContractViolation('NEW_RUN_REQUIRED')
    policy=read_verified_json(PROJECT_ROOT/'implementation/config/campaign_policy.json')['budget']
    require_disk_reservation(WORK_ROOT,planned_bytes=100*1024**2,emergency_free_bytes=policy['minimum_emergency_free_bytes'])
    ledger=GpuBudgetLedger(WORK_ROOT/'ledger/gpu_budget.json',BudgetCaps(policy['campaign_gpu_hours_cap'],policy['prior_gpu_hours_user_reported'],policy['per_root_gpu_hours_cap']))
    root.mkdir(parents=True);rows=[];status='NOT_RUN';unchanged=False;parameters=None
    try:
        with gpu2_training_session(ledger=ledger,lock_path=WORK_ROOT/'locks/gpu2.lock',run_id=root.name,
            root_id=4101,phase='T3',reserved_seconds=1200,checkpoint_grace_seconds=120,
            experiment_freeze_path=PARENT) as session, BoundaryStopController() as stop:
            parent=session.experiment_freeze
            tokenizer,meta=load_verified_tokenizer(Path(parent['resolved_artifacts']['tokenizer']))
            if meta['tokenizer_file_sha256']!=parent['tokenizer_file_sha256']: raise ContractViolation('TOKENIZER_MISMATCH')
            ref=contract['checkpoint']
            state,_=load_checkpoint_payload(Path(ref['path']),expected_state_sha256=ref['state_sha256'],expected_manifest_sha256=ref['manifest_sha256'])
            if state['lineage']['freeze_sha256']!=PARENT_SHA: raise ContractViolation('LINEAGE_MISMATCH')
            model=build_random_model(seed=4101).to('cuda:0');model.load_state_dict(state['model'],strict=True);del state
            model.eval();model.requires_grad_(False);parameters=sum(p.numel() for p in model.parameters())
            if semantic_model_fingerprint(model)!=ref['model_fingerprint']: raise ContractViolation('MODEL_MISMATCH')
            def score(row):
                ledger.require_time_remaining(session.lease,checkpoint_grace_seconds=120)
                if stop.requested: raise ContractViolation('INTERRUPTED')
                return {**row,'probability':score_probability_partition(model,tokenizer,row['prefix'],row['answers'],context_length=256)}
            a=[score(r) for r in contract['records'][:8]];b=[score(r) for r in contract['records'][:8]]
            if a!=b: raise ContractViolation('GPU_REPEAT_MISMATCH')
            publish_json_once(root/'gpu_smoke.json',{'status':'PASS','first':_hash(a),'second':_hash(b),'gpu':session.binding.as_dict()})
            for row in contract['records']:
                rows.append(score(row))
                if len(rows)%60==0:
                    publish_json_once(root/f'rows_{len(rows):04d}.json',{'completed':len(rows),'rows':rows[-60:]})
                    ledger.heartbeat(session.lease);print(len(rows),'/',len(expected_records),flush=True)
            unchanged=semantic_model_fingerprint(model)==ref['model_fingerprint']
            if not unchanged: raise ContractViolation('WEIGHTS_CHANGED')
            verify_implementation_manifest(manifest_path);status='COMPLETE'
    except Exception as exc: status=f'{type(exc).__name__}: {exc}'
    result={'status':status,'parameters':parameters,'completed':len(rows),'checkpoint':contract['checkpoint'],
        'contract_sha256':sha256_file(contract_path),'code_sha256':code,'cpu_sha256':sha256_file(cpu_path),
        'updates':0,'weights_unchanged':unchanged,'metrics':summarizer(rows) if len(rows)==len(expected_records) else None,
        'budget':ledger.snapshot(),'main_enabled':False,'primary_gate_replaced':False}
    print(publish_json_once(root/'run_summary.json',result),flush=True)
    return 0 if status=='COMPLETE' else 20

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
    if args.action=='prepare': print(prepare())
    else: raise SystemExit(run())
