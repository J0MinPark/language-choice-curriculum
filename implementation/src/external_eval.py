"""Bounded read-only paired-definition diagnostic; no optimizer or training path."""
import zipfile
from pathlib import Path
from .artifacts import read_verified_json, sha256_file, publish_json_once, publish_bytes_once
from .contracts import WORK_ROOT, PROJECT_ROOT, ContractViolation
from .semantic_experiment_freeze import _hash

SOURCE = WORK_ROOT/'runs/accuracy_selected_9e209c3/run_summary.json'
SOURCE_SHA = 'bf7a8c230414636f132d4cc8d0d44e2a19c6eaf8d7d0e741f5fc724eefc3eb6a'
PARENT = WORK_ROOT/'freeze/accuracy_selected_360_v3/experiment_freeze.json'
PARENT_SHA = '6f6cc0d52b4162f1018b560c6f95419cb9258deda7e053e0a12bdc7950e39dfd'
PLANNED = WORK_ROOT/'evaluations/external_matched_v1/planned_records.json'
PLANNED_SHA = '49bd8789b49b0f33ee31d94294bb15d5cbdc44e9657cf3f5fc8ee6186c1f84d7'


def definition_evidence(archive, records):
    with zipfile.ZipFile(archive) as z:
        license_bytes = z.read('wordnet/LICENSE')
        definitions = {}
        for line in z.read('wordnet/data.noun').decode().splitlines():
            if line[:1].isdigit() and ' | ' in line:
                definition = line.split(' | ',1)[1].split('; "',1)[0].strip()
                definitions.setdefault(definition, []).append(line.split()[0])
    evidence = []
    for row in records:
        if row['external_definition'] not in definitions:
            raise ContractViolation('EXTERNAL_DEFINITION_SOURCE_NOT_VERIFIED')
        evidence.append({'external_id':row['external_id'],
            'definition_sha256':_hash(row['external_definition']),
            'wordnet3_noun_offsets':definitions[row['external_definition']]})
    return evidence, license_bytes


def build_contract():
    parent=read_verified_json(PARENT,expected_sha256=PARENT_SHA)
    planned=read_verified_json(PLANNED,expected_sha256=PLANNED_SHA)
    mapping=read_verified_json(WORK_ROOT/'reviews/external_benchmark_v1/accepted_mapping_v1.json',
        expected_sha256='ba04ba1b8eb5229758eb31ee21faffeb0a878c73940196f5c316ee6d9bc4c801')
    rights=read_verified_json(WORK_ROOT/'sources/wordnet_rights_v1/manifest.json',
        expected_sha256='fdf0fbc2ada8f7e2d2d43065af448f038b824a1e42d5a1c4af7c6767a875ba3c')
    for ref in (rights['archive'],rights['license']):
        if sha256_file(Path(ref['path']))!=ref['sha256']:
            raise ContractViolation('WORDNET_RIGHTS_ARTIFACT_MISMATCH')
    evidence,license_bytes=definition_evidence(rights['archive']['path'],mapping['mappings'])
    source=read_verified_json(SOURCE,expected_sha256=SOURCE_SHA)
    checkpoint=[e['data']['checkpoint'] for e in source['events']
        if e['kind']=='CHECKPOINT' and e['data']['reason']=='FIXED_ENDPOINT'][-1]
    concepts={c['concept_id']:c for c in parent['concepts']}
    ref=parent['artifacts']['evaluation_plan']
    prompt=read_verified_json(Path(ref['path']),expected_sha256=ref['sha256'])['prompt_plan']
    records=[]
    for original in planned['records']:
        if original['record_sha256']!=_hash({k:v for k,v in original.items() if k!='record_sha256'}):
            raise ContractViolation('EXTERNAL_RECORD_DIGEST_MISMATCH')
        for condition in ('external_definition','original_definition'):
            row={k:v for k,v in original.items() if k not in ('record_sha256','prefix_tokens')}
            row.update(condition=condition,record_id=condition+'|'+original['record_id'])
            if condition=='original_definition':
                row['prefix']=prompt['templates'][row['wrapper']]['en'][row['mode']].format(
                    gloss=concepts[row['jm_concept_id']]['glosses']['en'],
                    target_language_name='' if row['requested_language'] is None else prompt['target_language_names']['en'][row['requested_language']])
            records.append({**row,'record_sha256':_hash(row)})
    return {'schema_version':'external-read-only-eval-v1','parent_freeze_sha256':PARENT_SHA,
        'source_summary_sha256':SOURCE_SHA,'checkpoint':checkpoint,'records':records,
        'records_sha256':_hash(records),'definition_source_evidence':evidence,
        'wordnet_license':rights['license'],'wordnet_archive':rights['archive'],
        'planned_source_sha256':PLANNED_SHA,'updates':0,'training_enabled':False,'main_enabled':False,
        'evaluation_scope':'31 reviewed matched concepts; EN input; original versus external definition; not full THINGS benchmark or primary KO gate',
        'context_length':256,'max_new_tokens':64,'reserved_seconds':1200,
        'automatic_expansion':False}


def summarize(rows):
    cells={}
    for row in rows:
        if row['mode']!='REQUESTED':continue
        key='|'.join((row['condition'],row['wrapper'],row['requested_language']))
        c=cells.setdefault(key,{'correct':0,'total':0})
        c['correct']+=int(row['generation']['request_compatible']);c['total']+=1
    for c in cells.values():c['accuracy']=c['correct']/c['total']
    return cells


def run(contract_path,manifest_path,cpu_path,directory):
    from .gpu_guard import require_literal_gpu2_mask
    require_literal_gpu2_mask()
    from .integrity import verify_implementation_manifest, verify_cpu_check_evidence
    from .budget import BudgetCaps,GpuBudgetLedger,require_disk_reservation
    from .train import gpu2_training_session,BoundaryStopController
    from .checkpoint import load_checkpoint_payload
    from .build_tokenizer import load_verified_tokenizer
    from .model import build_random_model
    from .score import greedy_first_line,score_probability_partition,semantic_model_fingerprint,preserved_evaluation_state
    contract=read_verified_json(contract_path)
    if contract!=build_contract():raise ContractViolation('EXTERNAL_EVALUATION_CONTRACT_MISMATCH')
    code=verify_implementation_manifest(manifest_path)['implementation_manifest_sha256']
    cpu=verify_cpu_check_evidence(cpu_path)
    if cpu['implementation']['implementation_manifest_sha256']!=code:raise ContractViolation('CPU_CODE_MISMATCH')
    directory=Path(directory).absolute()
    if directory!=directory.resolve() or not directory.is_relative_to(WORK_ROOT) or directory.exists():
        raise ContractViolation('EXTERNAL_EVAL_REQUIRES_NEW_WORK_DIRECTORY')
    policy=read_verified_json(PROJECT_ROOT/'implementation/config/campaign_policy.json')['budget']
    require_disk_reservation(directory.parent,planned_bytes=20*1024**2,emergency_free_bytes=policy['minimum_emergency_free_bytes'])
    directory.mkdir(parents=True)
    publish_bytes_once(directory/'WordNet-LICENSE',Path(contract['wordnet_license']['path']).read_bytes())
    ledger=GpuBudgetLedger(WORK_ROOT/'ledger/gpu_budget.json',BudgetCaps(policy['campaign_gpu_hours_cap'],policy['prior_gpu_hours_user_reported'],policy['per_root_gpu_hours_cap']))
    rows=[];status='NOT_RUN';binding=None;smoke=False;unchanged=False
    try:
        with gpu2_training_session(ledger=ledger,lock_path=WORK_ROOT/'locks/gpu2.lock',run_id=directory.name,
            root_id=4101,phase='T3',reserved_seconds=1200,checkpoint_grace_seconds=120,
            experiment_freeze_path=PARENT) as session, BoundaryStopController() as stop:
            binding=session.binding.as_dict();parent=session.experiment_freeze
            tokenizer,meta=load_verified_tokenizer(Path(parent['resolved_artifacts']['tokenizer']))
            if meta['tokenizer_file_sha256']!=parent['tokenizer_file_sha256']:raise ContractViolation('TOKENIZER_BINDING_MISMATCH')
            ref=contract['checkpoint']
            state,_=load_checkpoint_payload(Path(ref['path']),expected_state_sha256=ref['state_sha256'],expected_manifest_sha256=ref['manifest_sha256'])
            if state['lineage']['freeze_sha256']!=PARENT_SHA:raise ContractViolation('SOURCE_FREEZE_MISMATCH')
            model=build_random_model(seed=4101).to('cuda:0');model.load_state_dict(state['model'],strict=True);del state
            if semantic_model_fingerprint(model)!=ref['model_fingerprint']:raise ContractViolation('SOURCE_MODEL_MISMATCH')
            def score(row):
                session.ledger.require_time_remaining(session.lease,checkpoint_grace_seconds=120)
                if stop.requested:raise ContractViolation('INTERRUPTED_AT_BOUNDARY')
                result={**row,'generation':greedy_first_line(model,tokenizer,row['prefix'],row['answers'],row['requested_language'],context_length=256,max_new_tokens=64)}
                if row['mode']=='ANY':result['probability']=score_probability_partition(model,tokenizer,row['prefix'],row['answers'],context_length=256)
                return result
            # Same exact prompts twice on GPU, including likelihood events. No optimizer,
            # no new checkpoint, and no claim that this is a training-resume test.
            with preserved_evaluation_state(model):
                test=contract['records'][:10]
                first=[score(r) for r in test];second=[score(r) for r in test]
                if first!=second:raise ContractViolation('GPU_EVALUATION_REPEAT_MISMATCH')
                smoke=True
                publish_json_once(directory/'gpu_smoke.json',{'status':'PASS','record_ids':[r['record_id'] for r in test],
                    'first_sha256':_hash(first),'second_sha256':_hash(second),'gpu_binding':binding,'code':code,
                    'contract_sha256':sha256_file(contract_path),'scope':'read-only evaluation repeatability, not training replay'})
                for row in contract['records']:
                    rows.append(score(row))
                    if len(rows)%100==0:print(f'EXTERNAL_EVAL {len(rows)}/{len(contract["records"])}',flush=True)
            unchanged=semantic_model_fingerprint(model)==ref['model_fingerprint']
            if not unchanged:raise ContractViolation('EVALUATION_CHANGED_WEIGHTS')
            verify_implementation_manifest(manifest_path)
            status='COMPLETE'
    except Exception as exc:
        status=f'{type(exc).__name__}: {exc}'
    result={'status':status,'scope':contract['evaluation_scope'],'contract_sha256':sha256_file(contract_path),
        'checkpoint':contract['checkpoint'],'code_sha256':code,'cpu_check_sha256':cpu['cpu_check_sha256'],
        'rows':rows,'cells':summarize(rows),'expected_rows':len(contract['records']),'gpu_binding':binding,
        'gpu_repeat_test_pass':smoke,'weights_unchanged':unchanged,'updates':0,'budget':ledger.snapshot(),
        'primary_gate_replaced':False,'training_started':False,'automatic_expansion':False,'main_enabled':False}
    print(publish_json_once(directory/'run_summary.json',result),flush=True)
    return result
