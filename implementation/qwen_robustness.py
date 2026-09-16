"""Fixed base-model, read-only robustness exploration. No training path."""
import argparse
from pathlib import Path
from .src.artifacts import read_verified_json, sha256_file, publish_json_once
from .src.contracts import WORK_ROOT, PROJECT_ROOT, ContractViolation
from .src.semantic_experiment_freeze import _hash

MODEL = 'Qwen/Qwen3-0.6B-Base'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
DIRECTORY = WORK_ROOT/'sources/qwen3_06b_base_fixed'
CONTRACT = WORK_ROOT/'freeze/qwen_robustness_v1/contract.json'
CONTRACT_SHA = '43664153b33c553f5a77895de0405989626ed6dc907ad92a96ae6a38ab4214fe'
MANIFEST = WORK_ROOT/'integrity/qwen_robustness_v1_r2/implementation_manifest.json'
CPU = WORK_ROOT/'reports/cpu_qwen_robustness_v1_r2/cpu_checks.json'
RUN = WORK_ROOT/'runs/qwen_robustness_v1_r2'
LANGS = ('ko', 'en', 'zh', 'fr')
PARENT = WORK_ROOT/'freeze/accuracy_selected_360_v3/experiment_freeze.json'
PARENT_SHA = '6f6cc0d52b4162f1018b560c6f95419cb9258deda7e053e0a12bdc7950e39dfd'


def records():
    parent = read_verified_json(PARENT, expected_sha256=PARENT_SHA)
    ref = parent['artifacts']['evaluation_plan']
    prompt = read_verified_json(Path(ref['path']), expected_sha256=ref['sha256'])['prompt_plan']
    rows = []
    for c in parent['concepts']:
        for inp in LANGS:
            for wrapper in ('dev1', 'dev2'):
                for target in LANGS:
                    rows.append({'record_id':'|'.join((c['concept_id'], inp, wrapper, target)),
                        'concept_id':c['concept_id'], 'input_language':inp, 'wrapper':wrapper,
                        'requested_language':target, 'answers':c['answers'],
                        'prefix':prompt['templates'][wrapper][inp]['REQUESTED'].format(
                            gloss=c['glosses'][inp], target_language_name=prompt['target_language_names'][inp][target])})
    if len(rows) != 1920 or len({r['record_id'] for r in rows}) != 1920:
        raise ContractViolation('COHORT_MISMATCH')
    return rows


def summarize(rows, *, complete=False):
    cells = {}; seen = set()
    for row in rows:
        if row['input_language'] not in LANGS or row['requested_language'] not in LANGS or row['wrapper'] not in ('dev1','dev2'):
            raise ContractViolation('INVALID_RESULT_CELL')
        if row['record_id'] in seen:
            raise ContractViolation('DUPLICATE_RESULT')
        seen.add(row['record_id'])
        key = '|'.join((row['input_language'], row['wrapper'], row['requested_language']))
        c = cells.setdefault(key, {'total':0, 'A_count':0, 'C_count':0, 'unregistered':0, 'empty':0, 'matrix':{}})
        g = row['generation']; members = g['membership']
        if len(members)!=len(set(members)) or any(m not in LANGS for m in members) or (members and g['empty']):
            raise ContractViolation('INVALID_OUTPUT_MEMBERSHIP')
        column = ('SHARED:' + '+'.join(sorted(members)) if len(members)>1 else members[0]) if members else ('EMPTY' if g['empty'] else 'UNREGISTERED')
        c['total'] += 1; c['matrix'][column] = c['matrix'].get(column, 0)+1
        bucket = ('A_count' if row['requested_language'] in members else 'C_count') if members else ('empty' if g['empty'] else 'unregistered')
        c[bucket] += 1
    for c in cells.values():
        c['A'] = c['A_count']/c['total']
        c['unregistered_or_empty_rate'] = (c['unregistered']+c['empty'])/c['total']
    if complete and (len(cells)!=32 or any(c['total']!=60 for c in cells.values())):
        raise ContractViolation('INCOMPLETE_CELLS')
    robust = {f'{i}|{l}':min(cells[f'{i}|{w}|{l}']['A'] for w in ('dev1','dev2')) for i in LANGS for l in LANGS} if complete else None
    validity = complete and all(c['unregistered']+c['empty']<=6 for c in cells.values())
    return {'cells':cells, 'A_robust':robust, 'output_validity_pass':validity,
        'robustness_pass':bool(validity and all(v>=.9 for v in robust.values())),
        'interpretation_status': 'INCOMPLETE' if not complete else ('BLOCKED_OUTPUT_VALIDITY' if not validity else 'REGISTERED_TASK_ONLY')}


def save_progress(directory, rows, total):
    """Use an object envelope: the shared artifact writer rejects list roots."""
    done = len(rows)
    if not 0 < done <= total or done % 60:
        raise ContractViolation('INVALID_PROGRESS_BOUNDARY')
    return publish_json_once(directory/f'rows_{done:04d}.json', {
        'schema_version':'qwen-progress-v1', 'completed':done, 'total':total,
        'rows':rows[-60:], 'contract_sha256':CONTRACT_SHA})


def progress():
    """Read-only terminal display; final errors must not look like a live run."""
    final = RUN/'run_summary.json'
    if final.exists():
        result = read_verified_json(final)
        done, status = len(result['rows']), result['status']
    else:
        files = sorted(RUN.glob('rows_*.json'))
        done = read_verified_json(files[-1])['completed'] if files else 0
        status = '마지막 저장 위치 · 실행 여부는 GPU/프로세스 상태 확인'
    width=40; filled=done*width//1920
    print('['+'█'*filled+'░'*(width-filled)+']', f'{done}/1920 ({done/1920:.1%})')
    print(status)


def prepare():
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    from .src.budget import require_disk_reservation
    if CONTRACT.exists():
        raise ContractViolation('CONTRACT_ALREADY_EXISTS')
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ContractViolation('MODEL_REVISION_MISMATCH')
    files = [f for f in info.siblings if f.rfilename in ('LICENSE','README.md','config.json','generation_config.json',
        'merges.txt','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json')]
    if len(files)!=9 or sum(f.size for f in files)>1_300_000_000:
        raise ContractViolation('MODEL_DOWNLOAD_SIZE_MISMATCH')
    require_disk_reservation(WORK_ROOT, planned_bytes=sum(f.size for f in files)+100*1024**2,
        emergency_free_bytes=5*1024**3)
    refs = []
    for f in files:
        path = Path(hf_hub_download(MODEL, f.rfilename, revision=REVISION, local_dir=DIRECTORY))
        digest = sha256_file(path)
        if path.stat().st_size!=f.size or (f.lfs and digest!=f.lfs.sha256):
            raise ContractViolation('MODEL_FILE_MISMATCH')
        refs.append({'path':str(path), 'sha256':digest, 'bytes':f.size})
    tokenizer = AutoTokenizer.from_pretrained(DIRECTORY, local_files_only=True, trust_remote_code=False)
    rows = records()
    for r in rows:
        ids = tokenizer.encode(r['prefix'], add_special_tokens=False)
        if not ids or len(ids)+64>32768 or tokenizer.decode(ids, clean_up_tokenization_spaces=False)!=r['prefix']:
            raise ContractViolation('PROMPT_TOKENIZATION_MISMATCH')
        r['prefix_token_ids'] = ids
    contract = {'model':MODEL, 'revision':REVISION, 'type':'base', 'files':refs,
        'parent_sha256':PARENT_SHA, 'records':rows, 'records_sha256':_hash(rows),
        'generation':'existing greedy_first_line; no cache; exact prefix; no special tokens; newline required; max 64; no EOS override',
        'dtype':'float32', 'attention':'eager', 'context_length':32768, 'reserved_seconds':3600,
        'plan_sha256':sha256_file(PROJECT_ROOT/'implementation/PRETRAINED_ROBUSTNESS_EXPLORATION_V1.md'),
        'training':False, 'automatic_expansion':False, 'main_enabled':False}
    print(publish_json_once(CONTRACT, contract), flush=True)


def run():
    from .src.gpu_guard import require_literal_gpu2_mask, require_physical_gpu2, assert_model_on_bound_device
    require_literal_gpu2_mask()
    from .src.integrity import verify_implementation_manifest, verify_cpu_check_evidence
    from .src.budget import BudgetCaps, GpuBudgetLedger, ExclusiveGpu2Lock, require_disk_reservation
    from .src.train import BoundaryStopController
    from .src.score import greedy_first_line, semantic_model_fingerprint
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    contract = read_verified_json(CONTRACT, expected_sha256=CONTRACT_SHA)
    code = verify_implementation_manifest(MANIFEST)['implementation_manifest_sha256']
    cpu = verify_cpu_check_evidence(CPU)
    if cpu['implementation']['implementation_manifest_sha256']!=code:
        raise ContractViolation('CPU_CODE_MISMATCH')
    expected = records()
    if (contract['model']!=MODEL or contract['revision']!=REVISION or contract['records_sha256']!=_hash(contract['records'])
        or [{k:v for k,v in r.items() if k!='prefix_token_ids'} for r in contract['records']]!=expected
        or contract['plan_sha256']!=sha256_file(PROJECT_ROOT/'implementation/PRETRAINED_ROBUSTNESS_EXPLORATION_V1.md')):
        raise ContractViolation('CONTRACT_MISMATCH')
    for ref in contract['files']:
        if sha256_file(Path(ref['path']))!=ref['sha256']:
            raise ContractViolation('MODEL_FILE_MISMATCH')
    if RUN.exists():
        raise ContractViolation('NEW_RUN_REQUIRED')
    require_disk_reservation(WORK_ROOT, planned_bytes=100*1024**2, emergency_free_bytes=5*1024**3)
    policy = read_verified_json(PROJECT_ROOT/'implementation/config/campaign_policy.json')['budget']
    ledger = GpuBudgetLedger(WORK_ROOT/'ledger/gpu_budget.json', BudgetCaps(policy['campaign_gpu_hours_cap'],
        policy['prior_gpu_hours_user_reported'], policy['per_root_gpu_hours_cap']))
    RUN.mkdir(parents=True); rows=[]; status='NOT_RUN'; binding=None; unchanged=False
    with ExclusiveGpu2Lock(WORK_ROOT/'locks/gpu2.lock'):
        lease = ledger.reserve(run_id=RUN.name, root_id=None, phase='QWEN_BASE_READ_ONLY', reserved_seconds=3600)
        try:
            binding = require_physical_gpu2()
            torch.manual_seed(4101); torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32=False
            tokenizer = AutoTokenizer.from_pretrained(DIRECTORY, local_files_only=True, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(DIRECTORY, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.float32, attn_implementation='eager').to('cuda:0').eval()
            model.requires_grad_(False); assert_model_on_bound_device(model, binding)
            before = semantic_model_fingerprint(model)
            with BoundaryStopController() as stop:
                def score(r):
                    ledger.require_time_remaining(lease, checkpoint_grace_seconds=120)
                    if stop.requested:
                        raise ContractViolation('INTERRUPTED')
                    if tokenizer.encode(r['prefix'], add_special_tokens=False)!=r['prefix_token_ids']:
                        raise ContractViolation('PREFIX_CHANGED')
                    return {**r, 'generation':greedy_first_line(model, tokenizer, r['prefix'], r['answers'],
                        r['requested_language'], context_length=32768, max_new_tokens=64)}
                first = [score(r) for r in contract['records'][:8]]
                second = [score(r) for r in contract['records'][:8]]
                if first!=second:
                    raise ContractViolation('GPU_REPEAT_MISMATCH')
                publish_json_once(RUN/'gpu_smoke.json', {'status':'PASS', 'first':_hash(first), 'second':_hash(second), 'gpu':binding.as_dict()})
                for r in contract['records']:
                    row=score(r); rows.append(row)
                    if len(rows)%60==0:
                        save_progress(RUN, rows, len(contract['records']))
                        ledger.heartbeat(lease)
                        print(f'QWEN_EVAL {len(rows)}/1920', flush=True)
            unchanged = semantic_model_fingerprint(model)==before
            if not unchanged:
                raise ContractViolation('WEIGHTS_CHANGED')
            if [r['record_id'] for r in rows]!=[r['record_id'] for r in contract['records']]:
                raise ContractViolation('OUTPUT_RECORD_ORDER_MISMATCH')
            summarize(rows, complete=True)
            verify_implementation_manifest(MANIFEST)
            status='COMPLETE'
        except Exception as exc:
            status=f'{type(exc).__name__}: {exc}'
        finally:
            try:
                ledger.finish(lease, status)
            except Exception as exc:
                status=f'LEDGER_FINALIZATION_ERROR: {type(exc).__name__}: {exc}; evaluation={status}'
    result={'status':status, 'contract_sha256':sha256_file(CONTRACT), 'code_sha256':code,
        'cpu_sha256':sha256_file(CPU), 'model':MODEL, 'revision':REVISION, 'rows':rows,
        **summarize(rows, complete=status=='COMPLETE'), 'weights_unchanged':unchanged,
        'gpu':binding.as_dict() if binding else None, 'updates':0, 'budget':ledger.snapshot(),
        'automatic_expansion':False, 'main_enabled':False}
    print(publish_json_once(RUN/'run_summary.json', result), flush=True)
    return 0 if status=='COMPLETE' else 20


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare','run','progress'))
    args=parser.parse_args()
    raise SystemExit({'prepare':prepare, 'run':run, 'progress':progress}[args.action]())
