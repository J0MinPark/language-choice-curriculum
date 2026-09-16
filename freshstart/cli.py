from __future__ import annotations
import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from . import LANGUAGES, VERSION
from .core import ContractError
from .data import audit_dataset
from .krdict import collect
from .reports import write_json, render_json_report, measurement, sha256_file
from .schedule import make_assignment, history_schedule, audit_schedules

ROOT=Path(__file__).resolve().parents[1]

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def load_config():
    cfg=read_json(ROOT/'spec/pilot.json')
    if cfg.get('version')!=VERSION or cfg.get('languages')!=list(LANGUAGES): raise ContractError('BLOCKED_SPEC_VERSION')
    if len(cfg.get('roots',[]))!=4 or len(set(cfg['roots']))!=4: raise ContractError('INVALID_ROOT_PLAN')
    if cfg['policy'].get('main_enabled'): raise ContractError('MAIN_NOT_AUTHORIZED_IN_V4')
    return cfg

def new_report_path(kind):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return ROOT/'work/reports'/f'{kind}_{stamp}_{uuid.uuid4().hex[:8]}.json'

def emit(result, path):
    if path.exists() and read_json(path)!=result:
        raise ContractError('REPORT_EXISTS_USE_NEW_RUN_PATH')
    write_json(path,result); render_json_report(path,path.with_suffix('.md'))
    print(json.dumps({'status':result['status'],'report_json':str(path),'report_markdown':str(path.with_suffix('.md'))},ensure_ascii=False))

def main(argv=None):
    p=argparse.ArgumentParser(description='v4 reference tools. Full LM training is an implementation task for Claude, not included here.')
    sub=p.add_subparsers(dest='command',required=True)
    pf=sub.add_parser('preflight',help='Check credentials locally; no network, no GPU')
    pf.add_argument('--snapshot-manifest',help='Optional lawful local source manifest; provenance not yet validated')
    f=sub.add_parser('fetch',help='Bounded exact-query API collection; no retries/no training')
    f.add_argument('--queries',default=str(ROOT/'templates/queries.txt'))
    f.add_argument('--out',required=True)
    d=sub.add_parser('data-check'); d.add_argument('--concepts',required=True);d.add_argument('--etymology',required=True)
    d.add_argument('--out',required=True)
    m=sub.add_parser('measure');m.add_argument('--rows',required=True);m.add_argument('--cohort',required=True)
    m.add_argument('--metadata',required=True);m.add_argument('--out',required=True)
    r=sub.add_parser('render');r.add_argument('--json',required=True);r.add_argument('--out',required=True)
    s=sub.add_parser('schedule-smoke');s.add_argument('--out',required=True)
    args=p.parse_args(argv)
    try:
        cfg=load_config()
        if args.command=='preflight':
            key=os.environ.get('KRDICT_API_KEY','')
            present=bool(key); format_valid=bool(re.fullmatch(r'[0-9a-fA-F]{32}',key))
            snapshot=args.snapshot_manifest or cfg['data'].get('raw_snapshot_manifest')
            snapshot_present=bool(snapshot and Path(snapshot).is_file())
            ready=format_valid or snapshot_present
            result={'version':VERSION,'status':'READY_FOR_SOURCE_CHECK' if ready else 'BLOCKED_CREDENTIALS',
                    'key_present':present,'key_format_valid':format_valid,
                    'key_validated_by_provider':False,'snapshot_manifest_present':snapshot_present,
                    'network_requests':0,'GPU_jobs_started':0,
                    'source_payload_status':'NOT_RUN','data_qa_status':'NOT_RUN',
                    'training_implementation_status':'NOT_INCLUDED_IMPLEMENT_USING_SPEC',
                    'pilot_authorized_after_gates':cfg['policy']['pilot_authorized_after_gates'],
                    'main_enabled':False,'config_sha256':sha256_file(ROOT/'spec/pilot.json'),
                    'next_action':'Validate raw snapshot provenance or bounded API response, then candidate QA.' if ready else 'Obtain KRDICT key and export it privately in the terminal, or supply a lawful raw snapshot. Do not retry missing credentials.'}
            emit(result,new_report_path('preflight'))
            return 0 if ready else 10
        if args.command=='fetch':
            result=collect(Path(args.queries),Path(args.out),cfg['budget']['api_requests_cap'])
            emit(result,Path(args.out).parent/f'collection_{Path(args.out).name}.json')
            return 0 if result['status']=='COLLECTED_UNREVIEWED' else 10
        if args.command=='data-check':
            result=audit_dataset(read_jsonl(args.concepts),read_jsonl(args.etymology),cfg['data'])
            result.update({'version':VERSION,'concept_file_sha256':sha256_file(Path(args.concepts)),
                           'etymology_file_sha256':sha256_file(Path(args.etymology))})
            emit(result,Path(args.out)); return 0 if result['status']=='PASS' else 10
        if args.command=='measure':
            result=measurement(read_jsonl(args.rows),read_json(args.cohort)['identifiable_concept_ids'],read_json(args.metadata),cfg['measurement'])
            result['evaluation_file_sha256']=sha256_file(Path(args.rows))
            result['cohort_file_sha256']=sha256_file(Path(args.cohort))
            emit(result,Path(args.out));return 0 if result['status']=='PASS' else 10
        if args.command=='render':
            render_json_report(Path(args.json),Path(args.out));return 0
        if args.command=='schedule-smoke':
            pairs=[(f'TEST_{i:02d}a',f'TEST_{i:02d}b') for i in range(30)]
            assignment=make_assignment(pairs,4101)
            result=audit_schedules(history_schedule(assignment,'A'),history_schedule(assignment,'B'))
            result.update({'version':VERSION,'data_kind':'SYNTHETIC_TEST_FIXTURE','GPU_jobs_started':0})
            emit(result,Path(args.out));return 0
    except (ContractError,KeyError,ValueError,OSError) as e:
        # Do not emit arbitrary exception strings containing possible secret-bearing URLs.
        result={'version':VERSION,'status':'BLOCKED_INPUT_OR_CONTRACT','error_type':type(e).__name__}
        if isinstance(e,ContractError): result['safe_error_code']=str(e)
        emit(result,new_report_path('error'));return 20
    return 0

if __name__=='__main__': raise SystemExit(main())
