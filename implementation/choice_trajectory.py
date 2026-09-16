"""CPU-only retrospective trajectories from checksum-bound saved evaluations."""
import csv
import math
import statistics
from pathlib import Path
from .src.artifacts import read_verified_json,publish_json_once,publish_bytes_once,sha256_file
from .src.contracts import WORK_ROOT,PROJECT_ROOT,ContractViolation
from .qwen_robustness import PARENT,PARENT_SHA,LANGS

OUT=WORK_ROOT/'reports/choice_trajectory_v1'
SOURCE=WORK_ROOT/'runs/pilot_jm02_4e30ec2/run_summary_000263.json'
PHASES=('T0','T1','T2','T3')
WRAPPERS=('dev1','dev2')


def primary_cells(score_rows,record_rows,ids,checkpoint_sha):
    bank={r['record_id']:r for r in record_rows}
    if len(bank)!=len(record_rows):raise ContractViolation('DUPLICATE_RECORD')
    cells={i:{w:{'requested':{}} for w in WRAPPERS} for i in ids};seen=set()
    for r in score_rows:
        if r['input_language']!='ko' or r['format']!='RD' or r['wrapper'] not in WRAPPERS:continue
        if r['record_id'] in seen:raise ContractViolation('DUPLICATE_SCORE')
        seen.add(r['record_id']);rec=bank[r['record_id']]
        if r['checkpoint_sha256']!=checkpoint_sha:raise ContractViolation('MIXED_CHECKPOINTS')
        if r['concept_id'] not in cells or any(r[k]!=rec[k] for k in ('concept_id','input_language','format','wrapper','mode')):
            raise ContractViolation('RECORD_SCOPE_MISMATCH')
        cell=cells[r['concept_id']][r['wrapper']];g=r['generation']
        if r['mode']=='ANY':
            if 'Q' in cell:raise ContractViolation('DUPLICATE_ANY')
            p=r['probability'];q=p['Q_membership']
            if q is None or set(q)!=set(LANGS) or not all(math.isfinite(v) and 0<=v<=1 for v in q.values()) or not math.isclose(sum(q.values()),1,abs_tol=1e-12):
                raise ContractViolation('INVALID_Q')
            cell.update(Q=q,Z=p['Z'],any_membership=g['membership'],any_class=g['class'])
        else:
            target=rec['requested_language']
            if target in cell['requested']:raise ContractViolation('DUPLICATE_REQUEST')
            if g['request_compatible']!=(target in g['membership']):raise ContractViolation('GENERATION_MEMBERSHIP_MISMATCH')
            cell['requested'][target]={'success':g['request_compatible'],'class':g['class'],'membership':g['membership']}
    if any('Q' not in c or 'ko' not in c['requested'] for d in cells.values() for c in d.values()):
        raise ContractViolation('INCOMPLETE_PRIMARY_CELLS')
    return cells


def aggregate(cells):
    ids=sorted(cells);n=len(ids);wrappers={}
    for w in WRAPPERS:
        active=set(cells[ids[0]][w]['requested'])
        if any(set(cells[i][w]['requested'])!=active for i in ids):raise ContractViolation('INCONSISTENT_REQUEST_SET')
        wrappers[w]={'A':{l:sum(cells[i][w]['requested'][l]['success'] for i in ids)/n for l in sorted(active)},
            'A_counts':{l:sum(cells[i][w]['requested'][l]['success'] for i in ids) for l in sorted(active)},
            'Q_mean':{l:statistics.mean(cells[i][w]['Q'][l] for i in ids) for l in LANGS},
            'Z_median':statistics.median(cells[i][w]['Z'] for i in ids),
            'low_Z_fraction':sum(cells[i][w]['Z']<.05 for i in ids)/n,
            'ANY_registered_language_counts':{l:sum(l in cells[i][w]['any_membership'] for i in ids) for l in LANGS},
            'ANY_unregistered_or_empty_count':sum(not cells[i][w]['any_membership'] for i in ids)}
    return {'n_concepts':n,'wrappers':wrappers}


def main():
    if OUT.exists():raise ContractViolation('NEW_ANALYSIS_DIRECTORY_REQUIRED')
    parent=read_verified_json(PARENT,expected_sha256=PARENT_SHA);concepts={c['concept_id']:c for c in parent['concepts']};ids=sorted(concepts)
    if len(ids)!=60:raise ContractViolation('COHORT_MISMATCH')
    source=read_verified_json(SOURCE);events=[e['data'] for e in source['events'] if e['kind']=='EVALUATION']
    if len(events)!=60:raise ContractViolation('UNEXPECTED_EVALUATION_COUNT')
    snapshots=[];refs=[];seen=set();endpoint_cells={}
    for event in events:
        phase,step=event['phase'],event['phase_step'];key=(phase,step)
        if key in seen or phase not in PHASES:raise ContractViolation('DUPLICATE_OR_UNKNOWN_TIMEPOINT')
        seen.add(key)
        score=read_verified_json(Path(event['scores']['path']),expected_sha256=event['scores']['sha256'])
        records=read_verified_json(Path(event['records']['path']),expected_sha256=event['records']['sha256'])['records']
        ref=event['checkpoint'];manifest=read_verified_json(Path(ref['path'])/'manifest.json',expected_sha256=ref['manifest_sha256'])
        if score['provenance']['checkpoint_sha256']!=ref['state_sha256'] or score['provenance']['model_fingerprint']!=ref['model_fingerprint']:
            raise ContractViolation('SCORE_CHECKPOINT_BINDING_MISMATCH')
        for r in records:
            if r['answers']!=concepts[r['concept_id']]['answers']:raise ContractViolation('ANSWER_SET_CHANGED')
        cells=primary_cells(score['rows'],records,ids,ref['state_sha256'])
        point={'phase':phase,'phase_step':step,'global_step':manifest['progress']['global_step'],
            'after_T0_updates':manifest['progress']['global_step']-3245,'endpoint':event['endpoint'],
            'aggregate':aggregate(cells),'cells':cells,'source_scores':event['scores'],'source_records':event['records'],
            'checkpoint_state_sha256':ref['state_sha256']}
        snapshots.append(point);refs.extend([event['scores'],event['records']])
        if event['endpoint']:endpoint_cells[phase]=cells
    snapshots.sort(key=lambda x:x['global_step'])
    if set(seen)!={(p,s) for p in PHASES for s in range(200,3001,200)}:raise ContractViolation('MISSING_TIMEPOINT')
    drift_path=WORK_ROOT/'runs/checkpoint_drift_probe_v1/run_summary.json';drift=read_verified_json(drift_path)
    eight={};eight_rows={p:[] for p in PHASES}
    for ref in drift['new_raw_chunks']:
        chunk=read_verified_json(Path(ref['path']),expected_sha256=ref['sha256'])
        if chunk['label'] in eight_rows:eight_rows[chunk['label']].extend(chunk['rows'])
    from .prompt_average_probe import summarize
    for phase in PHASES:
        c=drift['checkpoints'][phase];m=summarize(eight_rows[phase])
        if m!=c['measurement']:raise ContractViolation('EIGHT_WRAPPER_SUMMARY_MISMATCH')
        endpoint=next(x for x in snapshots if x['phase']==phase and x['endpoint'])
        if c['entry']['checkpoint']['state_sha256']!=endpoint['checkpoint_state_sha256']:raise ContractViolation('ENDPOINT_MODEL_MISMATCH')
        for row in eight_rows[phase]:
            if row['wrapper'] in ('w1','w6'):
                w={'w1':'dev1','w6':'dev2'}[row['wrapper']]
                if row['probability']['Q_membership']!=endpoint_cells[phase][row['concept_id']][w]['Q']:
                    raise ContractViolation('ORIGINAL_WRAPPER_Q_NOT_REPRODUCED')
        eight[phase]={'Q_by_concept':m['all_concept_mean_Q'],
            'Q_mean':{l:statistics.mean(q[l] for q in m['all_concept_mean_Q'].values()) for l in LANGS},
            'Z_median_by_wrapper':{w:v['median'] for w,v in m['Z_by_wrapper'].items()},
            'worst_gate':c['worst_gate']}
    matched=[]
    for i in ids:
        ncorrect=sum(endpoint_cells['T3'][i][w]['requested']['ko']['success'] for w in WRAPPERS)
        q=eight['T3']['Q_by_concept'][i]
        matched.append({'concept_id':i,'ko_expression':concepts[i]['answers']['ko'],'T3_KO_success_wrappers':ncorrect,
            'T0_KO_success_wrappers':sum(endpoint_cells['T0'][i][w]['requested']['ko']['success'] for w in WRAPPERS),
            'T3_Q8_KO':q['ko'],'T3_Q8_FR':q['fr'],'T3_Q8_max_language':max(q,key=q.get),
            'T3_ANY_FR_wrappers':sum('fr' in endpoint_cells['T3'][i][w]['any_membership'] for w in WRAPPERS)})
    retained=[r for r in matched if r['T0_KO_success_wrappers']==2 and r['T3_KO_success_wrappers']==2]
    heterogeneity={}
    for phase in PHASES:
        vals=sorted(eight[phase]['Q_by_concept'][i]['ko'] for i in ids)
        heterogeneity[phase]={'KO_Q8_min':min(vals),'KO_Q8_max':max(vals),'KO_Q8_median':statistics.median(vals),'KO_Q8_gt_05_count':sum(v>.5 for v in vals)}
    increases={a+'->'+b:sum(eight[b]['Q_by_concept'][i]['ko']-eight[a]['Q_by_concept'][i]['ko']>1e-8 for i in ids) for a,b in zip(PHASES,PHASES[1:])}
    result={'status':'COMPLETE_CPU_ONLY','root_id':4101,'n_concepts':60,'saved_timepoints':len(snapshots),
        'dense_timepoints_after_T0':sum(p['after_T0_updates']>=0 for p in snapshots),
        'source':{'path':str(SOURCE),'sha256':sha256_file(SOURCE)},'source_artifacts':refs,
        'drift_source':{'path':str(drift_path),'sha256':sha256_file(drift_path)},
        'code_sha256':sha256_file(Path(__file__)),'snapshots':snapshots,'eight_wrapper_endpoints':eight,
        'matched_concepts':matched,'endpoint_heterogeneity':heterogeneity,'nonmonotone_KO_increase_counts':increases,
        'retained_KO_both_wrappers':{'count':len(retained),'FR_max_Q8_count':sum(r['T3_Q8_max_language']=='fr' for r in retained),
            'KO_Q8_lt_01_count':sum(r['T3_Q8_KO']<.1 for r in retained),'median_T3_Q8_KO':statistics.median(r['T3_Q8_KO'] for r in retained)},
        'scope_limits':['Requested accuracy is registered-expression production, not general language ability.',
          'Q8 endpoints use 8 ANY templates; requested accuracy uses the 2 original corresponding REQUESTED templates.',
          'Dense timepoints use only original dev1/dev2; they are not 8-template measurements or stable-trait estimates.',
          'One root, known concepts, exposure/language order/recency/updates confounded; no causal independent-history claim.',
          'Accuracy decreases from T0; preserved capability is not equivalence within +/-2 percentage points.',
          'The 29/31 English-input requested-language error statistic is excluded from KO-input ANY trajectories.'],
        'training_started':False,'GPU_used':False,'H_started':False,'main_enabled':False}
    OUT.mkdir(parents=True)
    ref=publish_json_once(OUT/'trajectory_summary.json',result)
    with (OUT/'concept_endpoint_trajectories.csv').open('x',newline='') as f:
        fields=['concept_id','ko_expression','phase']+['Q8_'+l for l in LANGS]+['KO_requested_dev1','KO_requested_dev2']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i in ids:
            for phase in PHASES:
                writer.writerow({'concept_id':i,'ko_expression':concepts[i]['answers']['ko'],'phase':phase,
                    **{'Q8_'+l:eight[phase]['Q_by_concept'][i][l] for l in LANGS},
                    **{'KO_requested_'+w:endpoint_cells[phase][i][w]['requested']['ko']['success'] for w in WRAPPERS}})
    print(ref)

if __name__=='__main__':main()
