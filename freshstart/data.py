"""Schema/coverage auditing is NOT independent verification of human annotations."""
from __future__ import annotations
from collections import Counter
from . import LANGUAGES
from .core import canonical, membership, ContractError

ETY_LABELS={'BORROWING_DOCUMENTED','SHARED_SOURCE_DOCUMENTED','DISTINCT_ROUTES_REVIEWED','UNRESOLVED'}

def validate_concept(row: dict) -> dict:
    for key in ('concept_id','snapshot_id','source_urls','source_record_hashes','answers','glosses','qa','cluster_id'):
        if key not in row: raise ContractError('MISSING_CONCEPT_FIELD:'+key)
    if not row['concept_id'] or not row['snapshot_id'] or not row['source_urls'] or not row['source_record_hashes']:
        raise ContractError('MISSING_PROVENANCE')
    if set(row['glosses'])!=set(LANGUAGES): raise ContractError('FOUR_GLOSSES_REQUIRED')
    m=membership(row['answers'])
    for l in LANGUAGES:
        if not isinstance(row['glosses'][l],str) or not row['glosses'][l].strip(): raise ContractError('EMPTY_GLOSS:'+l)
    if row.get('synthetic_fixture',False): raise ContractError('TEST_FIXTURE_IS_NOT_TRAINING_DATA')
    qa=row['qa']
    if qa.get('status')!='APPROVED_BY_RESEARCHER' or not qa.get('reviewer') or not qa.get('review_date'):
        raise ContractError('DATA_QA_PENDING')
    if not qa.get('source_alignment_checked') or not qa.get('answer_copy_checked'):
        raise ContractError('DATA_QA_INCOMPLETE')
    return {'concept_id':row['concept_id'],'identifiable':all(len(ls)==1 for ls in m.values()),
            'memberships':{y:list(ls) for y,ls in m.items()}}

def validate_etymology(row: dict) -> str:
    label=row.get('relation')
    if label not in ETY_LABELS: raise ContractError('UNKNOWN_ETYMOLOGY_LABEL')
    if row.get('pair')!=['en','fr']: raise ContractError('PRIMARY_ETYMOLOGY_PAIR_MUST_BE_EN_FR')
    if label!='UNRESOLVED':
        for k in ('evidence_urls','evidence_record_hashes','reviewer','review_date','sense_alignment_note','historical_scope','family_id'):
            if not row.get(k): raise ContractError('ETYMOLOGY_EVIDENCE_MISSING:'+k)
        if row.get('evidence_origin')=='model_generated': raise ContractError('MODEL_GENERATED_ETYMOLOGY_NOT_VERIFIED')
    return label

def audit_dataset(concepts: list[dict], etymology: list[dict], requirements: dict) -> dict:
    ids=[r.get('concept_id') for r in concepts]
    if len(set(ids))!=len(ids): raise ContractError('DUPLICATE_CONCEPT_ID')
    checked=[validate_concept(r) for r in concepts]
    eids=[r.get('concept_id') for r in etymology]
    if len(set(eids))!=len(eids) or set(eids)!=set(ids): raise ContractError('ETYMOLOGY_ROW_SET_MISMATCH')
    labels={r['concept_id']:validate_etymology(r) for r in etymology}
    identifiable=[r['concept_id'] for r in checked if r['identifiable']]
    counts=Counter(labels[c] for c in identifiable)
    related=counts['BORROWING_DOCUMENTED']+counts['SHARED_SOURCE_DOCUMENTED']
    control=counts['DISTINCT_ROUTES_REVIEWED']
    reasons=[]
    if len(ids)<requirements['min_total']: reasons.append('TOTAL_COVERAGE')
    if len(ids)%2: reasons.append('EVEN_COHORT_REQUIRED_FOR_PAIRS')
    if len(identifiable)<requirements['min_identifiable']: reasons.append('IDENTIFIABLE_COVERAGE')
    if related<requirements['min_related']: reasons.append('RELATED_COVERAGE')
    if control<requirements['min_distinct_routes']: reasons.append('REVIEWED_CONTROL_COVERAGE')
    return {'status':'PASS' if not reasons else 'BLOCKED_DATA_COVERAGE', 'reasons':reasons,
            'n_total':len(ids),'n_identifiable':len(identifiable),'n_shared':len(ids)-len(identifiable),
            'identifiable_concept_ids':sorted(identifiable),'relation_counts_identifiable':dict(counts),
            'requirements':requirements,
            'limits':'Checks schema and recorded sign-offs only; does not verify etymological truth or establish statistical power.'}
