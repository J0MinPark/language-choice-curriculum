"""Target-slot schedule ONLY; prompts, batches and optimizer steps are separate."""
from __future__ import annotations
import random
from collections import Counter
from . import LANGUAGES
from .core import ContractError

def make_assignment(pairs: list[tuple[str,str]], seed: int) -> dict[str,int]:
    if not pairs:
        raise ContractError("EMPTY_PAIRS")
    flat=[x for pair in pairs for x in pair]
    if len(set(flat))!=len(flat):
        raise ContractError("REPEATED_CONCEPT_IN_PAIRS")
    rng=random.Random(seed)
    out={}
    for a,b in pairs:
        g=rng.choice([-1,1]); out[a]=g; out[b]=-g
    return out

def history_schedule(assignment: dict[str,int], branch: str) -> list[dict]:
    if branch not in ('A','B') or not assignment or any(g not in (-1,1) for g in assignment.values()):
        raise ContractError("INVALID_SCHEDULE_ARGUMENT")
    if sum(assignment.values()) != 0:
        raise ContractError("UNBALANCED_ASSIGNMENT")
    concepts=sorted(assignment)
    counts=Counter(); rows=[]; round_id=0
    def emit(kind, language_for):
        nonlocal round_id
        for c in concepts:
            l=language_for(c)
            occurrence=counts[(c,l)]
            rows.append({"round":round_id,"concept_id":c,"target_language":l,
                         "occurrence":occurrence,"kind":kind,
                         "record_id":f"{c}|{l}|{occurrence}"})
            counts[(c,l)]+=1
        round_id+=1
    for m in range(72):
        def ml(c,m=m):
            orientation=assignment[c]*(1 if branch=='A' else -1)
            return ('en' if m<36 else 'fr') if orientation==1 else ('fr' if m<36 else 'en')
        emit('mutable', ml)
        if m%2==1:
            for l in ('ko','zh'):
                emit('anchor',lambda c,l=l:l)
    for _ in range(4):
        for l in LANGUAGES:
            emit('tail',lambda c,l=l:l)
    return rows

def audit_schedules(a: list[dict], b: list[dict]) -> dict:
    if not a or len(a)!=len(b): raise ContractError('SCHEDULE_LENGTH')
    def counts(rows): return Counter((r['concept_id'],r['target_language']) for r in rows)
    ca,cb=counts(a),counts(b)
    if ca!=cb or set(ca.values())!={40}: raise ContractError('TARGET_COUNTS')
    concepts={c for c,l in ca}
    if any({l for cc,l in ca if cc==c}!=set(LANGUAGES) for c in concepts): raise ContractError('INCORRECT_LANGUAGE_SET')
    for rows in (a,b):
        if {r['round'] for r in rows}!=set(range(160)): raise ContractError('INVALID_ROUND_IDS')
    if len(ca)!=4*len(concepts) or len(a)!=160*len(concepts): raise ContractError('NOT_4LANG_160_ROUNDS')
    if Counter(r['record_id'] for r in a)!=Counter(r['record_id'] for r in b): raise ContractError('RECORD_MULTISET')
    for kind in ('anchor','tail'):
        if [r for r in a if r['kind']==kind]!=[r for r in b if r['kind']==kind]: raise ContractError(kind.upper()+'_DIFFERS')
    def last(rows):
        out={}
        for r in rows: out[(r['concept_id'],r['target_language'])]=r['round']
        return out
    if last(a)!=last(b): raise ContractError('LAST_TARGET_ROUND_DIFFERS')
    return {'status':'PASS','concepts':len(concepts),'target_records_per_branch':len(a),
            'rounds':160,'per_concept_per_language':40,
            'scope':'target-slot schedule only; optimizer-step/batch and prompt records not tested here'}
