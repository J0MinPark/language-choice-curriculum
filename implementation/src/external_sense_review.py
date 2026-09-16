"""Import human review without rewriting it; apply the approved exact-sense rule."""
import csv
import io
from collections import Counter
from pathlib import Path

from .artifacts import read_verified_json, read_regular_file_bytes, publish_json_once, publish_bytes_once
from .contracts import ContractViolation, WORK_ROOT
from .external_benchmark import parse_things

FIELDS = ['external_id','external_synset','external_word','external_definition','jm_concept_id',
          'ko','en','zh','fr','jm_english_definition','review_status','reviewer','review_comment']
SCOPE_CONFLICTS = {'cucumber', 'strawberry'}


def expected_rows(audit, things, concepts):
    result = []
    for match in audit['lexical_overlap_candidates']:
        r = things[match['external_id']]
        for cid in match['current_concept_ids']:
            c = concepts[cid]
            result.append({'external_id':r['id'], 'external_synset':r['synset'],
                'external_word':r['word'], 'external_definition':r['description'],
                'jm_concept_id':cid, **c['answers'], 'jm_english_definition':c['glosses']['en']})
    return result


def apply_review(raw, expected):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if reader.fieldnames != FIELDS:
        raise ContractViolation('REVIEW_COLUMNS_CHANGED')
    rows = list(reader)
    key = lambda r: (r['external_id'], r['jm_concept_id'])
    originals = {key(r):r for r in expected}
    if len(rows) != len(originals) or len({key(r) for r in rows}) != len(rows):
        raise ContractViolation('REVIEW_MISSING_OR_DUPLICATE_ROWS')
    decisions = []
    for row in rows:
        original = originals.get(key(row))
        if original is None or any(row[k] != original[k] for k in FIELDS[:-3]):
            raise ContractViolation('REVIEW_ORIGINAL_FIELDS_CHANGED')
        if row['review_status'] not in ('승인','수정 필요','판정 유보') or row['reviewer'] != 'jm02':
            raise ContractViolation('REVIEW_STATUS_OR_REVIEWER_INVALID')
        if row['review_status'] != '승인' and not row['review_comment'].strip():
            raise ContractViolation('REVIEW_REASON_REQUIRED')
        status = {'승인':'ACCEPTED_EXACT', '수정 필요':'EXCLUDED_CURRENT_LINK',
                  '판정 유보':'HELD_FOR_SENSE_REVIEW'}[row['review_status']]
        reason = 'HUMAN_REVIEW'
        if row['external_id'] in SCOPE_CONFLICTS and row['review_status'] == '승인':
            status = 'HELD_POLICY_CONFLICT'
            reason = 'PLANT_AND_FRUIT_VS_FRUIT_ONLY; inclusion is not equivalence'
        decisions.append({'human_review':row, 'effective_status':status, 'reason':reason})
    accepted = [d for d in decisions if d['effective_status'] == 'ACCEPTED_EXACT']
    for field in ('jm_concept_id','external_id','external_synset'):
        values = [d['human_review'][field] for d in accepted]
        if len(values) != len(set(values)):
            raise ContractViolation('ACCEPTED_MAPPING_NOT_ONE_TO_ONE')
    return {'status':'REVIEW_IMPORTED_WITH_POLICY_HOLDS',
        'policy':'EXACT_CONTEXTUAL_SENSE_ONLY_V1',
        'human_counts':dict(Counter(r['review_status'] for r in rows)),
        'effective_counts':dict(Counter(d['effective_status'] for d in decisions)),
        'decisions':decisions, 'accepted_mapping_one_to_one':True,
        'original_jm60_unchanged':True, 'external_dataset_rows_deleted':0,
        'automatic_relinking':False, 'training_enabled':False,
        'human_approval_implies_source_license_clearance':False}


def main():
    root = WORK_ROOT / 'reviews/external_benchmark_v1'
    raw = read_regular_file_bytes(root/'pending_sense_review.csv')
    audit = read_verified_json(WORK_ROOT/'reports/external_benchmark_intake_v1.json',
        expected_sha256='9c6c269a727b73e7ce5b0c6a014f0ab7d08ff15c2bec028dee5bbd34e66176e8')
    ref = audit['dataset']
    from .artifacts import sha256_file
    if sha256_file(Path(ref['path'])) != ref['sha256']:
        raise ContractViolation('EXTERNAL_SOURCE_HASH_MISMATCH')
    things = {r['id']:r for r in parse_things(read_regular_file_bytes(ref['path']))}
    freeze = read_verified_json(WORK_ROOT/'freeze/accuracy_selected_360_v3/experiment_freeze.json',
        expected_sha256='6f6cc0d52b4162f1018b560c6f95419cb9258deda7e053e0a12bdc7950e39dfd')
    expected = expected_rows(audit, things, {c['concept_id']:c for c in freeze['concepts']})
    result = apply_review(raw, expected)
    result['review_snapshot'] = publish_bytes_once(root/'reviewed_jm02_snapshot_v1.csv',raw)
    result['basis'] = {'intake_report_sha256':sha256_file(WORK_ROOT/'reports/external_benchmark_intake_v1.json'),
                       'dataset_sha256':ref['sha256']}
    print(publish_json_once(root/'sense_decisions_v1.json',result))
    accepted = [d['human_review'] for d in result['decisions'] if d['effective_status']=='ACCEPTED_EXACT']
    print(publish_json_once(root/'accepted_mapping_v1.json',{'status':'REVIEWED_MAPPING_ONLY',
        'decision_report_sha256':sha256_file(root/'sense_decisions_v1.json'), 'mappings':accepted,
        'evaluation':'NOT_RUN', 'training_enabled':False,
        'scope':'selected overlap subset, not full external benchmark or new 31-concept replacement for jm02'}))


if __name__ == '__main__':
    main()
