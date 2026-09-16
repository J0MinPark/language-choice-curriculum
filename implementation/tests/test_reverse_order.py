import json
import unittest
from collections import Counter
from dataclasses import replace
from unittest import mock

from implementation.src.contracts import LANGUAGES, PROJECT_ROOT, ContractViolation
from implementation.src.records import (REVERSE_LANGUAGE_ORDER as REVERSE, stage_active_languages,
    StageConceptView, materialize_record, RecordSlot)
from implementation.src.schedule import build_stage_plan, audit_stage_plan
from implementation.src.pilot_streams import evaluation_records
from implementation.src.score import _validate_real_record
from implementation.src.pilot_runtime import PilotRuntime, phase_order
from implementation.src.pilot_execution_freeze import REVERSE_REVISION
from implementation.tests.test_train_checkpoint import fixture_concept


class ReverseOrderTests(unittest.TestCase):
    def test_order_pairs_factor_roles_without_changing_concept_or_batch_order(self):
        ids=[f'c{i}' for i in range(60)];mapping=dict(zip(LANGUAGES,REVERSE))
        expected={'T1':{'ko':240,'fr':240},'T2':{'ko':120,'fr':120,'zh':240},
                  'T3':{'ko':80,'fr':80,'zh':80,'en':240}}
        for stage in expected:
            f=build_stage_plan(ids,root_id=4101,stage=stage,optimizer_updates=15)
            r=build_stage_plan(ids,root_id=4101,stage=stage,optimizer_updates=15,language_order=REVERSE)
            self.assertEqual(len(f.batches),len(r.batches))
            self.assertEqual(Counter(x.target_language for x in r.slots),expected[stage])
            for a,b in zip(f.slots,r.slots):
                self.assertEqual((a.concept_id,mapping[a.input_language],mapping[a.target_language],a.mode,a.wrapper,a.occurrence),
                                 (b.concept_id,b.input_language,b.target_language,b.mode,b.wrapper,b.occurrence))
            bad=list(r.slots);bad[0]=replace(bad[0],target_language='en' if stage=='T1' else 'ko')
            with self.assertRaises(ContractViolation):audit_stage_plan(bad,root_id=4101,stage=stage,language_order=REVERSE)

    def test_training_uses_actual_french_answer_and_blocks_future_english(self):
        c=fixture_concept('c');policy=PROJECT_ROOT/'implementation/config/evaluation_plan_v4_1_2.json'
        slot=RecordSlot('r','c','ko','fr','REQUESTED','train1',0)
        record=materialize_record(c,slot,stage='T1',language_order=REVERSE,policy_path=policy)
        self.assertEqual(record.canonical_answer,c['answers']['fr'])
        with self.assertRaisesRegex(ContractViolation,'FUTURE_LANGUAGE'):materialize_record(c,slot,stage='T1',policy_path=policy)
        with self.assertRaisesRegex(ContractViolation,'FUTURE_LANGUAGE'):materialize_record(c,replace(slot,target_language='en'),stage='T1',language_order=REVERSE,policy_path=policy)
        view=StageConceptView(c,('ko','fr'),language_order=REVERSE)
        with self.assertRaisesRegex(ContractViolation,'FUTURE_INPUT'):view.gloss('en')
        with self.assertRaises(ContractViolation):stage_active_languages('T1',('fr','ko','zh','en'))

    def test_evaluation_order_is_bound_and_never_exposes_future_languages(self):
        c=fixture_concept('c');plan=json.loads((PROJECT_ROOT/'implementation/config/evaluation_plan_v4_1_2.json').read_text())
        for stage in ('T1','T2','T3'):
            records=evaluation_records([c],plan,stage,endpoint=True,language_order=REVERSE)
            active=set(stage_active_languages(stage,REVERSE))
            self.assertEqual({r['input_language'] for r in records},active)
            self.assertEqual({r['requested_language'] for r in records if r['mode']=='REQUESTED'},active)
            self.assertTrue(all(set(r['answers'])==set(LANGUAGES) for r in records))
            for r in records:_validate_real_record(r,concepts={'c':c},evaluation_plan=plan,stage=stage,language_order=REVERSE)
        row=next(r for r in evaluation_records([c],plan,'T1',language_order=REVERSE) if r.get('requested_language')=='fr')
        with self.assertRaisesRegex(ValueError,'FUTURE_REQUESTED'):_validate_real_record(row,concepts={'c':c},evaluation_plan=plan,stage='T1')

    def test_single_paired_run_has_no_new_init_or_history(self):
        execution={'roots':[4101],'start_phase':'T1','history_enabled':False}
        self.assertEqual(phase_order(execution),[(4101,p) for p in ('T1','T2','T3')])
        r=PilotRuntime.__new__(PilotRuntime);r.freeze={'exploratory_revision':REVERSE_REVISION,'execution':execution}
        r._root_gates=mock.Mock(side_effect=AssertionError('outcome gate is diagnostic'))
        r._training_gates(4101,'T1')
        for root,phase in ((4102,'T1'),(4101,'H_A')):
            with self.assertRaises(ContractViolation):r._training_gates(root,phase)


if __name__=='__main__':unittest.main()
