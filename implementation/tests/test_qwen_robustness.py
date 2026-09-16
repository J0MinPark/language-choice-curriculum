import unittest
import tempfile
from pathlib import Path
from implementation.qwen_robustness import LANGS, summarize, save_progress
from implementation.src.artifacts import read_verified_json, publish_json_once
from implementation.src.contracts import ContractViolation


class RobustnessTests(unittest.TestCase):
    def rows(self):
        return [{'record_id':f'{i}|{w}|{l}|{n}', 'input_language':i, 'wrapper':w,
            'requested_language':l, 'generation':{'membership':[l], 'empty':False}}
            for i in LANGS for w in ('dev1','dev2') for l in LANGS for n in range(60)]

    def test_minimum_and_denominator(self):
        rows=self.rows()
        for r in rows[:6]: r['generation']={'membership':['fr'], 'empty':False}
        result=summarize(rows, complete=True)
        self.assertEqual(result['A_robust']['ko|ko'], .9)
        self.assertEqual(result['cells']['ko|dev1|ko']['C_count'], 6)
        self.assertTrue(result['robustness_pass'])

    def test_unregistered_not_language_failure_and_seven_blocks(self):
        rows=self.rows()
        for r in rows[:7]: r['generation']={'membership':[], 'empty':False}
        result=summarize(rows, complete=True)
        self.assertFalse(result['output_validity_pass'])
        cell=result['cells']['ko|dev1|ko']
        self.assertEqual((cell['C_count'],cell['unregistered'],cell['total']), (0,7,60))
        self.assertAlmostEqual(result['A_robust']['ko|ko'],53/60)

    def test_shared_empty_missing_duplicate(self):
        rows=self.rows()
        rows[0]['generation']={'membership':['ko','en'], 'empty':False}
        rows[1]['generation']={'membership':[], 'empty':True}
        result=summarize(rows, complete=True)
        self.assertEqual(result['cells']['ko|dev1|ko']['matrix']['SHARED:en+ko'],1)
        self.assertEqual(result['cells']['ko|dev1|ko']['empty'],1)
        with self.assertRaises(ContractViolation): summarize(rows[:-1], complete=True)
        with self.assertRaises(ContractViolation): summarize(rows+[rows[0]])

    def test_all_progress_boundaries_and_final_publication(self):
        rows=self.rows()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            recovered=[]
            for done in range(60,1921,60):
                ref=save_progress(directory,rows[:done],1920)
                doc=read_verified_json(Path(ref['path']),expected_sha256=ref['sha256'])
                self.assertEqual((doc['completed'],doc['total']),(done,1920))
                recovered.extend(doc['rows'])
            self.assertEqual(recovered,rows)
            result={'status':'COMPLETE','rows':recovered,**summarize(recovered,complete=True)}
            ref=publish_json_once(directory/'run_summary.json',result)
            self.assertEqual(read_verified_json(Path(ref['path']))['status'],'COMPLETE')
            with self.assertRaises(ContractViolation): save_progress(directory,rows[:60],1920)
            with self.assertRaises(ContractViolation): save_progress(directory,rows[:59],1920)

    def test_invalid_cells_and_memberships(self):
        rows=self.rows(); rows[0]['wrapper']='unknown'
        with self.assertRaises(ContractViolation): summarize(rows,complete=True)
        rows=self.rows(); rows[0]['generation']['membership']=['ko','ko']
        with self.assertRaises(ContractViolation): summarize(rows,complete=True)
