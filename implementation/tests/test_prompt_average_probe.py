import copy
import unittest
from implementation.prompt_average_probe import TEMPLATES, PRIMARY, summarize
from implementation.src.contracts import ContractViolation

class PromptAverageTests(unittest.TestCase):
    def rows(self):
        return [{'wrapper':w,'concept_id':str(i),'input_language':'ko','mode':'ANY',
                 'probability':{'Z':.98,'Q_membership':{'ko':.1+i*.001,'en':.2-i*.0005,
                    'zh':.1+i*.0001,'fr':.6-i*.0006}}}
                for w in TEMPLATES for i in range(60)]

    def test_all_splits_and_alignment(self):
        rows=self.rows();s=summarize(rows)
        self.assertEqual(s,summarize(list(reversed(rows))))
        self.assertEqual(len(s['all_35_splits']),35)
        self.assertEqual(len({tuple(x['left']) for x in s['all_35_splits']}),35)
        self.assertEqual(set(s['primary_split']['left']),set(PRIMARY))
        for split in s['all_35_splits']:
            self.assertEqual(len(split['left']),4)
            self.assertFalse(set(split['left'])&set(split['right']))
            self.assertEqual(set(split['left'])|set(split['right']),set(TEMPLATES))
        self.assertEqual(s['all_languages_pass_split_count'],35)
        self.assertTrue(s['all_wrappers_high_Z'])
        self.assertEqual(s['interpretation_status'],'PRIMARY_MEAN_STABILITY_SUPPORTED')

    def test_bad_Z_is_not_removed_or_used_as_weight(self):
        rows=self.rows()
        for r in rows:
            if r['wrapper']=='w8':
                r['probability']['Z']=.1
                r['probability']['Q_membership']={'ko':.4,'en':.2,'zh':.1,'fr':.3}
        s=summarize(rows)
        self.assertEqual(s['interpretation_status'],'Z_NOT_MAINTAINED')
        self.assertEqual(len(s['all_35_splits']),35)
        self.assertAlmostEqual(s['all_concept_mean_Q']['0']['ko'],(.1*7+.4)/8)
        self.assertFalse(s['existing_pilot_gate_replaced'])

    def test_undefined_constant_and_invalid_rows(self):
        rows=self.rows()
        for r in rows:r['probability']['Q_membership']={l:.25 for l in ('ko','en','zh','fr')}
        s=summarize(rows)
        self.assertEqual(s['split_summary_by_language']['ko']['undefined_count'],35)
        self.assertFalse(s['primary_split']['all_languages_stability_pass'])
        with self.assertRaises(ContractViolation):summarize(rows+[copy.deepcopy(rows[0])])
        with self.assertRaises(ContractViolation):summarize(rows[:-1])
        rows=self.rows();rows[0]['probability']['Q_membership']['ko']=float('nan')
        with self.assertRaises(ContractViolation):summarize(rows)
