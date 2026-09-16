import copy
import unittest
from implementation.prompt_average_probe import TEMPLATES,summarize
from implementation.checkpoint_drift_probe import worst_gate,pair_drift
from implementation.src.contracts import ContractViolation

class CheckpointDriftTests(unittest.TestCase):
    def rows(self):
        return [{'wrapper':w,'concept_id':str(i),'input_language':'ko','mode':'ANY',
                 'probability':{'Z':.98,'Q_membership':{'ko':.1+i*.001,'en':.2-i*.0005,
                    'zh':.1+i*.0001,'fr':.6-i*.0006}}}
                for w in TEMPLATES for i in range(60)]

    def test_one_bad_split_blocks_even_when_primary_passes(self):
        s=summarize(self.rows());self.assertEqual(worst_gate(s)['status'],'PASS')
        s['all_35_splits'][0]['languages']['ko']['spearman']=.59
        g=worst_gate(s);self.assertEqual(g['status'],'BLOCKED_WORST_SPLIT_MEASUREMENT')
        self.assertEqual(g['languages']['ko']['min_rho'],.59)
        s['all_35_splits'][0]['languages']['ko']['spearman']=None
        self.assertEqual(worst_gate(s)['languages']['ko']['undefined_splits'],1)
        s['Z_by_wrapper']['w1']['median']=.89
        self.assertEqual(worst_gate(s)['status'],'BLOCKED_HIGH_Z')
        s=summarize(self.rows());s['all_35_splits'][0]=copy.deepcopy(s['all_35_splits'][1])
        with self.assertRaises(ContractViolation):worst_gate(s)

    def test_known_drift_units_and_no_invented_H_bound(self):
        before=self.rows();after=copy.deepcopy(before)
        for r in after:
            r['probability']['Q_membership']['ko']+=.05
            r['probability']['Q_membership']['en']-=.05
        a,b=summarize(before),summarize(after);d=pair_drift(before,after,a,b)
        self.assertAlmostEqual(d['mean_vector_MAE'],.025)
        self.assertAlmostEqual(d['mean_TV'],.05)
        self.assertAlmostEqual(d['languages']['ko']['mean_signed_delta'],.05)
        self.assertAlmostEqual(d['languages']['en']['mean_signed_delta'],-.05)
        self.assertAlmostEqual(d['languages']['ko']['worst_split_delta_disagreement'],0.)
        self.assertEqual(len(d['all_35_delta_sensitivity']),35)
        self.assertFalse(d['is_H_lower_bound'])
        self.assertEqual(d,pair_drift(list(reversed(before)),list(reversed(after)),a,b))

    def test_identical_checkpoint_has_zero_drift_and_undefined_delta_rank(self):
        rows=self.rows();s=summarize(rows);d=pair_drift(rows,rows,s,s)
        self.assertEqual(d['mean_TV'],0.)
        self.assertIsNone(d['all_35_delta_sensitivity'][0]['languages']['ko']['delta_spearman'])
