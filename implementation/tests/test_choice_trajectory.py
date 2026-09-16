import copy
import unittest
from implementation.choice_trajectory import primary_cells,aggregate
from implementation.src.contracts import ContractViolation

class ChoiceTrajectoryTests(unittest.TestCase):
    def fixture(self):
        records=[];scores=[]
        for w in ('dev1','dev2'):
            for mode in ('ANY','REQUESTED'):
                r={'record_id':w+'|'+mode,'concept_id':'c','input_language':'ko','format':'RD','wrapper':w,'mode':mode}
                if mode=='REQUESTED':r['requested_language']='ko'
                records.append(r)
                g={'membership':['fr'] if mode=='ANY' else ['ko'],'class':'REGISTERED','request_compatible':None if mode=='ANY' else True}
                scores.append({**r,'checkpoint_sha256':'same','generation':g,
                    'probability':{'Q_membership':{'ko':.05,'en':.05,'zh':.05,'fr':.85},'Z':.98} if mode=='ANY' else None})
        return records,scores

    def test_request_success_is_not_any_choice(self):
        r,s=self.fixture();cells=primary_cells(s,r,['c'],'same');a=aggregate(cells)
        self.assertEqual(a['wrappers']['dev1']['A_counts']['ko'],1)
        self.assertEqual(a['wrappers']['dev1']['ANY_registered_language_counts']['ko'],0)
        self.assertEqual(a['wrappers']['dev1']['Q_mean']['ko'],.05)
        self.assertEqual(cells,primary_cells(list(reversed(s)),list(reversed(r)),['c'],'same'))

    def test_missing_duplicate_and_mixed_checkpoint_fail(self):
        r,s=self.fixture()
        with self.assertRaises(ContractViolation):primary_cells(s[:-1],r,['c'],'same')
        with self.assertRaises(ContractViolation):primary_cells(s+[s[0]],r,['c'],'same')
        mixed=copy.deepcopy(s);mixed[0]['checkpoint_sha256']='other'
        with self.assertRaises(ContractViolation):primary_cells(mixed,r,['c'],'same')
        bad=copy.deepcopy(s);bad[1]['generation']['request_compatible']=False
        with self.assertRaises(ContractViolation):primary_cells(bad,r,['c'],'same')
