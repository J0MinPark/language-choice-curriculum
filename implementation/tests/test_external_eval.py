import tempfile
import unittest
import zipfile
from pathlib import Path
from implementation.src.external_eval import definition_evidence,summarize
from implementation.src.contracts import ContractViolation


class ExternalEvalTests(unittest.TestCase):
    def test_rights_match_requires_exact_definition(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.zip'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('wordnet/LICENSE','fixture notice')
                z.writestr('wordnet/data.noun','00001234 dummy | a fruit; "an example"\n')
            evidence,notice=definition_evidence(path,[{'external_id':'a','external_definition':'a fruit'}])
            self.assertEqual(evidence[0]['wordnet3_noun_offsets'],['00001234'])
            self.assertEqual(notice,b'fixture notice')
            with self.assertRaisesRegex(ContractViolation,'SOURCE_NOT_VERIFIED'):
                definition_evidence(path,[{'external_id':'a','external_definition':'a plant'}])

    def test_conditions_not_pooled_and_any_not_accuracy(self):
        rows=[{'condition':c,'wrapper':'dev1','requested_language':'zh','mode':'REQUESTED',
               'generation':{'request_compatible':v}} for c,v in [('original_definition',True),('external_definition',False)]]
        rows.append({'mode':'ANY'})
        cells=summarize(rows)
        self.assertEqual(cells['original_definition|dev1|zh']['accuracy'],1)
        self.assertEqual(cells['external_definition|dev1|zh']['accuracy'],0)
