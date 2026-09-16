import csv
import io
import unittest
from implementation.src.external_sense_review import FIELDS, apply_review
from implementation.src.contracts import ContractViolation


def encode(rows):
    out=io.StringIO(); writer=csv.DictWriter(out,fieldnames=FIELDS)
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


class SenseReviewTests(unittest.TestCase):
    def setUp(self):
        self.base={k:k for k in FIELDS[:-3]}
        self.row={**self.base,'review_status':'승인','reviewer':'jm02','review_comment':''}

    def test_immutable_fields_and_duplicate_rejected(self):
        with self.assertRaisesRegex(ContractViolation,'ORIGINAL_FIELDS'):
            apply_review(encode([{**self.row,'ko':'changed'}]),[self.base])
        with self.assertRaisesRegex(ContractViolation,'DUPLICATE'):
            apply_review(encode([self.row,self.row]),[self.base])

    def test_policy_hold_preserves_human_approval(self):
        base={**self.base,'external_id':'cucumber'}
        row={**self.row,'external_id':'cucumber'}
        result=apply_review(encode([row]),[base])
        self.assertEqual(result['human_counts'],{'승인':1})
        self.assertEqual(result['effective_counts'],{'HELD_POLICY_CONFLICT':1})
        self.assertFalse(result['training_enabled'])

    def test_duplicate_accepted_target_rejected(self):
        base2={**self.base,'external_id':'other','external_synset':'other.n.01'}
        row2={**self.row,**base2}
        with self.assertRaisesRegex(ContractViolation,'ONE_TO_ONE'):
            apply_review(encode([self.row,row2]),[self.base,base2])

    def test_exclusion_is_not_data_deletion(self):
        row={**self.row,'review_status':'수정 필요','review_comment':'different sense'}
        result=apply_review(encode([row]),[self.base])
        self.assertEqual(result['effective_counts'],{'EXCLUDED_CURRENT_LINK':1})
        self.assertEqual(result['external_dataset_rows_deleted'],0)
        self.assertFalse(result['automatic_relinking'])
