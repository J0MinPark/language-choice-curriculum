import unittest
from implementation.corpus_expression_audit import counts


class ExpressionCountsTest(unittest.TestCase):
    def test_overlap_boundaries_case_and_document_separation(self):
        self.assertEqual(counts(['banana', 'ban', 'ana'], 'ana')['substring_occurrences'], 3)
        self.assertEqual(counts(['egg Egg eggs'], 'egg')['unicode_word_bounded_occurrences'], 1)
        self.assertEqual(counts(['eg', 'g'], 'egg')['substring_occurrences'], 0)
        self.assertEqual(counts(['눈 눈이'], '눈')['substring_occurrences'], 2)
        with self.assertRaises(ValueError):
            counts(['text'], '')
