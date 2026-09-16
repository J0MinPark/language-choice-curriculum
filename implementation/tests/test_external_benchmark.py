import unittest
from implementation.src.external_benchmark import parse_things, prompt_for, preflight, DEFINITION
from implementation.src.contracts import ContractViolation


class Tokenizer:
    def encode(self, text):
        return list(text.encode())


class ExternalBenchmarkTests(unittest.TestCase):
    def test_parser_preserves_senses_and_synonyms(self):
        data = f'Word\tuniqueID\tWordnet ID4\tWordNet Synonyms\t"{DEFINITION}"\nfish\tfish1\tfish.n.01\tfish, aquatic_animal\ta water animal\n'.encode()
        row = parse_things(data)[0]
        self.assertEqual(row['synonyms'], ['fish', 'aquatic animal'])
        self.assertEqual(row['source_line'], 2)
        with self.assertRaisesRegex(ContractViolation, 'DUPLICATE'):
            parse_things(data + data.splitlines(keepends=True)[1])
        with self.assertRaisesRegex(ContractViolation, 'EMPTY_WORD'):
            parse_things(data.replace(b'a water animal', b''))

    def test_support_cannot_contain_query_alias_or_sense(self):
        query = {'id':'a','word':'car','synonyms':['auto'],'synset':'car.n.01','description':'vehicle'}
        support = {**query,'id':'b','word':'auto'}
        with self.assertRaisesRegex(ContractViolation, 'LEAKAGE'):
            prompt_for(query, [support])
        self.assertEqual(prompt_for(query, []), 'vehicle ⇒')

    def test_overflow_retained_and_no_training_or_score_claim(self):
        rows = [{'id':str(i),'word':f'w{i}','synonyms':[], 'synset':str(i),
                 'description':'a long independent description'} for i in range(26)]
        result = preflight(rows, Tokenizer(), [{'concept_id':'c','answers':{'en':'w0'}}])
        self.assertEqual(result['contexts']['24']['overflow_count'],26)
        self.assertEqual(result['n_records'],26)
        self.assertEqual(len(result['lexical_overlap_candidates']),1)
        self.assertFalse(result['training_enabled'])
        self.assertFalse(result['gpu_evaluation_enabled'])
        self.assertIsNone(result['accuracy'])


if __name__ == '__main__':
    unittest.main()
