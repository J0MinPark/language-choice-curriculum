import copy
import math
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from implementation.qwen_remeasurement import cached_generation, probability_summary
from implementation.src.score import greedy_first_line, score_probability_partition
from implementation.src.contracts import ContractViolation

class BytesTokenizer:
    def encode(self,text,add_special_tokens=False): return list(text.encode('ascii'))
    def decode(self,ids,**kwargs): return bytes(ids).decode('ascii')

class RemeasurementTests(unittest.TestCase):
    def test_cache_matches_reference_and_probability_matches_manual_logits(self):
        torch.manual_seed(12)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=128,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
            max_position_embeddings=128,attention_dropout=0.)).eval()
        tok=BytesTokenizer(); answers={'ko':'a','en':'bb','zh':'ccc','fr':'dddd'}
        row={'prefix':'Question:\n','answers':answers,'requested_language':'ko'}
        a=cached_generation(model,tok,row,max_new_tokens=8)
        b=greedy_first_line(model,tok,row['prefix'],answers,'ko',context_length=128,max_new_tokens=8)
        self.assertEqual(a,b)
        result=score_probability_partition(model,tok,row['prefix'],answers,context_length=128)
        self.assertAlmostEqual(sum(result['Q_membership'].values()),1.)
        for event in result['events']:
            full=event['prefix_token_ids']+event['continuation_token_ids']
            with torch.no_grad(): logits=model(torch.tensor([full[:-1]]),use_cache=False).logits
            start=len(event['prefix_token_ids'])-1
            manual=sum(float(torch.log_softmax(logits[0,start+j],dim=-1)[t])
                for j,t in enumerate(event['continuation_token_ids']))
            self.assertAlmostEqual(event['log_p'],manual,places=5)
            self.assertAlmostEqual(event['raw_p'],math.exp(manual),places=8)

    def test_summary_aligns_ids_and_detects_constant_duplicate_missing(self):
        rows=[]
        for w in ('dev1','dev2'):
            for i in range(60):
                q=(i+1)/1000
                rows.append({'condition':'zero_shot','input_language':'ko','wrapper':w,'concept_id':str(i),
                    'probability':{'Q_membership':{'ko':q,'en':q,'zh':q,'fr':1-3*q},'Z':.01,'logZ':math.log(.01)}})
        result=probability_summary(rows)
        self.assertEqual(result,probability_summary(list(reversed(rows))))
        self.assertEqual(result['zero_shot|ko']['languages']['ko']['spearman'],1.)
        self.assertFalse(result['zero_shot|ko']['original_measurement_thresholds_pass'])
        with self.assertRaises(ContractViolation): probability_summary(rows+[rows[0]])
        with self.assertRaises(ContractViolation): probability_summary(rows[:-1])
        constant=copy.deepcopy(rows)
        for r in constant: r['probability']['Q_membership']={l:.25 for l in ('ko','en','zh','fr')}
        self.assertIsNone(probability_summary(constant)['zero_shot|ko']['languages']['ko']['spearman'])

    def test_model40_conditions_keep_exact_target_and_newline_difference(self):
        from unittest.mock import patch
        from implementation.model40_remeasurement import records
        row={'condition':'zero_shot','prefix':'Definition: x\nAnswer:\n','record_id':'c|ko|dev1|zero_shot|ANY','concept_id':'c','answers':{'ko':'a','en':'b','zh':'c','fr':'d'}}
        with patch('implementation.model40_remeasurement.qwen_records',return_value={'probability':[row]}):
            got=records()
        self.assertEqual(got[0],row)
        self.assertEqual(got[1]['prefix']+'\n',row['prefix'])
        self.assertEqual(got[1]['answers'],row['answers'])
        self.assertEqual(got[1]['condition'],'original')
