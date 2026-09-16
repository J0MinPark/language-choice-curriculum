import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from freshstart import LANGUAGES, VERSION
from freshstart.core import *
from freshstart.data import validate_concept, validate_etymology, audit_dataset
from freshstart.krdict import parse_response, collect, SameHostRedirect
from freshstart.reports import readiness, spearman, measurement, write_json, render_json_report
from freshstart.schedule import make_assignment, history_schedule, audit_schedules

ANS={'ko':'테스트어','en':'word','zh':'汉字','fr':'mot'}
GATES={'rho_min':.6,'wrapper_abs_max':.15,'z_threshold':.05,'low_z_fraction_max':.1}

def concept(cid):
    return {'concept_id':cid,'snapshot_id':'fixture_snapshot','source_urls':['https://example.test/source'],
            'source_record_hashes':['f'*64],'answers':{l:w+cid for l,w in ANS.items()},
            'glosses':{l:'fixture definition '+l for l in LANGUAGES},
            'qa':{'status':'APPROVED_BY_RESEARCHER','reviewer':'TEST_ONLY','review_date':'2026-09-14',
                  'source_alignment_checked':True,'answer_copy_checked':True},'cluster_id':cid}

def ety(cid, label='UNRESOLVED'):
    return {'concept_id':cid,'pair':['en','fr'],'relation':label,'evidence_urls':['https://example.test/etymology'],
            'evidence_record_hashes':['e'*64],'reviewer':'TEST_ONLY','review_date':'2026-09-14',
            'sense_alignment_note':'fixture','historical_scope':'fixture','family_id':cid,
            'evidence_origin':'recorded_source'}

def measurement_fixture():
    rows=[];ids=[f'c{i}' for i in range(20)];meta={'root_id':'4101','checkpoint_sha256':'a'*64}
    for i,c in enumerate(ids):
        q={'ko':.10+.001*i,'en':.20+.002*i,'zh':.30-.001*i,'fr':.40-.002*i}
        for w in ('dev1','dev2'):
            rows.append({'format_version':VERSION,'root_id':'4101','checkpoint_sha256':'a'*64,
                         'stage':'T3','input_language':'ko','format':'RD','mode':'ANY','split':'dev',
                         'wrapper':w,'concept_id':c,'Q_language':q.copy(),'Z':.8})
    return rows,ids,meta

class ProbabilityTests(unittest.TestCase):
    def test_four_language_sum(self):
        r=event_partition(ANS,{y:math.log(.1) for y in ANS.values()})
        self.assertAlmostEqual(sum(r['Q_language'].values()),1);self.assertAlmostEqual(r['Z'],.4)
    def test_shared_counted_once(self):
        a=dict(ANS,fr='word');r=event_partition(a,{y:math.log(.1) for y in set(a.values())})
        self.assertAlmostEqual(r['Z'],.3);self.assertIsNone(r['Q_language']);self.assertIn('en+fr',r['Q_membership'])
    def test_shared_request_is_compatible_not_identifiable(self):
        r=generated_label('word',dict(ANS,fr='word'),'fr');self.assertTrue(r['request_compatible']);self.assertEqual(r['membership'],['en','fr'])
    def test_unregistered_is_retained(self):
        r=generated_label('other',ANS,'fr');self.assertTrue(r['unregistered']);self.assertFalse(r['request_compatible'])
    def test_wrong_registered_language(self):
        self.assertTrue(generated_label('word',ANS,'fr')['registered_other_language'])
    def test_any_not_language_error(self):self.assertIsNone(generated_label('word',ANS,None)['registered_other_language'])
    def test_low_z_high_q(self):
        p={y:-50. for y in ANS.values()};p['word']=-20.
        r=event_partition(ANS,p);self.assertLess(r['Z'],.05);self.assertGreater(r['Q_language']['en'],.99)
    def test_underflow_keeps_finite_log(self):
        r=event_partition(ANS,{y:-1000. for y in ANS.values()});self.assertEqual(r['Z'],0.);self.assertEqual(r['logZ_status'],'FINITE');self.assertIsNotNone(r['Q_language'])
    def test_all_zero_is_undefined(self):
        r=event_partition(ANS,{y:-math.inf for y in ANS.values()});self.assertIsNone(r['Q_language'])
    def test_excess_mass_rejected(self):
        with self.assertRaises(ContractError):event_partition(ANS,{y:math.log(.4) for y in ANS.values()})
    def test_wrong_event_set_rejected(self):
        with self.assertRaises(ContractError):event_partition(ANS,{'word':-1.})
    def test_nan_rejected(self):
        p={y:-2. for y in ANS.values()};p['word']=math.nan
        with self.assertRaises(ContractError):event_partition(ANS,p)
    def test_three_languages_rejected(self):
        with self.assertRaises(ContractError):membership({k:v for k,v in ANS.items() if k!='fr'})
    def test_nfc_and_accents(self):
        self.assertEqual(canonical('e\u0301cole'),'école');self.assertNotEqual(canonical('école'),canonical('ecole'))
    def test_multiline_rejected(self):
        with self.assertRaises(ContractError):canonical('word\nother')
    def test_prefix_event_rejected(self):
        with self.assertRaises(ContractError):validate_token_events({'a':[1],'b':[1,2]})
    def test_collision_rejected(self):
        with self.assertRaises(ContractError):validate_token_events({'a':[1,2],'b':[1,2]})
    def test_disjoint_token_events(self):validate_token_events({'a':[1,9],'b':[2,9]})
    def test_continuation_boundary(self):
        enc=lambda s:list(s.encode());self.assertEqual(continuation_tokens(enc,'P: ','word'),list(b'word\n'))
    def test_unstable_boundary_rejected(self):
        with self.assertRaises(ContractError):continuation_tokens(lambda s:[1] if s=='P' else [2,3],'P','a')
    def test_sign_invariant_under_joint_flip(self):
        a={l:.1+.1*i for i,l in enumerate(LANGUAGES)};b={l:.4-.1*i for i,l in enumerate(LANGUAGES)}
        self.assertEqual(signed_change(a,b,1),signed_change(b,a,-1));self.assertAlmostEqual(sum(signed_change(a,b,1).values()),0.)
    def test_vector_mae_units(self):self.assertAlmostEqual(vector_mae([[0,0,0,0]],[[.1,.1,-.1,-.1]]),.1)
    def test_similarity_not_etymology(self):
        self.assertEqual(spelling_similarity('word','word'),1.);self.assertEqual(token_jaccard([1,2],[2,3]),1/3)

class DataTests(unittest.TestCase):
    def test_pending_human_review_blocks(self):
        c=concept('a');c['qa']['status']='PENDING'
        with self.assertRaises(ContractError):validate_concept(c)
    def test_synthetic_training_rejected(self):
        c=concept('a');c['synthetic_fixture']=True
        with self.assertRaises(ContractError):validate_concept(c)
    def test_missing_language_definition(self):
        c=concept('a');del c['glosses']['fr']
        with self.assertRaises(ContractError):validate_concept(c)
    def test_unknown_remains_unknown(self):self.assertEqual(validate_etymology(ety('a')),'UNRESOLVED')
    def test_documented_needs_evidence(self):
        e=ety('a','BORROWING_DOCUMENTED');e['evidence_urls']=[]
        with self.assertRaises(ContractError):validate_etymology(e)
    def test_no_model_generated_etymology(self):
        e=ety('a','SHARED_SOURCE_DOCUMENTED');e['evidence_origin']='model_generated'
        with self.assertRaises(ContractError):validate_etymology(e)
    def test_coverage_counts_separately(self):
        cs=[concept(str(i)) for i in range(6)];es=[ety(str(i),'BORROWING_DOCUMENTED' if i<2 else 'DISTINCT_ROUTES_REVIEWED') for i in range(6)]
        req={'min_total':6,'min_identifiable':4,'min_related':2,'min_distinct_routes':2}
        r=audit_dataset(cs,es,req);self.assertEqual(r['status'],'PASS');self.assertEqual(r['n_total'],6)
    def test_unknown_not_negative_control(self):
        cs=[concept(str(i)) for i in range(6)];es=[ety(str(i)) for i in range(6)]
        req={'min_total':6,'min_identifiable':4,'min_related':1,'min_distinct_routes':1}
        r=audit_dataset(cs,es,req);self.assertEqual(r['status'],'BLOCKED_DATA_COVERAGE')
    def test_duplicate_concept_rejected(self):
        with self.assertRaises(ContractError):audit_dataset([concept('a'),concept('a')],[ety('a')],{})

class ReportTests(unittest.TestCase):
    def test_readiness_not_mean(self):
        r=readiness({'ko':.95,'en':.95,'zh':.95,'fr':.85},list(LANGUAGES));self.assertEqual(r['status'],'BLOCKED_READINESS')
    def test_readiness_all_requested_languages(self):self.assertEqual(readiness({l:.9 for l in LANGUAGES},list(LANGUAGES))['status'],'PASS')
    def test_missing_readiness_language(self):
        with self.assertRaises(ContractError):readiness({'ko':.95},list(LANGUAGES))
    def test_constant_spearman_undefined(self):self.assertIsNone(spearman([1,1,1],[1,2,3]))
    def test_spearman_ties_matches_scipy(self):
        try:from scipy.stats import spearmanr
        except ImportError:self.skipTest('SciPy not installed')
        x=[1,1,4,2,9,6];y=[2,1,5,2,8,8]
        self.assertAlmostEqual(spearman(x,y),spearmanr(x,y).statistic)
    def test_measurement_single_cell_pass(self):
        r,ids,m=measurement_fixture();self.assertEqual(measurement(r,ids,m,GATES)['status'],'PASS')
    def test_row_permutation_invariant(self):
        r,ids,m=measurement_fixture();self.assertEqual(measurement(r,ids,m,GATES),measurement(list(reversed(r)),ids,m,GATES))
    def test_different_checkpoint_rejected(self):
        r,ids,m=measurement_fixture();r[0]['checkpoint_sha256']='b'*64
        with self.assertRaises(ContractError):measurement(r,ids,m,GATES)
    def test_mixed_input_rejected(self):
        r,ids,m=measurement_fixture();r[0]['input_language']='en'
        with self.assertRaises(ContractError):measurement(r,ids,m,GATES)
    def test_duplicate_row_rejected(self):
        r,ids,m=measurement_fixture();r.append(r[0].copy())
        with self.assertRaises(ContractError):measurement(r,ids,m,GATES)
    def test_missing_row_rejected(self):
        r,ids,m=measurement_fixture();r.pop()
        with self.assertRaises(ContractError):measurement(r,ids,m,GATES)
    def test_shared_cannot_be_silently_zero_filled(self):
        r,ids,m=measurement_fixture();r[0]['Q_language']=None
        with self.assertRaises(ContractError):measurement(r,ids,m,GATES)
    def test_low_z_blocks(self):
        r,ids,m=measurement_fixture()
        for row in r:row['Z']=.01
        self.assertEqual(measurement(r,ids,m,GATES)['status'],'BLOCKED_MEASUREMENT')
    def test_report_render_does_not_recompute(self):
        with tempfile.TemporaryDirectory() as d:
            jp=Path(d)/'r.json';mp=Path(d)/'r.md';payload={'status':'BLOCKED_MEASUREMENT','spearman':.523456789}
            write_json(jp,payload);render_json_report(jp,mp)
            displayed=json.loads(mp.read_text().split('```json\n')[1].split('\n```')[0]);self.assertEqual(payload,displayed)

class ScheduleTests(unittest.TestCase):
    def setUp(self):self.g=make_assignment([('a','b'),('c','d')],1)
    def test_rounds_counts(self):
        r=audit_schedules(history_schedule(self.g,'A'),history_schedule(self.g,'B'));self.assertEqual(r['rounds'],160);self.assertEqual(r['target_records_per_branch'],640)
    def test_repeat_seed_exact(self):self.assertEqual(self.g,make_assignment([('a','b'),('c','d')],1))
    def test_invalid_seedpair_repeated_concept(self):
        with self.assertRaises(ContractError):make_assignment([('a','b'),('a','c')],1)
    def test_tail_mutation_rejected(self):
        a=history_schedule(self.g,'A');b=history_schedule(self.g,'B');b[-1]['round']=999
        with self.assertRaises(ContractError):audit_schedules(a,b)
    def test_two_language_schedule_rejected(self):
        a=[r for r in history_schedule(self.g,'A') if r['target_language'] in ('en','fr')]
        with self.assertRaises(ContractError):audit_schedules(a,a)
        a=history_schedule(self.g,'A');b=history_schedule(self.g,'B')
        for rows in (a,b):
            for r in rows:
                if r['target_language']=='ko':r['target_language']='es'
        with self.assertRaises(ContractError):audit_schedules(a,b)

class ApiTests(unittest.TestCase):
    def test_no_credentials_no_http(self):
        with patch('urllib.request.build_opener') as mock:
            r=collect(Path('NOT_READ'),Path('NOT_WRITTEN'),key='')
            self.assertEqual(r['requests_made'],0);self.assertEqual(r['status'],'BLOCKED_CREDENTIALS');mock.assert_not_called()
    def test_error_xml(self):
        with self.assertRaisesRegex(ContractError,'020'):parse_response(b'<error><error_code>020</error_code></error>','en')
    def test_doctype_rejected(self):
        with self.assertRaises(ContractError):parse_response(b'<!DOCTYPE a><channel/>','en')
    def test_malformed_doc_sample_not_repaired(self):
        with self.assertRaises(ContractError):parse_response(b'<channel><item></channel>','en')
    def test_html_not_data(self):
        with self.assertRaises(ContractError):parse_response(b'<html/>','en')
    def test_parse_optional_translation_missing(self):
        s='<channel><total>1</total><item><target_code>1</target_code><word>시험</word><pos>명사</pos><sense><sense_order>1</sense_order><definition>테스트 정의</definition></sense></item></channel>'
        r=parse_response(s.encode(),'en');self.assertEqual(r['candidate_senses'][0]['translations'],[])
    def test_parse_sense_translation(self):
        s='<channel><total>1</total><item><target_code>1</target_code><word>시험</word><pos>명사</pos><sense><sense_order>2</sense_order><definition>테스트 정의</definition><translation><trans_lang>영어</trans_lang><trans_word>test</trans_word><trans_dfn>fixture</trans_dfn></translation></sense></item></channel>'
        r=parse_response(s.encode(),'en');self.assertEqual(r['candidate_senses'][0]['sense_order'],'2');self.assertEqual(r['candidate_senses'][0]['qa_status'],'PENDING')
    def test_redirect_blocked_before_following(self):
        with self.assertRaises(ContractError):SameHostRedirect().redirect_request(None,None,302,'',{},'https://elsewhere.test/')

if __name__=='__main__':unittest.main()
