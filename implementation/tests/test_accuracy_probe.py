import copy
import json
from pathlib import Path
import unittest

import torch

from implementation.src.contracts import PROJECT_ROOT, ContractViolation
from implementation.src.accuracy_probe import mix_prompt, probe_stream, compare_arms, continuation_lineage, continuation_result
from implementation.src.pilot_execution_freeze import build_execution_freeze, RETRAIN_REVISION
from implementation.src.records import RecordSlot, materialize_record
from implementation.tests.test_train_checkpoint import fixture_concept, ByteTokenizer
from implementation.accuracy_diagnostic import diagnostic_records, summarize


class AccuracyProbeTests(unittest.TestCase):
    def test_single_continuation_reports_deltas_without_fabricated_control(self):
        baseline={"readiness":{"status":"PASS","cells":{"dev1:zh":{"compatible_numerator":55}}},
            "measurement":{"status":"BLOCKED_MEASUREMENT"}}
        endpoint=copy.deepcopy(baseline)
        endpoint["readiness"]["cells"]["dev1:zh"]["compatible_numerator"]=56
        result=continuation_result(baseline,{"lower_lr":endpoint})
        self.assertEqual(result["compatible_count_change_by_cell"],{"dev1:zh":1})
        self.assertFalse(result["concurrent_control"])
        self.assertFalse(result["all_original_gates_pass"])
        self.assertNotIn("promising_exploratory_candidate",result)
        with self.assertRaisesRegex(ContractViolation,"ONLY_SELECTED_ARM"):
            continuation_result(baseline,{"constant":endpoint})

    def test_parent_weight_fingerprint_is_not_reused_for_updated_model(self):
        source={"model_fingerprint":"parent-weights","initial_model_fingerprint":"initial-weights",
            "phase":"T3","code_sha256":"old-code"}
        new=continuation_lineage(source,code_sha256="new-code")
        self.assertNotIn("model_fingerprint",new)
        self.assertEqual(new["source_lineage"]["model_fingerprint"],"parent-weights")
        self.assertEqual(new["initial_model_fingerprint"],"initial-weights")
        self.assertEqual(source["code_sha256"],"old-code")

    def setUp(self):
        self.path=PROJECT_ROOT/"implementation/config/evaluation_plan_v4_1_2.json"
        self.plan=json.loads(self.path.read_text())

    def test_deferred_5400_cannot_be_published(self):
        with self.assertRaisesRegex(ContractViolation,"DEFERRED_5400"):
            build_execution_freeze(revision=RETRAIN_REVISION)

    def test_prompt_mixture_no_heldout_or_answer_changes(self):
        concept=fixture_concept("c")
        concept["glosses"]["ko"]+="\n두 번째 정의 줄"
        changed=0
        for i in range(40):
            slot=RecordSlot(str(i),"c","ko","zh","REQUESTED","train1",i)
            base=materialize_record(concept,slot,stage="T3",policy_path=self.path)
            mixed=mix_prompt(base,concept,self.plan["prompt_plan"])
            self.assertEqual(mixed,mix_prompt(base,concept,self.plan["prompt_plan"]))
            self.assertEqual(mixed.canonical_answer,base.canonical_answer)
            self.assertEqual(mixed.record_id,base.record_id)
            self.assertIn(concept["glosses"]["ko"],mixed.prefix)
            if mixed!=base:
                changed+=1; self.assertNotEqual(mixed.content_sha256,base.content_sha256)
                self.assertIn("중국어",mixed.prefix)
            foreign=materialize_record(concept,RecordSlot(str(i),"c","fr","zh","REQUESTED","train1",i),
                stage="T3",policy_path=self.path)
            self.assertEqual(mix_prompt(foreign,concept,self.plan["prompt_plan"]),foreign)
        self.assertGreater(changed,0); self.assertLess(changed,40)

    def test_same_schedule_targets_and_resumable_loader_all_arms(self):
        freeze={"concepts":[fixture_concept(f"c{i}") for i in range(60)],
            "execution":{"additional_updates":360},"resolved_artifacts":{"evaluation_plan":str(self.path)}}
        a,aa=probe_stream(freeze,ByteTokenizer(),torch.Generator(),prompt_mix=False)
        b,bb=probe_stream(freeze,ByteTokenizer(),torch.Generator(),prompt_mix=True)
        self.assertEqual(len(a),360)
        self.assertEqual(aa["schedule_sha256"],bb["schedule_sha256"])
        self.assertEqual(aa["target_slots_sha256"],bb["target_slots_sha256"])
        self.assertEqual(aa["counts"],bb["counts"])
        self.assertNotEqual(aa["record_bank_sha256"],bb["record_bank_sha256"])
        self.assertGreater(bb["augmented_records"],0)
        a.next_batch(); state=a.state_dict()
        a.load_state_dict(state)
        with self.assertRaisesRegex(ContractViolation,"PLAN_MISMATCH"):
            b.load_state_dict(state)

    def test_selection_requires_both_languages_and_preserved_anchors(self):
        def gates(ko=57,en=53,zh=48,fr=54):
            return {"readiness":{"status":"BLOCKED_READINESS","cells":{
                f"{w}:{l}":{"compatible_numerator":n} for w in ("dev1","dev2")
                for l,n in (("ko",ko),("en",en),("zh",zh),("fr",fr))}},"measurement":{"status":"BLOCKED_MEASUREMENT"}}
        arms={"constant":gates(),"low_lr":gates(zh=50,fr=55),"prompt_mix":gates(ko=54,zh=56,fr=58)}
        out=compare_arms(gates(),arms)
        self.assertTrue(out["low_lr"]["promising_exploratory_candidate"])
        self.assertFalse(out["low_lr"]["all_original_gates_pass"])
        self.assertFalse(out["prompt_mix"]["promising_exploratory_candidate"])
        self.assertFalse(out["constant"]["promising_exploratory_candidate"])
        arms["low_lr"]=gates(zh=56,fr=54)
        self.assertFalse(compare_arms(gates(),arms)["low_lr"]["promising_exploratory_candidate"])

    def test_diagnostic_is_train_dev_only_not_test(self):
        records=diagnostic_records([fixture_concept("c")],self.plan)
        self.assertEqual(len(records),16)
        self.assertEqual({r["wrapper"] for r in records},{"train1","train2","dev1","dev2"})
        cells=summarize([{**r,"generation":{"request_compatible":True,"first_line":"x"}} for r in records])
        self.assertTrue(all(c["accuracy"]==1 for c in cells.values()))


if __name__=="__main__": unittest.main()
