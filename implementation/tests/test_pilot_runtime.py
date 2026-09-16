import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from implementation.src.contracts import ContractViolation, PROJECT_ROOT
from implementation.src.pilot_journal import PilotJournal
from implementation.src.pilot_execution_freeze import history_pairs, verify_execution_freeze, PARENT, PARENT_SHA
from implementation.src.pilot_streams import corpus_stream, evaluation_records, lexical_stream
from implementation.src.pilot_runtime import PilotRuntime, phase_order, gate_results, history_contrast
from implementation.src.records import STAGE_ACTIVE_LANGUAGES
from implementation.src.schedule import build_history_plan_from_verified_freeze
from implementation.tests.test_train_checkpoint import fixture_concept, ByteTokenizer


class PilotJournalTests(unittest.TestCase):
    def test_atomic_resume_and_binding_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            j=PilotJournal(d,{"code":"a"})
            j.append("CHECKPOINT",{"root_id":4101,"phase":"T0","phase_step":200})
            self.assertEqual(PilotJournal(d,{"code":"a"}).latest("CHECKPOINT",4101,"T0")["phase_step"],200)
            with self.assertRaisesRegex(ContractViolation,"BINDING_MISMATCH"):
                PilotJournal(d,{"code":"b"})
            p=Path(d)/"000000.json"
            value=json.loads(p.read_text()); value["data"]["phase_step"]=3000
            p.chmod(0o600); p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ContractViolation,"BINDING_MISMATCH"): PilotJournal(d,{"code":"a"})

    def test_gap_and_write_collision_fail(self):
        with tempfile.TemporaryDirectory() as d:
            a=PilotJournal(d,{})
            b=PilotJournal(d,{})
            a.append("START",{})
            with self.assertRaises(Exception): b.append("START",{})
            a.append("STOP",{})
            (Path(d)/"000000.json").unlink()
            with self.assertRaisesRegex(ContractViolation,"SEQUENCE_BROKEN"): PilotJournal(d,{})


class PilotExecutionTests(unittest.TestCase):
    def test_altered_freeze_cannot_enable_main_or_shorten_endpoint(self):
        from implementation.src import pilot_execution_freeze as f
        expected={"parent_freeze":{"path":str(PARENT),"sha256":PARENT_SHA},"execution":{"lexical_updates":3000},"main_enabled":False}
        for key,value in (("main_enabled",True),("execution",{"lexical_updates":2})):
            changed={**expected,key:value}
            with mock.patch.object(f,"read_verified_json",return_value=changed),mock.patch.object(f,"build_execution_freeze",return_value=expected):
                with self.assertRaisesRegex(ContractViolation,"FREEZE_MISMATCH"): verify_execution_freeze(Path("fixture"))
        with self.assertRaisesRegex(ContractViolation,"REQUIRES_REAL"): verify_execution_freeze(Path("fixture"),production=False)

    def test_pairing_is_preoutcome_and_both_branches_counterbalanced(self):
        ids=[f"c{i:03d}" for i in range(60)]
        pairs=history_pairs(ids)
        self.assertEqual(pairs,history_pairs(list(reversed(ids))))
        frozen={"verified":True,"status":"PASS","freeze_gate_status":"PASS","data_kind":"REAL",
            "concepts":[{"concept_id":i} for i in ids],"history_pairs":pairs}
        plan=build_history_plan_from_verified_freeze(frozen,root_id=4101)
        self.assertEqual(len(plan.branches["A"].batches),300)
        self.assertEqual(len(plan.branches["B"].uses),9600)

    def test_full_corpus_partial_batch_and_resume(self):
        tokens=np.arange(2000000,dtype=np.uint16)
        s,a=corpus_stream(tokens,"a"*64,torch.Generator().manual_seed(1))
        self.assertEqual(len(s),245)
        self.assertEqual(a["unique_corpus_tokens"],2000000)
        self.assertEqual(a["model_tokens_with_overlap"],2000000)
        self.assertEqual(len(s.batches[244].record_ids),5)
        self.assertEqual(int(s.batches[244].attention_mask.sum()),1152)
        s.next_batch(); state=s.state_dict()
        r,_=corpus_stream(tokens,"a"*64,torch.Generator())
        r.load_state_dict(state)
        self.assertEqual(s.next_batch().record_ids,r.next_batch().record_ids)
        state["plan_sha256"]="b"*64
        with self.assertRaisesRegex(ContractViolation,"PLAN_MISMATCH"): r.load_state_dict(state)

    def test_evaluation_records_do_not_expose_future_inputs(self):
        plan=json.loads((PROJECT_ROOT/"implementation/config/evaluation_plan_v4_1_2.json").read_text())
        concepts=[fixture_concept("c")]
        for stage in ("T0","T1","T2","T3"):
            rows=evaluation_records(concepts,plan,stage,endpoint=True)
            active=STAGE_ACTIVE_LANGUAGES[stage]
            self.assertEqual({r["input_language"] for r in rows},set(active))
            self.assertEqual({r["requested_language"] for r in rows if r["mode"]=="REQUESTED"},set(active))
            self.assertTrue(all(set(r["answers"])=={"ko","en","zh","fr"} for r in rows))
            self.assertTrue(all("requested_language" not in r for r in rows if r["mode"]=="ANY"))
            self.assertEqual(len({r["record_id"] for r in rows}),len(rows))

    def test_first_root_and_all_roots_gate_execution(self):
        runtime=PilotRuntime.__new__(PilotRuntime)
        runtime._root_gates=mock.Mock(return_value={"readiness":{"status":"BLOCKED_READINESS"},"measurement":{"status":"PASS"}})
        runtime._training_gates(4101,"T0")
        runtime._root_gates.assert_not_called()
        with self.assertRaisesRegex(ContractViolation,"BLOCKED_TRAINING_GATES"): runtime._training_gates(4102,"INIT")
        runtime._root_gates=mock.Mock(side_effect=lambda r:{"readiness":{"status":"PASS" if r!=4104 else "BLOCKED_READINESS"},"measurement":{"status":"PASS"}})
        with self.assertRaisesRegex(ContractViolation,"ROOT_4104"): runtime._training_gates(4101,"H_A")

    def test_phase_order_has_independent_roots_and_one_common_h_base(self):
        order=phase_order()
        self.assertEqual(order[:6],[(4101,p) for p in ("INIT","CORPUS","T0","T1","T2","T3")])
        self.assertEqual(order[24:27],[(4101,p) for p in ("H_BASE","H_A","H_B")])
        self.assertEqual(len(order),36)
        self.assertEqual(len(set(order)),36)

    def test_history_contrast_sign_and_lineage(self):
        from implementation.src.score import partition_probability_events
        import math
        answers={l:l for l in ("ko","en","zh","fr")}
        row={"concept_id":"c","input_language":"ko","wrapper":"dev1","split":"dev","mode":"ANY"}
        pa=partition_probability_events(answers,{l:math.log(p) for l,p in zip(answers,(.1,.2,.1,.1))})
        pb=partition_probability_events(answers,{l:math.log(p) for l,p in zip(answers,(.1,.1,.1,.2))})
        a={"provenance":{"branch":"A","root_id":4101},"rows":[{**row,"probability":pa}]}
        b={"provenance":{"branch":"B","root_id":4101},"rows":[{**row,"probability":pb}]}
        result=history_contrast(a,b,{"c":-1})["rows"][0]
        self.assertAlmostEqual(sum(result["h"].values()),0)
        self.assertAlmostEqual(result["r"],.2)
        b["provenance"]["root_id"]=4102
        with self.assertRaisesRegex(ContractViolation,"LINEAGE_MISMATCH"): history_contrast(a,b,{"c":1})

    def test_retention_never_deletes_endpoint_or_latest_two(self):
        from implementation.src.artifacts import sha256_file
        with tempfile.TemporaryDirectory() as d:
            r=PilotRuntime.__new__(PilotRuntime); r.directory=Path(d)
            r.journal=PilotJournal(Path(d)/"journal",{})
            paths=[]
            for i,reason in enumerate(("PERIODIC","PERIODIC","PERIODIC","FIXED_ENDPOINT")):
                p=Path(d)/"checkpoints"/str(i)/"state.pt"; p.parent.mkdir(parents=True); p.write_bytes(str(i).encode())
                paths.append(p)
                r.journal.append("CHECKPOINT",{"root_id":4101,"phase":"CORPUS" if i==0 else "T0","reason":reason,
                    "checkpoint":{"path":str(p.parent),"state_sha256":sha256_file(p)}})
            r._prune_intermediates(4101,"T0")
            self.assertFalse(paths[0].exists())
            self.assertTrue(all(p.exists() for p in paths[1:]))
            r._prune_intermediates(4101,"T0")
            self.assertEqual(len([e for e in r.journal.events if e["kind"]=="STATE_PRUNED"]),1)

    def test_launcher_detaches_and_does_not_claim_training_started(self):
        from implementation import pilot_cli
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/"work") as d, \
             mock.patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":"2","CUDA_DEVICE_ORDER":"PCI_BUS_ID"}), \
             mock.patch("socket.gethostname",return_value="fixture-server"), \
             mock.patch.object(pilot_cli.subprocess,"Popen") as popen:
            popen.return_value.pid=12345
            rc=pilot_cli.main(["--freeze","f","--implementation-manifest","i","--cpu-checks","c",
                "--replay","r","--run-directory",d,"--launch"])
            self.assertEqual(rc,0)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(popen.call_args.args[0][0],"nohup")
            value=json.loads(next(Path(d).glob("launch_*.json")).read_text())
            self.assertEqual(value["status"],"STARTING_NOT_YET_VALIDATED")

    def test_unexpected_exception_never_publishes_complete(self):
        from implementation.src import pilot_runtime as module
        with tempfile.TemporaryDirectory() as d:
            r=PilotRuntime.__new__(PilotRuntime)
            r.directory=Path(d); r.binding={}; r.resume=False
            r.freeze={"resolved_artifacts":{"corpus":"fixture"}}
            r.ledger=mock.Mock(); r.ledger.snapshot.return_value={}
            with mock.patch.object(module,"load_corpus_token_memmap",side_effect=TypeError("fixture unexpected failure")):
                result=r.run()
            self.assertNotEqual(result["status"],"PILOT_COMPLETE")
            self.assertTrue(result["uncommitted_updates_possible"])
            self.assertEqual(result["events"][-1]["kind"],"STOP")

    def test_evaluation_publishes_object_wrapped_record_artifact(self):
        from implementation.src import pilot_runtime as module
        from implementation.src.pilot_runtime import bound_json
        with tempfile.TemporaryDirectory() as d:
            r=PilotRuntime.__new__(PilotRuntime); r.directory=Path(d)
            r.journal=PilotJournal(Path(d)/"journal",{})
            r.plan=json.loads((PROJECT_ROOT/"implementation/config/evaluation_plan_v4_1_2.json").read_text())
            r.freeze={"concepts":[fixture_concept("c")],"freeze_sha256":"a"*64,
                "tokenizer_file_sha256":"a"*64,"artifacts":{"pilot_config":{"sha256":"a"*64},"evaluation_plan":{"sha256":"a"*64}}}
            r.code="a"*64; r.freeze_path=Path("fixture"); r.tokenizer=None
            ref={"path":"fixture","state_sha256":"a"*64,"manifest_sha256":"a"*64,"model_fingerprint":"a"*64}
            with mock.patch.object(module,"score_checkpoint",return_value={"rows":[],"fixture":True}), \
                 mock.patch.object(module,"gate_results",return_value={"readiness":{"status":"FIXTURE_ONLY"}}):
                event=r._evaluate((None,None,None,None,None),ref,4101,"T0",200,False,mock.Mock(),mock.Mock())
            data=bound_json(event["records"])
            self.assertEqual(data["schema_version"],"pilot-evaluation-records-v1")
            self.assertEqual(len(data["records"]),4)


if __name__ == "__main__": unittest.main()
