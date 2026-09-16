"""Exploratory retraining must not relax gates or silently alter old runs."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from implementation.src.contracts import ContractViolation
from implementation.src.pilot_execution_freeze import RETRAIN_SOURCE
from implementation.src.pilot_journal import PilotJournal
from implementation.src.pilot_runtime import PilotRuntime, phase_learning_rate, phase_order
from implementation.src.schedule import build_stage_plan, audit_concept_factor_balance


class RetrainingTests(unittest.TestCase):
    def test_joint_cells_exact_deterministic_and_train_only(self):
        for stage in ("T2", "T3"):
            a = build_stage_plan(["a","b","c","d"],root_id=4101,stage=stage,
                optimizer_updates=12,concept_factor_balance=True)
            b = build_stage_plan(["d","c","b","a"],root_id=4101,stage=stage,
                optimizer_updates=12,concept_factor_balance=True)
            self.assertEqual(a.plan_sha256,b.plan_sha256)
            self.assertEqual(a.counts["concept_factor_balance"]["status"],"PASS")
            self.assertEqual({s.wrapper for s in a.slots},{"train1","train2"})
            if stage == "T2":
                self.assertNotIn("fr",{s.target_language for s in a.slots})
            with self.assertRaisesRegex(ContractViolation,"CONCEPT_FACTOR_BALANCE"):
                audit_concept_factor_balance(a.slots[:-1],stage=stage)
        with self.assertRaisesRegex(ContractViolation,"CONCEPT_FACTOR_CYCLE_NOT_EXACT"):
            build_stage_plan([str(i) for i in range(60)],root_id=4101,stage="T3",
                optimizer_updates=3000,concept_factor_balance=True)

    def test_learning_rate_boundary_and_resume(self):
        from implementation.src.model import ConstantSchedule
        import torch
        execution={"learning_rate_reduction":{"stages":["T2","T3"],"after_updates":3000,"learning_rate":0.0001}}
        for phase in ("T2","T3"):
            self.assertEqual(phase_learning_rate(execution,phase,2999),0.0003)
            self.assertEqual(phase_learning_rate(execution,phase,3000),0.0001)
            self.assertEqual(phase_learning_rate(execution,phase,5200),0.0001)
        self.assertEqual(phase_learning_rate(execution,"T1",3000),0.0003)
        optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=0.0003)
        scheduler=ConstantSchedule(optimizer,0.0003,"T2")
        saved=scheduler.state_dict()
        scheduler.load_state_dict(saved)
        scheduler.transition(phase="T2",learning_rate=phase_learning_rate(execution,"T2",3000))
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"],0.0001)

    def test_exploration_does_not_expand_to_roots_or_history(self):
        self.assertEqual(phase_order({"roots":[4101],"start_phase":"T2","history_enabled":False}),
            [(4101,"T2"),(4101,"T3")])
        self.assertEqual(len(phase_order()),36)

    def test_import_pins_lineage_preserves_state_and_is_idempotent(self):
        from implementation.src import pilot_runtime as m
        with tempfile.TemporaryDirectory() as d:
            r=PilotRuntime.__new__(PilotRuntime); r.directory=Path(d); r.emergency=0; r.code="new-code"
            r.freeze={"execution":{"import_full_state":RETRAIN_SOURCE},"freeze_sha256":"new-freeze",
                "tokenizer_file_sha256":"tokenizer", "artifacts":{"pilot_config":{"sha256":"config"},
                "evaluation_plan":{"sha256":"evaluation"}}}
            r.journal=PilotJournal(Path(d)/"journal",{})
            lineage={"root_id":4101,"phase":"T1","stage":"T1","branch":None,"data_kind":"REAL",
                "freeze_sha256":RETRAIN_SOURCE["freeze_sha256"],"code_sha256":RETRAIN_SOURCE["code_sha256"],
                "config_sha256":"config","tokenizer_sha256":"tokenizer","evaluation_plan_sha256":"evaluation"}
            original={"lineage":lineage,"progress":{"phase_step":3000},"model":{"fixture":1},
                "optimizer":{"fixture":2},"scheduler":{"fixture":3},"rng":{"fixture":4},"semantic_fingerprints":{"all":"same"}}
            payload=copy.deepcopy(original)
            ref={"path":"fixture"}
            r._load_checkpoint=mock.Mock(side_effect=lambda *a:(copy.deepcopy(payload),{}))
            with mock.patch.object(m,"load_checkpoint_payload",return_value=(payload,{})) as load, \
                 mock.patch.object(m,"require_disk_reservation"), \
                 mock.patch.object(m,"save_checkpoint") as save:
                save.return_value.as_dict.return_value=ref
                r._import_full_state()
                for k in ("model","optimizer","scheduler","rng","progress"):
                    self.assertEqual(payload[k],original[k])
                self.assertEqual(payload["lineage"]["source_lineage"],original["lineage"])
                self.assertEqual(payload["lineage"]["freeze_sha256"],"new-freeze")
                self.assertEqual(load.call_args.kwargs["expected_state_sha256"],RETRAIN_SOURCE["state_sha256"])
                r._import_full_state(); self.assertEqual(save.call_count,1)
            r.journal=PilotJournal(Path(d)/"other-journal",{})
            payload=copy.deepcopy(original); payload["lineage"]["root_id"]=4102
            with mock.patch.object(m,"load_checkpoint_payload",return_value=(payload,{})):
                with self.assertRaisesRegex(ContractViolation,"SOURCE_LINEAGE_MISMATCH"):
                    r._import_full_state()


if __name__ == "__main__": unittest.main()
