import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from implementation.src.contracts import PROJECT_ROOT, ContractViolation
from implementation.src.accuracy_probe import probe_stream, probe_source_event, compare_arms, prune_probe_midpoint
from implementation.src.pilot_journal import PilotJournal
from implementation.src.artifacts import sha256_file
from implementation.tests.test_train_checkpoint import fixture_concept,ByteTokenizer


class RefinementTests(unittest.TestCase):
    def test_variants_keep_global_exposure_but_balance_local_cells(self):
        f={"concepts":[fixture_concept(f"c{i}") for i in range(60)],"execution":{"additional_updates":360},
            "resolved_artifacts":{"evaluation_plan":str(PROJECT_ROOT/"implementation/config/evaluation_plan_v4_1_2.json")}}
        _,base=probe_stream(f,ByteTokenizer(),torch.Generator(),prompt_mix=True)
        _,balanced=probe_stream(f,ByteTokenizer(),torch.Generator(),prompt_mix=True,concept_factor_balance=True)
        _,full=probe_stream(f,ByteTokenizer(),torch.Generator(),prompt_mix=True,all_ko_prompts=True)
        for key in ("target","target_input","target_mode","target_wrapper","concept"):
            self.assertEqual(base["counts"][key],balanced["counts"][key])
        self.assertEqual(balanced["counts"]["concept_factor_balance"]["status"],"PASS")
        self.assertEqual(full["augmented_records"],2880)
        self.assertEqual(base["target_slots_sha256"],full["target_slots_sha256"])
        self.assertNotEqual(base["schedule_sha256"],balanced["schedule_sha256"])

    def test_source_is_selected_fixed_endpoint_not_other_arm_or_midpoint(self):
        events=[{"kind":"CHECKPOINT","data":{"arm":"chosen","reason":"FIXED_ENDPOINT","checkpoint":{"id":"final"}}},
                {"kind":"CHECKPOINT","data":{"arm":"other","reason":"FIXED_ENDPOINT","checkpoint":{"id":"wrong"}}},
                {"kind":"EVALUATION","data":{"arm":"chosen","scores":{"id":"scores"}}}]
        e=probe_source_event({"events":events},{"source_arm":"chosen"})
        self.assertEqual(e["checkpoint"]["id"],"final")

    def test_refinement_selection_cannot_hide_readiness_or_measurement_failure(self):
        def g(zh=53,fr=57,status="PASS"):
            return {"readiness":{"status":status,"cells":{f"{w}:{l}":{"compatible_numerator":n}
                for w in ("dev1","dev2") for l,n in (("ko",58),("en",56),("zh",zh),("fr",fr))}},
                "measurement":{"status":"BLOCKED_MEASUREMENT"}}
        base=g(status="BLOCKED_READINESS")
        out=compare_arms(base,{"constant":base,"candidate":g(zh=55)},selection_version=2)
        self.assertTrue(out["candidate"]["promising_exploratory_candidate"])
        self.assertFalse(out["candidate"]["all_original_gates_pass"])
        self.assertFalse(compare_arms(base,{"constant":base,"candidate":g(zh=55,status="BLOCKED_READINESS")},selection_version=2)["candidate"]["promising_exploratory_candidate"])

    def test_only_verified_in_run_midpoint_is_pruned(self):
        from implementation.src import accuracy_probe as m
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'checkpoints/mid/state.pt';p.parent.mkdir(parents=True);p.write_bytes(b'mid')
            ref={"path":str(p.parent),"state_sha256":sha256_file(p)}
            end={"path":str(root/'checkpoints/end'),"state_sha256":"a"*64,"manifest_sha256":"b"*64}
            j=PilotJournal(root/'journal',{})
            with mock.patch.object(m,"load_checkpoint_payload",return_value=({},{})):
                with self.assertRaisesRegex(ContractViolation,"SCOPE_MISMATCH"):
                    prune_probe_midpoint(root,ref,ref,j,"arm")
                self.assertTrue(p.exists())
                prune_probe_midpoint(root,ref,end,j,"arm")
            self.assertFalse(p.exists());self.assertEqual(j.events[-1]['kind'],'STATE_PRUNED')


if __name__=="__main__":unittest.main()
