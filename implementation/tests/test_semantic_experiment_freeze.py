import copy
import unittest
from pathlib import Path
from unittest import mock

from implementation.src import semantic_experiment_freeze as experiment
from implementation.src.contracts import ContractViolation


class SemanticExperimentFreezeTests(unittest.TestCase):
    def test_untrusted_artifact_is_rejected_before_reparsing_or_gpu(self):
        manifest = {"freeze_id": "untrusted", **{key: "fixture" for key in (
            "schema_version", "protocol_version", "annotation_freeze_id", "cohort_sha256",
            "tokenizer_file_sha256", "corpus_token_count", "boundary_audit_sha256",
            "allowed_execution_phases", "scientific_training_authorized")}}
        ref = {"path": "fixture", "sha256": "a" * 64, "bytes": 1}
        with mock.patch.object(experiment, "_read", return_value=(manifest, ref)), \
             mock.patch.object(experiment, "build_semantic_experiment_freeze") as build:
            with self.assertRaisesRegex(ContractViolation, "TRUSTED_ARTIFACT_ANCHOR_MISSING"):
                experiment.verify_semantic_experiment_freeze(Path("fixture"))
        build.assert_not_called()

    def test_synthetic_mode_is_not_a_production_bypass(self):
        with self.assertRaisesRegex(ContractViolation, "SEMANTIC_EXPERIMENT_REQUIRES_PRODUCTION_SOURCES"):
            experiment.verify_semantic_experiment_freeze(Path("fixture"), production=False)

    def test_gpu_session_rejects_scientific_phase_for_replay_only_freeze(self):
        from implementation.src import train
        frozen = {"schema_version": experiment.SCHEMA, "status": "PASS", "freeze_gate_status": "PASS", "data_kind": "REAL", "verified": True, "allowed_execution_phases": ["GPU_REPLAY"]}
        ledger = mock.Mock()
        with mock.patch.object(train, "require_literal_gpu2_mask"), \
             mock.patch("implementation.src.prepare_data.verify_experiment_freeze", return_value=frozen):
            with self.assertRaisesRegex(ContractViolation, "SEMANTIC_EXPERIMENT_PHASE_NOT_AUTHORIZED"):
                with train.gpu2_training_session(ledger=ledger, lock_path=Path("fixture"), run_id="fixture", root_id=4101, phase="T0", reserved_seconds=1800, checkpoint_grace_seconds=120, experiment_freeze_path=Path("fixture")):
                    self.fail("Replay-only freeze authorized study training")
        ledger.reserve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
