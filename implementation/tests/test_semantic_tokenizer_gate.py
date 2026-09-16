import tempfile
import unittest
from pathlib import Path
from unittest import mock

from implementation.src import semantic_tokenizer_gate as gate


class SemanticTokenizerGateTests(unittest.TestCase):
    def test_successful_probe_cannot_override_failed_authoritative_gate(self):
        frozen = {"concepts": [], "cohort_sha256": "a" * 64}
        ref = {"path": "synthetic-fixture", "sha256": "b" * 64, "bytes": 1}
        info = {"evaluation_plan_path": "synthetic-plan", "evaluation_plan_sha256": "c" * 64, "evaluation_plan": {"bytes": 1, "sha256": "c" * 64}, "tokenizer_file_sha256": "d" * 64}
        published = {}
        def publish(path, payload):
            published[Path(path).name] = payload
            return ref
        for status in ("PASS", "BLOCKED_TOKEN_BOUNDARY"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                audit = {"status": status, "expression_only_metrics": [{}] * 60, "contexts_checked": 7200, "contexts_passed": 7200 if status == "PASS" else 1800, "contexts_failed": 0 if status == "PASS" else 5400, "failure_counts": {} if status == "PASS" else {"UNSTABLE_TOKEN_BOUNDARY": 5400}}
                audit["evaluation_plan_sha256"] = "c" * 64
                with mock.patch.object(gate, "_output", return_value=Path(directory) / "output"), \
                     mock.patch.object(gate, "audit_semantic_freeze", return_value={"artifact": ref}), \
                     mock.patch.object(gate, "_bound", return_value=frozen), \
                     mock.patch.object(gate, "load_verified_tokenizer", return_value=(object(), info)), \
                     mock.patch.object(gate, "_read", return_value=({}, ref)), \
                     mock.patch.object(gate, "audit_lexical_token_boundaries", return_value=audit), \
                     mock.patch.object(gate, "load_evaluation_plan", return_value={}), \
                     mock.patch.object(gate, "probe_trailing_space_removal", return_value={"status": "PASS_DIAGNOSTIC_ONLY"}) as probe, \
                     mock.patch.object(gate, "publish_json_once", side_effect=publish):
                    result = gate.run_gate(Path("synthetic-freeze"), Path("synthetic-tokenizer"), Path(directory) / "output")
                self.assertEqual(result["automatic_advance_allowed"], status == "PASS")
                self.assertEqual(probe.call_count, 0 if status == "PASS" else 1)
                self.assertFalse(published["gate.json"]["training_started"])
                self.assertFalse(published["gate.json"]["training_eligible"])


if __name__ == "__main__":
    unittest.main()
