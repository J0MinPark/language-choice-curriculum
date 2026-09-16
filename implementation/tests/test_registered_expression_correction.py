"""Live provenance regression; offline source snapshots are required."""
import copy
import json
import unittest
from pathlib import Path

from implementation.src.contracts import ContractViolation, WORK_ROOT
from implementation.src import registered_expression_correction as correction

EVIDENCE = WORK_ROOT / "evidence/oewn_2025_egg_ordinal38_20260915T105411Z_dc343f2/oewn_evidence_manifest.json"


@unittest.skipUnless(EVIDENCE.is_file(), "Live frozen evidence is not installed")
class RegisteredExpressionCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = correction.build_registered_expression_correction(EVIDENCE)

    def test_only_ordinal_38_en_changes_and_source_identity_is_preserved(self):
        payload = self.payload
        original = json.loads(Path(payload["source_artifacts"]["old_selection_plan"]["path"]).read_bytes())
        for old, new in zip(original["selected"], payload["selected"], strict=True):
            if old["selection_ordinal"] != 38:
                self.assertEqual(old, new)
                continue
            self.assertEqual(old["candidate_sha256"], new["candidate_sha256"])
            for lang in ("ko", "zh", "fr"):
                self.assertEqual(old["terms"][lang], new["terms"][lang])
            self.assertEqual(new["terms"]["en"]["canonical"], "egg")
            self.assertNotEqual(old["terms"]["en"]["term_id"], new["terms"]["en"]["term_id"])
            self.assertEqual(old["terms"]["en"], new["terms"]["en"]["source_term"])
        self.assertEqual(payload["optional_etymology"]["status"], "NOT_RUN")
        self.assertFalse(payload["training_eligible"])

    def test_resealed_answer_change_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["selected"][37]["terms"]["en"]["canonical"] = "Egg"
        payload["cohort_sha256"] = correction._hash(payload["selected"])
        payload["correction_sha256"] = correction._hash({k: v for k, v in payload.items() if k != "correction_sha256"})
        with self.assertRaisesRegex(ContractViolation, "CORRECTION_CONTENT_MISMATCH"):
            correction.validate_registered_expression_correction(payload)

    def test_training_claim_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["training_eligible"] = True
        with self.assertRaisesRegex(ContractViolation, "CORRECTION_CONTENT_MISMATCH"):
            correction.validate_registered_expression_correction(payload)


if __name__ == "__main__":
    unittest.main()
