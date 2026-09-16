from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest import mock

from implementation.src import semantic_annotation_freeze as freeze
from implementation.src.contracts import WORK_ROOT, ContractViolation

CORRECTION = WORK_ROOT / "annotations/registered_expression_correction_v4_1_1_jm02/selection_correction.json"


class SemanticApprovalContractTests(unittest.TestCase):
    def setUp(self):
        # Synthetic contract objects never published as research artifacts.
        self.packet = {"rows": [{"review_subject_sha256": "a" * 64}]}
        self.ref = {"path": "synthetic-test-only", "sha256": "b" * 64, "bytes": 123}

    def approved(self):
        approval = freeze.approval_template(self.packet, self.ref)
        approval.update(status="APPROVED_BY_RESEARCHER", reviewer_id="jm02",
                        review_date=datetime.now(timezone.utc).date().isoformat(),
                        meaning_alignment_checked=True, term_quality_checked=True,
                        source_bindings_checked=True)
        return approval

    def test_pending_intake_cannot_pass_formal_approval(self):
        for payload in (freeze.approval_template(self.packet, self.ref), {"status": "APPROVED"}):
            with self.assertRaisesRegex(ContractViolation, "BLOCKED_EXPLICIT_SOURCE_BOUND_APPROVAL"):
                freeze.validate_approval(self.packet, self.ref, payload)

    def test_complete_explicit_approval_is_accepted_without_etymology(self):
        freeze.validate_approval(self.packet, self.ref, self.approved())

    def test_subject_drift_or_machine_origin_or_incomplete_checks_rejected(self):
        for key, value in (
            ("approved_subjects", ["c" * 64]),
            ("evidence_origin", "ai-reviewed-dictionary-record"),
            ("source_bindings_checked", False),
            ("meaning_alignment_checked", 1),
            ("identity_authentication_claimed", True),
            ("reviewer_id", "auto"),
        ):
            with self.subTest(key=key):
                approval = self.approved()
                approval[key] = value
                with self.assertRaisesRegex(ContractViolation, "SEMANTIC_FREEZE_APPROVAL_SCOPE_MISMATCH"):
                    freeze.validate_approval(self.packet, self.ref, approval)

    def test_invalid_review_dates_rejected(self):
        for value in (None, "2999-01-01", "2020-01-01", "20260915"):
            approval = self.approved()
            approval["review_date"] = value
            with self.assertRaisesRegex(ContractViolation, "SEMANTIC_FREEZE_INVALID_REVIEW_DATE"):
                freeze.validate_approval(self.packet, self.ref, approval)

    def test_freeze_roundtrip_detects_resealed_changes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            packet = {
                **self.packet, "cohort_sha256": "c" * 64,
                "source_manifest_artifact": {"fixture": True},
                "source_merge_sha256": "d" * 64,
            }
            packet_path = root / "synthetic_packet.json"
            packet_path.write_bytes(freeze.canonical_json_bytes(packet))
            with mock.patch.object(freeze, "WORK_ROOT", root):
                _, packet_ref = freeze._read(packet_path)
                approval = self.approved()
                approval["packet_artifact"] = packet_ref
                approval_path = root / "synthetic_approval.json"
                approval_path.write_bytes(freeze.canonical_json_bytes(approval))
                output = root / "synthetic_freeze.json"
                with mock.patch.object(freeze, "audit_review_packet", return_value=(packet, packet_ref)):
                    result = freeze.freeze_semantic_annotations(packet_path, approval_path, output)
                    self.assertEqual(result["status"], "PASS_SEMANTIC_ANNOTATION_FREEZE")
                    self.assertFalse(result["training_eligible"])
                    with self.assertRaisesRegex(ContractViolation, "ARTIFACT_EXISTS"):
                        freeze.freeze_semantic_annotations(packet_path, approval_path, output)
                    damaged = json.loads(output.read_bytes())
                    damaged["training_eligible"] = True
                    damaged["freeze_sha256"] = freeze._hash({k: v for k, v in damaged.items() if k != "freeze_sha256"})
                    output.chmod(0o644)
                    output.write_bytes(freeze.canonical_json_bytes(damaged))
                    with self.assertRaisesRegex(ContractViolation, "SEMANTIC_FREEZE_CONTENT_MISMATCH"):
                        freeze.audit_semantic_freeze(output)


@unittest.skipUnless(CORRECTION.is_file(), "Frozen KRDICT/correction snapshots not installed")
class SemanticReviewPacketLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = freeze.build_review_packet(CORRECTION)

    def test_all_expressions_have_exact_source_spans_and_four_language_glosses(self):
        self.assertEqual(self.packet["concept_count"], 60)
        self.assertEqual(self.packet["expression_count"], 240)
        self.assertGreaterEqual(self.packet["identifiable_count"], 40)
        for row in self.packet["rows"]:
            self.assertEqual(set(row["terms"]), {"ko", "en", "zh", "fr"})
            self.assertNotIn("etymology", row)
            self.assertNotIn("family_id", row)
            for term in row["terms"].values():
                self.assertEqual(term["source_word_raw"][term["source_span_start"]:term["source_span_end"]], term["source_text_exact"])
                self.assertTrue(term["source_refs"])
                self.assertTrue(term["gloss"])
        egg = self.packet["rows"][37]["terms"]["en"]
        self.assertEqual(egg["answer"], "egg")
        self.assertEqual(egg["source_word_raw"], "hen's egg")
        self.assertIsNotNone(egg["supplementary_evidence"])
        self.assertFalse(self.packet["training_eligible"])

    def test_packet_hash_covers_source_refs_and_summary_contains_egg(self):
        core = {k: v for k, v in self.packet.items() if k != "packet_sha256"}
        self.assertEqual(self.packet["packet_sha256"], freeze._hash(core))
        damaged = copy.deepcopy(core)
        damaged["rows"][37]["terms"]["en"]["source_refs"][0]["raw_sha256"] = "0" * 64
        self.assertNotEqual(self.packet["packet_sha256"], freeze._hash(damaged))
        self.assertIn("| 38 | 달걀 | egg | 鸡蛋 | œuf |", freeze.render_review_packet(self.packet))

    def test_reparse_detects_changed_raw_gloss_independently_of_prior_intake(self):
        actual = freeze.merge_krdict_snapshot
        def damaged(*args, **kwargs):
            merge = actual(*args, **kwargs)
            candidate = next(r for r in merge["eligible_candidates"] if r["candidate_id"].endswith(":60487:1"))
            candidate["options"]["en"][0]["gloss"] = "Different sense"
            return merge
        with mock.patch.object(freeze, "merge_krdict_snapshot", side_effect=damaged):
            with self.assertRaisesRegex(ContractViolation, "SEMANTIC_FREEZE_SOURCE_TEXT_MISMATCH"):
                freeze.build_review_packet(CORRECTION)


if __name__ == "__main__":
    unittest.main()
