from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from implementation.src import ai_prereview
from implementation.src import review_feedback_intake as intake
from implementation.src.contracts import ContractViolation, canonical_json_bytes
from implementation.tests.test_ai_prereview import _advisory_rows, _source_fixture


DEFERRED = frozenset({2, 9, 10, 19, 21, 35, 38, 59})
DEFAULT_ADJUDICATION = (
    "APPROVED_REMAINDER_BY_REVIEWER_CONFIRMATION",
    "Approved as part of jm02's explicitly confirmed remainder.",
)
SPECIAL_ADJUDICATIONS = {
    2: (
        "DEFERRED_DEEP_RELATION_UNESTABLISHED",
        "Language-family contrast alone is insufficient to establish distinct routes; the deeper bee/abeille relation remains unresolved.",
    ),
    9: (
        "DEFERRED_MIXED_BORROWING_AND_INHERITANCE_CAT",
        "A common cattus source with a borrowing-and-inheritance mixture requires further classification.",
    ),
    10: (
        "DEFERRED_POSSIBLE_COMMON_ANCESTOR_SNOW",
        "A possible common ancestor makes a distinct-routes decision unsupported by the reviewed material.",
    ),
    19: (
        "DEFERRED_ENGLISH_ORIGIN_UNCERTAIN_BIRD",
        "The deeper origin of English bird remains uncertain, so language-family contrast alone does not establish distinct routes.",
    ),
    21: (
        "DEFERRED_MIXED_BORROWING_AND_INHERITANCE_PEAR",
        "English borrowing and French inheritance from the pira lineage form a mixed path requiring further classification.",
    ),
    22: (
        "APPROVED_SOURCE_SENSE_CLARIFIES_FISH",
        "The reviewed KRDICT source sense denotes fish used as food, so 고기/fish/鱼/poisson is retained.",
    ),
    35: (
        "DEFERRED_MIXED_BORROWING_AND_INHERITANCE_BUTTER",
        "English borrowing and French inheritance from the butyrum lineage form a mixed path requiring further classification.",
    ),
    38: (
        "DEFERRED_EN_TERM_NATURALNESS_AND_ETYMOLOGY_EGG",
        "Meaning alignment is retained, but English hen's egg naturalness and the EN–FR etymology remain deferred.",
    ),
    42: (
        "APPROVED_SOURCE_SENSE_CLARIFIES_LIPS",
        "The reviewed KRDICT source sense denotes lips, so 입/lips/嘴唇/lèvre is retained.",
    ),
    50: (
        "APPROVED_SOURCE_SENSE_CLARIFIES_STEP_COUNTER",
        "The reviewed KRDICT source sense is the counter for steps, so 발/step/步/pas is retained.",
    ),
    59: (
        "DEFERRED_SENSE_SPECIFIC_ETYMOLOGY_CARD",
        "The captured English etymology locator may concern the playing-card lineage rather than the reviewed information/ID-card sense.",
    ),
}


def _feedback_rows(plan):
    rows = []
    advisory_rows = _advisory_rows(plan)
    for ordinal, (selection, advisory) in enumerate(
        zip(plan["selected"], advisory_rows), 1
    ):
        deferred = ordinal in DEFERRED
        term_quality = {
            language: {
                "disposition": "APPROVED",
                "value": advisory["suggested_term_quality"][language],
            }
            for language in ("ko", "en", "zh", "fr")
        }
        if ordinal == 38:
            term_quality["en"] = {"disposition": "DEFERRED", "value": None}
        etymology = (
            {
                "disposition": "DEFERRED",
                "subtype": None,
                "direction": None,
            }
            if deferred
            else {
                "disposition": "APPROVED",
                "subtype": advisory["suggested_etymology_subtype"],
                "direction": advisory["suggested_relation_direction"],
            }
        )
        reason_code, note = SPECIAL_ADJUDICATIONS.get(ordinal, DEFAULT_ADJUDICATION)
        rows.append(
            {
                "selection_ordinal": ordinal,
                "candidate_id": selection["candidate_id"],
                "overall_disposition": "DEFERRED" if deferred else "APPROVED",
                "meaning": {
                    "disposition": "APPROVED",
                    "value": advisory["suggested_meaning_alignment"],
                },
                "term_quality": term_quality,
                "etymology": etymology,
                "reason_code": reason_code,
                "intake_note": note,
            }
        )
    return rows


class ReviewFeedbackIntakeTests(unittest.TestCase):
    def setUp(self):
        self.live_source_profile = intake.EXPECTED_REVIEWED_SOURCE_PROFILE
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plan, terms, pairs = _source_fixture()
        self.source_patch = mock.patch.object(
            ai_prereview,
            "_validate_csv_sources",
            return_value=(terms, pairs, "snapshot"),
        )
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

        self.review_root = self.root / "ai_prereview_fixture"
        self.review_root.mkdir()
        self.bundle_dir = self.review_root / "advisory_bundle_fixture"
        self.bundle_dir.mkdir()
        self.plan_path = self.bundle_dir / "selection_plan.json"
        self.advisory_path = self.bundle_dir / "ai_prereview.jsonl"
        self.bundle_manifest_path = self.bundle_dir / "ai_prereview_manifest.json"
        self.summary_path = self.review_root / "ai_prereview_summary_fixture.md"
        self.commitment_path = self.review_root / "selection_commitment.json"
        self.plan_path.write_bytes(canonical_json_bytes(self.plan))
        self.commitment_path.write_bytes(canonical_json_bytes(self.plan))
        self.advisory_path.write_bytes(
            ai_prereview._canonical_jsonl(_advisory_rows(self.plan))
        )
        self.summary_path.write_text(
            "# Fixed 60-item AI pre-review\n\nHuman feedback is recorded separately.\n",
            encoding="utf-8",
        )
        source_raw = self.commitment_path.read_bytes()
        plan_raw = self.plan_path.read_bytes()
        advisory_raw = self.advisory_path.read_bytes()
        manifest = ai_prereview._bundle_manifest(
            source_plan_ref={
                "path": str(self.commitment_path),
                "sha256": hashlib.sha256(source_raw).hexdigest(),
                "bytes": len(source_raw),
            },
            plan_ref={
                "path": str(self.plan_path),
                "sha256": hashlib.sha256(plan_raw).hexdigest(),
                "bytes": len(plan_raw),
            },
            advisory_ref={
                "path": str(self.advisory_path),
                "sha256": hashlib.sha256(advisory_raw).hexdigest(),
                "bytes": len(advisory_raw),
            },
            pairwise_count=60,
            evidence_count=0,
            captured_count=0,
        )
        self.bundle_manifest_path.write_bytes(canonical_json_bytes(manifest))
        for path in (
            self.plan_path,
            self.advisory_path,
            self.bundle_manifest_path,
            self.summary_path,
            self.commitment_path,
        ):
            path.chmod(0o444)
        self.bundle_dir.chmod(0o555)
        fixture_paths = {
            "selection_plan": self.plan_path,
            "advisory": self.advisory_path,
            "bundle_manifest": self.bundle_manifest_path,
            "summary": self.summary_path,
        }
        self.fixture_source_profile = tuple(
            (
                source_name,
                str(fixture_paths[source_name]),
                hashlib.sha256(fixture_paths[source_name].read_bytes()).hexdigest(),
                fixture_paths[source_name].stat().st_size,
            )
            for source_name in intake.SOURCE_NAMES
        )
        self.profile_patch = mock.patch.object(
            intake,
            "EXPECTED_REVIEWED_SOURCE_PROFILE",
            self.fixture_source_profile,
        )
        self.profile_patch.start()
        self.addCleanup(self.profile_patch.stop)
        self.feedback = _feedback_rows(self.plan)

    def _build(self):
        return intake.build_review_feedback_intake(
            self.plan_path,
            self.advisory_path,
            self.bundle_manifest_path,
            self.summary_path,
            self.feedback,
            reviewer_id="jm02",
            review_date="2026-09-15",
            recorded_at_utc="2026-09-15T00:00:00Z",
            scope_root=self.root,
        )

    def _write_bundle_with_intake_bytes(self, payload, intake_raw, name):
        output = self.root / name
        output.mkdir()
        intake_path = output / intake.INTAKE_FILENAME
        manifest_path = output / intake.BUNDLE_MANIFEST_FILENAME
        intake_path.write_bytes(intake_raw)
        intake_ref = {
            "path": str(intake_path),
            "sha256": hashlib.sha256(intake_raw).hexdigest(),
            "bytes": len(intake_raw),
        }
        manifest = intake._build_bundle_manifest(payload, intake_ref)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        intake_path.chmod(0o444)
        manifest_path.chmod(0o444)
        output.chmod(0o555)
        return manifest_path

    def test_exact_jm02_interpretation_and_nonproduction_contract(self):
        payload = self._build()
        self.assertEqual(payload["schema_version"], "review-feedback-intake-v1")
        self.assertEqual(payload["reviewer_id"], "jm02")
        self.assertEqual(len(payload["rows"]), 60)
        self.assertEqual(
            payload["decision_counts"],
            {
                "overall": {
                    "APPROVED": 52,
                    "DEFERRED": 8,
                    "REVISION_REQUIRED": 0,
                },
                "meaning": {
                    "APPROVED": 60,
                    "DEFERRED": 0,
                    "REVISION_REQUIRED": 0,
                },
                "term_quality": {
                    "APPROVED": 239,
                    "DEFERRED": 1,
                    "REVISION_REQUIRED": 0,
                },
                "etymology": {
                    "APPROVED": 52,
                    "DEFERRED": 8,
                    "REVISION_REQUIRED": 0,
                },
            },
        )
        self.assertEqual(
            {
                row["selection_ordinal"]
                for row in payload["rows"]
                if row["overall_disposition"] == "DEFERRED"
            },
            set(DEFERRED),
        )
        self.assertEqual(
            payload["reason_code_counts"],
            {
                DEFAULT_ADJUDICATION[0]: 49,
                **{
                    reason_code: 1
                    for reason_code, _note in SPECIAL_ADJUDICATIONS.values()
                },
            },
        )
        self.assertEqual(
            {
                row["selection_ordinal"]: (
                    row["reason_code"],
                    row["intake_note"],
                )
                for row in payload["rows"]
            },
            {
                ordinal: SPECIAL_ADJUDICATIONS.get(ordinal, DEFAULT_ADJUDICATION)
                for ordinal in range(1, 61)
            },
        )
        self.assertTrue(
            all(
                set(row["etymology"]) == {"disposition", "subtype", "direction"}
                for row in payload["rows"]
            )
        )
        self.assertEqual(
            [
                row["selection_ordinal"]
                for row in payload["rows"]
                if row["term_quality"]["en"]["disposition"] == "DEFERRED"
            ],
            [38],
        )
        self.assertEqual(
            {
                row["selection_ordinal"]
                for row in payload["rows"]
                if row["etymology"]["disposition"] == "DEFERRED"
            },
            set(DEFERRED),
        )
        self.assertTrue(
            all(row["meaning"]["disposition"] == "APPROVED" for row in payload["rows"])
        )
        audit_summary = intake.validate_review_feedback_intake(
            payload, scope_root=self.root
        )
        for field in (
            "identity_authentication_claimed",
            "formal_research_approval_claimed",
            "production_evidence_verified",
            "direct_csv_input_allowed",
            "review_csvs_modified",
            "evidence_records_created",
            "annotation_freeze_created",
            "training_eligible",
        ):
            self.assertIs(payload[field], False)
            self.assertIs(audit_summary[field], False)
        self.assertNotIn(b"APPROVED_BY_RESEARCHER", canonical_json_bytes(payload))

    def test_source_refs_bind_exact_path_hash_and_size(self):
        payload = self._build()
        for name, path in (
            ("selection_plan", self.plan_path),
            ("advisory", self.advisory_path),
            ("bundle_manifest", self.bundle_manifest_path),
            ("summary", self.summary_path),
        ):
            ref = payload["source_artifacts"][name]
            self.assertEqual(ref["path"], str(path))
            self.assertEqual(ref["bytes"], path.stat().st_size)
            self.assertEqual(len(ref["sha256"]), 64)

        self.summary_path.chmod(0o644)
        self.summary_path.write_text("changed", encoding="utf-8")
        self.summary_path.chmod(0o444)
        with self.assertRaisesRegex(
            ContractViolation, "ARTIFACT_SIZE_MISMATCH|REVIEW_INTAKE_SOURCE_HASH_MISMATCH"
        ):
            intake.validate_review_feedback_intake(payload, scope_root=self.root)

    def test_live_source_profile_matches_exact_read_only_artifacts(self):
        self.assertEqual(
            {entry[0] for entry in self.live_source_profile},
            set(intake.SOURCE_NAMES),
        )
        for source_name, raw_path, expected_hash, expected_bytes in (
            self.live_source_profile
        ):
            with self.subTest(source_name=source_name):
                path = Path(raw_path)
                self.assertTrue(path.is_absolute())
                self.assertEqual(path.resolve(strict=True), path)
                raw = path.read_bytes()
                self.assertEqual(len(raw), expected_bytes)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_hash)
                self.assertEqual(path.stat().st_mode & 0o222, 0)

    def test_live_source_profile_rejects_an_alternate_valid_bundle(self):
        with mock.patch.object(
            intake,
            "EXPECTED_REVIEWED_SOURCE_PROFILE",
            self.live_source_profile,
        ), self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_SOURCE_PROFILE_MISMATCH"
        ):
            self._build()

    def test_same_path_initially_altered_summary_is_rejected_by_source_pin(self):
        self.summary_path.chmod(0o644)
        self.summary_path.write_text(
            "# Different 60-item display\n\nStill structurally valid text.\n",
            encoding="utf-8",
        )
        self.summary_path.chmod(0o444)
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_SOURCE_PROFILE_MISMATCH"
        ):
            self._build()

    def test_exact_ai_bundle_manifest_and_summary_path_are_required(self):
        copied_manifest = self.review_root / "copied_manifest.json"
        copied_manifest.write_bytes(self.bundle_manifest_path.read_bytes())
        copied_manifest.chmod(0o444)
        with self.assertRaisesRegex(
            ContractViolation,
            "AI_ADVISORY_BUNDLE_PATH_MISMATCH|REVIEW_INTAKE_AI_BUNDLE_PATH_MISMATCH",
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                copied_manifest,
                self.summary_path,
                self.feedback,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

        wrong_summary = self.review_root / "ai_prereview_summary_other.md"
        wrong_summary.write_bytes(self.summary_path.read_bytes())
        wrong_summary.chmod(0o444)
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_SUMMARY_PATH_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                wrong_summary,
                self.feedback,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

    def test_source_size_is_capped_before_payload_read(self):
        self.summary_path.chmod(0o644)
        with self.summary_path.open("wb") as handle:
            handle.truncate(intake._SOURCE_MAX_BYTES["summary"] + 1)
        self.summary_path.chmod(0o444)
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_SOURCE_TOO_LARGE"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                self.feedback,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

    def test_selection_and_advisory_must_be_canonical_and_match(self):
        noncanonical_plan = self.root / "noncanonical-plan.json"
        noncanonical_plan.write_text(
            json.dumps(self.plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_NONCANONICAL_SELECTION_PLAN"
        ):
            intake.build_review_feedback_intake(
                noncanonical_plan,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                self.feedback,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

        changed_rows = _advisory_rows(self.plan)
        changed_rows[0]["candidate_id"] = changed_rows[1]["candidate_id"]
        changed_advisory = self.root / "changed-advisory.jsonl"
        changed_advisory.write_bytes(ai_prereview._canonical_jsonl(changed_rows))
        with self.assertRaisesRegex(
            ContractViolation, "AI_ADVISORY_CANDIDATE_ORDER_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                changed_advisory,
                self.bundle_manifest_path,
                self.summary_path,
                self.feedback,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

    def test_all_sixty_feedback_rows_and_component_consistency_are_required(self):
        shortened = self.feedback[:-1]
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_ROW_COUNT_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                shortened,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

        changed = copy.deepcopy(self.feedback)
        changed[0]["candidate_id"] = changed[1]["candidate_id"]
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_FEEDBACK_ORDER_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

        changed = copy.deepcopy(self.feedback)
        changed[1]["overall_disposition"] = "APPROVED"
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_OVERALL_DISPOSITION_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

    def test_deferred_rows_require_notes_and_no_formal_token_can_be_injected(self):
        changed = copy.deepcopy(self.feedback)
        changed[1]["intake_note"] = ""
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_REASON_NOTE_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

    def test_approved_values_exactly_match_advisory_and_deferred_values_are_null(self):
        changed = copy.deepcopy(self.feedback)
        changed[0]["meaning"]["value"] = "ALIGNED"
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_MEANING_VALUE_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

        changed = copy.deepcopy(self.feedback)
        changed[0]["etymology"]["confidence"] = "LOW"
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_ETYMOLOGY_SCHEMA_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

    def test_reason_codes_and_exact_adjudication_map_are_closed(self):
        changed = copy.deepcopy(self.feedback)
        changed[0]["reason_code"] = intake._SPECIAL_ADJUDICATIONS[22][0]
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_REASON_CODE_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

        changed = copy.deepcopy(self.feedback)
        changed[21]["overall_disposition"] = "DEFERRED"
        changed[21]["etymology"] = {
            "disposition": "DEFERRED",
            "subtype": None,
            "direction": None,
        }
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_CONFIRMED_DISPOSITION_MISMATCH"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

        changed = copy.deepcopy(self.feedback)
        changed[1]["etymology"]["subtype"] = "INDETERMINATE"
        with self.assertRaisesRegex(
            ContractViolation,
            "REVIEW_INTAKE_UNACCEPTED_ETYMOLOGY_VALUES_FORBIDDEN",
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

        changed = copy.deepcopy(self.feedback)
        changed[37]["term_quality"]["en"]["value"] = "UNRESOLVED"
        with self.assertRaisesRegex(
            ContractViolation,
            "REVIEW_INTAKE_UNACCEPTED_TERM_QUALITY_VALUE_FORBIDDEN",
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

        changed = copy.deepcopy(self.feedback)
        changed[0]["intake_note"] = "approved_by_researcher"
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_FORMAL_APPROVAL_TOKEN_FORBIDDEN"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                changed,
                reviewer_id="jm02",
                review_date="2026-09-15",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

    def test_identity_evidence_csv_freeze_and_training_claims_fail_closed(self):
        for field, code in (
            ("identity_authentication_claimed", "REVIEW_INTAKE_IDENTITY_CLAIM_FORBIDDEN"),
            ("formal_research_approval_claimed", "REVIEW_INTAKE_FORMAL_APPROVAL_FORBIDDEN"),
            ("production_evidence_verified", "REVIEW_INTAKE_PRODUCTION_EVIDENCE_FORBIDDEN"),
            ("direct_csv_input_allowed", "REVIEW_INTAKE_DIRECT_CSV_INPUT_FORBIDDEN"),
            ("review_csvs_modified", "REVIEW_INTAKE_CSV_MODIFICATION_FORBIDDEN"),
            ("evidence_records_created", "REVIEW_INTAKE_EVIDENCE_RECORD_FORBIDDEN"),
            ("annotation_freeze_created", "REVIEW_INTAKE_FREEZE_CLAIM_FORBIDDEN"),
            ("training_eligible", "REVIEW_INTAKE_TRAINING_ELIGIBILITY_FORBIDDEN"),
        ):
            changed = self._build()
            changed[field] = True
            with self.subTest(field=field), self.assertRaisesRegex(
                ContractViolation, code
            ):
                intake.validate_review_feedback_intake(changed, scope_root=self.root)

        changed = self._build()
        changed["formal_status"] = "APPROVED"
        with self.assertRaisesRegex(ContractViolation, "REVIEW_INTAKE_SCHEMA_MISMATCH"):
            intake.validate_review_feedback_intake(changed, scope_root=self.root)

        changed = self._build()
        changed["reviewer_id"] = "someone_else"
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_REVIEWER_ID_MISMATCH"
        ):
            intake.validate_review_feedback_intake(changed, scope_root=self.root)

    def test_publication_is_canonical_read_only_write_once_and_auditable(self):
        payload = self._build()
        output = self.root / "jm02_review_feedback_intake"
        result = intake.publish_review_feedback_intake(
            payload, output, scope_root=self.root
        )
        self.assertEqual(result["status"], intake.AUDIT_STATUS)
        self.assertEqual(result["row_count"], 60)
        intake_path = output / intake.INTAKE_FILENAME
        manifest_path = output / intake.BUNDLE_MANIFEST_FILENAME
        self.assertEqual(intake_path.read_bytes(), canonical_json_bytes(payload))
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {intake.INTAKE_FILENAME, intake.BUNDLE_MANIFEST_FILENAME},
        )
        self.assertEqual(os.stat(output).st_mode & 0o222, 0)
        self.assertEqual(os.stat(intake_path).st_mode & 0o222, 0)
        self.assertEqual(os.stat(manifest_path).st_mode & 0o222, 0)

        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_OUTPUT_EXISTS"
        ):
            intake.publish_review_feedback_intake(payload, output, scope_root=self.root)

        output.chmod(0o755)
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_READ_ONLY_BUNDLE_REQUIRED"
        ):
            intake.audit_review_feedback_intake(manifest_path, scope_root=self.root)

    def test_bundle_manifest_is_internal_and_extra_members_are_rejected(self):
        payload = self._build()
        output = self.root / "exact_bundle"
        result = intake.publish_review_feedback_intake(
            payload, output, scope_root=self.root
        )
        manifest_path = output / intake.BUNDLE_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["intake_artifact"]["path"],
            str(output / intake.INTAKE_FILENAME),
        )
        self.assertEqual(manifest["source_artifacts"], payload["source_artifacts"])
        self.assertEqual(
            result["write_semantics"], "ATOMIC_DIRECTORY_NOREPLACE_READ_ONLY"
        )

        output.chmod(0o755)
        extra = output / "extra"
        extra.write_bytes(b"not part of the bundle")
        extra.chmod(0o444)
        output.chmod(0o555)
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_READ_ONLY_BUNDLE_REQUIRED"
        ):
            intake.audit_review_feedback_intake(
                manifest_path, scope_root=self.root
            )

    def test_publication_rechecks_all_source_hashes_before_atomic_rename(self):
        payload = self._build()
        output = self.root / "changed_source_bundle"
        original_publish = intake.publish_bytes_once
        calls = 0

        def publish_then_change_source(*args, **kwargs):
            nonlocal calls
            result = original_publish(*args, **kwargs)
            calls += 1
            if calls == 2:
                self.summary_path.chmod(0o644)
                self.summary_path.write_text("changed during staging", encoding="utf-8")
                self.summary_path.chmod(0o444)
            return result

        with mock.patch.object(
            intake, "publish_bytes_once", side_effect=publish_then_change_source
        ), self.assertRaisesRegex(
            ContractViolation,
            "ARTIFACT_SIZE_MISMATCH|REVIEW_INTAKE_SOURCE_HASH_MISMATCH",
        ):
            intake.publish_review_feedback_intake(
                payload, output, scope_root=self.root
            )
        self.assertFalse(output.exists())

    def test_publication_normalizes_non_json_payload_failures(self):
        cyclic = self._build()
        cyclic["notice"] = cyclic
        non_json = self._build()
        non_json["notice"] = object()
        for name, payload in (("cyclic", cyclic), ("non_json", non_json)):
            output = self.root / f"invalid_{name}_output"
            with self.subTest(name=name), self.assertRaisesRegex(
                ContractViolation, "REVIEW_INTAKE_INVALID_JSON"
            ):
                intake.publish_review_feedback_intake(
                    payload, output, scope_root=self.root
                )
            self.assertFalse(output.exists())

    def test_audit_rejects_noncanonical_and_duplicate_key_outer_manifest(self):
        payload = self._build()
        intake_raw = canonical_json_bytes(payload)
        intake_ref = {
            "path": str(self.root / intake.INTAKE_FILENAME),
            "sha256": hashlib.sha256(intake_raw).hexdigest(),
            "bytes": len(intake_raw),
        }
        manifest = intake._build_bundle_manifest(payload, intake_ref)
        noncanonical = self.root / "noncanonical-manifest.json"
        noncanonical.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        noncanonical.chmod(0o444)
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_NONCANONICAL_BUNDLE_MANIFEST"
        ):
            intake.audit_review_feedback_intake(noncanonical, scope_root=self.root)

        duplicate = self.root / "duplicate-manifest.json"
        duplicate.write_bytes(
            canonical_json_bytes(manifest).replace(
                b"{",
                b'{"schema_version":"review-feedback-intake-bundle-manifest-v1",',
                1,
            )
        )
        duplicate.chmod(0o444)
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_DUPLICATE_JSON_KEY"
        ):
            intake.audit_review_feedback_intake(duplicate, scope_root=self.root)

    def test_audit_rejects_noncanonical_duplicate_and_nonfinite_inner_intake(self):
        payload = self._build()
        canonical = canonical_json_bytes(payload)
        duplicate = canonical.replace(
            b"{",
            b'{"schema_version":"review-feedback-intake-v1",',
            1,
        )
        nonfinite = canonical.replace(b'"row_count":60', b'"row_count":NaN', 1)
        cases = (
            (
                "noncanonical_inner",
                json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                "REVIEW_INTAKE_NONCANONICAL_JSON",
            ),
            (
                "duplicate_inner",
                duplicate,
                "REVIEW_INTAKE_DUPLICATE_JSON_KEY",
            ),
            (
                "nonfinite_inner",
                nonfinite,
                "REVIEW_INTAKE_NONFINITE_JSON_NUMBER",
            ),
        )
        for name, raw, code in cases:
            with self.subTest(name=name):
                manifest_path = self._write_bundle_with_intake_bytes(
                    payload, raw, name
                )
                with self.assertRaisesRegex(ContractViolation, code):
                    intake.audit_review_feedback_intake(
                        manifest_path, scope_root=self.root
                    )

    def test_unicode_surrogates_fail_closed_at_json_and_text_boundaries(self):
        payload = self._build()

        intake_raw = canonical_json_bytes(payload)
        intake_ref = {
            "path": str(self.root / intake.INTAKE_FILENAME),
            "sha256": hashlib.sha256(intake_raw).hexdigest(),
            "bytes": len(intake_raw),
        }
        outer_raw = canonical_json_bytes(
            intake._build_bundle_manifest(payload, intake_ref)
        ).replace(b'"reviewer_id":"jm02"', b'"reviewer_id":"\\ud800"', 1)
        outer_manifest = self.root / "surrogate-outer-manifest.json"
        outer_manifest.write_bytes(outer_raw)
        outer_manifest.chmod(0o444)
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_UNICODE_SURROGATE_FORBIDDEN"
        ):
            intake.audit_review_feedback_intake(
                outer_manifest, scope_root=self.root
            )

        inner_raw = canonical_json_bytes(payload).replace(
            b'"reviewer_id":"jm02"', b'"reviewer_id":"\\ud800"', 1
        )
        inner_manifest = self._write_bundle_with_intake_bytes(
            payload, inner_raw, "surrogate_inner"
        )
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_UNICODE_SURROGATE_FORBIDDEN"
        ):
            intake.audit_review_feedback_intake(
                inner_manifest, scope_root=self.root
            )

        changed = copy.deepcopy(payload)
        changed["reviewer_id"] = "jm\ud800"
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_UNICODE_SURROGATE_FORBIDDEN"
        ):
            intake.validate_review_feedback_intake(changed, scope_root=self.root)

    def test_unicode_surrogates_fail_closed_in_each_json_source(self):
        for source_path in (
            self.plan_path,
            self.advisory_path,
            self.bundle_manifest_path,
        ):
            with self.subTest(source_path=source_path.name):
                original = source_path.read_bytes()
                source_path.chmod(0o644)
                source_path.write_bytes(
                    original.replace(b"{", b'{"surrogate":"\\ud800",', 1)
                )
                source_path.chmod(0o444)
                try:
                    with self.assertRaisesRegex(
                        ContractViolation,
                        "REVIEW_INTAKE_UNICODE_SURROGATE_FORBIDDEN",
                    ):
                        self._build()
                finally:
                    source_path.chmod(0o644)
                    source_path.write_bytes(original)
                    source_path.chmod(0o444)

    def test_review_date_cannot_precede_selection_commitment(self):
        with self.assertRaisesRegex(
            ContractViolation, "REVIEW_INTAKE_DATE_PRECEDES_SELECTION"
        ):
            intake.build_review_feedback_intake(
                self.plan_path,
                self.advisory_path,
                self.bundle_manifest_path,
                self.summary_path,
                self.feedback,
                reviewer_id="jm02",
                review_date="2026-09-13",
                recorded_at_utc="2026-09-15T00:00:00Z",
                scope_root=self.root,
            )

    def test_review_and_recording_timestamps_cannot_be_future_or_predate_plan(self):
        cases = (
            (
                "2099-01-01",
                "2026-09-15T00:00:00Z",
                "REVIEW_INTAKE_REVIEW_DATE_IN_FUTURE",
            ),
            (
                "2026-09-15",
                "2099-01-01T00:00:00Z",
                "REVIEW_INTAKE_RECORDED_AT_IN_FUTURE",
            ),
            (
                "2026-09-14",
                "2026-09-14T18:00:00Z",
                "REVIEW_INTAKE_RECORDED_AT_PRECEDES_SELECTION",
            ),
        )
        for review_date, recorded_at, code in cases:
            with self.subTest(code=code), self.assertRaisesRegex(
                ContractViolation, code
            ):
                intake.build_review_feedback_intake(
                    self.plan_path,
                    self.advisory_path,
                    self.bundle_manifest_path,
                    self.summary_path,
                    self.feedback,
                    reviewer_id="jm02",
                    review_date=review_date,
                    recorded_at_utc=recorded_at,
                    scope_root=self.root,
                )


if __name__ == "__main__":
    unittest.main()
