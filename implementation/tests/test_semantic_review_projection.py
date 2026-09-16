from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from implementation.src import ai_prereview
from implementation.src import review_feedback_intake as intake
from implementation.src import semantic_review_projection as projection
from implementation.src.contracts import ContractViolation, canonical_json_bytes
from implementation.tests.test_ai_prereview import _advisory_rows, _source_fixture
from implementation.tests.test_review_feedback_intake import _feedback_rows


class SemanticReviewProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plan, terms, pairs = _source_fixture()
        source_patch = mock.patch.object(
            ai_prereview,
            "_validate_csv_sources",
            return_value=(terms, pairs, "snapshot"),
        )
        source_patch.start()
        self.addCleanup(source_patch.stop)

        review_root = self.root / "ai_prereview_fixture"
        review_root.mkdir()
        advisory_bundle = review_root / "advisory_bundle_fixture"
        advisory_bundle.mkdir()
        self.plan_path = advisory_bundle / "selection_plan.json"
        self.advisory_path = advisory_bundle / "ai_prereview.jsonl"
        self.advisory_manifest_path = advisory_bundle / "ai_prereview_manifest.json"
        self.summary_path = review_root / "ai_prereview_summary_fixture.md"
        commitment_path = review_root / "selection_commitment.json"

        self.plan_path.write_bytes(canonical_json_bytes(self.plan))
        commitment_path.write_bytes(canonical_json_bytes(self.plan))
        self.advisory_path.write_bytes(
            ai_prereview._canonical_jsonl(_advisory_rows(self.plan))
        )
        self.summary_path.write_text(
            "# Fixed 60-item AI pre-review\n\n"
            "Human feedback is recorded separately.\n",
            encoding="utf-8",
        )
        commitment_raw = commitment_path.read_bytes()
        plan_raw = self.plan_path.read_bytes()
        advisory_raw = self.advisory_path.read_bytes()
        advisory_manifest = ai_prereview._bundle_manifest(
            source_plan_ref={
                "path": str(commitment_path),
                "sha256": hashlib.sha256(commitment_raw).hexdigest(),
                "bytes": len(commitment_raw),
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
        self.advisory_manifest_path.write_bytes(
            canonical_json_bytes(advisory_manifest)
        )
        source_paths = {
            "selection_plan": self.plan_path,
            "advisory": self.advisory_path,
            "bundle_manifest": self.advisory_manifest_path,
            "summary": self.summary_path,
        }
        for path in (*source_paths.values(), commitment_path):
            path.chmod(0o444)
        advisory_bundle.chmod(0o555)
        fixture_profile = tuple(
            (
                source_name,
                str(source_paths[source_name]),
                hashlib.sha256(source_paths[source_name].read_bytes()).hexdigest(),
                source_paths[source_name].stat().st_size,
            )
            for source_name in intake.SOURCE_NAMES
        )
        profile_patch = mock.patch.object(
            intake, "EXPECTED_REVIEWED_SOURCE_PROFILE", fixture_profile
        )
        profile_patch.start()
        self.addCleanup(profile_patch.stop)

        intake_payload = intake.build_review_feedback_intake(
            self.plan_path,
            self.advisory_path,
            self.advisory_manifest_path,
            self.summary_path,
            _feedback_rows(self.plan),
            reviewer_id="jm02",
            review_date="2026-09-15",
            recorded_at_utc="2026-09-15T00:00:00Z",
            scope_root=self.root,
        )
        self.intake_bundle = self.root / "review_feedback_intake_bundle"
        intake.publish_review_feedback_intake(
            intake_payload, self.intake_bundle, scope_root=self.root
        )
        self.intake_manifest_path = (
            self.intake_bundle / intake.BUNDLE_MANIFEST_FILENAME
        )
        self.intake_payload = intake_payload

    def _build(self) -> dict:
        return projection.build_semantic_review_projection(
            self.intake_manifest_path, scope_root=self.root
        )

    @staticmethod
    def _reseal(value: dict) -> dict:
        logical = {
            key: item for key, item in value.items() if key != "projection_sha256"
        }
        value["projection_sha256"] = hashlib.sha256(
            canonical_json_bytes(logical)
        ).hexdigest()
        return value

    def test_exact_core_counts_and_only_ordinal_38_en_blocks(self) -> None:
        result = self._build()
        source_row_38 = self.intake_payload["rows"][37]
        self.assertEqual(result["row_count"], 60)
        self.assertEqual(
            result["decision_counts"],
            {
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
                "core_overall": {
                    "APPROVED": 59,
                    "DEFERRED": 1,
                    "REVISION_REQUIRED": 0,
                },
            },
        )
        self.assertEqual(result["core_review_status"], "BLOCKED_TERM_QUALITY")
        self.assertEqual(
            result["core_decision_blockers"],
            [
                {
                    "selection_ordinal": 38,
                    "review_order": source_row_38["review_order"],
                    "candidate_id": source_row_38["candidate_id"],
                    "component": "term_quality",
                    "language": "en",
                    "disposition": "DEFERRED",
                }
            ],
        )
        row_38 = result["rows"][37]
        self.assertEqual(row_38["selection_ordinal"], 38)
        self.assertEqual(row_38["term_quality"]["en"], {"disposition": "DEFERRED", "value": None})
        self.assertEqual(row_38["core_disposition"], "DEFERRED")
        self.assertEqual(
            row_38["core_decision_reasons"], ["TERM_QUALITY_EN_DEFERRED"]
        )

    def test_source_overall_is_preserved_but_core_is_recomputed_without_etymology(self) -> None:
        result = self._build()
        source_deferred = {
            row["selection_ordinal"]
            for row in result["rows"]
            if row["source_intake_overall_disposition"] == "DEFERRED"
        }
        core_deferred = {
            row["selection_ordinal"]
            for row in result["rows"]
            if row["core_disposition"] == "DEFERRED"
        }
        self.assertEqual(source_deferred, {2, 9, 10, 19, 21, 35, 38, 59})
        self.assertEqual(core_deferred, {38})
        self.assertEqual(result["rows"][1]["core_disposition"], "APPROVED")
        self.assertEqual(result["core_mapping_rule"], projection.MAPPING_RULE)
        forbidden = {
            "etymology",
            "family_id",
            "etymology_family_id",
            "synonym_cluster_id",
            "relation_direction",
            "shared_source",
        }
        for row in result["rows"]:
            self.assertFalse(set(row).intersection(forbidden))
            self.assertEqual(set(row["term_quality"]), {"ko", "en", "zh", "fr"})

    def test_optional_etymology_and_safety_claims_are_exactly_nonproduction(self) -> None:
        result = self._build()
        self.assertEqual(
            result["optional_etymology"],
            {
                "status": "NOT_RUN",
                "reason": "INSUFFICIENT_SOURCE_BOUND_HUMAN_REVIEWED_ETYMOLOGY_DATA",
                "reviewed_count": 0,
                "artifact": None,
                "blocks_core": False,
            },
        )
        self.assertEqual(
            result["formal_evidence"],
            {
                "status": "PENDING",
                "artifact": None,
                "required_before_training": True,
            },
        )
        self.assertTrue(result["non_production_projection"])
        for field in (
            "training_eligible",
            "direct_trainer_input_allowed",
            "annotation_freeze_created",
            "identity_authentication_claimed",
            "formal_research_approval_claimed",
        ):
            self.assertFalse(result[field])

    def test_all_intake_and_source_refs_are_preserved_by_sha_and_bytes(self) -> None:
        result = self._build()
        audited = intake.audit_review_feedback_intake(
            self.intake_manifest_path, scope_root=self.root
        )
        expected = {
            "review_feedback_intake": audited["intake_artifact"],
            "review_feedback_intake_manifest": audited["manifest_artifact"],
            "selection_plan": self.intake_payload["source_artifacts"]["selection_plan"],
            "advisory": self.intake_payload["source_artifacts"]["advisory"],
            "advisory_bundle_manifest": self.intake_payload["source_artifacts"]["bundle_manifest"],
            "summary": self.intake_payload["source_artifacts"]["summary"],
        }
        self.assertEqual(result["source_artifacts"], expected)
        for ref in result["source_artifacts"].values():
            raw = Path(ref["path"]).read_bytes()
            self.assertEqual(len(raw), ref["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), ref["sha256"])

    def test_build_is_deterministic_and_mandatorily_calls_exact_intake_audit(self) -> None:
        original = intake.audit_review_feedback_intake
        with mock.patch.object(
            intake, "audit_review_feedback_intake", wraps=original
        ) as audited:
            first = self._build()
        second = self._build()
        self.assertEqual(audited.call_count, 2)
        self.assertEqual(first, second)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        logical = {
            key: value for key, value in first.items() if key != "projection_sha256"
        }
        self.assertEqual(
            first["projection_sha256"],
            hashlib.sha256(canonical_json_bytes(logical)).hexdigest(),
        )

    def test_validation_rejects_tampering_even_when_attacker_reseals_hash(self) -> None:
        original = self._build()
        changed = copy.deepcopy(original)
        changed["rows"][1]["core_disposition"] = "DEFERRED"
        self._reseal(changed)
        with self.assertRaisesRegex(
            ContractViolation, "SEMANTIC_PROJECTION_CONTENT_MISMATCH"
        ):
            projection.validate_semantic_review_projection(
                changed, scope_root=self.root
            )

        forbidden = copy.deepcopy(original)
        forbidden["rows"][0]["family_id"] = "invented-family"
        self._reseal(forbidden)
        with self.assertRaisesRegex(
            ContractViolation, "SEMANTIC_PROJECTION_ROW_SCHEMA_MISMATCH"
        ):
            projection.validate_semantic_review_projection(
                forbidden, scope_root=self.root
            )

    def test_canonical_projection_file_audits_and_noncanonical_json_fails(self) -> None:
        value = self._build()
        canonical_path = self.root / "semantic_projection.json"
        canonical_raw = canonical_json_bytes(value)
        canonical_path.write_bytes(canonical_raw)
        audit = projection.audit_semantic_review_projection_file(
            canonical_path, scope_root=self.root
        )
        self.assertEqual(audit["status"], projection.AUDIT_STATUS)
        self.assertEqual(audit["projection_artifact"]["bytes"], len(canonical_raw))
        self.assertEqual(
            audit["projection_artifact"]["sha256"],
            hashlib.sha256(canonical_raw).hexdigest(),
        )

        pretty = self.root / "pretty_projection.json"
        pretty.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(
            ContractViolation, "SEMANTIC_PROJECTION_NONCANONICAL_JSON"
        ):
            projection.audit_semantic_review_projection_file(
                pretty, scope_root=self.root
            )

    def test_projection_and_source_manifest_symlinks_are_rejected(self) -> None:
        with self.subTest(kind="source_manifest"):
            source_alias = self.root / "intake_manifest_alias.json"
            source_alias.symlink_to(self.intake_manifest_path)
            with self.assertRaisesRegex(
                ContractViolation, "SEMANTIC_PROJECTION_SOURCE_NOT_REGULAR_FILE"
            ):
                projection.build_semantic_review_projection(
                    source_alias, scope_root=self.root
                )

        value = self._build()
        real_projection = self.root / "real_projection.json"
        real_projection.write_bytes(canonical_json_bytes(value))
        projection_alias = self.root / "projection_alias.json"
        projection_alias.symlink_to(real_projection)
        with self.subTest(kind="projection"), self.assertRaisesRegex(
            ContractViolation, "SEMANTIC_PROJECTION_ARTIFACT_NOT_REGULAR_FILE"
        ):
            projection.audit_semantic_review_projection_file(
                projection_alias, scope_root=self.root
            )

    def test_duplicate_key_nonfinite_and_surrogate_json_fail_closed(self) -> None:
        value = self._build()
        raw = canonical_json_bytes(value)
        cases = (
            (
                "duplicate",
                raw.replace(
                    b"{",
                    b'{"schema_version":"semantic-review-core-projection-v2",',
                    1,
                ),
                "SEMANTIC_PROJECTION_DUPLICATE_JSON_KEY",
            ),
            (
                "nonfinite",
                raw.replace(b'"row_count":60', b'"row_count":NaN', 1),
                "SEMANTIC_PROJECTION_NONFINITE_JSON_NUMBER",
            ),
            (
                "surrogate",
                raw.replace(b'"reviewer_id":"jm02"', b'"reviewer_id":"\\ud800"', 1),
                "SEMANTIC_PROJECTION_UNICODE_SURROGATE_FORBIDDEN",
            ),
        )
        for name, invalid_raw, code in cases:
            path = self.root / f"{name}.json"
            path.write_bytes(invalid_raw)
            with self.subTest(name=name), self.assertRaisesRegex(
                ContractViolation, code
            ):
                projection.audit_semantic_review_projection_file(
                    path, scope_root=self.root
                )


if __name__ == "__main__":
    unittest.main()
