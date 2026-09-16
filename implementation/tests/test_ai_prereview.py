from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from implementation.src import ai_prereview as advisory
from implementation.src.contracts import ContractViolation, canonical_json_bytes


def _source_fixture():
    terms = {}
    pairs = {}
    selected = []
    for ordinal in range(1, 61):
        candidate_id = f"snapshot:c{ordinal:03d}"
        candidate_sha = hashlib.sha256(candidate_id.encode()).hexdigest()
        pairs[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_sha,
            "review_order": str(ordinal),
        }
        chosen = {}
        for language in ("ko", "en", "zh", "fr"):
            term_id = f"{candidate_id}|{language}|1"
            canonical = f"{language}-{ordinal}"
            term_sha = hashlib.sha256(term_id.encode()).hexdigest()
            terms[term_id] = {
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_sha,
                "lang": language,
                "term_id": term_id,
                "term_sha256": term_sha,
                "term_canonical": canonical,
                "source_gloss": f"{language} source gloss {ordinal}",
                "source_option_id": "a" * 64,
                "source_span_start": "0",
                "source_span_end": str(len(canonical)),
            }
            chosen[language] = {
                "canonical": canonical,
                "source_gloss": f"{language} source gloss {ordinal}",
                "source_option_id": "a" * 64,
                "source_span_start": 0,
                "source_span_end": len(canonical),
                "term_id": term_id,
                "term_sha256": term_sha,
            }
        selected.append(
            {
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_sha,
                "pairwise_distinct_canonical": True,
                "review_order": ordinal,
                "selection_ordinal": ordinal,
                "terms": chosen,
            }
        )
    dummy_ref = {"path": "/placeholder", "sha256": "b" * 64, "bytes": 1}
    plan = {
        "advisory_only": True,
        "coverage_targets_not_claims": dict(advisory.ANTICIPATED_QUOTA_TARGETS),
        "created_at_utc": "2026-09-14T18:40:20Z",
        "forbidden_outputs": list(advisory.FORBIDDEN_OUTPUTS),
        "human_approval_claimed": False,
        "implementation_git_commit": "c" * 40,
        "inputs": {
            name: dict(dummy_ref) for name in advisory.REQUIRED_INPUT_NAMES
        },
        "model_results_seen": False,
        "pairwise_distinct_precheck_count": 60,
        "replacement_policy": advisory.REPLACEMENT_POLICY,
        "requires_human_review": True,
        "schema_version": advisory.SELECTION_PLAN_SCHEMA,
        "selected": selected,
        "selected_count": 60,
        "selection_locked_before_evidence_review": True,
        "selection_method": advisory.SELECTION_METHOD,
        "selection_rationale": "Fixed before evidence review.",
        "snapshot_id": "snapshot",
        "training_eligible": False,
    }
    return plan, terms, pairs


def _advisory_rows(plan):
    return [
        {
            "schema_version": advisory.ADVISORY_ROW_SCHEMA,
            "selection_ordinal": ordinal,
            "candidate_id": selection["candidate_id"],
            "advisory_only": True,
            "requires_human_review": True,
            "suggested_meaning_alignment": "UNRESOLVED",
            "suggested_term_quality": {
                language: "UNRESOLVED" for language in ("ko", "en", "zh", "fr")
            },
            "suggested_etymology_subtype": "INDETERMINATE",
            "suggested_relation_direction": "UNKNOWN",
            "suggested_shared_source": "",
            "suggested_confidence": "LOW",
            "evidence_candidates": [],
            "notes": "AI lead only; human review remains required.",
        }
        for ordinal, selection in enumerate(plan["selected"], 1)
    ]


class SelectionPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.terms, self.pairs = _source_fixture()
        self.source_patch = mock.patch.object(
            advisory,
            "_validate_csv_sources",
            return_value=(self.terms, self.pairs, "snapshot"),
        )
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def test_exact_sixty_four_terms_and_quotas_pass(self):
        result = advisory.validate_ai_prereview_selection_plan(self.plan)
        self.assertEqual(result["selection_count"], 60)
        self.assertEqual(result["pairwise_distinct_count"], 60)
        self.assertFalse(result["training_eligible"])
        self.assertTrue(result["requires_human_review"])
        self.assertEqual(
            result["anticipated_quota_targets"], advisory.ANTICIPATED_QUOTA_TARGETS
        )

    def test_model_results_replacement_order_and_source_mutations_fail(self):
        mutations = []
        changed = copy.deepcopy(self.plan)
        changed["model_results_seen"] = True
        mutations.append((changed, "AI_PLAN_MODEL_RESULTS_FORBIDDEN"))
        changed = copy.deepcopy(self.plan)
        changed["replacement_policy"] = "AUTO_REPLACE"
        mutations.append((changed, "AI_PLAN_AUTOMATIC_REPLACEMENT_FORBIDDEN"))
        changed = copy.deepcopy(self.plan)
        changed["selected"][1]["selection_ordinal"] = 1
        mutations.append((changed, "AI_PLAN_SELECTION_ORDER_MISMATCH"))
        changed = copy.deepcopy(self.plan)
        changed["selected"][0]["terms"]["en"]["canonical"] = "changed"
        mutations.append((changed, "AI_PLAN_SELECTED_TERM_SOURCE_MISMATCH"))
        changed = copy.deepcopy(self.plan)
        changed["selected"][0]["terms"]["en"]["source_span_start"] = False
        mutations.append((changed, "AI_PLAN_SELECTED_TERM_SOURCE_MISMATCH"))
        changed = copy.deepcopy(self.plan)
        changed["selected"] = changed["selected"][:-1]
        mutations.append((changed, "AI_PLAN_COUNT_MISMATCH"))
        for value, code in mutations:
            with self.subTest(code=code), self.assertRaisesRegex(
                ContractViolation, code
            ):
                advisory.validate_ai_prereview_selection_plan(value)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ContractViolation, "DUPLICATE_JSON_KEY"):
            advisory._loads_object(b'{"a":1,"a":2}', "BAD_JSON")


class AdvisoryRowsTests(unittest.TestCase):
    def setUp(self):
        self.plan, _terms, _pairs = _source_fixture()
        self.rows = _advisory_rows(self.plan)

    def test_exact_same_order_is_advisory_only(self):
        rows, evidence, captures = advisory._validate_advisory_rows(
            self.rows, self.plan, scope_root=Path(tempfile.gettempdir())
        )
        self.assertEqual(len(rows), 60)
        self.assertEqual((evidence, captures), (0, 0))
        self.assertNotIn("training_eligible", rows[0])
        self.assertNotIn("reviewer", rows[0])
        self.assertNotIn("evidence_origin", rows[0])

    def test_reordering_and_human_or_production_fields_fail_structurally(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["candidate_id"] = changed[1]["candidate_id"]
        with self.assertRaisesRegex(
            ContractViolation, "AI_ADVISORY_CANDIDATE_ORDER_MISMATCH"
        ):
            advisory._validate_advisory_rows(
                changed, self.plan, scope_root=Path(tempfile.gettempdir())
            )

    def test_shared_source_matches_the_suggested_subtype(self):
        changed = copy.deepcopy(self.rows)
        changed[0].update(
            {
                "suggested_etymology_subtype": "COGNATE_INHERITED",
                "suggested_relation_direction": "COMMON_ANCESTOR_TO_BOTH",
                "suggested_shared_source": "Proto-Indo-European *example",
            }
        )
        rows, _evidence, _captures = advisory._validate_advisory_rows(
            changed, self.plan, scope_root=Path(tempfile.gettempdir())
        )
        self.assertEqual(
            rows[0]["suggested_shared_source"], "Proto-Indo-European *example"
        )
        changed[0]["suggested_shared_source"] = ""
        with self.assertRaisesRegex(
            ContractViolation, "AI_ADVISORY_SHARED_SOURCE_REQUIRED"
        ):
            advisory._validate_advisory_rows(
                changed, self.plan, scope_root=Path(tempfile.gettempdir())
            )
        unrelated = copy.deepcopy(self.rows)
        unrelated[0]["suggested_shared_source"] = "Latin"
        with self.assertRaisesRegex(
            ContractViolation, "AI_ADVISORY_SHARED_SOURCE_NOT_ALLOWED"
        ):
            advisory._validate_advisory_rows(
                unrelated, self.plan, scope_root=Path(tempfile.gettempdir())
            )
        for forbidden in ("reviewer", "evidence_origin", "training_eligible"):
            changed = copy.deepcopy(self.rows)
            changed[0][forbidden] = "human"
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(
                ContractViolation, "AI_ADVISORY_ROW_SCHEMA_MISMATCH"
            ):
                advisory._validate_advisory_rows(
                    changed, self.plan, scope_root=Path(tempfile.gettempdir())
                )
        changed = copy.deepcopy(self.rows)
        changed[0]["notes"] = "APPROVED_BY_RESEARCHER"
        with self.assertRaisesRegex(
            ContractViolation, "AI_ADVISORY_HUMAN_APPROVAL_CLAIM_FORBIDDEN"
        ):
            advisory._validate_advisory_rows(
                changed, self.plan, scope_root=Path(tempfile.gettempdir())
            )

    def test_https_lead_and_optional_capture_are_never_production_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            capture = root / "capture.bin"
            capture.write_bytes(b"dictionary excerpt")
            capture_ref = {
                "path": str(capture),
                "sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
                "bytes": capture.stat().st_size,
            }
            self.rows[0]["evidence_candidates"] = [
                {
                    "source_url": "https://example.test/dictionary/entry",
                    "locator": "sense 1",
                    "summary": "Candidate source; not yet human-verified.",
                    "capture_artifact": capture_ref,
                },
                {
                    "source_url": "https://example.test/history/entry",
                    "locator": "paragraph 2",
                    "summary": "URL-only lead.",
                    "capture_artifact": None,
                },
            ]
            _rows, evidence, captures = advisory._validate_advisory_rows(
                self.rows, self.plan, scope_root=root
            )
            self.assertEqual((evidence, captures), (2, 1))
            manifest = advisory._bundle_manifest(
                source_plan_ref=capture_ref,
                plan_ref=capture_ref,
                advisory_ref=capture_ref,
                pairwise_count=60,
                evidence_count=evidence,
                captured_count=captures,
            )
            self.assertFalse(manifest["evidence_candidates_are_production_evidence"])
            self.assertFalse(manifest["capture_artifacts_are_production_evidence"])
            changed = copy.deepcopy(self.rows)
            changed[0]["evidence_candidates"][0]["source_url"] = "http://example.test/"
            with self.assertRaisesRegex(
                ContractViolation, "AI_ADVISORY_INVALID_EVIDENCE_URL"
            ):
                advisory._validate_advisory_rows(changed, self.plan, scope_root=root)


class BundlePublicationTests(unittest.TestCase):
    @staticmethod
    def _stage(root: Path, name: str) -> Path:
        stage = root / name
        stage.mkdir()
        (stage / "payload").write_bytes(name.encode())
        return stage

    def test_existing_empty_directory_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            stage = self._stage(root, "stage")
            destination = root / "bundle"
            destination.mkdir()
            with self.assertRaisesRegex(
                ContractViolation, "AI_ADVISORY_OUTPUT_EXISTS"
            ):
                advisory._publish_directory_noreplace(stage, destination)
            self.assertTrue(stage.is_dir())
            self.assertEqual(list(destination.iterdir()), [])

    def test_concurrent_directory_publication_has_one_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            stages = [self._stage(root, "stage-a"), self._stage(root, "stage-b")]
            destination = root / "bundle"
            barrier = threading.Barrier(2)

            def publish(stage):
                barrier.wait()
                try:
                    advisory._publish_directory_noreplace(stage, destination)
                except ContractViolation as exc:
                    return exc.code
                return "PASS"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(publish, stages))
            self.assertCountEqual(outcomes, ["PASS", "AI_ADVISORY_OUTPUT_EXISTS"])
            self.assertTrue((destination / "payload").is_file())

    def test_write_once_bundle_round_trips_and_cannot_be_replaced(self):
        plan, _terms, _pairs = _source_fixture()
        rows = _advisory_rows(plan)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "selection_commitment.json"
            source_bytes = canonical_json_bytes(plan)
            source.write_bytes(source_bytes)
            source_ref = {
                "path": str(source),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bytes": len(source_bytes),
                "summary": {},
            }
            draft = root / "draft.jsonl"
            draft.write_bytes(advisory._canonical_jsonl(rows))
            output = root / "bundle"
            with mock.patch.object(
                advisory,
                "load_and_validate_ai_prereview_selection_plan",
                return_value=(plan, source_ref),
            ):
                result = advisory.publish_ai_prereview_bundle(
                    source, draft, output_dir=output, scope_root=root
                )
            self.assertEqual(result["selection_count"], 60)
            self.assertFalse(result["training_eligible"])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "selection_plan.json",
                    "ai_prereview.jsonl",
                    "ai_prereview_manifest.json",
                },
            )
            with mock.patch.object(
                advisory,
                "load_and_validate_ai_prereview_selection_plan",
                return_value=(plan, source_ref),
            ), self.assertRaisesRegex(
                ContractViolation, "AI_ADVISORY_OUTPUT_EXISTS"
            ):
                advisory.publish_ai_prereview_bundle(
                    source, draft, output_dir=output, scope_root=root
                )

            expected_summary = {
                "pairwise_distinct_count": 60,
            }
            with mock.patch.object(
                advisory,
                "validate_ai_prereview_selection_plan",
                return_value=expected_summary,
            ):
                audit = advisory.audit_ai_prereview_bundle(
                    output / "ai_prereview_manifest.json", scope_root=root
                )
            self.assertEqual(audit["status"], "PASS_AI_ADVISORY_ONLY")
            self.assertFalse(audit["evidence_candidates_are_production_evidence"])

            # Read-only mode is part of the audit contract, not merely a
            # publication convention.
            output.chmod(0o755)
            with mock.patch.object(
                advisory,
                "validate_ai_prereview_selection_plan",
                return_value=expected_summary,
            ), self.assertRaisesRegex(
                ContractViolation, "AI_ADVISORY_READ_ONLY_BUNDLE_REQUIRED"
            ):
                advisory.audit_ai_prereview_bundle(
                    output / "ai_prereview_manifest.json", scope_root=root
                )


if __name__ == "__main__":
    unittest.main()
