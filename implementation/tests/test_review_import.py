from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from implementation.src.contracts import PROJECT_ROOT, ContractViolation, canonical_json_bytes
from implementation.src.prepare_data import merge_krdict_snapshot
from implementation.src.review_import import (
    ETYMOLOGY_6_TO_4,
    ReviewTables,
    build_review_tables,
    compile_filled_review_bundle,
    en_fr_pair_evidence_subject_sha256,
    index_evidence_records,
    load_and_validate_review_csv_bundle,
    render_review_csv_bundle,
    validate_pair_evidence_set,
    validate_relation_direction,
    validate_review_tables,
)


def logical_hash(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def option(language: str, answer: str, index: int) -> dict:
    core = language + "\0" + answer + "\0" + str(index)
    return {
        "answer": " ".join(answer.split()),
        "word_raw": answer,
        "definition_raw": f"{language} definition {index}",
        "gloss": f"{language} definition {index}",
        "option_id": hashlib.sha256(core.encode()).hexdigest(),
        "source_refs": [
            {
                "raw_sha256": hashlib.sha256((core + " source").encode()).hexdigest(),
                "language": language,
            }
        ],
    }


def fixture_merge(n: int = 1, overrides: dict[int, dict[str, str]] | None = None) -> dict:
    candidates = []
    overrides = overrides or {}
    for index in range(n):
        words = {
            "ko": f"한국어{index}",
            "en": f"english{index}",
            "zh": f"中文{index}",
            "fr": f"français{index}",
            **overrides.get(index, {}),
        }
        candidate_core = {
            "schema_version": "4.0.0",
            "data_kind": "SYNTHETIC_TEST_FIXTURE",
            "snapshot_id": "fixture-snapshot",
            "target_code": str(1000 + index),
            "sense_order": "1",
            "candidate_id": f"fixture-snapshot:{1000 + index}:1",
            "queries": [f"query-{index}"],
            "options": {
                language: [option(language, words[language], index)]
                for language in ("ko", "en", "zh", "fr")
            },
            "source_record_hashes": [hashlib.sha256(f"source-{index}".encode()).hexdigest()],
            "entry_urls": [f"https://example.invalid/{index}"],
            "machine_status": "PENDING_HUMAN_REVIEW",
        }
        candidates.append(
            {**candidate_core, "candidate_sha256": logical_hash(candidate_core)}
        )
    merge_core = {
        "schema_version": "4.0.0",
        "implementation_revision": "test",
        "status": "MERGED_PENDING_REVIEW",
        "data_kind": "SYNTHETIC_TEST_FIXTURE",
        "snapshot_id": "fixture-snapshot",
        "source_manifest_sha256": "a" * 64,
        "source_set_sha256": "b" * 64,
        "eligible_candidates": candidates,
        "quarantine": [],
        "summary": {"eligible": n, "quarantined": 0},
    }
    return {**merge_core, "merge_sha256": logical_hash(merge_core)}


def mutable_tables(tables: ReviewTables) -> ReviewTables:
    return ReviewTables(
        tuple(copy.deepcopy(row) for row in tables.terms),
        tuple(copy.deepcopy(row) for row in tables.pairs),
        tuple(copy.deepcopy(row) for row in tables.screen),
    )


def evidence_record(
    evidence_id: str,
    subject_kind: str,
    subject_id: str,
    supports_label: str,
    *,
    payload_sha256: str = "c" * 64,
) -> dict:
    core = {
        "evidence_id": evidence_id,
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "supports_label": supports_label,
        "source_url": f"https://example.invalid/evidence/{evidence_id}",
        "evidence_origin": "human-reviewed-dictionary-record",
        "source_name": "Synthetic Evidence Dictionary",
        "source_version": "fixture-v1",
        "source_license": "Synthetic fixture license",
        "source_license_url": "https://example.invalid/license",
        "sense_locator": f"sense-{evidence_id}",
        "payload_path": "evidence/payload.txt",
        "payload_sha256": payload_sha256,
        "payload_bytes": 1,
        "retrieved_at": "2026-01-01T00:00:00Z",
        "conflicts_with": [],
    }
    return {**core, "record_sha256": logical_hash(core)}


class ReviewGenerationTests(unittest.TestCase):
    def test_blank_generation_is_nonapproving_and_round_trips_strict_csv(self):
        merge = fixture_merge(
            overrides={
                0: {
                    "ko": "역",
                    "en": "role; part",
                    "zh": "(电影、戏剧中)角，角色",
                    "fr": "rôle, emploi",
                }
            }
        )
        tables = build_review_tables(merge)
        self.assertEqual(len(tables.pairs), 1)
        self.assertEqual(
            [row["term_canonical"] for row in tables.terms if row["lang"] == "zh"],
            ["(电影、戏剧中)角", "角色"],
        )
        for row in tables.terms:
            self.assertEqual(row["term_review_status"], "")
        for row in tables.pairs:
            self.assertEqual(row["concept_review_status"], "")
            self.assertEqual(row["pilot_selected"], "")
        rendered = render_review_csv_bundle(merge)
        self.assertEqual(rendered["status"], "PENDING_HUMAN_REVIEW")
        self.assertFalse(rendered["training_eligible"])
        self.assertEqual(set(rendered["files"]), {
            "terms_long.csv",
            "pair_selection_sheet.csv",
            "untranslated_screen.csv",
        })
        loaded = load_and_validate_review_csv_bundle(
            merge,
            rendered["files"]["terms_long.csv"],
            rendered["files"]["pair_selection_sheet.csv"],
            rendered["files"]["untranslated_screen.csv"],
        )
        self.assertEqual(len(loaded.terms), len(tables.terms))
        audit = validate_review_tables(merge, loaded)
        self.assertEqual(audit["status"], "BLOCKED_DATA_QA")
        self.assertEqual(audit["automatically_approved"], 0)

    def test_source_columns_and_selected_term_qa_fail_closed(self):
        merge = fixture_merge()
        tables = mutable_tables(build_review_tables(merge))
        tables.terms[0]["term_canonical"] = "tampered"
        with self.assertRaisesRegex(ContractViolation, "CSV_SOURCE_COLUMN_MISMATCH"):
            validate_review_tables(merge, tables)

        tables = mutable_tables(build_review_tables(merge))
        pair = tables.pairs[0]
        pair["candidate_decision"] = "SELECT_PAIR"
        for language in ("ko", "en", "zh", "fr"):
            pair[f"selected_{language}_term_id"] = next(
                row["term_id"] for row in tables.terms if row["lang"] == language
            )
        with self.assertRaisesRegex(ContractViolation, "SELECTED_TERM_QA_PENDING"):
            validate_review_tables(merge, tables)

    def test_mapping_and_direction_contract(self):
        self.assertEqual(
            ETYMOLOGY_6_TO_4,
            {
                "BORROWING_DIRECT": "BORROWING_DOCUMENTED",
                "BORROWING_PARALLEL": "SHARED_SOURCE_DOCUMENTED",
                "NEOCLASSICAL_SHARED": "SHARED_SOURCE_DOCUMENTED",
                "COGNATE_INHERITED": "SHARED_SOURCE_DOCUMENTED",
                "DISTINCT_ROUTES_REVIEWED": "DISTINCT_ROUTES_REVIEWED",
                "INDETERMINATE": "UNRESOLVED",
            },
        )
        validate_relation_direction("BORROWING_DIRECT", "EN_TO_FR", "")
        validate_relation_direction(
            "BORROWING_PARALLEL", "COMMON_SOURCE_TO_BOTH", "Latin source"
        )
        with self.assertRaisesRegex(ContractViolation, "INVALID_ETYMOLOGY_DIRECTION"):
            validate_relation_direction("BORROWING_PARALLEL", "EN_TO_FR", "Latin")
        with self.assertRaisesRegex(ContractViolation, "SHARED_SOURCE_REQUIRED"):
            validate_relation_direction("COGNATE_INHERITED", "COMMON_ANCESTOR_TO_BOTH", "")

    def test_evidence_origin_is_a_closed_human_reviewed_vocabulary(self):
        for origin in (
            "model_generated",
            "model-generated",
            "model generated",
            "llm-generated",
            "ChatGPT",
            "human-reviewed-dictionary-record-but-model-generated",
        ):
            record = evidence_record(
                "origin-test", "TERM", "term-id", "ATTESTED_SAME_SENSE"
            )
            record["evidence_origin"] = origin
            record["record_sha256"] = logical_hash(
                {key: value for key, value in record.items() if key != "record_sha256"}
            )
            with self.subTest(origin=origin), self.assertRaisesRegex(
                ContractViolation, "INVALID_EVIDENCE_ORIGIN"
            ):
                index_evidence_records([record], production=False)

    def test_evidence_source_version_and_license_are_hash_bound(self):
        baseline = evidence_record(
            "source-metadata", "TERM", "term-id", "ATTESTED_SAME_SENSE"
        )
        self.assertIn(
            "source-metadata", index_evidence_records([baseline], production=False)
        )
        for field, value, code in (
            ("source_name", "", "INVALID_EVIDENCE_SOURCE_NAME"),
            ("source_version", "", "INVALID_EVIDENCE_SOURCE_VERSION"),
            ("source_license", "", "INVALID_EVIDENCE_SOURCE_LICENSE"),
            (
                "source_license_url",
                "http://example.invalid/license",
                "INVALID_EVIDENCE_SOURCE_LICENSE_URL",
            ),
        ):
            with self.subTest(field=field):
                changed = dict(baseline)
                changed[field] = value
                changed["record_sha256"] = logical_hash(
                    {
                        key: item
                        for key, item in changed.items()
                        if key != "record_sha256"
                    }
                )
                with self.assertRaisesRegex(ContractViolation, code):
                    index_evidence_records([changed], production=False)


class RealReviewRegressionTests(unittest.TestCase):
    def test_real_catalog_counts_and_known_boundary_cases(self):
        manifest = (
            PROJECT_ROOT
            / "work/sources/krdict/hardened_v4r1_20260914T151550Z/source_manifest.json"
        )
        if not manifest.is_file():
            self.skipTest("real immutable KRDICT snapshot is not present")
        merge = merge_krdict_snapshot(manifest)
        tables = build_review_tables(merge)
        summary = render_review_csv_bundle(merge)["summary"]
        self.assertEqual(len(tables.pairs), 409)
        self.assertEqual(
            Counter(row["lang"] for row in tables.terms),
            {"ko": 409, "en": 531, "zh": 608, "fr": 783},
        )
        zh_counts = defaultdict(int)
        for row in tables.terms:
            if row["lang"] == "zh":
                zh_counts[row["candidate_id"]] += 1
        self.assertEqual(sum(value > 1 for value in zh_counts.values()), 160)
        self.assertEqual(summary["exact_en_fr_pair_count"], 50)
        self.assertEqual(summary["exact_en_fr_term_count"], 100)
        self.assertEqual(summary["case_variant_pair_count"], 1)
        self.assertEqual(summary["case_variant_term_count"], 2)
        self.assertEqual(summary["romanized_term_count"], 30)
        self.assertEqual(summary["priority_union_term_count"], 116)

        order_15_zh = [
            row["term_canonical"]
            for row in tables.terms
            if row["review_order"] == "15" and row["lang"] == "zh"
        ]
        self.assertEqual(order_15_zh, ["(电影、戏剧中)角", "角色"])
        screen_by_order = defaultdict(list)
        for row in tables.screen:
            screen_by_order[row["review_order"]].append(row)
        self.assertTrue(
            any("CASE_VARIANT_EN_FR" in row["screen_flags_json"] for row in screen_by_order["313"])
        )
        self.assertTrue(
            any("ROMANIZED_KO_EXACT" in row["screen_flags_json"] for row in screen_by_order["271"])
        )
        self.assertTrue(
            any("ROMANIZED_KO_EXACT" in row["screen_flags_json"] for row in screen_by_order["337"])
        )


class PairEvidenceConflictContractTests(unittest.TestCase):
    subject_id = "d" * 64

    def _symmetric_selected_and_alternative(self) -> list[dict]:
        selected = evidence_record(
            "selected-support",
            "EN_FR_PAIR",
            self.subject_id,
            "BORROWING_DIRECT",
        )
        alternative = evidence_record(
            "alternative-support",
            "EN_FR_PAIR",
            self.subject_id,
            "DISTINCT_ROUTES_REVIEWED",
        )
        selected["conflicts_with"] = ["alternative-support"]
        alternative["conflicts_with"] = ["selected-support"]
        for row in (selected, alternative):
            row["record_sha256"] = logical_hash(
                {key: value for key, value in row.items() if key != "record_sha256"}
            )
        return [selected, alternative]

    def test_selected_and_alternative_symmetric_medium_passes(self):
        evidence = self._symmetric_selected_and_alternative()
        checked = validate_pair_evidence_set(
            evidence,
            subject_id=self.subject_id,
            selected_subtype="BORROWING_DIRECT",
            confidence="MEDIUM",
        )
        self.assertEqual(
            [row["supports_label"] for row in checked],
            ["BORROWING_DIRECT", "DISTINCT_ROUTES_REVIEWED"],
        )

    def test_indeterminate_may_record_a_completed_search_without_evidence(self):
        self.assertEqual(
            validate_pair_evidence_set(
                [],
                subject_id=self.subject_id,
                selected_subtype="INDETERMINATE",
                confidence="LOW",
            ),
            [],
        )
        with self.assertRaisesRegex(
            ContractViolation, "INVALID_ETYMOLOGY_EVIDENCE"
        ):
            validate_pair_evidence_set(
                [],
                subject_id=self.subject_id,
                selected_subtype="BORROWING_DIRECT",
                confidence="LOW",
            )

    def test_selected_subtype_support_is_required(self):
        evidence = self._symmetric_selected_and_alternative()
        evidence[0]["supports_label"] = "NEOCLASSICAL_SHARED"
        evidence[0]["record_sha256"] = logical_hash(
            {
                key: value
                for key, value in evidence[0].items()
                if key != "record_sha256"
            }
        )
        with self.assertRaisesRegex(
            ContractViolation, "SELECTED_ETYMOLOGY_SUPPORT_REQUIRED"
        ):
            validate_pair_evidence_set(
                evidence,
                subject_id=self.subject_id,
                selected_subtype="BORROWING_DIRECT",
                confidence="MEDIUM",
            )

    def test_alternative_requires_a_conflict_with_selected_support(self):
        evidence = self._symmetric_selected_and_alternative()
        for row in evidence:
            row["conflicts_with"] = []
            row["record_sha256"] = logical_hash(
                {key: value for key, value in row.items() if key != "record_sha256"}
            )
        with self.assertRaisesRegex(
            ContractViolation, "ALTERNATIVE_ETYMOLOGY_CONFLICT_REQUIRED"
        ):
            validate_pair_evidence_set(
                evidence,
                subject_id=self.subject_id,
                selected_subtype="BORROWING_DIRECT",
                confidence="MEDIUM",
            )

    def test_symmetric_conflict_forbids_high_confidence(self):
        with self.assertRaisesRegex(
            ContractViolation, "CONFLICTING_EVIDENCE_HIGH_CONFIDENCE_FORBIDDEN"
        ):
            validate_pair_evidence_set(
                self._symmetric_selected_and_alternative(),
                subject_id=self.subject_id,
                selected_subtype="BORROWING_DIRECT",
                confidence="HIGH",
            )


class CompletedBundleTests(unittest.TestCase):
    def test_exact_60_40_12_12_and_compile(self):
        merge = fixture_merge(60)
        tables = mutable_tables(build_review_tables(merge))
        evidence_records = {}
        term_by_candidate_lang = {
            (row["candidate_id"], row["lang"]): row for row in tables.terms
        }
        for index, pair in enumerate(tables.pairs):
            pair["candidate_decision"] = "SELECT_PAIR"
            pair["pilot_selected"] = "TRUE"
            for language in ("ko", "en", "zh", "fr"):
                pair[f"selected_{language}_term_id"] = term_by_candidate_lang[
                    (pair["candidate_id"], language)
                ]["term_id"]
            for language in ("en", "fr"):
                term = term_by_candidate_lang[(pair["candidate_id"], language)]
                quality_id = f"quality-{index}-{language}"
                evidence_records[quality_id] = evidence_record(
                    quality_id,
                    "TERM",
                    term["term_id"],
                    "ATTESTED_SAME_SENSE",
                )
                term.update(
                    {
                        "segmentation_decision": "APPROVED",
                        "translation_quality": "ATTESTED_SAME_SENSE",
                        "is_transliteration": "FALSE",
                        "quality_evidence_ids_json": json.dumps([quality_id]),
                        "quality_note": "Human fixture attestation.",
                        "term_review_status": "APPROVED_BY_RESEARCHER",
                        "term_reviewer": "synthetic-test-reviewer",
                        "term_review_date": "2026-01-01",
                    }
                )
            pair.update(
                {
                    "selection_rationale": "Exact synthetic source span.",
                    "meaning_alignment_decision": "ALIGNED",
                    "sense_alignment_note": "Same synthetic fixture sense.",
                    "source_alignment_checked": "TRUE",
                    "answer_copy_checked": "TRUE",
                    "synonym_cluster_id": f"synonym-{index}",
                    "historical_scope": "Synthetic test scope.",
                    "confidence": "MEDIUM",
                    "family_id": f"family-{index}",
                    "concept_review_status": "APPROVED_BY_RESEARCHER",
                    "concept_reviewer": "synthetic-test-reviewer",
                    "concept_review_date": "2026-01-01",
                    "etymology_review_status": "APPROVED_BY_RESEARCHER",
                    "etymology_reviewer": "synthetic-test-reviewer",
                    "etymology_review_date": "2026-01-01",
                }
            )
            if index < 12:
                subtype = "BORROWING_DIRECT"
                pair.update(
                    {
                        "etymology_subtype": subtype,
                        "etymology_primary": "BORROWING_DOCUMENTED",
                        "relation_direction": "EN_TO_FR",
                        "shared_source": "",
                        "evidence_search_note": "",
                    }
                )
            elif index < 24:
                subtype = "DISTINCT_ROUTES_REVIEWED"
                pair.update(
                    {
                        "etymology_subtype": subtype,
                        "etymology_primary": "DISTINCT_ROUTES_REVIEWED",
                        "relation_direction": "NONE",
                        "shared_source": "",
                        "evidence_search_note": "",
                    }
                )
            else:
                subtype = "INDETERMINATE"
                pair.update(
                    {
                        "etymology_subtype": subtype,
                        "etymology_primary": "UNRESOLVED",
                        "relation_direction": "UNKNOWN",
                        "shared_source": "",
                        "evidence_search_note": "Sources searched; evidence insufficient.",
                    }
                )
            pair_id = f"pair-{index}"
            selected_en = term_by_candidate_lang[(pair["candidate_id"], "en")]
            selected_fr = term_by_candidate_lang[(pair["candidate_id"], "fr")]
            pair_subject = en_fr_pair_evidence_subject_sha256(
                candidate_id=pair["candidate_id"],
                en_term_id=selected_en["term_id"],
                en_term_sha256=selected_en["term_sha256"],
                en_canonical_answer=selected_en["term_canonical"],
                fr_term_id=selected_fr["term_id"],
                fr_term_sha256=selected_fr["term_sha256"],
                fr_canonical_answer=selected_fr["term_canonical"],
            )
            evidence_records[pair_id] = evidence_record(
                pair_id, "EN_FR_PAIR", pair_subject, subtype
            )
            pair["etymology_evidence_ids_json"] = json.dumps([pair_id])
        audit = validate_review_tables(merge, tables, require_complete=True)
        self.assertEqual(audit["status"], "STRUCTURE_PASS_PENDING_EVIDENCE")
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["n_identifiable"], 60)
        self.assertEqual(audit["n_related_identifiable"], 12)
        self.assertEqual(audit["n_distinct_routes_identifiable"], 12)
        compiled = compile_filled_review_bundle(
            merge,
            tables,
            evidence_records,
            {"c" * 64},
            production=False,
        )
        self.assertEqual(compiled["status"], "PASS")
        self.assertFalse(compiled["training_eligible"])
        self.assertEqual(len(compiled["cohort_ids"]), 60)
        self.assertEqual(len(compiled["concept_reviews"]), 60)
        self.assertEqual(len(compiled["etymology_reviews"]), 60)
        self.assertIn("relation_subtype", compiled["etymology_reviews"][0])
        frozen_concept = compiled["validated"]["concepts"][0]
        self.assertEqual(set(frozen_concept["selected_term_quality"]), {"en", "fr"})
        self.assertEqual(
            frozen_concept["selected_term_quality"]["en"]["translation_quality"],
            "ATTESTED_SAME_SENSE",
        )
        self.assertEqual(
            compiled["concept_reviews"][0]["meaning_alignment_decision"], "ALIGNED"
        )
        frozen_etymology = compiled["validated"]["etymology"][0]
        self.assertEqual(frozen_etymology["relation_subtype"], "BORROWING_DIRECT")
        self.assertEqual(frozen_etymology["relation_direction"], "EN_TO_FR")
        with self.assertRaisesRegex(
            ContractViolation, "PRODUCTION_EVIDENCE_SCOPE_REQUIRED"
        ):
            compile_filled_review_bundle(
                merge,
                tables,
                evidence_records,
                {"c" * 64},
                production=True,
            )
        with self.assertRaisesRegex(
            ContractViolation, "UNBOUND_REVIEW_EVIDENCE_PAYLOAD"
        ):
            compile_filled_review_bundle(
                merge, tables, evidence_records, set(), production=False
            )

        indeterminate_high = mutable_tables(tables)
        indeterminate_high.pairs[24]["confidence"] = "HIGH"
        with self.assertRaisesRegex(
            ContractViolation, "INDETERMINATE_HIGH_CONFIDENCE_FORBIDDEN"
        ):
            validate_review_tables(merge, indeterminate_high, require_complete=True)

        conflict_tables = mutable_tables(tables)
        conflict_tables.pairs[0]["etymology_evidence_ids_json"] = json.dumps(
            ["pair-0", "pair-0-conflict"]
        )
        conflict_records = copy.deepcopy(evidence_records)
        pair_zero = conflict_records["pair-0"]
        pair_zero["conflicts_with"] = ["pair-0-conflict"]
        pair_zero["record_sha256"] = logical_hash(
            {key: value for key, value in pair_zero.items() if key != "record_sha256"}
        )
        conflict = evidence_record(
            "pair-0-conflict",
            "EN_FR_PAIR",
            pair_zero["subject_id"],
            "DISTINCT_ROUTES_REVIEWED",
        )
        conflict["conflicts_with"] = ["pair-0"]
        conflict["record_sha256"] = logical_hash(
            {key: value for key, value in conflict.items() if key != "record_sha256"}
        )
        conflict_records["pair-0-conflict"] = conflict

        mixed = compile_filled_review_bundle(
            merge,
            conflict_tables,
            conflict_records,
            {"c" * 64},
            production=False,
        )
        mixed_row = mixed["validated"]["etymology"][0]
        self.assertEqual(mixed_row["relation_subtype"], "BORROWING_DIRECT")
        self.assertEqual(mixed_row["relation"], "BORROWING_DOCUMENTED")
        self.assertEqual(
            {row["supports_label"] for row in mixed_row["evidence"]},
            {"BORROWING_DIRECT", "DISTINCT_ROUTES_REVIEWED"},
        )

        high_conflict_tables = mutable_tables(conflict_tables)
        high_conflict_tables.pairs[0]["confidence"] = "HIGH"
        with self.assertRaisesRegex(
            ContractViolation, "CONFLICTING_EVIDENCE_HIGH_CONFIDENCE_FORBIDDEN"
        ):
            compile_filled_review_bundle(
                merge,
                high_conflict_tables,
                conflict_records,
                {"c" * 64},
                production=False,
            )


if __name__ == "__main__":
    unittest.main()
