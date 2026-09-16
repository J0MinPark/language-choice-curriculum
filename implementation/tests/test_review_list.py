from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from implementation.src.artifacts import publish_bytes_once, publish_json_once
from implementation.src.contracts import ContractViolation
from implementation.src.review_list import export_etymology_review_list


def option(language: str, answer: str) -> dict:
    return {
        "answer": answer,
        "definition_raw": language + " definition",
        "option_id": hashlib.sha256((language + answer).encode()).hexdigest(),
        "source_refs": [{"raw_sha256": "f" * 64}],
    }


class ReviewListTests(unittest.TestCase):
    def test_exports_nonapproving_four_language_worksheet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            task = {
                "task_type": "CONCEPT_AND_EN_FR_ETYMOLOGY",
                "candidate_id": "snapshot:1:1",
                "candidate_sha256": "b" * 64,
                "status": "PENDING",
                "answer_options": {
                    "ko": [option("ko", "시험")],
                    "en": [option("en", "test")],
                    "zh": [option("zh", "测试")],
                    "fr": [option("fr", "essai")],
                },
            }
            pending = root / "pending.jsonl"
            publish_bytes_once(
                pending,
                (json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n").encode(),
            )
            summary = root / "summary.json"
            publish_json_once(
                summary,
                {
                    "schema_version": "4.0.0",
                    "status": "PENDING_HUMAN_REVIEW",
                    "data_kind": "REAL_KRDICT_API",
                    "summary": {
                        "pending": 1,
                        "automatically_approved": 0,
                        "human_signoff_required": True,
                    },
                },
            )
            result = export_etymology_review_list(
                pending,
                summary,
                output_dir=root,
                scope_root=root,
            )
            self.assertEqual(result["row_count"], 1)
            self.assertFalse(result["training_eligible"])
            rows = list(
                csv.DictReader(
                    io.StringIO((root / "etymology_review_list.csv").read_text())
                )
            )
            self.assertEqual(rows[0]["en_expression"], "test")
            self.assertEqual(rows[0]["fr_expression"], "essai")
            self.assertEqual(rows[0]["en_fr_relation"], "")
            self.assertEqual(rows[0]["reviewer"], "")

    def test_count_mismatch_and_multiple_options_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            task = {
                "task_type": "CONCEPT_AND_EN_FR_ETYMOLOGY",
                "candidate_id": "snapshot:1:1",
                "candidate_sha256": "b" * 64,
                "status": "PENDING",
                "answer_options": {
                    "ko": [option("ko", "시험")],
                    "en": [option("en", "test"), option("en", "exam")],
                    "zh": [option("zh", "测试")],
                    "fr": [option("fr", "essai")],
                },
            }
            pending = root / "pending.jsonl"
            publish_bytes_once(pending, (json.dumps(task) + "\n").encode())
            summary = root / "summary.json"
            publish_json_once(
                summary,
                {
                    "schema_version": "4.0.0",
                    "status": "PENDING_HUMAN_REVIEW",
                    "data_kind": "REAL_KRDICT_API",
                    "summary": {
                        "pending": 1,
                        "automatically_approved": 0,
                        "human_signoff_required": True,
                    },
                },
            )
            with self.assertRaisesRegex(
                ContractViolation, "WORKSHEET_REQUIRES_ONE_OPTION_PER_LANGUAGE"
            ):
                export_etymology_review_list(
                    pending,
                    summary,
                    output_dir=root,
                    scope_root=root,
                )


if __name__ == "__main__":
    unittest.main()
