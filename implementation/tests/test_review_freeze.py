from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from implementation.src import review_freeze
from implementation.src.contracts import ContractViolation, canonical_json_bytes
from implementation.src.review_import import (
    answer_pair_sha256,
    en_fr_pair_evidence_subject_sha256,
)


def logical_hash(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def paths(root: Path) -> tuple[Path, Path, Path, Path]:
    values = tuple(
        root / name for name in ("source.json", "terms.csv", "pairs.csv", "screen.csv")
    )
    for path in values:
        path.write_text("fixture\n", encoding="utf-8")
    return values


def evidence_record(
    evidence_id: str,
    kind: str,
    subject: str,
    label: str,
    payload: Path,
    root: Path,
) -> dict:
    core = {
        "evidence_id": evidence_id,
        "subject_kind": kind,
        "subject_id": subject,
        "supports_label": label,
        "source_url": f"https://example.invalid/{evidence_id}",
        "evidence_origin": "human-reviewed-dictionary-record",
        "source_name": "Synthetic Evidence Dictionary",
        "source_version": "fixture-v1",
        "source_license": "Synthetic fixture license",
        "source_license_url": "https://example.invalid/license",
        "sense_locator": f"sense-{evidence_id}",
        "payload_path": str(payload.relative_to(root)),
        "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "payload_bytes": payload.stat().st_size,
        "retrieved_at": "2026-01-01T00:00:00Z",
        "conflicts_with": [],
    }
    return {**core, "record_sha256": logical_hash(core)}


def table_and_evidence(root: Path):
    payload = root / "evidence.txt"
    payload.write_bytes(b"verified evidence payload\n")
    en_id, fr_id = "candidate|en|1", "candidate|fr|1"
    en = {
        "term_id": en_id,
        "term_sha256": "1" * 64,
        "term_canonical": "word",
        "quality_evidence_ids_json": '["quality-en"]',
    }
    fr = {
        "term_id": fr_id,
        "term_sha256": "2" * 64,
        "term_canonical": "mot",
        "quality_evidence_ids_json": '["quality-fr"]',
    }
    pair = {
        "candidate_id": "candidate",
        "pilot_selected": "TRUE",
        "selected_en_term_id": en_id,
        "selected_fr_term_id": fr_id,
        "etymology_subtype": "BORROWING_DIRECT",
        "etymology_evidence_ids_json": '["pair-evidence"]',
    }
    records = [
        evidence_record(
            "quality-en", "TERM", en_id, "ATTESTED_SAME_SENSE", payload, root
        ),
        evidence_record(
            "quality-fr", "TERM", fr_id, "ATTESTED_SAME_SENSE", payload, root
        ),
        evidence_record(
            "pair-evidence",
            "EN_FR_PAIR",
            en_fr_pair_evidence_subject_sha256(
                candidate_id="candidate",
                en_term_id=en_id,
                en_term_sha256=en["term_sha256"],
                en_canonical_answer=en["term_canonical"],
                fr_term_id=fr_id,
                fr_term_sha256=fr["term_sha256"],
                fr_canonical_answer=fr["term_canonical"],
            ),
            "BORROWING_DIRECT",
            payload,
            root,
        ),
    ]
    return SimpleNamespace(terms=(en, fr), pairs=(pair,)), records


def compiled_for(records):
    return {
        "status": "PASS",
        "concept_reviews": [{"candidate_id": "c1"}, {"candidate_id": "c2"}],
        "etymology_reviews": [{"candidate_id": "c1"}, {"candidate_id": "c2"}],
        "cohort_ids": ["c1", "c2"],
        "evidence_records": records,
        "evidence_record_hashes": {row["record_sha256"] for row in records},
        "evidence_payload_hashes": {row["payload_sha256"] for row in records},
        "validated": {"status": "PASS"},
    }


class ReviewFreezeBoundaryTests(unittest.TestCase):
    def test_actual_payload_is_hashed_and_full_records_reach_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, terms, pairs, screen = paths(root)
            tables, records = table_and_evidence(root)
            compiler = mock.Mock(return_value=compiled_for(records))
            freezer = mock.Mock(return_value={"status": "PASS", "freeze_id": "fixture"})
            with mock.patch.object(
                review_freeze, "merge_krdict_snapshot", return_value={}
            ), mock.patch.object(
                review_freeze,
                "_load_review_import_api",
                return_value=(lambda *_args: tables, compiler),
            ), mock.patch.object(
                review_freeze, "freeze_reviewed_dataset", freezer
            ):
                result = review_freeze.freeze_completed_review_csv_bundle(
                    source,
                    terms,
                    pairs,
                    screen,
                    evidence_records=records,
                    output_dir=root / "freeze",
                    requirements={"fixture": True},
                    production=False,
                    scope_root=root,
                )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(
                compiler.call_args.kwargs["evidence_payload_hashes"],
                {row["payload_sha256"] for row in records},
            )
            self.assertEqual(
                freezer.call_args.kwargs["evidence_record_hashes"],
                {row["record_sha256"] for row in records},
            )
            self.assertEqual(
                {row["evidence_id"] for row in freezer.call_args.kwargs["evidence_records"]},
                {"quality-en", "quality-fr", "pair-evidence"},
            )

    def test_caller_hash_assertion_does_not_replace_file_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, terms, pairs, screen = paths(root)
            tables, records = table_and_evidence(root)
            (root / "evidence.txt").write_bytes(b"tampered\n")
            with mock.patch.object(
                review_freeze, "merge_krdict_snapshot", return_value={}
            ), mock.patch.object(
                review_freeze,
                "_load_review_import_api",
                return_value=(lambda *_args: tables, mock.Mock()),
            ):
                with self.assertRaisesRegex(
                    ContractViolation, "EVIDENCE_PAYLOAD_(SIZE|SHA256)_MISMATCH"
                ):
                    review_freeze.freeze_completed_review_csv_bundle(
                        source,
                        terms,
                        pairs,
                        screen,
                        evidence_records=records,
                        evidence_payload_hashes={records[0]["payload_sha256"]},
                        output_dir=root / "freeze",
                        requirements={"fixture": True},
                        production=False,
                        scope_root=root,
                    )

    def test_record_hash_and_selected_pair_binding_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths(root)
            tables, records = table_and_evidence(root)
            broken_hash = [dict(row) for row in records]
            broken_hash[0]["sense_locator"] = "changed-after-hash"
            with self.assertRaisesRegex(
                ContractViolation, "EVIDENCE_RECORD_HASH_MISMATCH"
            ):
                review_freeze._index_and_bind_evidence(
                    broken_hash,
                    root,
                    review_freeze._referenced_evidence_bindings(tables),
                    production=True,
                )

            wrong_pair = [dict(row) for row in records]
            wrong_pair[2]["subject_id"] = answer_pair_sha256("word", "mot")
            core = {k: v for k, v in wrong_pair[2].items() if k != "record_sha256"}
            wrong_pair[2]["record_sha256"] = logical_hash(core)
            with self.assertRaisesRegex(
                ContractViolation, "EVIDENCE_SUBJECT_OR_LABEL_MISMATCH"
            ):
                review_freeze._index_and_bind_evidence(
                    wrong_pair,
                    root,
                    review_freeze._referenced_evidence_bindings(tables),
                    production=True,
                )

    def test_one_evidence_id_cannot_be_reused_for_two_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths(root)
            tables, _records = table_and_evidence(root)
            reused_terms = [dict(row) for row in tables.terms]
            reused_terms[1]["quality_evidence_ids_json"] = '["quality-en"]'
            reused = SimpleNamespace(terms=tuple(reused_terms), pairs=tables.pairs)
            with self.assertRaisesRegex(
                ContractViolation, "EVIDENCE_REUSED_ACROSS_SUBJECTS"
            ):
                review_freeze._referenced_evidence_bindings(reused)

    def test_pair_binding_preserves_a_symmetric_alternative_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths(root)
            tables, records = table_and_evidence(root)
            pair = dict(tables.pairs[0])
            pair["etymology_evidence_ids_json"] = (
                '["pair-evidence","pair-alternative"]'
            )
            selected = records[2]
            selected["conflicts_with"] = ["pair-alternative"]
            selected["record_sha256"] = logical_hash(
                {key: value for key, value in selected.items() if key != "record_sha256"}
            )
            payload = root / "evidence.txt"
            alternative = evidence_record(
                "pair-alternative",
                "EN_FR_PAIR",
                selected["subject_id"],
                "DISTINCT_ROUTES_REVIEWED",
                payload,
                root,
            )
            alternative["conflicts_with"] = ["pair-evidence"]
            alternative["record_sha256"] = logical_hash(
                {
                    key: value
                    for key, value in alternative.items()
                    if key != "record_sha256"
                }
            )
            records.append(alternative)
            mixed_tables = SimpleNamespace(terms=tables.terms, pairs=(pair,))

            indexed, _payloads, _records = review_freeze._index_and_bind_evidence(
                records,
                root,
                review_freeze._referenced_evidence_bindings(mixed_tables),
                production=True,
            )
            self.assertEqual(
                indexed["pair-alternative"]["supports_label"],
                "DISTINCT_ROUTES_REVIEWED",
            )


if __name__ == "__main__":
    unittest.main()
