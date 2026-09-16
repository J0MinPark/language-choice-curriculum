import unittest

from implementation.src.contracts import ContractViolation
from implementation.src.review_import import (
    answer_pair_sha256,
    en_fr_pair_evidence_subject_sha256,
)


class PairEvidenceSubjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.subject = {
            "candidate_id": "candidate-1",
            "en_term_id": "candidate-1|en|1",
            "en_term_sha256": "a" * 64,
            "en_canonical_answer": "word",
            "fr_term_id": "candidate-1|fr|1",
            "fr_term_sha256": "b" * 64,
            "fr_canonical_answer": "mot",
        }

    def digest(self, **changes: str) -> str:
        values = {**self.subject, **changes}
        return en_fr_pair_evidence_subject_sha256(**values)

    def test_digest_is_deterministic_and_not_the_legacy_surface_hash(self) -> None:
        digest = self.digest()
        self.assertEqual(digest, self.digest())
        self.assertNotEqual(
            digest,
            answer_pair_sha256(
                self.subject["en_canonical_answer"],
                self.subject["fr_canonical_answer"],
            ),
        )

    def test_different_candidate_changes_digest(self) -> None:
        self.assertNotEqual(
            self.digest(),
            self.digest(
                candidate_id="candidate-2",
                en_term_id="candidate-2|en|1",
                fr_term_id="candidate-2|fr|1",
            ),
        )

    def test_different_selected_term_id_changes_digest(self) -> None:
        baseline = self.digest()
        for field, value in (
            ("en_term_id", "candidate-1|en|2"),
            ("fr_term_id", "candidate-1|fr|2"),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(baseline, self.digest(**{field: value}))

    def test_different_term_hash_changes_digest(self) -> None:
        baseline = self.digest()
        for field, value in (
            ("en_term_sha256", "c" * 64),
            ("fr_term_sha256", "d" * 64),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(baseline, self.digest(**{field: value}))

    def test_different_canonical_answer_changes_digest(self) -> None:
        baseline = self.digest()
        for field, value in (
            ("en_canonical_answer", "term"),
            ("fr_canonical_answer", "parole"),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(baseline, self.digest(**{field: value}))

    def test_invalid_candidate_term_prefix_is_rejected(self) -> None:
        for field, value in (
            ("en_term_id", "other-candidate|en|1"),
            ("fr_term_id", "candidate-1|en|1"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ContractViolation, "EVIDENCE_TERM_ID_CANDIDATE_MISMATCH"
                ):
                    self.digest(**{field: value})

    def test_invalid_term_hash_is_rejected(self) -> None:
        for field in ("en_term_sha256", "fr_term_sha256"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ContractViolation, "INVALID_EVIDENCE_TERM_SHA256"
                ):
                    self.digest(**{field: "not-a-sha256"})

    def test_noncanonical_answer_is_rejected(self) -> None:
        for field, value in (
            ("en_canonical_answer", " word"),
            ("fr_canonical_answer", "mot  composé"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ContractViolation, "NONCANONICAL_EVIDENCE_ANSWER"
                ):
                    self.digest(**{field: value})


if __name__ == "__main__":
    unittest.main()
