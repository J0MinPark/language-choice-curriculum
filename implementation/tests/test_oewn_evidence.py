from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from implementation.src import oewn_evidence as evidence
from implementation.src.contracts import ContractViolation


class OewnEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "download"
        self.source.mkdir()

        entries_lines = ["# filler"] * 9819 + list(evidence._EXPECTED_LEMMA_LINES)
        synset_lines = ["# filler"] * 18795 + list(
            evidence._EXPECTED_SYNSET_LINES
        )
        payloads = {
            "entries_e": ("\n".join(entries_lines) + "\n").encode(),
            "noun_food": ("\n".join(synset_lines) + "\n").encode(),
            "license": (
                "Creative Commons Attribution 4.0 International License\n"
                "Princeton WordNet\n"
            ).encode(),
            "wndb_license": (
                "WordNet 3.1 Copyright 2011 by Princeton University\n"
            ).encode(),
        }
        specs = {}
        self.source_files: dict[str, Path] = {}
        for name, original in evidence.SOURCE_SPECS.items():
            raw = payloads[name]
            path = self.source / str(original["filename"])
            path.write_bytes(raw)
            self.source_files[name] = path
            specs[name] = {
                **original,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        patcher = mock.patch.object(evidence, "SOURCE_SPECS", specs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _publish(self, name: str = "bundle") -> tuple[Path, dict]:
        output = self.root / name
        result = evidence.publish_oewn_evidence_bundle(
            self.source_files,
            output,
            captured_at_utc="2026-09-15T10:50:00Z",
            scope_root=self.root,
        )
        return output, result

    def test_publish_and_audit_exact_read_only_bundle(self) -> None:
        output, result = self._publish()
        self.assertEqual(result["status"], evidence.AUDIT_STATUS)
        self.assertEqual(result["registered_expression"], "egg")
        self.assertEqual(result["source_artifact_count"], 4)
        self.assertFalse(result["identity_authentication_claimed"])
        self.assertFalse(result["training_eligible"])
        self.assertEqual(os.stat(output).st_mode & 0o222, 0)
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {
                evidence.MANIFEST_FILENAME,
                "entries-e.yaml",
                "noun.food.yaml",
                "LICENSE.md",
                "WNDB_License.txt",
            },
        )
        audited = evidence.audit_oewn_evidence_bundle(
            output / evidence.MANIFEST_FILENAME, scope_root=self.root
        )
        self.assertEqual(audited, result)

    def test_source_hash_mismatch_and_existing_output_are_rejected(self) -> None:
        damaged = self.source_files["entries_e"]
        raw = bytearray(damaged.read_bytes())
        raw[0] ^= 1
        damaged.write_bytes(raw)
        with self.assertRaisesRegex(
            ContractViolation, "OEWN_EVIDENCE_SOURCE_HASH_MISMATCH"
        ):
            evidence.publish_oewn_evidence_bundle(
                self.source_files,
                self.root / "damaged",
                captured_at_utc="2026-09-15T10:50:00Z",
                scope_root=self.root,
            )

        # Restore the fixture before exercising write-once publication.
        raw[0] ^= 1
        damaged.write_bytes(raw)
        output, _result = self._publish("once")
        with self.assertRaisesRegex(
            ContractViolation, "OEWN_EVIDENCE_OUTPUT_EXISTS"
        ):
            evidence.publish_oewn_evidence_bundle(
                self.source_files,
                output,
                captured_at_utc="2026-09-15T10:50:00Z",
                scope_root=self.root,
            )

    def test_published_payload_tamper_and_extra_file_are_rejected(self) -> None:
        output, _result = self._publish("tamper")
        output.chmod(0o755)
        payload = output / "entries-e.yaml"
        payload.chmod(0o644)
        raw = bytearray(payload.read_bytes())
        raw[0] ^= 1
        payload.write_bytes(raw)
        payload.chmod(0o444)
        output.chmod(0o555)
        with self.assertRaisesRegex(
            ContractViolation, "OEWN_EVIDENCE_SOURCE_HASH_MISMATCH"
        ):
            evidence.audit_oewn_evidence_bundle(
                output / evidence.MANIFEST_FILENAME, scope_root=self.root
            )

        other, _result = self._publish("extra")
        other.chmod(0o755)
        extra = other / "unexpected"
        extra.write_text("x", encoding="utf-8")
        extra.chmod(0o444)
        other.chmod(0o555)
        with self.assertRaisesRegex(
            ContractViolation, "OEWN_EVIDENCE_READ_ONLY_BUNDLE_REQUIRED"
        ):
            evidence.audit_oewn_evidence_bundle(
                other / evidence.MANIFEST_FILENAME, scope_root=self.root
            )

    def test_manifest_is_self_hashed_and_strict(self) -> None:
        output, _result = self._publish("manifest")
        manifest_path = output / evidence.MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evidence_scope"] = "SOMETHING_ELSE"
        with self.assertRaisesRegex(
            ContractViolation, "OEWN_EVIDENCE_MANIFEST_POLICY_MISMATCH"
        ):
            evidence.validate_oewn_evidence_manifest(
                manifest, manifest_path=manifest_path, scope_root=self.root
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["unexpected"] = True
        with self.assertRaisesRegex(
            ContractViolation, "OEWN_EVIDENCE_MANIFEST_SCHEMA_MISMATCH"
        ):
            evidence.validate_oewn_evidence_manifest(
                manifest, manifest_path=manifest_path, scope_root=self.root
            )

    def test_invalid_or_future_capture_time_is_rejected(self) -> None:
        for timestamp, code in (
            ("not-a-time", "OEWN_EVIDENCE_INVALID_CAPTURE_TIME"),
            ("2999-01-01T00:00:00Z", "OEWN_EVIDENCE_CAPTURE_TIME_IN_FUTURE"),
        ):
            with self.subTest(timestamp=timestamp):
                with self.assertRaisesRegex(ContractViolation, code):
                    evidence.build_oewn_evidence_manifest(
                        self.root / "future", captured_at_utc=timestamp
                    )


if __name__ == "__main__":
    unittest.main()
