from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from implementation.src.contracts import ContractViolation
from implementation.src.design_revision import (
    AMENDMENT_RELATIVE_PATH,
    EXPECTED_AMENDMENT_BYTES,
    EXPECTED_AMENDMENT_SHA256,
    _load_json_bytes,
    audit_v4_1_design_amendment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DesignRevisionTests(unittest.TestCase):
    def _copy_bundle(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        for relative in (
            "spec/RESEARCH_SPEC_V4_KO.md",
            "spec/pilot.json",
            "implementation/protocol/RESEARCH_SPEC_V4_1_KO.md",
            "implementation/config/pilot_v4_1.json",
            AMENDMENT_RELATIVE_PATH,
        ):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / relative, destination)
        return temporary, root

    def test_live_bundle_is_exact_and_non_executable(self):
        result = audit_v4_1_design_amendment()
        self.assertEqual(result["status"], "PASS_V4_1_PROSPECTIVE_DESIGN_AMENDMENT")
        self.assertEqual(result["amendment_sha256"], EXPECTED_AMENDMENT_SHA256)
        self.assertEqual(result["amendment_bytes"], EXPECTED_AMENDMENT_BYTES)
        self.assertFalse(result["core_etymology_required"])
        self.assertEqual(result["ignored_core_etymology_deferrals"], 8)
        self.assertEqual(
            result["current_core_term_blockers"][0]["selection_ordinal"], 38
        )
        self.assertEqual(result["prediction_status"], "NOT_RUN_PREDICTION_BY_DESIGN")
        self.assertFalse(result["implementation_ready"])
        self.assertFalse(result["training_eligible"])
        self.assertFalse(result["legacy_freeze_reuse_allowed"])

    def test_base_or_revised_artifact_tamper_is_rejected(self):
        for relative, expected_code in (
            ("spec/pilot.json", "DESIGN_REVISION_ARTIFACT_SIZE_MISMATCH"),
            (
                "implementation/protocol/RESEARCH_SPEC_V4_1_KO.md",
                "DESIGN_REVISION_ARTIFACT_SIZE_MISMATCH",
            ),
            (
                "implementation/config/pilot_v4_1.json",
                "DESIGN_REVISION_ARTIFACT_SIZE_MISMATCH",
            ),
        ):
            with self.subTest(relative=relative):
                _temporary, root = self._copy_bundle()
                with (root / relative).open("ab") as handle:
                    handle.write(b"x")
                with self.assertRaisesRegex(ContractViolation, expected_code):
                    audit_v4_1_design_amendment(project_root=root)

    def test_same_size_artifact_tamper_is_rejected_by_hash(self):
        _temporary, root = self._copy_bundle()
        path = root / "implementation/config/pilot_v4_1.json"
        raw = bytearray(path.read_bytes())
        index = raw.index(b"4.1.0")
        raw[index] = ord("5")
        path.write_bytes(raw)
        with self.assertRaisesRegex(
            ContractViolation, "DESIGN_REVISION_ARTIFACT_HASH_MISMATCH"
        ):
            audit_v4_1_design_amendment(project_root=root)

    def test_amendment_tamper_or_alias_path_is_rejected(self):
        _temporary, root = self._copy_bundle()
        amendment = root / AMENDMENT_RELATIVE_PATH
        raw = bytearray(amendment.read_bytes())
        raw[-2] = 0x20 if raw[-2] != 0x20 else 0x09
        amendment.write_bytes(raw)
        with self.assertRaisesRegex(
            ContractViolation, "DESIGN_REVISION_AMENDMENT_HASH_MISMATCH"
        ):
            audit_v4_1_design_amendment(project_root=root)

        _temporary, root = self._copy_bundle()
        alias = root / "implementation/protocol/alias.json"
        alias.write_bytes((root / AMENDMENT_RELATIVE_PATH).read_bytes())
        with self.assertRaisesRegex(
            ContractViolation, "DESIGN_REVISION_AMENDMENT_PATH_MISMATCH"
        ):
            audit_v4_1_design_amendment(alias, project_root=root)

        _temporary, root = self._copy_bundle()
        amendment = root / AMENDMENT_RELATIVE_PATH
        saved = root / "saved-amendment.json"
        amendment.rename(saved)
        amendment.symlink_to(saved)
        with self.assertRaisesRegex(
            ContractViolation, "DESIGN_REVISION_AMENDMENT_NOT_REGULAR_FILE"
        ):
            audit_v4_1_design_amendment(project_root=root)

    def test_symlinked_bound_artifact_is_rejected(self):
        _temporary, root = self._copy_bundle()
        config = root / "implementation/config/pilot_v4_1.json"
        saved = root / "saved-config.json"
        config.rename(saved)
        config.symlink_to(saved)
        with self.assertRaisesRegex(
            ContractViolation, "DESIGN_REVISION_ARTIFACT_NOT_REGULAR_FILE"
        ):
            audit_v4_1_design_amendment(project_root=root)

    def test_json_parser_rejects_duplicate_keys_nonfinite_and_nonobject(self):
        for raw, code in (
            (b'{"a":1,"a":2}', "DESIGN_REVISION_DUPLICATE_JSON_KEY"),
            (b'{"a":NaN}', "DESIGN_REVISION_NONFINITE_JSON_NUMBER"),
            (json.dumps([1, 2]).encode(), "TEST_INVALID_JSON"),
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ContractViolation, code):
                    _load_json_bytes(raw, "TEST_INVALID_JSON")

    def test_core_config_has_no_etymology_quota_and_does_not_claim_execution(self):
        config = json.loads(
            (PROJECT_ROOT / "implementation/config/pilot_v4_1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("min_related", config["data"])
        self.assertNotIn("min_distinct_routes", config["data"])
        self.assertFalse(config["data"]["etymology_required"])
        self.assertFalse(config["prediction"]["enabled_for_current_pilot"])
        self.assertFalse(
            config["exploratory"]["documented_lexical_relation"][
                "enabled_for_current_pilot"
            ]
        )
        self.assertEqual(config["design_status"], "APPROVED_DESIGN_PENDING_IMPLEMENTATION")
        self.assertFalse(config["policy"]["main_enabled"])


if __name__ == "__main__":
    unittest.main()
