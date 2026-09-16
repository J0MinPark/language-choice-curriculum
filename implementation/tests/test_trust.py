from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from implementation.src.contracts import ContractViolation
from implementation.src.trust import (
    SUPPORTED_ARTIFACT_KINDS,
    load_trusted_artifact_registry,
    require_trusted_artifact_anchor,
)


MANIFEST_SHA256 = (
    "65e899adebb855fc330db6765d9531f24df7d36769d7cf9e5c2acb4f430dcbc5"
)
BINDINGS = {
    "implementation_revision": "4.0.0-r1",
    "collection_method": "OFFLINE_IMPORT_EXISTING_IMMUTABLE_RAW",
    "data_kind": "REAL_KRDICT_API",
    "endpoint": "https://krdict.korean.go.kr/api/search",
    "plan_sha256": (
        "2b00f62e1f0d594a3e075cbd765ef014c33259bc932f42420094eff290b00620"
    ),
    "query_source_sha256": (
        "745b8f8af0b5bd71862f030888c9b93c32a02388d70b37f609e463f8e769ddd1"
    ),
    "requests_planned": 597,
    "source_set_sha256": (
        "b100064abad6c8368bdc7f137b8ec887eff4e90482be47026995ab9c33e0bf69"
    ),
}


class TrustedArtifactAnchorTests(unittest.TestCase):
    def _require(self, **changes: object) -> dict[str, object]:
        values = {
            "artifact_kind": "KRDICT_SOURCE_COLLECTION",
            "artifact_id": "krdict-b100064abad6c836",
            "manifest_sha256": MANIFEST_SHA256,
            "bindings": copy.deepcopy(BINDINGS),
        }
        values.update(changes)
        return require_trusted_artifact_anchor(**values)

    def _write_registry(self, root: Path, value: object) -> Path:
        path = root / "registry.json"
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def test_exact_retained_krdict_anchor_passes(self) -> None:
        anchor = self._require()
        self.assertEqual(anchor["artifact_id"], "krdict-b100064abad6c836")
        self.assertEqual(anchor["manifest_sha256"], MANIFEST_SHA256)
        self.assertEqual(anchor["bindings"], BINDINGS)

    def test_registry_documents_git_authority_without_signature_claim(self) -> None:
        registry = load_trusted_artifact_registry()
        self.assertEqual(
            registry["trust_model"]["authority"],
            "REVIEWED_GIT_COMMIT_AND_VERIFIED_IMPLEMENTATION_INTEGRITY_MANIFEST",
        )
        self.assertIs(registry["trust_model"]["human_signature_claimed"], False)
        self.assertIs(registry["trust_model"]["keyed_signature_claimed"], False)
        self.assertEqual(
            registry["trust_model"]["unkeyed_hash_role"],
            "INTEGRITY_BINDING_ONLY_NOT_AUTHENTICATION",
        )
        self.assertEqual(len(registry["anchors"]), 2)
        replay_anchor = registry["anchors"][0]
        self.assertEqual(replay_anchor["artifact_id"], "semantic-experiment-2e91e64b9599f8507292")
        self.assertEqual(replay_anchor["bindings"]["allowed_execution_phases"], ["GPU_REPLAY"])
        self.assertFalse(replay_anchor["bindings"]["scientific_training_authorized"])
        self.assertEqual(
            SUPPORTED_ARTIFACT_KINDS,
            {
                "KRDICT_SOURCE_COLLECTION",
                "ANNOTATION_FREEZE",
                "EXPERIMENT_FREEZE",
            },
        )

    def test_any_identity_or_binding_change_fails_closed(self) -> None:
        changed_bindings = copy.deepcopy(BINDINGS)
        changed_bindings["requests_planned"] = 596
        extra_bindings = copy.deepcopy(BINDINGS)
        extra_bindings["runtime_override"] = True
        cases = (
            {"artifact_kind": "OTHER"},
            {"artifact_id": "krdict-untrusted"},
            {"manifest_sha256": "f" * 64},
            {"bindings": changed_bindings},
            {"bindings": extra_bindings},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ContractViolation):
                    self._require(**changes)

    def test_missing_supported_anchor_fails_closed(self) -> None:
        for artifact_kind in ("ANNOTATION_FREEZE", "EXPERIMENT_FREEZE"):
            with self.subTest(artifact_kind=artifact_kind):
                with self.assertRaisesRegex(
                    ContractViolation, "TRUSTED_ARTIFACT_ANCHOR_MISSING"
                ):
                    self._require(
                        artifact_kind=artifact_kind,
                        artifact_id="not-reviewed",
                    )

    def test_manifest_and_exact_binding_mismatches_are_distinct(self) -> None:
        with self.assertRaisesRegex(
            ContractViolation, "TRUSTED_ARTIFACT_MANIFEST_SHA256_MISMATCH"
        ):
            self._require(manifest_sha256="f" * 64)

        changed_type = copy.deepcopy(BINDINGS)
        changed_type["requests_planned"] = "597"
        with self.assertRaisesRegex(
            ContractViolation, "TRUSTED_ARTIFACT_BINDINGS_MISMATCH"
        ):
            self._require(bindings=changed_type)

        missing_field = copy.deepcopy(BINDINGS)
        del missing_field["endpoint"]
        with self.assertRaisesRegex(
            ContractViolation, "TRUSTED_ARTIFACT_BINDINGS_MISMATCH"
        ):
            self._require(bindings=missing_field)

    def test_malformed_registry_schema_revision_and_entries_are_rejected(self) -> None:
        base = load_trusted_artifact_registry()
        malformed: list[dict[str, object]] = []

        wrong_schema = copy.deepcopy(base)
        wrong_schema["schema_version"] = "trusted-artifact-anchors-v2"
        malformed.append(wrong_schema)

        wrong_revision = copy.deepcopy(base)
        wrong_revision["registry_revision"] = "2"
        malformed.append(wrong_revision)

        wrong_implementation = copy.deepcopy(base)
        wrong_implementation["implementation_revision"] = "4.0.0-r2"
        malformed.append(wrong_implementation)

        integer_signature_claim = copy.deepcopy(base)
        integer_signature_claim["trust_model"]["human_signature_claimed"] = 0
        malformed.append(integer_signature_claim)

        extra_registry_field = copy.deepcopy(base)
        extra_registry_field["self_sha256"] = "0" * 64
        malformed.append(extra_registry_field)

        invalid_entry = copy.deepcopy(base)
        invalid_entry["anchors"][0]["artifact_kind"] = "OTHER"
        malformed.append(invalid_entry)

        extra_entry_field = copy.deepcopy(base)
        extra_entry_field["anchors"][0]["approved_by"] = "human"
        malformed.append(extra_entry_field)

        empty_bindings = copy.deepcopy(base)
        empty_bindings["anchors"][0]["bindings"] = {}
        malformed.append(empty_bindings)

        duplicate_entry = copy.deepcopy(base)
        duplicate_entry["anchors"].append(
            copy.deepcopy(duplicate_entry["anchors"][0])
        )
        malformed.append(duplicate_entry)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ordinal, registry in enumerate(malformed):
                with self.subTest(ordinal=ordinal):
                    path = self._write_registry(root, registry)
                    with self.assertRaises(ContractViolation):
                        load_trusted_artifact_registry(path)

    def test_duplicate_json_object_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                '{"schema_version":"trusted-artifact-anchors-v1",'
                '"schema_version":"trusted-artifact-anchors-v1",'
                '"registry_revision":"1",'
                '"implementation_revision":"4.0.0-r1",'
                '"trust_model":{},"anchors":[]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ContractViolation, "DUPLICATE_TRUST_REGISTRY_KEY"
            ):
                load_trusted_artifact_registry(path)

    def test_parser_value_errors_are_normalized_to_contract_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                '{"schema_version":"trusted-artifact-anchors-v1",'
                f'"registry_revision":{("9" * 5000)},'
                '"implementation_revision":"4.0.0-r1",'
                '"trust_model":{},"anchors":[]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ContractViolation, "INVALID_TRUST_REGISTRY_JSON"
            ):
                load_trusted_artifact_registry(path)


if __name__ == "__main__":
    unittest.main()
