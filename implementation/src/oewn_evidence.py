"""Immutable Open English WordNet evidence for the ordinal-38 correction.

This module captures four files from one commit-pinned Open English WordNet
release into an atomic, read-only directory.  Its deliberately narrow claim is
that ``egg`` is a standard English noun lemma for the reviewed food sense.  It
does not claim that the previously selected phrase is ungrammatical, make an
etymology claim, authenticate a reviewer, or authorize training.
"""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    publish_bytes_once,
    publish_verified_file_once,
    read_regular_file_bytes_exact,
)
from .contracts import (
    WORK_ROOT,
    ContractViolation,
    canonical_json_bytes,
    require_relative_to,
    require_sha256,
    sha256_bytes,
)


SCHEMA_VERSION = "oewn-registered-expression-evidence-v1"
STATUS = "CAPTURED_IMMUTABLE_SOURCE_EVIDENCE"
EVIDENCE_SCOPE = "STANDARD_LEMMA_FOR_REVIEWED_SENSE"
AUDIT_STATUS = "PASS_OEWN_REGISTERED_EXPRESSION_EVIDENCE"
MANIFEST_FILENAME = "oewn_evidence_manifest.json"
RELEASE_TAG = "2025-edition"
GIT_COMMIT = "dc343f2683279ecbb13fab4e2fd778d7b162d287"
REPOSITORY_URL = "https://github.com/globalwordnet/english-wordnet"
RELEASE_URL = f"{REPOSITORY_URL}/releases/tag/{RELEASE_TAG}"

SOURCE_SPECS = {
    "entries_e": {
        "filename": "entries-e.yaml",
        "relative_url": "src/yaml/entries-e.yaml",
        "sha256": "a22fd769bcd0e25578bfcef2c49c9166f151d2b94c5e3c9147d68b48bd06dce7",
        "bytes": 888886,
    },
    "noun_food": {
        "filename": "noun.food.yaml",
        "relative_url": "src/yaml/noun.food.yaml",
        "sha256": "280089ff57ea13e53581c10e10812f4c08afda4639023e65f4e84770ae75e6c7",
        "bytes": 503122,
    },
    "license": {
        "filename": "LICENSE.md",
        "relative_url": "LICENSE.md",
        "sha256": "672cc8b5663e8dc74c4b07a9dcf477193853575b119908fd3dc0aeeb60a9dbbb",
        "bytes": 19863,
    },
    "wndb_license": {
        "filename": "WNDB_License.txt",
        "relative_url": "WNDB_License.txt",
        "sha256": "df30ec18fbabcdaf031b79ea026d3e6b959010cffe6dd7be9ac137822175b904",
        "bytes": 1740,
    },
}
RAW_FILE_URLS = {
    name: (
        "https://raw.githubusercontent.com/globalwordnet/english-wordnet/"
        f"{GIT_COMMIT}/{spec['relative_url']}"
    )
    for name, spec in SOURCE_SPECS.items()
}
LIMITATIONS = [
    "EVIDENCE_SUPPORTS_EGG_AS_A_STANDARD_ENGLISH_LEMMA_FOR_THE_REVIEWED_FOOD_SENSE_ONLY",
    "EVIDENCE_DOES_NOT_ESTABLISH_THAT_THE_IMMUTABLE_KRDICT_SOURCE_PHRASE_IS_INCORRECT",
    "NO_ETYMOLOGY_RELATIONSHIP_IS_ASSERTED_OR_REQUIRED",
    "REVIEWER_IDENTITY_IS_NOT_AUTHENTICATED",
    "EVIDENCE_DOES_NOT_AUTHORIZE_TRAINING",
]

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "captured_at_utc",
        "evidence_scope",
        "source",
        "source_artifacts",
        "locators",
        "licensing",
        "limitations",
        "identity_authentication_claimed",
        "training_eligible",
        "manifest_sha256",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "resource_name",
        "edition",
        "release_tag",
        "git_commit",
        "repository_url",
        "release_url",
        "raw_file_urls",
    }
)
_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256", "bytes"})
_LOCATOR_FIELDS = frozenset({"lemma_entry", "food_synset"})
_LEMMA_LOCATOR_FIELDS = frozenset(
    {
        "artifact",
        "line_start",
        "line_end",
        "line_numbering",
        "lemma",
        "part_of_speech",
        "sense_id",
        "synset_id",
    }
)
_SYNSET_LOCATOR_FIELDS = frozenset(
    {
        "artifact",
        "line_start",
        "line_end",
        "line_numbering",
        "synset_id",
        "part_of_speech",
        "members",
        "sense_scope",
    }
)
_LICENSING_FIELDS = frozenset(
    {
        "resource_license",
        "underlying_data_license",
        "license_artifact",
        "underlying_license_artifact",
        "attribution_required",
    }
)
_CAPTURED_AT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

_EXPECTED_LEMMA_LINES = (
    "egg:",
    "  n:",
    "    sense:",
    "    - id: 'egg%1:05:00::'",
    "      synset: 01463098-n",
    "    - derivation:",
    "      - 'egg%2:35:00::'",
    "      - 'egg%2:35:01::'",
    "      id: 'egg%1:13:00::'",
    "      synset: 07856780-n",
    "    - id: 'egg%1:08:00::'",
)
_EXPECTED_SYNSET_LINES = (
    "07856780-n:",
    "  definition:",
    "  - oval reproductive body of a fowl (especially a hen) used as food",
    "  hypernym:",
    "  - 07581905-n",
    "  ili: i78362",
    "  members:",
    "  - egg",
    "  - eggs",
    "  mero_part:",
    "  - 07857013-n",
    "  - 07857321-n",
    "  - 09455334-n",
    "  mero_substance:",
    "  - 14752903-n",
    "  partOfSpeech: n",
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation("OEWN_EVIDENCE_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_token: str) -> None:
    raise ContractViolation("OEWN_EVIDENCE_NONFINITE_JSON_NUMBER")


def _loads_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ContractViolation:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ContractViolation("OEWN_EVIDENCE_INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise ContractViolation("OEWN_EVIDENCE_INVALID_JSON")
    return value


def _exact_mapping(
    value: Any, fields: frozenset[str], code: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ContractViolation(code)
    return value


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or _CAPTURED_AT_RE.fullmatch(value) is None:
        raise ContractViolation("OEWN_EVIDENCE_INVALID_CAPTURE_TIME")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ContractViolation("OEWN_EVIDENCE_INVALID_CAPTURE_TIME") from exc
    if parsed > datetime.now(timezone.utc):
        raise ContractViolation("OEWN_EVIDENCE_CAPTURE_TIME_IN_FUTURE")
    return value


def _read_expected_source(path: Path, name: str) -> bytes:
    spec = SOURCE_SPECS[name]
    raw = read_regular_file_bytes_exact(path, expected_bytes=int(spec["bytes"]))
    if sha256_bytes(raw) != spec["sha256"]:
        raise ContractViolation("OEWN_EVIDENCE_SOURCE_HASH_MISMATCH")
    return raw


def _validate_source_semantics(raw_by_name: Mapping[str, bytes]) -> None:
    try:
        lemma_lines = raw_by_name["entries_e"].decode("utf-8").splitlines()
        synset_lines = raw_by_name["noun_food"].decode("utf-8").splitlines()
        license_text = raw_by_name["license"].decode("utf-8")
        underlying_text = raw_by_name["wndb_license"].decode("utf-8")
    except (UnicodeDecodeError, KeyError) as exc:
        raise ContractViolation("OEWN_EVIDENCE_SOURCE_SEMANTICS_MISMATCH") from exc
    if tuple(lemma_lines[9819:9830]) != _EXPECTED_LEMMA_LINES:
        raise ContractViolation("OEWN_EVIDENCE_LEMMA_LOCATOR_MISMATCH")
    if tuple(synset_lines[18795:18811]) != _EXPECTED_SYNSET_LINES:
        raise ContractViolation("OEWN_EVIDENCE_SYNSET_LOCATOR_MISMATCH")
    if (
        "Creative Commons Attribution 4.0 International License"
        not in license_text
        or "Princeton WordNet" not in license_text
        or "WordNet 3.1 Copyright 2011 by Princeton University"
        not in underlying_text
    ):
        raise ContractViolation("OEWN_EVIDENCE_LICENSE_MISMATCH")


def _artifact_ref(path: Path, name: str) -> dict[str, Any]:
    spec = SOURCE_SPECS[name]
    return {
        "path": str(path),
        "sha256": spec["sha256"],
        "bytes": spec["bytes"],
    }


def _manifest_core(
    destination: Path,
    *,
    captured_at_utc: str,
) -> dict[str, Any]:
    source_artifacts = {
        name: _artifact_ref(destination / spec["filename"], name)
        for name, spec in SOURCE_SPECS.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "captured_at_utc": captured_at_utc,
        "evidence_scope": EVIDENCE_SCOPE,
        "source": {
            "resource_name": "Open English WordNet",
            "edition": "2025",
            "release_tag": RELEASE_TAG,
            "git_commit": GIT_COMMIT,
            "repository_url": REPOSITORY_URL,
            "release_url": RELEASE_URL,
            "raw_file_urls": dict(RAW_FILE_URLS),
        },
        "source_artifacts": source_artifacts,
        "locators": {
            "lemma_entry": {
                "artifact": "entries_e",
                "line_start": 9820,
                "line_end": 9830,
                "line_numbering": "ONE_BASED_INCLUSIVE",
                "lemma": "egg",
                "part_of_speech": "n",
                "sense_id": "egg%1:13:00::",
                "synset_id": "07856780-n",
            },
            "food_synset": {
                "artifact": "noun_food",
                "line_start": 18796,
                "line_end": 18811,
                "line_numbering": "ONE_BASED_INCLUSIVE",
                "synset_id": "07856780-n",
                "part_of_speech": "n",
                "members": ["egg", "eggs"],
                "sense_scope": "FOOD_EGG_OF_FOWL_ESPECIALLY_HEN",
            },
        },
        "licensing": {
            "resource_license": "CC BY 4.0",
            "underlying_data_license": "Princeton WordNet License",
            "license_artifact": "license",
            "underlying_license_artifact": "wndb_license",
            "attribution_required": [
                "Princeton WordNet",
                "Open English WordNet team",
            ],
        },
        "limitations": list(LIMITATIONS),
        "identity_authentication_claimed": False,
        "training_eligible": False,
    }


def build_oewn_evidence_manifest(
    destination: Path,
    *,
    captured_at_utc: str,
) -> dict[str, Any]:
    """Build the exact self-hashed manifest for a final bundle location."""
    captured = _validate_timestamp(captured_at_utc)
    core = _manifest_core(Path(destination), captured_at_utc=captured)
    return {**core, "manifest_sha256": sha256_bytes(canonical_json_bytes(core))}


def validate_oewn_evidence_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Validate manifest shape, fixed metadata, payloads, and exact locators."""
    value = _exact_mapping(
        manifest, _TOP_LEVEL_FIELDS, "OEWN_EVIDENCE_MANIFEST_SCHEMA_MISMATCH"
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != STATUS
        or value.get("evidence_scope") != EVIDENCE_SCOPE
        or value.get("identity_authentication_claimed") is not False
        or value.get("training_eligible") is not False
    ):
        raise ContractViolation("OEWN_EVIDENCE_MANIFEST_POLICY_MISMATCH")
    _validate_timestamp(value.get("captured_at_utc"))
    core = {key: item for key, item in value.items() if key != "manifest_sha256"}
    claimed = value.get("manifest_sha256")
    if not isinstance(claimed, str):
        raise ContractViolation("OEWN_EVIDENCE_INVALID_MANIFEST_HASH")
    require_sha256(claimed, "OEWN_EVIDENCE_INVALID_MANIFEST_HASH")
    if sha256_bytes(canonical_json_bytes(core)) != claimed:
        raise ContractViolation("OEWN_EVIDENCE_MANIFEST_HASH_MISMATCH")

    source = _exact_mapping(
        value.get("source"), _SOURCE_FIELDS, "OEWN_EVIDENCE_SOURCE_SCHEMA_MISMATCH"
    )
    expected_source = _manifest_core(
        manifest_path.parent,
        captured_at_utc=str(value["captured_at_utc"]),
    )["source"]
    if source != expected_source:
        raise ContractViolation("OEWN_EVIDENCE_SOURCE_METADATA_MISMATCH")

    artifacts = value.get("source_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(SOURCE_SPECS):
        raise ContractViolation("OEWN_EVIDENCE_ARTIFACT_SCHEMA_MISMATCH")
    raw_by_name: dict[str, bytes] = {}
    resolved_root = Path(scope_root).resolve(strict=True)
    for name, spec in SOURCE_SPECS.items():
        ref = _exact_mapping(
            artifacts.get(name),
            _ARTIFACT_REF_FIELDS,
            "OEWN_EVIDENCE_ARTIFACT_REF_MISMATCH",
        )
        expected_path = manifest_path.parent / str(spec["filename"])
        if ref != _artifact_ref(expected_path, name):
            raise ContractViolation("OEWN_EVIDENCE_ARTIFACT_REF_MISMATCH")
        path = require_relative_to(
            Path(str(ref["path"])),
            resolved_root,
            "OEWN_EVIDENCE_ARTIFACT_OUTSIDE_SCOPE",
        )
        if path != expected_path or expected_path.is_symlink():
            raise ContractViolation("OEWN_EVIDENCE_ARTIFACT_PATH_MISMATCH")
        raw_by_name[name] = _read_expected_source(path, name)
    _validate_source_semantics(raw_by_name)

    locators = _exact_mapping(
        value.get("locators"),
        _LOCATOR_FIELDS,
        "OEWN_EVIDENCE_LOCATOR_SCHEMA_MISMATCH",
    )
    _exact_mapping(
        locators.get("lemma_entry"),
        _LEMMA_LOCATOR_FIELDS,
        "OEWN_EVIDENCE_LOCATOR_SCHEMA_MISMATCH",
    )
    _exact_mapping(
        locators.get("food_synset"),
        _SYNSET_LOCATOR_FIELDS,
        "OEWN_EVIDENCE_LOCATOR_SCHEMA_MISMATCH",
    )
    expected = _manifest_core(
        manifest_path.parent,
        captured_at_utc=str(value["captured_at_utc"]),
    )
    for field in ("locators", "licensing", "limitations"):
        if value.get(field) != expected[field]:
            raise ContractViolation("OEWN_EVIDENCE_MANIFEST_CONTENT_MISMATCH")
    _exact_mapping(
        value.get("licensing"),
        _LICENSING_FIELDS,
        "OEWN_EVIDENCE_LICENSING_SCHEMA_MISMATCH",
    )
    return {
        "status": AUDIT_STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "registered_expression": "egg",
        "sense_id": "egg%1:13:00::",
        "synset_id": "07856780-n",
        "source_artifact_count": len(SOURCE_SPECS),
        "manifest_sha256": claimed,
        "identity_authentication_claimed": False,
        "training_eligible": False,
    }


def _require_read_only_bundle(directory: Path) -> None:
    expected = {
        MANIFEST_FILENAME,
        *(str(spec["filename"]) for spec in SOURCE_SPECS.values()),
    }
    try:
        if directory.is_symlink() or stat.S_IMODE(
            directory.stat(follow_symlinks=False).st_mode
        ) & 0o222:
            raise ContractViolation("OEWN_EVIDENCE_READ_ONLY_BUNDLE_REQUIRED")
        with os.scandir(directory) as iterator:
            entries = list(iterator)
        if (
            {entry.name for entry in entries} != expected
            or any(not entry.is_file(follow_symlinks=False) for entry in entries)
        ):
            raise ContractViolation("OEWN_EVIDENCE_READ_ONLY_BUNDLE_REQUIRED")
        for name in expected:
            mode = stat.S_IMODE(
                (directory / name).stat(follow_symlinks=False).st_mode
            )
            if mode & 0o222:
                raise ContractViolation("OEWN_EVIDENCE_READ_ONLY_BUNDLE_REQUIRED")
    except ContractViolation:
        raise
    except OSError as exc:
        raise ContractViolation("OEWN_EVIDENCE_READ_ONLY_BUNDLE_REQUIRED") from exc


def audit_oewn_evidence_bundle(
    manifest_path: Path,
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Re-open and audit the complete immutable evidence directory."""
    resolved_root = Path(scope_root).resolve(strict=True)
    declared = Path(manifest_path)
    resolved = require_relative_to(
        declared, resolved_root, "OEWN_EVIDENCE_ARTIFACT_OUTSIDE_SCOPE"
    )
    if (
        resolved != declared.absolute()
        or declared.is_symlink()
        or resolved.name != MANIFEST_FILENAME
    ):
        raise ContractViolation("OEWN_EVIDENCE_MANIFEST_PATH_MISMATCH")
    _require_read_only_bundle(resolved.parent)
    metadata = resolved.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
        raise ContractViolation("OEWN_EVIDENCE_INVALID_MANIFEST_FILE")
    raw = read_regular_file_bytes_exact(resolved, expected_bytes=metadata.st_size)
    manifest = _loads_object(raw)
    if raw != canonical_json_bytes(manifest):
        raise ContractViolation("OEWN_EVIDENCE_NONCANONICAL_MANIFEST")
    result = validate_oewn_evidence_manifest(
        manifest, manifest_path=resolved, scope_root=resolved_root
    )
    final_raw = read_regular_file_bytes_exact(resolved, expected_bytes=len(raw))
    if final_raw != raw:
        raise ContractViolation("OEWN_EVIDENCE_CHANGED_DURING_AUDIT")
    return {
        **result,
        "manifest_artifact": {
            "path": str(resolved),
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
        },
        "write_semantics": "ATOMIC_DIRECTORY_NOREPLACE_READ_ONLY",
    }


def _cleanup_stage(stage: Path) -> None:
    try:
        os.chmod(stage, 0o700)
    except FileNotFoundError:
        return
    for name in (
        MANIFEST_FILENAME,
        *(str(spec["filename"]) for spec in SOURCE_SPECS.values()),
    ):
        try:
            (stage / name).unlink()
        except FileNotFoundError:
            pass
    try:
        stage.rmdir()
    except FileNotFoundError:
        pass


def publish_oewn_evidence_bundle(
    source_files: Mapping[str, Path],
    output_dir: Path,
    *,
    captured_at_utc: str,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Atomically publish verified OEWN payloads and their strict manifest."""
    if not isinstance(source_files, Mapping) or set(source_files) != set(SOURCE_SPECS):
        raise ContractViolation("OEWN_EVIDENCE_SOURCE_SET_MISMATCH")
    destination = require_relative_to(
        Path(output_dir), scope_root, "OEWN_EVIDENCE_OUTPUT_OUTSIDE_SCOPE"
    )
    if destination.exists() or destination.is_symlink():
        raise ContractViolation("OEWN_EVIDENCE_OUTPUT_EXISTS")
    raw_by_name = {
        name: _read_expected_source(Path(source_files[name]), name)
        for name in SOURCE_SPECS
    }
    _validate_source_semantics(raw_by_name)
    manifest = build_oewn_evidence_manifest(
        destination, captured_at_utc=captured_at_utc
    )
    manifest_raw = canonical_json_bytes(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    stage.mkdir(mode=0o700)
    try:
        for name, spec in SOURCE_SPECS.items():
            publish_verified_file_once(
                source_files[name],
                stage / str(spec["filename"]),
                expected_sha256=str(spec["sha256"]),
                expected_bytes=int(spec["bytes"]),
                mode=0o444,
            )
        publish_bytes_once(stage / MANIFEST_FILENAME, manifest_raw, mode=0o444)
        staged = {
            name: _read_expected_source(stage / str(spec["filename"]), name)
            for name, spec in SOURCE_SPECS.items()
        }
        _validate_source_semantics(staged)
        if read_regular_file_bytes_exact(
            stage / MANIFEST_FILENAME, expected_bytes=len(manifest_raw)
        ) != manifest_raw:
            raise ContractViolation("OEWN_EVIDENCE_PREPUBLICATION_RECHECK_FAILED")
        os.chmod(stage, 0o555)
        from .prepare_data import _rename_directory_noreplace

        try:
            _rename_directory_noreplace(stage, destination)
        except ContractViolation as exc:
            mapped = {
                "ANNOTATION_FREEZE_OUTPUT_EXISTS": "OEWN_EVIDENCE_OUTPUT_EXISTS",
                "ANNOTATION_FREEZE_ATOMIC_NOREPLACE_UNSUPPORTED": (
                    "OEWN_EVIDENCE_ATOMIC_NOREPLACE_UNSUPPORTED"
                ),
            }.get(exc.code, "OEWN_EVIDENCE_ATOMIC_PUBLISH_FAILED")
            raise ContractViolation(mapped) from exc
    except Exception:
        _cleanup_stage(stage)
        raise
    return audit_oewn_evidence_bundle(
        destination / MANIFEST_FILENAME, scope_root=scope_root
    )


__all__ = [
    "AUDIT_STATUS",
    "EVIDENCE_SCOPE",
    "GIT_COMMIT",
    "LIMITATIONS",
    "MANIFEST_FILENAME",
    "RAW_FILE_URLS",
    "RELEASE_TAG",
    "SCHEMA_VERSION",
    "SOURCE_SPECS",
    "STATUS",
    "audit_oewn_evidence_bundle",
    "build_oewn_evidence_manifest",
    "publish_oewn_evidence_bundle",
    "validate_oewn_evidence_manifest",
]
