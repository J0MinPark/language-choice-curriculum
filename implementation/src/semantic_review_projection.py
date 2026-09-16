"""Non-production v4.1 projection of the immutable jm02 review intake.

Only meaning alignment and the four registered-term quality dispositions are
projected.  The source intake's overall disposition is retained solely as
``source_intake_overall_disposition``; the v4.1 core disposition is recomputed
from meaning plus KO/EN/ZH/FR term quality.  No row produced here contains an
etymology, family, or synonym field.

The source bundle is always audited through
``review_feedback_intake.audit_review_feedback_intake`` before it is read.
This module publishes nothing and cannot make data training eligible, create
formal evidence, or create an annotation freeze.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import review_feedback_intake
from .contracts import (
    LANGUAGES,
    WORK_ROOT,
    ContractViolation,
    canonical_json_bytes,
    require_relative_to,
    require_sha256,
)


SCHEMA_VERSION = "semantic-review-core-projection-v2"
PROTOCOL_VERSION = "4.1.0"
STATUS = "CORE_REVIEW_PROJECTION_RECORDED_PENDING_FORMAL_EVIDENCE"
AUDIT_STATUS = "PASS_SEMANTIC_REVIEW_PROJECTION_NON_PRODUCTION"
FORMAL_EVIDENCE_STATUS = "PENDING"
MAPPING_RULE = (
    "core_disposition=max_severity(meaning.disposition,"
    "term_quality[ko,en,zh,fr].disposition), where "
    "APPROVED<DEFERRED<REVISION_REQUIRED; source intake etymology is excluded"
)
NOTICE = (
    "This deterministic projection is non-production. It preserves reported "
    "jm02 intake decisions but does not authenticate identity, verify formal "
    "production evidence, create an annotation freeze, or permit training."
)

DISPOSITIONS = ("APPROVED", "DEFERRED", "REVISION_REQUIRED")
_SEVERITY = {value: index for index, value in enumerate(DISPOSITIONS)}
_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256", "bytes"})
_SOURCE_NAMES = (
    "selection_plan",
    "advisory",
    "advisory_bundle_manifest",
    "summary",
)
_ALL_REF_NAMES = (
    "review_feedback_intake",
    "review_feedback_intake_manifest",
    *_SOURCE_NAMES,
)
_SOURCE_KEY_MAP = {
    "selection_plan": "selection_plan",
    "advisory": "advisory",
    "advisory_bundle_manifest": "bundle_manifest",
    "summary": "summary",
}
_MAX_BYTES = {
    "review_feedback_intake": 4 * 1024 * 1024,
    "review_feedback_intake_manifest": 1024 * 1024,
    "selection_plan": 2 * 1024 * 1024,
    "advisory": 4 * 1024 * 1024,
    "advisory_bundle_manifest": 1024 * 1024,
    "summary": 1024 * 1024,
    "projection": 4 * 1024 * 1024,
}
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "status",
        "non_production_projection",
        "training_eligible",
        "direct_trainer_input_allowed",
        "annotation_freeze_created",
        "identity_authentication_claimed",
        "formal_research_approval_claimed",
        "source_intake",
        "source_artifacts",
        "core_mapping_rule",
        "row_count",
        "decision_counts",
        "core_review_status",
        "core_decision_blockers",
        "formal_evidence",
        "optional_etymology",
        "rows",
        "notice",
        "projection_sha256",
    }
)
_SOURCE_INTAKE_FIELDS = frozenset(
    {
        "schema_version",
        "spec_version",
        "recorded_status",
        "reviewer_id",
        "review_date",
        "recorded_at_utc",
        "identity_authentication_claimed",
    }
)
_ROW_FIELDS = frozenset(
    {
        "selection_ordinal",
        "review_order",
        "candidate_id",
        "candidate_sha256",
        "source_intake_overall_disposition",
        "meaning",
        "term_quality",
        "core_disposition",
        "core_decision_reasons",
    }
)
_MEANING_FIELDS = frozenset({"disposition", "value"})
_TERM_FIELDS = frozenset({"disposition", "value"})
_COUNT_DIMENSIONS = ("meaning", "term_quality", "core_overall")
_OPTIONAL_ETYMOLOGY = {
    "status": "NOT_RUN",
    "reason": "INSUFFICIENT_SOURCE_BOUND_HUMAN_REVIEWED_ETYMOLOGY_DATA",
    "reviewed_count": 0,
    "artifact": None,
    "blocks_core": False,
}
_FORMAL_EVIDENCE = {
    "status": FORMAL_EVIDENCE_STATUS,
    "artifact": None,
    "required_before_training": True,
}
_FORBIDDEN_CORE_ROW_FIELDS = frozenset(
    {
        "etymology",
        "etymology_subtype",
        "etymology_primary",
        "relation_direction",
        "shared_source",
        "historical_scope",
        "family_id",
        "etymology_family_id",
        "synonym_cluster_id",
    }
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation("SEMANTIC_PROJECTION_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ContractViolation("SEMANTIC_PROJECTION_NONFINITE_JSON_NUMBER")


def _reject_unicode_surrogates(value: Any) -> None:
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in current):
                raise ContractViolation(
                    "SEMANTIC_PROJECTION_UNICODE_SURROGATE_FORBIDDEN"
                )
            continue
        if not isinstance(current, (Mapping, list, tuple)):
            continue
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, Mapping):
            for key, item in current.items():
                pending.extend((key, item))
        else:
            pending.extend(current)


def _canonical_bytes(value: Any) -> bytes:
    _reject_unicode_surrogates(value)
    try:
        return canonical_json_bytes(value)
    except ContractViolation:
        raise
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ContractViolation("SEMANTIC_PROJECTION_INVALID_JSON") from exc


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
        raise ContractViolation("SEMANTIC_PROJECTION_INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise ContractViolation("SEMANTIC_PROJECTION_INVALID_JSON")
    _reject_unicode_surrogates(value)
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _resolve_direct_regular_path(
    path: Path,
    *,
    scope_root: Path,
    outside_code: str,
    unsafe_code: str,
) -> Path:
    """Resolve a path while rejecting final or ancestor symlink aliases."""
    candidate = Path(path)
    try:
        lexical_absolute = Path(os.path.abspath(candidate))
        resolved = candidate.resolve(strict=True)
        require_relative_to(resolved, scope_root, outside_code)
        metadata = os.lstat(lexical_absolute)
    except ContractViolation:
        raise
    except OSError as exc:
        raise ContractViolation(unsafe_code) from exc
    if lexical_absolute != resolved or not stat.S_ISREG(metadata.st_mode):
        raise ContractViolation(unsafe_code)
    return resolved


def _read_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    scope_root: Path,
    outside_code: str = "SEMANTIC_PROJECTION_ARTIFACT_OUTSIDE_SCOPE",
    unsafe_code: str = "SEMANTIC_PROJECTION_ARTIFACT_NOT_REGULAR_FILE",
) -> tuple[Path, bytes]:
    resolved = _resolve_direct_regular_path(
        path,
        scope_root=scope_root,
        outside_code=outside_code,
        unsafe_code=unsafe_code,
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ContractViolation(unsafe_code) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractViolation(unsafe_code)
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise ContractViolation("SEMANTIC_PROJECTION_ARTIFACT_TOO_LARGE")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ContractViolation(
                    "SEMANTIC_PROJECTION_ARTIFACT_CHANGED_DURING_READ"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ContractViolation(
                "SEMANTIC_PROJECTION_ARTIFACT_CHANGED_DURING_READ"
            )
        after = os.fstat(descriptor)
    except ContractViolation:
        raise
    except OSError as exc:
        raise ContractViolation("SEMANTIC_PROJECTION_ARTIFACT_READ_FAILED") from exc
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ContractViolation("SEMANTIC_PROJECTION_ARTIFACT_CHANGED_DURING_READ")
    return resolved, b"".join(chunks)


def _checked_ref(
    value: Any,
    *,
    name: str,
    scope_root: Path,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_REF_FIELDS:
        raise ContractViolation("SEMANTIC_PROJECTION_INVALID_ARTIFACT_REF")
    path_text = value.get("path")
    expected_bytes = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path_text, str)
        or not path_text
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
        or expected_bytes > _MAX_BYTES[name]
        or not isinstance(digest, str)
    ):
        raise ContractViolation("SEMANTIC_PROJECTION_INVALID_ARTIFACT_REF")
    expected_hash = require_sha256(
        digest, "SEMANTIC_PROJECTION_INVALID_ARTIFACT_REF"
    )
    path, raw = _read_regular_bytes(
        Path(path_text), maximum_bytes=_MAX_BYTES[name], scope_root=scope_root
    )
    if path_text != str(path):
        raise ContractViolation("SEMANTIC_PROJECTION_ARTIFACT_PATH_NOT_CANONICAL")
    if len(raw) != expected_bytes:
        raise ContractViolation("SEMANTIC_PROJECTION_ARTIFACT_SIZE_MISMATCH")
    if _sha256(raw) != expected_hash:
        raise ContractViolation("SEMANTIC_PROJECTION_ARTIFACT_HASH_MISMATCH")
    return {"path": str(path), "sha256": expected_hash, "bytes": len(raw)}, raw


def _source_snapshot(
    intake_manifest_path: Path,
    *,
    scope_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    # Reject symlink aliases before the established audit resolves its input.
    manifest_path = _resolve_direct_regular_path(
        intake_manifest_path,
        scope_root=scope_root,
        outside_code="SEMANTIC_PROJECTION_SOURCE_OUTSIDE_SCOPE",
        unsafe_code="SEMANTIC_PROJECTION_SOURCE_NOT_REGULAR_FILE",
    )
    first_audit = review_feedback_intake.audit_review_feedback_intake(
        manifest_path, scope_root=scope_root
    )
    intake_ref, intake_raw = _checked_ref(
        first_audit.get("intake_artifact"),
        name="review_feedback_intake",
        scope_root=scope_root,
    )
    manifest_ref, _manifest_raw = _checked_ref(
        first_audit.get("manifest_artifact"),
        name="review_feedback_intake_manifest",
        scope_root=scope_root,
    )
    if manifest_ref["path"] != str(manifest_path):
        raise ContractViolation("SEMANTIC_PROJECTION_SOURCE_MANIFEST_MISMATCH")
    intake_payload = _loads_object(intake_raw)
    if intake_raw != _canonical_bytes(intake_payload):
        raise ContractViolation("SEMANTIC_PROJECTION_SOURCE_NONCANONICAL_JSON")

    source_values = intake_payload.get("source_artifacts")
    if not isinstance(source_values, Mapping) or set(source_values) != set(
        review_feedback_intake.SOURCE_NAMES
    ):
        raise ContractViolation("SEMANTIC_PROJECTION_SOURCE_REF_SET_MISMATCH")
    refs: dict[str, dict[str, Any]] = {
        "review_feedback_intake": intake_ref,
        "review_feedback_intake_manifest": manifest_ref,
    }
    for output_name in _SOURCE_NAMES:
        source_name = _SOURCE_KEY_MAP[output_name]
        checked, _raw = _checked_ref(
            source_values[source_name], name=output_name, scope_root=scope_root
        )
        refs[output_name] = checked

    second_audit = review_feedback_intake.audit_review_feedback_intake(
        manifest_path, scope_root=scope_root
    )
    if second_audit != first_audit:
        raise ContractViolation("SEMANTIC_PROJECTION_SOURCE_CHANGED_DURING_BUILD")
    return intake_payload, first_audit, refs


def _disposition(value: Any) -> str:
    if value not in _SEVERITY:
        raise ContractViolation("SEMANTIC_PROJECTION_INVALID_DISPOSITION")
    return str(value)


def _core_decision(
    meaning: Mapping[str, Any], term_quality: Mapping[str, Mapping[str, Any]]
) -> tuple[str, list[str]]:
    components: list[tuple[str, str]] = [
        ("MEANING", _disposition(meaning.get("disposition")))
    ]
    components.extend(
        (
            f"TERM_QUALITY_{language.upper()}",
            _disposition(term_quality[language].get("disposition")),
        )
        for language in LANGUAGES
    )
    core = max((value for _name, value in components), key=_SEVERITY.__getitem__)
    reasons = [f"{name}_{value}" for name, value in components if value != "APPROVED"]
    return core, reasons


def _empty_counts() -> dict[str, dict[str, int]]:
    return {
        dimension: {disposition: 0 for disposition in DISPOSITIONS}
        for dimension in _COUNT_DIMENSIONS
    }


def _derive_projection(
    intake_manifest_path: Path,
    *,
    scope_root: Path,
) -> dict[str, Any]:
    intake_payload, intake_audit, refs = _source_snapshot(
        intake_manifest_path, scope_root=scope_root
    )
    source_rows = intake_payload.get("rows")
    if not isinstance(source_rows, Sequence) or isinstance(
        source_rows, (str, bytes, bytearray)
    ):
        raise ContractViolation("SEMANTIC_PROJECTION_SOURCE_ROWS_INVALID")

    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    counts = _empty_counts()
    for source_row in source_rows:
        if not isinstance(source_row, Mapping):
            raise ContractViolation("SEMANTIC_PROJECTION_SOURCE_ROWS_INVALID")
        meaning_raw = source_row.get("meaning")
        term_raw = source_row.get("term_quality")
        if (
            not isinstance(meaning_raw, Mapping)
            or set(meaning_raw) != _MEANING_FIELDS
            or not isinstance(term_raw, Mapping)
            or set(term_raw) != set(LANGUAGES)
        ):
            raise ContractViolation("SEMANTIC_PROJECTION_SOURCE_COMPONENT_INVALID")
        meaning = {
            "disposition": _disposition(meaning_raw.get("disposition")),
            "value": meaning_raw.get("value"),
        }
        term_quality: dict[str, dict[str, Any]] = {}
        for language in LANGUAGES:
            value = term_raw[language]
            if not isinstance(value, Mapping) or set(value) != _TERM_FIELDS:
                raise ContractViolation(
                    "SEMANTIC_PROJECTION_SOURCE_COMPONENT_INVALID"
                )
            term_quality[language] = {
                "disposition": _disposition(value.get("disposition")),
                "value": value.get("value"),
            }
        core_disposition, reasons = _core_decision(meaning, term_quality)
        row = {
            "selection_ordinal": source_row.get("selection_ordinal"),
            "review_order": source_row.get("review_order"),
            "candidate_id": source_row.get("candidate_id"),
            "candidate_sha256": source_row.get("candidate_sha256"),
            "source_intake_overall_disposition": _disposition(
                source_row.get("overall_disposition")
            ),
            "meaning": meaning,
            "term_quality": term_quality,
            "core_disposition": core_disposition,
            "core_decision_reasons": reasons,
        }
        if set(row).intersection(_FORBIDDEN_CORE_ROW_FIELDS):
            raise ContractViolation("SEMANTIC_PROJECTION_FORBIDDEN_CORE_FIELD")
        rows.append(row)
        counts["meaning"][meaning["disposition"]] += 1
        counts["core_overall"][core_disposition] += 1
        for language in LANGUAGES:
            disposition = term_quality[language]["disposition"]
            counts["term_quality"][disposition] += 1
            if disposition != "APPROVED":
                blockers.append(
                    {
                        "selection_ordinal": row["selection_ordinal"],
                        "review_order": row["review_order"],
                        "candidate_id": row["candidate_id"],
                        "component": "term_quality",
                        "language": language,
                        "disposition": disposition,
                    }
                )
        if meaning["disposition"] != "APPROVED":
            blockers.append(
                {
                    "selection_ordinal": row["selection_ordinal"],
                    "review_order": row["review_order"],
                    "candidate_id": row["candidate_id"],
                    "component": "meaning",
                    "language": None,
                    "disposition": meaning["disposition"],
                }
            )

    core_review_status = "READY_FOR_FORMAL_EVIDENCE"
    if blockers:
        components = {str(item["component"]) for item in blockers}
        core_review_status = (
            "BLOCKED_TERM_QUALITY"
            if components == {"term_quality"}
            else "BLOCKED_CORE_SEMANTIC_REVIEW"
        )
    core = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": STATUS,
        "non_production_projection": True,
        "training_eligible": False,
        "direct_trainer_input_allowed": False,
        "annotation_freeze_created": False,
        "identity_authentication_claimed": False,
        "formal_research_approval_claimed": False,
        "source_intake": {
            "schema_version": intake_audit["schema_version"],
            "spec_version": intake_audit["spec_version"],
            "recorded_status": intake_audit["recorded_status"],
            "reviewer_id": intake_audit["reviewer_id"],
            "review_date": intake_audit["review_date"],
            "recorded_at_utc": intake_audit["recorded_at_utc"],
            "identity_authentication_claimed": False,
        },
        "source_artifacts": refs,
        "core_mapping_rule": MAPPING_RULE,
        "row_count": len(rows),
        "decision_counts": counts,
        "core_review_status": core_review_status,
        "core_decision_blockers": blockers,
        "formal_evidence": dict(_FORMAL_EVIDENCE),
        "optional_etymology": dict(_OPTIONAL_ETYMOLOGY),
        "rows": rows,
        "notice": NOTICE,
    }
    return {**core, "projection_sha256": _sha256(_canonical_bytes(core))}


def build_semantic_review_projection(
    intake_manifest_path: Path,
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Build the deterministic v4.1 semantic projection in memory only."""
    return _derive_projection(Path(intake_manifest_path), scope_root=scope_root)


def _validate_structure(payload: Mapping[str, Any]) -> None:
    _reject_unicode_surrogates(payload)
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ContractViolation("SEMANTIC_PROJECTION_SCHEMA_MISMATCH")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("status") != STATUS
        or payload.get("core_mapping_rule") != MAPPING_RULE
        or payload.get("notice") != NOTICE
    ):
        raise ContractViolation("SEMANTIC_PROJECTION_METADATA_MISMATCH")
    for field in (
        "non_production_projection",
        "identity_authentication_claimed",
        "formal_research_approval_claimed",
        "training_eligible",
        "direct_trainer_input_allowed",
        "annotation_freeze_created",
    ):
        expected = field == "non_production_projection"
        if payload.get(field) is not expected:
            raise ContractViolation("SEMANTIC_PROJECTION_SAFETY_CLAIM_MISMATCH")
    if payload.get("optional_etymology") != _OPTIONAL_ETYMOLOGY:
        raise ContractViolation("SEMANTIC_PROJECTION_ETYMOLOGY_STATUS_MISMATCH")
    if payload.get("formal_evidence") != _FORMAL_EVIDENCE:
        raise ContractViolation("SEMANTIC_PROJECTION_FORMAL_EVIDENCE_MISMATCH")
    source_intake = payload.get("source_intake")
    if not isinstance(source_intake, Mapping) or set(source_intake) != _SOURCE_INTAKE_FIELDS:
        raise ContractViolation("SEMANTIC_PROJECTION_SOURCE_METADATA_MISMATCH")
    if source_intake.get("identity_authentication_claimed") is not False:
        raise ContractViolation("SEMANTIC_PROJECTION_IDENTITY_CLAIM_FORBIDDEN")
    refs = payload.get("source_artifacts")
    if not isinstance(refs, Mapping) or set(refs) != set(_ALL_REF_NAMES):
        raise ContractViolation("SEMANTIC_PROJECTION_SOURCE_REF_SET_MISMATCH")
    for ref in refs.values():
        if not isinstance(ref, Mapping) or set(ref) != _ARTIFACT_REF_FIELDS:
            raise ContractViolation("SEMANTIC_PROJECTION_INVALID_ARTIFACT_REF")
    rows = payload.get("rows")
    row_count = payload.get("row_count")
    if (
        not isinstance(rows, list)
        or not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count != len(rows)
    ):
        raise ContractViolation("SEMANTIC_PROJECTION_ROW_COUNT_MISMATCH")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise ContractViolation("SEMANTIC_PROJECTION_ROW_SCHEMA_MISMATCH")
        if set(row).intersection(_FORBIDDEN_CORE_ROW_FIELDS):
            raise ContractViolation("SEMANTIC_PROJECTION_FORBIDDEN_CORE_FIELD")
        if (
            not isinstance(row.get("meaning"), Mapping)
            or set(row["meaning"]) != _MEANING_FIELDS
        ):
            raise ContractViolation("SEMANTIC_PROJECTION_ROW_SCHEMA_MISMATCH")
        terms = row.get("term_quality")
        if not isinstance(terms, Mapping) or set(terms) != set(LANGUAGES):
            raise ContractViolation("SEMANTIC_PROJECTION_ROW_SCHEMA_MISMATCH")
        if any(
            not isinstance(value, Mapping) or set(value) != _TERM_FIELDS
            for value in terms.values()
        ):
            raise ContractViolation("SEMANTIC_PROJECTION_ROW_SCHEMA_MISMATCH")


def validate_semantic_review_projection(
    payload: Mapping[str, Any],
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Re-derive a projection from its exact source bundle and compare it."""
    _validate_structure(payload)
    claimed = payload.get("projection_sha256")
    if not isinstance(claimed, str):
        raise ContractViolation("SEMANTIC_PROJECTION_HASH_MISMATCH")
    require_sha256(claimed, "SEMANTIC_PROJECTION_HASH_MISMATCH")
    logical = {key: value for key, value in payload.items() if key != "projection_sha256"}
    if _sha256(_canonical_bytes(logical)) != claimed:
        raise ContractViolation("SEMANTIC_PROJECTION_HASH_MISMATCH")
    manifest_ref = payload["source_artifacts"]["review_feedback_intake_manifest"]
    expected = _derive_projection(Path(manifest_ref["path"]), scope_root=scope_root)
    if dict(payload) != expected:
        raise ContractViolation("SEMANTIC_PROJECTION_CONTENT_MISMATCH")
    return {
        "status": AUDIT_STATUS,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "projection_status": STATUS,
        "projection_sha256": claimed,
        "row_count": payload["row_count"],
        "decision_counts": payload["decision_counts"],
        "core_review_status": payload["core_review_status"],
        "core_decision_blockers": payload["core_decision_blockers"],
        "formal_evidence_status": FORMAL_EVIDENCE_STATUS,
        "identity_authentication_claimed": False,
        "training_eligible": False,
    }


def canonical_semantic_review_projection_bytes(
    payload: Mapping[str, Any],
    *,
    scope_root: Path = WORK_ROOT,
) -> bytes:
    """Return canonical bytes only after exact source-backed validation."""
    validate_semantic_review_projection(payload, scope_root=scope_root)
    return _canonical_bytes(payload)


def audit_semantic_review_projection_file(
    projection_path: Path,
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Audit one existing canonical projection file without publishing it."""
    path, raw = _read_regular_bytes(
        Path(projection_path),
        maximum_bytes=_MAX_BYTES["projection"],
        scope_root=scope_root,
    )
    payload = _loads_object(raw)
    if raw != _canonical_bytes(payload):
        raise ContractViolation("SEMANTIC_PROJECTION_NONCANONICAL_JSON")
    result = validate_semantic_review_projection(payload, scope_root=scope_root)
    final_path, final_raw = _read_regular_bytes(
        path, maximum_bytes=_MAX_BYTES["projection"], scope_root=scope_root
    )
    if final_path != path or final_raw != raw:
        raise ContractViolation("SEMANTIC_PROJECTION_ARTIFACT_CHANGED_DURING_AUDIT")
    return {
        **result,
        "projection_artifact": {
            "path": str(path),
            "sha256": _sha256(raw),
            "bytes": len(raw),
        },
    }


__all__ = [
    "AUDIT_STATUS",
    "FORMAL_EVIDENCE_STATUS",
    "MAPPING_RULE",
    "NOTICE",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "STATUS",
    "audit_semantic_review_projection_file",
    "build_semantic_review_projection",
    "canonical_semantic_review_projection_bytes",
    "validate_semantic_review_projection",
]
