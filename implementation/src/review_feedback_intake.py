"""Write-once, non-production intake for externally supplied review feedback.

This module records what a reviewer reported against the fixed 60-item AI
pre-review.  It deliberately does *not* authenticate the reviewer, verify
production evidence, edit review CSVs, create an annotation freeze, or make
anything eligible for training.  An ``APPROVED`` disposition in this schema is
therefore only an intake classification; it is never the formal approval token
accepted by the production annotation workflow.

The intake binds the exact, already-audited AI pre-review bundle (its manifest,
selection plan, and advisory JSONL) plus the matching human-readable summary by
canonical absolute path, SHA-256, and byte count.  Publication atomically
installs one read-only directory containing the intake and an internal manifest;
neither file nor the directory may replace an existing path.  No function in
this module performs network, GPU, training, CSV-write, evidence publication,
or freeze operations.
"""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import ai_prereview
from .artifacts import (
    publish_bytes_once,
    read_regular_file_bytes_exact,
)
from .contracts import (
    LANGUAGES,
    VERSION,
    WORK_ROOT,
    ContractViolation,
    canonical_json_bytes,
    require_relative_to,
    require_sha256,
    sha256_bytes,
)


SCHEMA_VERSION = "review-feedback-intake-v1"
BUNDLE_MANIFEST_SCHEMA_VERSION = "review-feedback-intake-bundle-manifest-v1"
RECORDED_STATUS = (
    "HUMAN_REVIEW_INTAKE_RECORDED_PENDING_EVIDENCE_AND_DEFERRED_ITEMS"
)
AUDIT_STATUS = "PASS_REVIEW_FEEDBACK_INTAKE_NON_PRODUCTION"
ROW_COUNT = ai_prereview.SELECTION_COUNT
DISPOSITIONS = ("APPROVED", "DEFERRED", "REVISION_REQUIRED")
SOURCE_NAMES = ("selection_plan", "advisory", "bundle_manifest", "summary")
INTAKE_FILENAME = "review_feedback_intake.json"
BUNDLE_MANIFEST_FILENAME = "review_feedback_intake_manifest.json"

# jm02 reviewed one concrete, already-published display bundle.  The decisions
# below are ordinal-specific, so accepting merely *any* structurally valid
# 60-row advisory bundle would risk attaching those decisions to different
# concepts.  Keep the four canonical references in one immutable profile and
# require an exact match before reading or interpreting any feedback row.
EXPECTED_REVIEWED_SOURCE_PROFILE = (
    (
        "selection_plan",
        "/home/i2slab4/jm/test_jm/work/annotations/"
        "pending_krdict-b100064abad6c836/"
        "ai_prereview_20260914T183937Z_3ff1529/"
        "advisory_bundle_ca44d79/selection_plan.json",
        "01556dfd37271c8687151f923f70a015057754f1fa1a1abe6d2b5c3c27957a27",
        109878,
    ),
    (
        "advisory",
        "/home/i2slab4/jm/test_jm/work/annotations/"
        "pending_krdict-b100064abad6c836/"
        "ai_prereview_20260914T183937Z_3ff1529/"
        "advisory_bundle_ca44d79/ai_prereview.jsonl",
        "06a007556d684a22d70f8476440e0b521e01f058ecbe08f8e8e8b7377b2d9654",
        140557,
    ),
    (
        "bundle_manifest",
        "/home/i2slab4/jm/test_jm/work/annotations/"
        "pending_krdict-b100064abad6c836/"
        "ai_prereview_20260914T183937Z_3ff1529/"
        "advisory_bundle_ca44d79/ai_prereview_manifest.json",
        "9b5354b0f93799a85e2498af29c22ea36a967725f29217ccf74b9564071da720",
        1805,
    ),
    (
        "summary",
        "/home/i2slab4/jm/test_jm/work/annotations/"
        "pending_krdict-b100064abad6c836/"
        "ai_prereview_20260914T183937Z_3ff1529/"
        "ai_prereview_summary_ca44d79.md",
        "19a0a62e8c2630db530bf234500c37f880d0ce8e6e780a453f4b19db36f0e6e1",
        9209,
    ),
)

_SOURCE_MAX_BYTES = {
    "selection_plan": 2 * 1024 * 1024,
    "advisory": 4 * 1024 * 1024,
    "bundle_manifest": 1024 * 1024,
    "summary": 1024 * 1024,
}
_INTAKE_MAX_BYTES = 4 * 1024 * 1024
_BUNDLE_MANIFEST_MAX_BYTES = 1024 * 1024

NOTICE = (
    "APPROVED is an intake-only disposition. Reviewer identity is not "
    "authenticated here; production evidence, formal CSV review fields, and "
    "an annotation freeze remain separate required steps. This artifact is not "
    "training eligible."
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "spec_version",
        "status",
        "non_production_intake",
        "reviewer_id",
        "review_date",
        "recorded_at_utc",
        "identity_authentication_claimed",
        "formal_research_approval_claimed",
        "production_evidence_verified",
        "direct_csv_input_allowed",
        "review_csvs_modified",
        "evidence_records_created",
        "annotation_freeze_created",
        "training_eligible",
        "requires_formal_evidence_and_csv_workflow",
        "source_artifacts",
        "row_count",
        "decision_counts",
        "reason_code_counts",
        "rows",
        "notice",
    }
)
_ROW_FIELDS = frozenset(
    {
        "selection_ordinal",
        "review_order",
        "candidate_id",
        "candidate_sha256",
        "overall_disposition",
        "meaning",
        "term_quality",
        "etymology",
        "reason_code",
        "intake_note",
    }
)
_FEEDBACK_INPUT_FIELDS = frozenset(
    {
        "selection_ordinal",
        "candidate_id",
        "overall_disposition",
        "meaning",
        "term_quality",
        "etymology",
        "reason_code",
        "intake_note",
    }
)
_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256", "bytes"})
_COUNT_DIMENSIONS = ("overall", "meaning", "term_quality", "etymology")
_MEANING_FIELDS = frozenset({"disposition", "value"})
_TERM_QUALITY_FIELDS = frozenset({"disposition", "value"})
_ETYMOLOGY_FIELDS = frozenset({"disposition", "subtype", "direction"})
_REVIEWER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,63}")
_FORMAL_APPROVAL_TOKEN_RE = re.compile(
    r"approved[\s_-]*by[\s_-]*researcher", re.IGNORECASE
)

_ETYMOLOGY_LABEL_KO = {
    "BORROWING_DIRECT": "직접차용",
    "BORROWING_PARALLEL": "병렬차용",
    "NEOCLASSICAL_SHARED": "신고전 공통",
    "COGNATE_INHERITED": "계승동족어",
    "DISTINCT_ROUTES_REVIEWED": "별개경로",
    "INDETERMINATE": "미확정",
}

# This module records the exact adjudication confirmed by jm02, rather than a
# generic mechanism that could silently attribute a different decision map to
# that identifier.  The 49 ordinary approvals, three clarified source-sense
# approvals, and eight deferrals add to the fixed 60-row cohort.
_DEFAULT_APPROVAL = (
    "APPROVED_REMAINDER_BY_REVIEWER_CONFIRMATION",
    "Approved as part of jm02's explicitly confirmed remainder.",
)
_SPECIAL_ADJUDICATIONS = {
    2: (
        "DEFERRED_DEEP_RELATION_UNESTABLISHED",
        "Language-family contrast alone is insufficient to establish distinct routes; the deeper bee/abeille relation remains unresolved.",
    ),
    9: (
        "DEFERRED_MIXED_BORROWING_AND_INHERITANCE_CAT",
        "A common cattus source with a borrowing-and-inheritance mixture requires further classification.",
    ),
    10: (
        "DEFERRED_POSSIBLE_COMMON_ANCESTOR_SNOW",
        "A possible common ancestor makes a distinct-routes decision unsupported by the reviewed material.",
    ),
    19: (
        "DEFERRED_ENGLISH_ORIGIN_UNCERTAIN_BIRD",
        "The deeper origin of English bird remains uncertain, so language-family contrast alone does not establish distinct routes.",
    ),
    21: (
        "DEFERRED_MIXED_BORROWING_AND_INHERITANCE_PEAR",
        "English borrowing and French inheritance from the pira lineage form a mixed path requiring further classification.",
    ),
    22: (
        "APPROVED_SOURCE_SENSE_CLARIFIES_FISH",
        "The reviewed KRDICT source sense denotes fish used as food, so 고기/fish/鱼/poisson is retained.",
    ),
    35: (
        "DEFERRED_MIXED_BORROWING_AND_INHERITANCE_BUTTER",
        "English borrowing and French inheritance from the butyrum lineage form a mixed path requiring further classification.",
    ),
    38: (
        "DEFERRED_EN_TERM_NATURALNESS_AND_ETYMOLOGY_EGG",
        "Meaning alignment is retained, but English hen's egg naturalness and the EN–FR etymology remain deferred.",
    ),
    42: (
        "APPROVED_SOURCE_SENSE_CLARIFIES_LIPS",
        "The reviewed KRDICT source sense denotes lips, so 입/lips/嘴唇/lèvre is retained.",
    ),
    50: (
        "APPROVED_SOURCE_SENSE_CLARIFIES_STEP_COUNTER",
        "The reviewed KRDICT source sense is the counter for steps, so 발/step/步/pas is retained.",
    ),
    59: (
        "DEFERRED_SENSE_SPECIFIC_ETYMOLOGY_CARD",
        "The captured English etymology locator may concern the playing-card lineage rather than the reviewed information/ID-card sense.",
    ),
}
_DEFERRED_ORDINALS = frozenset({2, 9, 10, 19, 21, 35, 38, 59})
REASON_CODES = frozenset(
    {_DEFAULT_APPROVAL[0], *(value[0] for value in _SPECIAL_ADJUDICATIONS.values())}
)
_RESERVED_NOTE_CLAIMS = re.compile(
    r"(?:approved\W*by\W*researcher|formal\W+(?:research\W+)?approval|"
    r"training\W+(?:eligible|authorization|authorized)|production\W+evidence|"
    r"annotation\W+freeze|identity\W+authenticated|review\W+csvs?\W+modified|"
    r"evidence\W+records?\W+created)",
    re.IGNORECASE,
)

_BUNDLE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "spec_version",
        "status",
        "non_production_intake",
        "training_eligible",
        "intake_artifact",
        "source_artifacts",
        "row_count",
        "decision_counts",
        "reason_code_counts",
        "reviewer_id",
        "review_date",
        "write_semantics",
    }
)

_EXPECTED_REASON_CODE_COUNTS = {
    _DEFAULT_APPROVAL[0]: ROW_COUNT - len(_SPECIAL_ADJUDICATIONS),
    **{value[0]: 1 for value in _SPECIAL_ADJUDICATIONS.values()},
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation("REVIEW_INTAKE_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_token: str) -> None:
    raise ContractViolation("REVIEW_INTAKE_NONFINITE_JSON_NUMBER")


def _reject_unicode_surrogates(value: Any, seen: set[int] | None = None) -> None:
    """Reject Unicode surrogate code points before canonical UTF-8 encoding."""
    visited = set() if seen is None else seen
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in current):
                raise ContractViolation(
                    "REVIEW_INTAKE_UNICODE_SURROGATE_FORBIDDEN"
                )
            continue
        if not isinstance(current, (Mapping, list, tuple)):
            continue
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(current, Mapping):
            for key, item in current.items():
                pending.extend((key, item))
        else:
            pending.extend(current)


def _canonical_json_bytes_checked(value: Any, code: str) -> bytes:
    try:
        return canonical_json_bytes(value)
    except ContractViolation:
        raise
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ContractViolation(code) from exc


def _loads_object(raw: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ContractViolation:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ContractViolation(code) from exc
    if not isinstance(value, dict):
        raise ContractViolation(code)
    _reject_unicode_surrogates(value)
    return value


def _exact_bool(value: Any, expected: bool, code: str) -> None:
    if value is not expected:
        raise ContractViolation(code)


def _exact_int(value: Any, expected: int, code: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise ContractViolation(code)


def _safe_text(
    value: Any,
    code: str,
    *,
    allow_empty: bool = False,
    limit: int = 8000,
) -> str:
    _reject_unicode_surrogates(value)
    if (
        not isinstance(value, str)
        or "\x00" in value
        or len(value) > limit
        or (not allow_empty and not value.strip())
    ):
        raise ContractViolation(code)
    if _FORMAL_APPROVAL_TOKEN_RE.search(value):
        raise ContractViolation("REVIEW_INTAKE_FORMAL_APPROVAL_TOKEN_FORBIDDEN")
    return value


def _validate_reviewer_id(value: Any) -> str:
    reviewer_id = _safe_text(value, "REVIEW_INTAKE_INVALID_REVIEWER_ID", limit=64)
    if _REVIEWER_ID_RE.fullmatch(reviewer_id) is None:
        raise ContractViolation("REVIEW_INTAKE_INVALID_REVIEWER_ID")
    if reviewer_id != "jm02":
        raise ContractViolation("REVIEW_INTAKE_REVIEWER_ID_MISMATCH")
    return reviewer_id


def _validate_review_date(value: Any) -> str:
    text = _safe_text(value, "REVIEW_INTAKE_INVALID_REVIEW_DATE", limit=10)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ContractViolation("REVIEW_INTAKE_INVALID_REVIEW_DATE") from exc
    if parsed.isoformat() != text:
        raise ContractViolation("REVIEW_INTAKE_INVALID_REVIEW_DATE")
    if parsed > datetime.now(timezone.utc).date():
        raise ContractViolation("REVIEW_INTAKE_REVIEW_DATE_IN_FUTURE")
    return text


def _validate_recorded_at_utc(value: Any) -> tuple[str, datetime]:
    text = _safe_text(value, "REVIEW_INTAKE_INVALID_RECORDED_AT", limit=40)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractViolation("REVIEW_INTAKE_INVALID_RECORDED_AT") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractViolation("REVIEW_INTAKE_INVALID_RECORDED_AT")
    canonical = parsed.isoformat(timespec="seconds").replace("+00:00", "Z")
    if canonical != text:
        raise ContractViolation("REVIEW_INTAKE_INVALID_RECORDED_AT")
    if parsed > datetime.now(timezone.utc):
        raise ContractViolation("REVIEW_INTAKE_RECORDED_AT_IN_FUTURE")
    return text, parsed


def _artifact_ref(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _read_regular_file_bytes_capped(
    path: Path,
    *,
    maximum_bytes: int,
    too_large_code: str,
) -> bytes:
    """Read one regular-file descriptor after bounding its allocation size."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractViolation("REVIEW_INTAKE_ARTIFACT_NOT_REGULAR_FILE") from exc
    try:
        try:
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise ContractViolation("REVIEW_INTAKE_ARTIFACT_NOT_REGULAR_FILE") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractViolation("REVIEW_INTAKE_ARTIFACT_NOT_REGULAR_FILE")
        if metadata.st_size < 0 or metadata.st_size > maximum_bytes:
            raise ContractViolation(too_large_code)
        remaining = metadata.st_size
        chunks: list[bytes] = []
        while remaining:
            try:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
            except OSError as exc:
                raise ContractViolation("REVIEW_INTAKE_ARTIFACT_READ_FAILED") from exc
            if not chunk:
                raise ContractViolation("REVIEW_INTAKE_ARTIFACT_CHANGED_DURING_READ")
            chunks.append(chunk)
            remaining -= len(chunk)
        try:
            trailing = os.read(descriptor, 1)
        except OSError as exc:
            raise ContractViolation("REVIEW_INTAKE_ARTIFACT_READ_FAILED") from exc
        if trailing:
            raise ContractViolation("REVIEW_INTAKE_ARTIFACT_CHANGED_DURING_READ")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_source_path(
    path: Path,
    *,
    scope_root: Path,
    source_name: str,
    code: str,
) -> tuple[Path, bytes, dict[str, Any]]:
    try:
        resolved = require_relative_to(Path(path), scope_root, code)
    except OSError as exc:
        raise ContractViolation(code) from exc
    raw = _read_regular_file_bytes_capped(
        resolved,
        maximum_bytes=_SOURCE_MAX_BYTES[source_name],
        too_large_code="REVIEW_INTAKE_SOURCE_TOO_LARGE",
    )
    return resolved, raw, _artifact_ref(resolved, raw)


def _read_artifact_ref(
    value: Any,
    *,
    scope_root: Path,
    source_name: str,
) -> tuple[Path, bytes]:
    code = f"REVIEW_INTAKE_INVALID_{source_name.upper()}_REF"
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_REF_FIELDS:
        raise ContractViolation(code)
    raw_path = value.get("path")
    expected_bytes = value.get("bytes")
    claimed_hash = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractViolation(code)
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise ContractViolation(code)
    if expected_bytes > _SOURCE_MAX_BYTES[source_name]:
        raise ContractViolation("REVIEW_INTAKE_SOURCE_TOO_LARGE")
    if not isinstance(claimed_hash, str):
        raise ContractViolation(code)
    expected_hash = require_sha256(claimed_hash, code)
    try:
        path = require_relative_to(
            Path(raw_path),
            scope_root,
            "REVIEW_INTAKE_SOURCE_OUTSIDE_SCOPE",
        )
    except OSError as exc:
        raise ContractViolation("REVIEW_INTAKE_SOURCE_OUTSIDE_SCOPE") from exc
    if raw_path != str(path):
        raise ContractViolation("REVIEW_INTAKE_SOURCE_PATH_NOT_CANONICAL")
    raw = read_regular_file_bytes_exact(path, expected_bytes=expected_bytes)
    if sha256_bytes(raw) != expected_hash:
        raise ContractViolation("REVIEW_INTAKE_SOURCE_HASH_MISMATCH")
    return path, raw


def _require_expected_reviewed_source_profile(source_artifacts: Any) -> None:
    """Require the exact four artifacts against which jm02 made decisions."""
    try:
        expected = {
            name: {"path": path, "sha256": digest, "bytes": byte_count}
            for name, path, digest, byte_count in EXPECTED_REVIEWED_SOURCE_PROFILE
        }
    except (TypeError, ValueError) as exc:  # defensive code/profile integrity
        raise ContractViolation(
            "REVIEW_INTAKE_INVALID_EXPECTED_SOURCE_PROFILE"
        ) from exc
    if set(expected) != set(SOURCE_NAMES) or len(expected) != len(
        EXPECTED_REVIEWED_SOURCE_PROFILE
    ):
        raise ContractViolation("REVIEW_INTAKE_INVALID_EXPECTED_SOURCE_PROFILE")
    for source_name in SOURCE_NAMES:
        ref = expected[source_name]
        if (
            not isinstance(ref["path"], str)
            or not Path(ref["path"]).is_absolute()
            or not isinstance(ref["bytes"], int)
            or isinstance(ref["bytes"], bool)
            or ref["bytes"] < 0
            or ref["bytes"] > _SOURCE_MAX_BYTES[source_name]
        ):
            raise ContractViolation("REVIEW_INTAKE_INVALID_EXPECTED_SOURCE_PROFILE")
        require_sha256(
            ref["sha256"], "REVIEW_INTAKE_INVALID_EXPECTED_SOURCE_PROFILE"
        )
    if not isinstance(source_artifacts, Mapping) or set(source_artifacts) != set(
        SOURCE_NAMES
    ):
        raise ContractViolation("REVIEW_INTAKE_SOURCE_SET_MISMATCH")
    for source_name in SOURCE_NAMES:
        actual = source_artifacts[source_name]
        if (
            not isinstance(actual, Mapping)
            or set(actual) != _ARTIFACT_REF_FIELDS
            or any(
                actual.get(field) != value
                for field, value in expected[source_name].items()
            )
        ):
            raise ContractViolation("REVIEW_INTAKE_SOURCE_PROFILE_MISMATCH")


def _load_advisory_rows(raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractViolation("REVIEW_INTAKE_INVALID_ADVISORY_JSONL") from exc
    lines = text.splitlines()
    if len(lines) != ROW_COUNT or any(not line.strip() for line in lines):
        raise ContractViolation("REVIEW_INTAKE_ADVISORY_COUNT_MISMATCH")
    return [
        _loads_object(line.encode("utf-8"), "REVIEW_INTAKE_INVALID_ADVISORY_JSONL")
        for line in lines
    ]


def _load_and_validate_sources(
    source_artifacts: Any,
    *,
    scope_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(source_artifacts, Mapping) or set(source_artifacts) != set(
        SOURCE_NAMES
    ):
        raise ContractViolation("REVIEW_INTAKE_SOURCE_SET_MISMATCH")

    loaded: dict[str, tuple[Path, bytes]] = {}
    for source_name in SOURCE_NAMES:
        loaded[source_name] = _read_artifact_ref(
            source_artifacts[source_name],
            scope_root=scope_root,
            source_name=source_name,
        )
    if len({path for path, _raw in loaded.values()}) != len(SOURCE_NAMES):
        raise ContractViolation("REVIEW_INTAKE_SOURCE_PATHS_MUST_BE_DISTINCT")

    plan_path, plan_raw = loaded["selection_plan"]
    plan = _loads_object(plan_raw, "REVIEW_INTAKE_INVALID_SELECTION_PLAN_JSON")
    if (
        _canonical_json_bytes_checked(
            plan, "REVIEW_INTAKE_INVALID_SELECTION_PLAN_JSON"
        )
        != plan_raw
    ):
        raise ContractViolation("REVIEW_INTAKE_NONCANONICAL_SELECTION_PLAN")
    ai_prereview.validate_ai_prereview_selection_plan(plan, scope_root=scope_root)

    advisory_path, advisory_raw = loaded["advisory"]
    advisory_rows = _load_advisory_rows(advisory_raw)
    checked_rows, _evidence_count, _captured_count = (
        ai_prereview._validate_advisory_rows(  # noqa: SLF001 - shared contract
            advisory_rows,
            plan,
            scope_root=scope_root,
        )
    )
    if ai_prereview._canonical_jsonl(checked_rows) != advisory_raw:  # noqa: SLF001
        raise ContractViolation("REVIEW_INTAKE_NONCANONICAL_ADVISORY")

    bundle_manifest_path, bundle_manifest_raw = loaded["bundle_manifest"]
    bundle_manifest = _loads_object(
        bundle_manifest_raw, "REVIEW_INTAKE_INVALID_AI_BUNDLE_MANIFEST"
    )
    if (
        _canonical_json_bytes_checked(
            bundle_manifest, "REVIEW_INTAKE_INVALID_AI_BUNDLE_MANIFEST"
        )
        != bundle_manifest_raw
    ):
        raise ContractViolation("REVIEW_INTAKE_NONCANONICAL_AI_BUNDLE_MANIFEST")
    for field, maximum in (
        ("selection_commitment_source", _SOURCE_MAX_BYTES["selection_plan"]),
        ("selection_plan_artifact", _SOURCE_MAX_BYTES["selection_plan"]),
        ("advisory_artifact", _SOURCE_MAX_BYTES["advisory"]),
    ):
        ref = bundle_manifest.get(field)
        if (
            not isinstance(ref, Mapping)
            or not isinstance(ref.get("bytes"), int)
            or isinstance(ref.get("bytes"), bool)
            or ref["bytes"] < 0
        ):
            raise ContractViolation("REVIEW_INTAKE_INVALID_AI_BUNDLE_MANIFEST")
        if ref["bytes"] > maximum:
            raise ContractViolation("REVIEW_INTAKE_SOURCE_TOO_LARGE")

    # The original advisory auditor validates its full manifest semantics,
    # immutable source commitment, row evidence candidates, and read-only
    # three-file bundle.  The intake additionally requires the exact plan and
    # advisory paths supplied above to be the ones named by that manifest.
    advisory_audit = ai_prereview.audit_ai_prereview_bundle(
        bundle_manifest_path, scope_root=scope_root
    )
    if (
        advisory_audit.get("status") != "PASS_AI_ADVISORY_ONLY"
        or advisory_audit.get("manifest_sha256")
        != source_artifacts["bundle_manifest"]["sha256"]
    ):
        raise ContractViolation("REVIEW_INTAKE_AI_BUNDLE_AUDIT_FAILED")
    if (
        bundle_manifest_path.name != "ai_prereview_manifest.json"
        or plan_path != bundle_manifest_path.parent / "selection_plan.json"
        or advisory_path != bundle_manifest_path.parent / "ai_prereview.jsonl"
        or bundle_manifest.get("selection_plan_artifact")
        != source_artifacts["selection_plan"]
        or bundle_manifest.get("advisory_artifact") != source_artifacts["advisory"]
    ):
        raise ContractViolation("REVIEW_INTAKE_AI_BUNDLE_PATH_MISMATCH")

    summary_path, summary_raw = loaded["summary"]
    bundle_match = re.fullmatch(r"advisory_bundle_([A-Za-z0-9._-]+)", bundle_manifest_path.parent.name)
    if (
        bundle_match is None
        or summary_path.parent != bundle_manifest_path.parent.parent
        or summary_path.name
        != f"ai_prereview_summary_{bundle_match.group(1)}.md"
    ):
        raise ContractViolation("REVIEW_INTAKE_SUMMARY_PATH_MISMATCH")
    if not summary_raw:
        raise ContractViolation("REVIEW_INTAKE_INVALID_SUMMARY")
    try:
        summary_text = summary_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractViolation("REVIEW_INTAKE_INVALID_SUMMARY") from exc
    _reject_unicode_surrogates(summary_text)
    if not summary_text.strip() or "\x00" in summary_text:
        raise ContractViolation("REVIEW_INTAKE_INVALID_SUMMARY")
    if _FORMAL_APPROVAL_TOKEN_RE.search(summary_text):
        raise ContractViolation("REVIEW_INTAKE_FORMAL_APPROVAL_TOKEN_FORBIDDEN")
    try:
        summary_mode = stat.S_IMODE(summary_path.stat(follow_symlinks=False).st_mode)
    except OSError as exc:
        raise ContractViolation("REVIEW_INTAKE_SOURCE_READ_ONLY_REQUIRED") from exc
    if summary_mode & 0o222:
        raise ContractViolation("REVIEW_INTAKE_SOURCE_READ_ONLY_REQUIRED")

    # Structural validity is not identity: the ordinal-specific jm02 map must
    # never be applied to another otherwise valid advisory bundle or display.
    _require_expected_reviewed_source_profile(source_artifacts)
    return plan, checked_rows


def _validate_disposition(value: Any, code: str) -> str:
    if value not in DISPOSITIONS:
        raise ContractViolation(code)
    return str(value)


def _overall_from_components(
    meaning: str,
    term_quality: Mapping[str, Mapping[str, Any]],
    etymology: str,
) -> str:
    values = [
        meaning,
        etymology,
        *(str(term_quality[language]["disposition"]) for language in LANGUAGES),
    ]
    if "REVISION_REQUIRED" in values:
        return "REVISION_REQUIRED"
    if "DEFERRED" in values:
        return "DEFERRED"
    return "APPROVED"


def _empty_counts() -> dict[str, dict[str, int]]:
    return {
        dimension: {disposition: 0 for disposition in DISPOSITIONS}
        for dimension in _COUNT_DIMENSIONS
    }


def _expected_adjudication(ordinal: int) -> tuple[str, str]:
    return _SPECIAL_ADJUDICATIONS.get(ordinal, _DEFAULT_APPROVAL)


def _validate_reason_counts(value: Any, observed: Mapping[str, int]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(
        _EXPECTED_REASON_CODE_COUNTS
    ):
        raise ContractViolation("REVIEW_INTAKE_REASON_COUNT_SCHEMA_MISMATCH")
    for reason_code, expected in _EXPECTED_REASON_CODE_COUNTS.items():
        count = value.get(reason_code)
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count != expected
            or observed.get(reason_code) != expected
        ):
            raise ContractViolation("REVIEW_INTAKE_REASON_COUNT_MISMATCH")


def _validate_rows(
    rows: Any,
    plan: Mapping[str, Any],
    advisory_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, int]]:
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise ContractViolation("REVIEW_INTAKE_ROW_COUNT_MISMATCH")
    selected = plan.get("selected")
    if not isinstance(selected, list) or len(selected) != ROW_COUNT:
        raise ContractViolation("REVIEW_INTAKE_SELECTION_COUNT_MISMATCH")
    if len(advisory_rows) != ROW_COUNT:
        raise ContractViolation("REVIEW_INTAKE_ADVISORY_COUNT_MISMATCH")

    checked: list[dict[str, Any]] = []
    counts = _empty_counts()
    reason_counts = {reason_code: 0 for reason_code in _EXPECTED_REASON_CODE_COUNTS}
    for ordinal, (row, selection, advisory_row) in enumerate(
        zip(rows, selected, advisory_rows), 1
    ):
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise ContractViolation("REVIEW_INTAKE_ROW_SCHEMA_MISMATCH")
        _exact_int(
            row.get("selection_ordinal"),
            ordinal,
            "REVIEW_INTAKE_SELECTION_ORDER_MISMATCH",
        )
        candidate_id = row.get("candidate_id")
        if candidate_id != selection.get("candidate_id"):
            raise ContractViolation("REVIEW_INTAKE_CANDIDATE_ORDER_MISMATCH")
        _exact_int(
            row.get("review_order"),
            selection.get("review_order"),
            "REVIEW_INTAKE_REVIEW_ORDER_MISMATCH",
        )
        if row.get("candidate_sha256") != selection.get("candidate_sha256"):
            raise ContractViolation("REVIEW_INTAKE_CANDIDATE_HASH_MISMATCH")

        overall = _validate_disposition(
            row.get("overall_disposition"),
            "REVIEW_INTAKE_INVALID_OVERALL_DISPOSITION",
        )

        meaning_raw = row.get("meaning")
        if not isinstance(meaning_raw, Mapping) or set(meaning_raw) != _MEANING_FIELDS:
            raise ContractViolation("REVIEW_INTAKE_MEANING_SCHEMA_MISMATCH")
        meaning_disposition = _validate_disposition(
            meaning_raw.get("disposition"),
            "REVIEW_INTAKE_INVALID_MEANING_DISPOSITION",
        )
        meaning_value = meaning_raw.get("value")
        if meaning_disposition == "APPROVED":
            if meaning_value != advisory_row["suggested_meaning_alignment"]:
                raise ContractViolation("REVIEW_INTAKE_MEANING_VALUE_MISMATCH")
        elif meaning_value is not None:
            raise ContractViolation("REVIEW_INTAKE_UNACCEPTED_MEANING_VALUE_FORBIDDEN")
        meaning = {"disposition": meaning_disposition, "value": meaning_value}

        term_quality_raw = row.get("term_quality")
        if not isinstance(term_quality_raw, Mapping) or set(term_quality_raw) != set(
            LANGUAGES
        ):
            raise ContractViolation("REVIEW_INTAKE_TERM_QUALITY_SCHEMA_MISMATCH")
        term_quality: dict[str, dict[str, Any]] = {}
        for language in LANGUAGES:
            term_raw = term_quality_raw[language]
            if (
                not isinstance(term_raw, Mapping)
                or set(term_raw) != _TERM_QUALITY_FIELDS
            ):
                raise ContractViolation("REVIEW_INTAKE_TERM_QUALITY_SCHEMA_MISMATCH")
            disposition = _validate_disposition(
                term_raw.get("disposition"),
                "REVIEW_INTAKE_INVALID_TERM_QUALITY_DISPOSITION",
            )
            value = term_raw.get("value")
            if disposition == "APPROVED":
                if value != advisory_row["suggested_term_quality"][language]:
                    raise ContractViolation("REVIEW_INTAKE_TERM_QUALITY_VALUE_MISMATCH")
            elif value is not None:
                raise ContractViolation(
                    "REVIEW_INTAKE_UNACCEPTED_TERM_QUALITY_VALUE_FORBIDDEN"
                )
            term_quality[language] = {"disposition": disposition, "value": value}

        etymology_raw = row.get("etymology")
        if (
            not isinstance(etymology_raw, Mapping)
            or set(etymology_raw) != _ETYMOLOGY_FIELDS
        ):
            raise ContractViolation("REVIEW_INTAKE_ETYMOLOGY_SCHEMA_MISMATCH")
        etymology_disposition = _validate_disposition(
            etymology_raw.get("disposition"),
            "REVIEW_INTAKE_INVALID_ETYMOLOGY_DISPOSITION",
        )
        etymology_values = {
            "subtype": etymology_raw.get("subtype"),
            "direction": etymology_raw.get("direction"),
        }
        expected_etymology = {
            "subtype": advisory_row["suggested_etymology_subtype"],
            "direction": advisory_row["suggested_relation_direction"],
        }
        if etymology_disposition == "APPROVED":
            if etymology_values != expected_etymology:
                raise ContractViolation("REVIEW_INTAKE_ETYMOLOGY_VALUE_MISMATCH")
        elif any(value is not None for value in etymology_values.values()):
            raise ContractViolation("REVIEW_INTAKE_UNACCEPTED_ETYMOLOGY_VALUES_FORBIDDEN")
        etymology = {"disposition": etymology_disposition, **etymology_values}

        if overall != _overall_from_components(
            meaning_disposition, term_quality, etymology_disposition
        ):
            raise ContractViolation("REVIEW_INTAKE_OVERALL_DISPOSITION_MISMATCH")
        expected_overall = "DEFERRED" if ordinal in _DEFERRED_ORDINALS else "APPROVED"
        if overall != expected_overall:
            raise ContractViolation("REVIEW_INTAKE_CONFIRMED_DISPOSITION_MISMATCH")
        if meaning_disposition != "APPROVED":
            raise ContractViolation("REVIEW_INTAKE_CONFIRMED_MEANING_MISMATCH")
        for language in LANGUAGES:
            expected_term_disposition = (
                "DEFERRED" if ordinal == 38 and language == "en" else "APPROVED"
            )
            if term_quality[language]["disposition"] != expected_term_disposition:
                raise ContractViolation("REVIEW_INTAKE_CONFIRMED_TERM_QUALITY_MISMATCH")
        expected_etymology_disposition = (
            "DEFERRED" if ordinal in _DEFERRED_ORDINALS else "APPROVED"
        )
        if etymology_disposition != expected_etymology_disposition:
            raise ContractViolation("REVIEW_INTAKE_CONFIRMED_ETYMOLOGY_MISMATCH")

        expected_reason_code, expected_note = _expected_adjudication(ordinal)
        reason_code = row.get("reason_code")
        if reason_code != expected_reason_code or reason_code not in REASON_CODES:
            raise ContractViolation("REVIEW_INTAKE_REASON_CODE_MISMATCH")
        note = _safe_text(
            row.get("intake_note"),
            "REVIEW_INTAKE_INVALID_NOTE",
            allow_empty=True,
        )
        if _RESERVED_NOTE_CLAIMS.search(note):
            raise ContractViolation("REVIEW_INTAKE_RESERVED_CLAIM_FORBIDDEN")
        if note != expected_note:
            raise ContractViolation("REVIEW_INTAKE_REASON_NOTE_MISMATCH")

        counts["overall"][overall] += 1
        counts["meaning"][meaning_disposition] += 1
        counts["etymology"][etymology_disposition] += 1
        for language in LANGUAGES:
            counts["term_quality"][term_quality[language]["disposition"]] += 1
        reason_counts[reason_code] += 1
        checked.append(
            {
                "selection_ordinal": ordinal,
                "review_order": selection["review_order"],
                "candidate_id": selection["candidate_id"],
                "candidate_sha256": selection["candidate_sha256"],
                "overall_disposition": overall,
                "meaning": meaning,
                "term_quality": term_quality,
                "etymology": etymology,
                "reason_code": reason_code,
                "intake_note": note,
            }
        )
    if reason_counts != _EXPECTED_REASON_CODE_COUNTS:
        raise ContractViolation("REVIEW_INTAKE_REASON_COUNT_MISMATCH")
    return checked, counts, reason_counts


def _validate_counts(value: Any, expected: Mapping[str, Mapping[str, int]]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_COUNT_DIMENSIONS):
        raise ContractViolation("REVIEW_INTAKE_COUNT_SCHEMA_MISMATCH")
    for dimension in _COUNT_DIMENSIONS:
        dimension_counts = value.get(dimension)
        if not isinstance(dimension_counts, Mapping) or set(dimension_counts) != set(
            DISPOSITIONS
        ):
            raise ContractViolation("REVIEW_INTAKE_COUNT_SCHEMA_MISMATCH")
        for disposition in DISPOSITIONS:
            observed = dimension_counts.get(disposition)
            wanted = expected[dimension][disposition]
            if (
                not isinstance(observed, int)
                or isinstance(observed, bool)
                or observed != wanted
            ):
                raise ContractViolation("REVIEW_INTAKE_COUNT_MISMATCH")


def validate_review_feedback_intake(
    payload: Mapping[str, Any],
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Validate one exact 60-row non-production feedback intake object."""
    _reject_unicode_surrogates(payload)
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ContractViolation("REVIEW_INTAKE_SCHEMA_MISMATCH")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractViolation("REVIEW_INTAKE_VERSION_MISMATCH")
    if payload.get("spec_version") != VERSION:
        raise ContractViolation("REVIEW_INTAKE_SPEC_VERSION_MISMATCH")
    if payload.get("status") != RECORDED_STATUS:
        raise ContractViolation("REVIEW_INTAKE_STATUS_MISMATCH")
    _exact_bool(
        payload.get("non_production_intake"),
        True,
        "REVIEW_INTAKE_NON_PRODUCTION_REQUIRED",
    )
    for field, code in (
        ("identity_authentication_claimed", "REVIEW_INTAKE_IDENTITY_CLAIM_FORBIDDEN"),
        ("formal_research_approval_claimed", "REVIEW_INTAKE_FORMAL_APPROVAL_FORBIDDEN"),
        ("production_evidence_verified", "REVIEW_INTAKE_PRODUCTION_EVIDENCE_FORBIDDEN"),
        ("direct_csv_input_allowed", "REVIEW_INTAKE_DIRECT_CSV_INPUT_FORBIDDEN"),
        ("review_csvs_modified", "REVIEW_INTAKE_CSV_MODIFICATION_FORBIDDEN"),
        ("evidence_records_created", "REVIEW_INTAKE_EVIDENCE_RECORD_FORBIDDEN"),
        ("annotation_freeze_created", "REVIEW_INTAKE_FREEZE_CLAIM_FORBIDDEN"),
        ("training_eligible", "REVIEW_INTAKE_TRAINING_ELIGIBILITY_FORBIDDEN"),
    ):
        _exact_bool(payload.get(field), False, code)
    _exact_bool(
        payload.get("requires_formal_evidence_and_csv_workflow"),
        True,
        "REVIEW_INTAKE_FORMAL_WORKFLOW_REQUIRED",
    )
    _exact_int(
        payload.get("row_count"), ROW_COUNT, "REVIEW_INTAKE_ROW_COUNT_MISMATCH"
    )
    reviewer_id = _validate_reviewer_id(payload.get("reviewer_id"))
    review_date = _validate_review_date(payload.get("review_date"))
    recorded_at_text, recorded_at = _validate_recorded_at_utc(
        payload.get("recorded_at_utc")
    )
    if payload.get("notice") != NOTICE:
        raise ContractViolation("REVIEW_INTAKE_NOTICE_MISMATCH")

    plan, advisory_rows = _load_and_validate_sources(
        payload.get("source_artifacts"), scope_root=scope_root
    )
    if len(advisory_rows) != ROW_COUNT:  # defensive; source validator checks this
        raise ContractViolation("REVIEW_INTAKE_ADVISORY_COUNT_MISMATCH")
    try:
        plan_created_at = datetime.fromisoformat(
            str(plan["created_at_utc"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as exc:  # validated upstream
        raise ContractViolation("REVIEW_INTAKE_INVALID_SELECTION_PLAN_DATE") from exc
    if date.fromisoformat(review_date) < plan_created_at.date():
        raise ContractViolation("REVIEW_INTAKE_DATE_PRECEDES_SELECTION")
    if date.fromisoformat(review_date) > recorded_at.date():
        raise ContractViolation("REVIEW_INTAKE_DATE_AFTER_RECORDED_AT")
    if recorded_at < plan_created_at:
        raise ContractViolation("REVIEW_INTAKE_RECORDED_AT_PRECEDES_SELECTION")

    checked_rows, counts, reason_counts = _validate_rows(
        payload.get("rows"), plan, advisory_rows
    )
    _validate_counts(payload.get("decision_counts"), counts)
    _validate_reason_counts(payload.get("reason_code_counts"), reason_counts)
    return {
        "status": AUDIT_STATUS,
        "recorded_status": RECORDED_STATUS,
        "schema_version": SCHEMA_VERSION,
        "spec_version": VERSION,
        "non_production_intake": True,
        "reviewer_id": reviewer_id,
        "review_date": review_date,
        "recorded_at_utc": recorded_at_text,
        "identity_authentication_claimed": False,
        "formal_research_approval_claimed": False,
        "production_evidence_verified": False,
        "direct_csv_input_allowed": False,
        "review_csvs_modified": False,
        "evidence_records_created": False,
        "annotation_freeze_created": False,
        "training_eligible": False,
        "row_count": len(checked_rows),
        "decision_counts": counts,
        "reason_code_counts": reason_counts,
        "requires_formal_evidence_and_csv_workflow": True,
    }


def build_review_feedback_intake(
    selection_plan_path: Path,
    advisory_path: Path,
    advisory_bundle_manifest_path: Path,
    summary_path: Path,
    feedback_rows: Sequence[Mapping[str, Any]],
    *,
    reviewer_id: str,
    review_date: str,
    recorded_at_utc: str,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Build, but do not publish, a source-bound feedback intake object."""
    source_artifacts: dict[str, dict[str, Any]] = {}
    for source_name, source_path in (
        ("selection_plan", selection_plan_path),
        ("advisory", advisory_path),
        ("bundle_manifest", advisory_bundle_manifest_path),
        ("summary", summary_path),
    ):
        _resolved, _raw, ref = _read_source_path(
            source_path,
            scope_root=scope_root,
            source_name=source_name,
            code="REVIEW_INTAKE_SOURCE_OUTSIDE_SCOPE",
        )
        source_artifacts[source_name] = ref

    # Load the source plan before expanding caller feedback.  Caller rows bind
    # both ordinal and candidate ID; the immutable hash/order fields always
    # come from the selection commitment, never from free-form feedback.
    plan, advisory_rows = _load_and_validate_sources(
        source_artifacts, scope_root=scope_root
    )
    if not isinstance(feedback_rows, Sequence) or isinstance(
        feedback_rows, (str, bytes, bytearray)
    ):
        raise ContractViolation("REVIEW_INTAKE_ROW_COUNT_MISMATCH")
    if len(feedback_rows) != ROW_COUNT:
        raise ContractViolation("REVIEW_INTAKE_ROW_COUNT_MISMATCH")

    expanded_rows: list[dict[str, Any]] = []
    for ordinal, (feedback, selection) in enumerate(
        zip(feedback_rows, plan["selected"]), 1
    ):
        if not isinstance(feedback, Mapping) or set(feedback) != _FEEDBACK_INPUT_FIELDS:
            raise ContractViolation("REVIEW_INTAKE_FEEDBACK_SCHEMA_MISMATCH")
        if (
            feedback.get("selection_ordinal") != ordinal
            or feedback.get("candidate_id") != selection["candidate_id"]
        ):
            raise ContractViolation("REVIEW_INTAKE_FEEDBACK_ORDER_MISMATCH")
        expanded_rows.append(
            {
                **dict(feedback),
                "review_order": selection["review_order"],
                "candidate_sha256": selection["candidate_sha256"],
            }
        )

    checked_rows, counts, reason_counts = _validate_rows(
        expanded_rows, plan, advisory_rows
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "spec_version": VERSION,
        "status": RECORDED_STATUS,
        "non_production_intake": True,
        "reviewer_id": reviewer_id,
        "review_date": review_date,
        "recorded_at_utc": recorded_at_utc,
        "identity_authentication_claimed": False,
        "formal_research_approval_claimed": False,
        "production_evidence_verified": False,
        "direct_csv_input_allowed": False,
        "review_csvs_modified": False,
        "evidence_records_created": False,
        "annotation_freeze_created": False,
        "training_eligible": False,
        "requires_formal_evidence_and_csv_workflow": True,
        "source_artifacts": source_artifacts,
        "row_count": ROW_COUNT,
        "decision_counts": counts,
        "reason_code_counts": reason_counts,
        "rows": checked_rows,
        "notice": NOTICE,
    }
    validate_review_feedback_intake(payload, scope_root=scope_root)
    return payload


def _build_bundle_manifest(
    payload: Mapping[str, Any], intake_ref: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
        "spec_version": VERSION,
        "status": RECORDED_STATUS,
        "non_production_intake": True,
        "training_eligible": False,
        "intake_artifact": dict(intake_ref),
        "source_artifacts": dict(payload["source_artifacts"]),
        "row_count": ROW_COUNT,
        "decision_counts": dict(payload["decision_counts"]),
        "reason_code_counts": dict(payload["reason_code_counts"]),
        "reviewer_id": payload["reviewer_id"],
        "review_date": payload["review_date"],
        "write_semantics": "ATOMIC_DIRECTORY_NOREPLACE_READ_ONLY",
    }


def _cleanup_private_stage(stage: Path) -> None:
    for name in (BUNDLE_MANIFEST_FILENAME, INTAKE_FILENAME):
        try:
            (stage / name).unlink()
        except FileNotFoundError:
            pass
    try:
        stage.rmdir()
    except FileNotFoundError:
        pass


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    from .prepare_data import _rename_directory_noreplace

    try:
        _rename_directory_noreplace(source, destination)
    except ContractViolation as exc:
        mapped = {
            "ANNOTATION_FREEZE_OUTPUT_EXISTS": "REVIEW_INTAKE_OUTPUT_EXISTS",
            "ANNOTATION_FREEZE_ATOMIC_NOREPLACE_UNSUPPORTED": (
                "REVIEW_INTAKE_ATOMIC_NOREPLACE_UNSUPPORTED"
            ),
            "ANNOTATION_FREEZE_ATOMIC_PUBLISH_FAILED": (
                "REVIEW_INTAKE_ATOMIC_PUBLISH_FAILED"
            ),
            "ANNOTATION_FREEZE_PARENT_FSYNC_FAILED": (
                "REVIEW_INTAKE_PARENT_FSYNC_FAILED"
            ),
        }.get(exc.code, "REVIEW_INTAKE_ATOMIC_PUBLISH_FAILED")
        raise ContractViolation(mapped) from exc


def _require_exact_read_only_bundle(directory: Path) -> None:
    try:
        if directory.is_symlink():
            raise ContractViolation("REVIEW_INTAKE_READ_ONLY_BUNDLE_REQUIRED")
        directory_mode = stat.S_IMODE(
            directory.stat(follow_symlinks=False).st_mode
        )
        entries = []
        with os.scandir(directory) as iterator:
            for entry in iterator:
                entries.append(entry)
                if len(entries) > 2:
                    raise ContractViolation(
                        "REVIEW_INTAKE_READ_ONLY_BUNDLE_REQUIRED"
                    )
    except ContractViolation:
        raise
    except OSError as exc:
        raise ContractViolation("REVIEW_INTAKE_READ_ONLY_BUNDLE_REQUIRED") from exc
    expected = {INTAKE_FILENAME, BUNDLE_MANIFEST_FILENAME}
    if (
        directory_mode & 0o222
        or {entry.name for entry in entries} != expected
        or any(not entry.is_file(follow_symlinks=False) for entry in entries)
    ):
        raise ContractViolation("REVIEW_INTAKE_READ_ONLY_BUNDLE_REQUIRED")
    try:
        modes = [
            stat.S_IMODE((directory / name).stat(follow_symlinks=False).st_mode)
            for name in expected
        ]
    except OSError as exc:
        raise ContractViolation("REVIEW_INTAKE_READ_ONLY_BUNDLE_REQUIRED") from exc
    if any(mode & 0o222 for mode in modes):
        raise ContractViolation("REVIEW_INTAKE_READ_ONLY_BUNDLE_REQUIRED")


def _read_intake_ref(
    value: Any, *, scope_root: Path
) -> tuple[Path, bytes]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_REF_FIELDS:
        raise ContractViolation("REVIEW_INTAKE_INVALID_INTERNAL_ARTIFACT_REF")
    expected_bytes = value.get("bytes")
    raw_path = value.get("path")
    claimed_hash = value.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
        or expected_bytes > _INTAKE_MAX_BYTES
        or not isinstance(claimed_hash, str)
    ):
        raise ContractViolation("REVIEW_INTAKE_INVALID_INTERNAL_ARTIFACT_REF")
    expected_hash = require_sha256(
        claimed_hash, "REVIEW_INTAKE_INVALID_INTERNAL_ARTIFACT_REF"
    )
    path = require_relative_to(
        Path(raw_path), scope_root, "REVIEW_INTAKE_ARTIFACT_OUTSIDE_SCOPE"
    )
    if raw_path != str(path):
        raise ContractViolation("REVIEW_INTAKE_ARTIFACT_PATH_NOT_CANONICAL")
    raw = read_regular_file_bytes_exact(path, expected_bytes=expected_bytes)
    if sha256_bytes(raw) != expected_hash:
        raise ContractViolation("REVIEW_INTAKE_ARTIFACT_HASH_MISMATCH")
    return path, raw


def publish_review_feedback_intake(
    payload: Mapping[str, Any],
    output_dir: Path,
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Atomically publish an exact two-file, read-only intake directory."""
    _reject_unicode_surrogates(payload)
    try:
        snapshot_value = dict(payload)
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ContractViolation("REVIEW_INTAKE_INVALID_JSON") from exc
    snapshot_raw = _canonical_json_bytes_checked(
        snapshot_value, "REVIEW_INTAKE_INVALID_JSON"
    )
    snapshot = _loads_object(snapshot_raw, "REVIEW_INTAKE_INVALID_JSON")
    summary = validate_review_feedback_intake(snapshot, scope_root=scope_root)
    destination = require_relative_to(
        Path(output_dir), scope_root, "REVIEW_INTAKE_OUTPUT_OUTSIDE_SCOPE"
    )
    if destination.exists() or destination.is_symlink():
        raise ContractViolation("REVIEW_INTAKE_OUTPUT_EXISTS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    stage.mkdir(mode=0o700)
    intake_ref = _artifact_ref(destination / INTAKE_FILENAME, snapshot_raw)
    manifest = _build_bundle_manifest(snapshot, intake_ref)
    manifest_raw = _canonical_json_bytes_checked(
        manifest, "REVIEW_INTAKE_INVALID_BUNDLE_MANIFEST"
    )
    try:
        publish_bytes_once(stage / INTAKE_FILENAME, snapshot_raw, mode=0o444)
        publish_bytes_once(
            stage / BUNDLE_MANIFEST_FILENAME, manifest_raw, mode=0o444
        )
        # Re-open every external source, then every staged byte, before the
        # atomic directory rename.  This is the final pre-publication hash gate.
        second_summary = validate_review_feedback_intake(
            snapshot, scope_root=scope_root
        )
        staged_intake = read_regular_file_bytes_exact(
            stage / INTAKE_FILENAME, expected_bytes=len(snapshot_raw)
        )
        staged_manifest = read_regular_file_bytes_exact(
            stage / BUNDLE_MANIFEST_FILENAME, expected_bytes=len(manifest_raw)
        )
        if (
            second_summary != summary
            or staged_intake != snapshot_raw
            or staged_manifest != manifest_raw
            or sha256_bytes(staged_intake) != intake_ref["sha256"]
            or _loads_object(
                staged_manifest, "REVIEW_INTAKE_INVALID_BUNDLE_MANIFEST"
            )
            != manifest
        ):
            raise ContractViolation("REVIEW_INTAKE_PREPUBLICATION_RECHECK_FAILED")
        os.chmod(stage, 0o555)
        _publish_directory_noreplace(stage, destination)
    except Exception:
        try:
            os.chmod(stage, 0o700)
        except FileNotFoundError:
            pass
        _cleanup_private_stage(stage)
        raise
    audit = audit_review_feedback_intake(
        destination / BUNDLE_MANIFEST_FILENAME, scope_root=scope_root
    )
    if (
        audit["decision_counts"] != summary["decision_counts"]
        or audit["reason_code_counts"] != summary["reason_code_counts"]
    ):
        raise ContractViolation("REVIEW_INTAKE_PUBLICATION_MISMATCH")
    return audit


def audit_review_feedback_intake(
    manifest_path: Path,
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Audit both files, their exact paths, and every bound external source."""
    manifest_resolved = require_relative_to(
        Path(manifest_path), scope_root, "REVIEW_INTAKE_ARTIFACT_OUTSIDE_SCOPE"
    )
    manifest_raw = _read_regular_file_bytes_capped(
        manifest_resolved,
        maximum_bytes=_BUNDLE_MANIFEST_MAX_BYTES,
        too_large_code="REVIEW_INTAKE_BUNDLE_MANIFEST_TOO_LARGE",
    )
    manifest = _loads_object(
        manifest_raw, "REVIEW_INTAKE_INVALID_BUNDLE_MANIFEST"
    )
    if manifest_raw != _canonical_json_bytes_checked(
        manifest, "REVIEW_INTAKE_INVALID_BUNDLE_MANIFEST"
    ):
        raise ContractViolation("REVIEW_INTAKE_NONCANONICAL_BUNDLE_MANIFEST")
    if set(manifest) != _BUNDLE_MANIFEST_FIELDS:
        raise ContractViolation("REVIEW_INTAKE_BUNDLE_MANIFEST_SCHEMA_MISMATCH")
    if manifest_resolved.name != BUNDLE_MANIFEST_FILENAME:
        raise ContractViolation("REVIEW_INTAKE_BUNDLE_PATH_MISMATCH")
    _require_exact_read_only_bundle(manifest_resolved.parent)
    intake_path, intake_raw = _read_intake_ref(
        manifest.get("intake_artifact"), scope_root=scope_root
    )
    if intake_path != manifest_resolved.parent / INTAKE_FILENAME:
        raise ContractViolation("REVIEW_INTAKE_BUNDLE_PATH_MISMATCH")
    payload = _loads_object(intake_raw, "REVIEW_INTAKE_INVALID_JSON")
    if intake_raw != _canonical_json_bytes_checked(
        payload, "REVIEW_INTAKE_INVALID_JSON"
    ):
        raise ContractViolation("REVIEW_INTAKE_NONCANONICAL_JSON")
    summary = validate_review_feedback_intake(payload, scope_root=scope_root)
    expected_manifest = _build_bundle_manifest(
        payload, _artifact_ref(intake_path, intake_raw)
    )
    if manifest != expected_manifest:
        raise ContractViolation("REVIEW_INTAKE_BUNDLE_MANIFEST_CONTENT_MISMATCH")

    # Final full-byte recheck detects mutation during the audit itself.
    final_manifest = _read_regular_file_bytes_capped(
        manifest_resolved,
        maximum_bytes=_BUNDLE_MANIFEST_MAX_BYTES,
        too_large_code="REVIEW_INTAKE_BUNDLE_MANIFEST_TOO_LARGE",
    )
    final_intake = read_regular_file_bytes_exact(
        intake_path, expected_bytes=len(intake_raw)
    )
    if final_manifest != manifest_raw or final_intake != intake_raw:
        raise ContractViolation("REVIEW_INTAKE_BUNDLE_CHANGED_DURING_AUDIT")
    return {
        **summary,
        "intake_artifact": _artifact_ref(intake_path, intake_raw),
        "manifest_artifact": _artifact_ref(manifest_resolved, manifest_raw),
        "write_semantics": "ATOMIC_DIRECTORY_NOREPLACE_READ_ONLY",
    }


__all__ = [
    "AUDIT_STATUS",
    "DISPOSITIONS",
    "NOTICE",
    "BUNDLE_MANIFEST_FILENAME",
    "BUNDLE_MANIFEST_SCHEMA_VERSION",
    "INTAKE_FILENAME",
    "REASON_CODES",
    "RECORDED_STATUS",
    "ROW_COUNT",
    "SCHEMA_VERSION",
    "audit_review_feedback_intake",
    "build_review_feedback_intake",
    "publish_review_feedback_intake",
    "validate_review_feedback_intake",
]
