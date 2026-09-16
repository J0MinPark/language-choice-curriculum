"""Strict AI-advisory pre-review artifacts for the fixed 60-item cohort.

These artifacts are deliberately outside the human annotation and production
evidence contracts.  They may help a researcher prioritize checks, but they
cannot approve a term, create an evidence record, enter an annotation freeze,
or become trainer input.  The selected cohort is immutable for a given plan:
an excluded or unresolved item is never silently replaced.

No function in this module performs network access or modifies the three
review CSVs.  Optional capture artifacts must already exist as regular,
hash-bound files; even then they remain evidence *candidates*, not production
evidence.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .artifacts import (
    publish_bytes_once,
    read_regular_file_bytes,
    read_regular_file_bytes_exact,
    sha256_file,
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
from .review_import import (
    ETYMOLOGY_6_TO_4,
    PAIR_FIELDS,
    PAIR_HUMAN_FIELDS,
    SCREEN_FIELDS,
    TERM_FIELDS,
    TERM_HUMAN_FIELDS,
    TERM_IMMUTABLE_FIELDS,
    canonical,
)


SELECTION_PLAN_SCHEMA = "ai-prereview-selection-commitment-v1"
ADVISORY_ROW_SCHEMA = "ai-prereview-advisory-row-v1"
ADVISORY_MANIFEST_SCHEMA = "ai-prereview-manifest-v1"

SELECTION_COUNT = 60
REQUIRED_INPUT_NAMES = frozenset(
    {
        "cpu_checks",
        "implementation_manifest",
        "pair_selection_sheet",
        "pending_review",
        "source_manifest",
        "terms_long",
        "untranslated_screen",
    }
)
ANTICIPATED_QUOTA_TARGETS = {
    "exact_review_count": 60,
    "minimum_identifiable": 40,
    "minimum_related_en_fr": 12,
    "minimum_distinct_routes_en_fr": 12,
}
FORBIDDEN_OUTPUTS = (
    "APPROVED_BY_RESEARCHER",
    "reviewer=human",
    "production evidence_records.jsonl",
    "annotation freeze",
    "training authorization",
)
REPLACEMENT_POLICY = (
    "NO_AUTOMATIC_OR_SILENT_REPLACEMENT; any failed sense or evidence item "
    "remains blocked or unresolved and any future replacement requires a new "
    "pre-model-results commitment."
)
SELECTION_METHOD = (
    "MANUAL_PRESPECIFIED_ID_LIST_BEFORE_MODEL_RESULTS_AND_DETAILED_ETYMOLOGY_REVIEW"
)

MEANING_SUGGESTIONS = frozenset(
    {"ALIGNED", "PARTIAL", "NOT_ALIGNED", "UNRESOLVED"}
)
TERM_QUALITY_SUGGESTIONS = frozenset(
    {"ATTESTED_SAME_SENSE", "TRANSLATION_MISSING", "UNRESOLVED"}
)
CONFIDENCE_SUGGESTIONS = frozenset({"HIGH", "MEDIUM", "LOW"})
_DIRECTIONS = {
    "BORROWING_DIRECT": frozenset({"EN_TO_FR", "FR_TO_EN"}),
    "BORROWING_PARALLEL": frozenset({"COMMON_SOURCE_TO_BOTH"}),
    "NEOCLASSICAL_SHARED": frozenset({"COMMON_SOURCE_TO_BOTH"}),
    "COGNATE_INHERITED": frozenset({"COMMON_ANCESTOR_TO_BOTH"}),
    "DISTINCT_ROUTES_REVIEWED": frozenset({"NONE"}),
    "INDETERMINATE": frozenset({"UNKNOWN"}),
}

_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "advisory_only",
        "coverage_targets_not_claims",
        "created_at_utc",
        "forbidden_outputs",
        "human_approval_claimed",
        "implementation_git_commit",
        "inputs",
        "model_results_seen",
        "pairwise_distinct_precheck_count",
        "replacement_policy",
        "requires_human_review",
        "selected",
        "selected_count",
        "selection_locked_before_evidence_review",
        "selection_method",
        "selection_rationale",
        "snapshot_id",
        "training_eligible",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "candidate_id",
        "candidate_sha256",
        "pairwise_distinct_canonical",
        "review_order",
        "selection_ordinal",
        "terms",
    }
)
_SELECTED_TERM_FIELDS = frozenset(
    {
        "canonical",
        "source_gloss",
        "source_option_id",
        "source_span_end",
        "source_span_start",
        "term_id",
        "term_sha256",
    }
)
_ADVISORY_FIELDS = frozenset(
    {
        "schema_version",
        "selection_ordinal",
        "candidate_id",
        "advisory_only",
        "requires_human_review",
        "suggested_meaning_alignment",
        "suggested_term_quality",
        "suggested_etymology_subtype",
        "suggested_relation_direction",
        "suggested_shared_source",
        "suggested_confidence",
        "evidence_candidates",
        "notes",
    }
)
_EVIDENCE_CANDIDATE_FIELDS = frozenset(
    {"source_url", "locator", "summary", "capture_artifact"}
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "spec_version",
        "status",
        "advisory_only",
        "training_eligible",
        "requires_human_review",
        "model_results_seen",
        "human_approval_claimed",
        "automatic_approval_count",
        "selection_count",
        "pairwise_distinct_count",
        "anticipated_quota_targets",
        "selection_commitment_source",
        "selection_plan_artifact",
        "advisory_artifact",
        "evidence_candidate_count",
        "captured_candidate_count",
        "evidence_candidates_are_production_evidence",
        "capture_artifacts_are_production_evidence",
        "review_csvs_modified",
        "network_requests_by_this_tool",
        "notice",
    }
)
_NOTICE = (
    "AI suggestions, URLs, and optional captured payloads are advisory leads only. "
    "A URL or capture is not production evidence, no row is researcher approval, "
    "and all 60 items still require independent human review and a separate "
    "production evidence/freeze workflow."
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_token: str) -> None:
    raise ContractViolation("NONFINITE_JSON_NUMBER")


def _loads_object(raw: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ContractViolation:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation(code) from exc
    if not isinstance(value, dict):
        raise ContractViolation(code)
    return value


def _exact_bool(value: Any, expected: bool, code: str) -> None:
    if value is not expected:
        raise ContractViolation(code)


def _exact_int(value: Any, expected: int, code: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise ContractViolation(code)


def _safe_text(value: Any, code: str, *, allow_empty: bool = False, limit: int = 8000) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value.strip())
        or "\x00" in value
        or len(value) > limit
    ):
        raise ContractViolation(code)
    return value


def _artifact_ref(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _read_artifact_ref(
    value: Any,
    *,
    scope_root: Path,
    code: str = "INVALID_AI_PREREVIEW_ARTIFACT_REF",
) -> tuple[Path, bytes]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        raise ContractViolation(code)
    raw_path = value.get("path")
    expected_bytes = value.get("bytes")
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractViolation(code)
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise ContractViolation(code)
    claimed_hash = value.get("sha256")
    if not isinstance(claimed_hash, str):
        raise ContractViolation("INVALID_AI_PREREVIEW_ARTIFACT_HASH")
    expected_hash = require_sha256(
        claimed_hash, "INVALID_AI_PREREVIEW_ARTIFACT_HASH"
    )
    try:
        path = require_relative_to(
            Path(raw_path), scope_root, "AI_PREREVIEW_ARTIFACT_OUTSIDE_SCOPE"
        )
    except OSError as exc:
        raise ContractViolation("AI_PREREVIEW_ARTIFACT_OUTSIDE_SCOPE") from exc
    if raw_path != str(path):
        raise ContractViolation("AI_PREREVIEW_ARTIFACT_PATH_NOT_CANONICAL")
    raw = read_regular_file_bytes_exact(path, expected_bytes=expected_bytes)
    if sha256_bytes(raw) != expected_hash:
        raise ContractViolation("AI_PREREVIEW_ARTIFACT_HASH_MISMATCH")
    return path, raw


def _read_csv(raw: bytes, fields: Sequence[str], *, allow_empty: bool = False) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None or tuple(reader.fieldnames) != tuple(fields):
            raise ContractViolation("AI_PREREVIEW_CSV_HEADER_MISMATCH")
        rows = [dict(row) for row in reader]
    except ContractViolation:
        raise
    except (UnicodeError, csv.Error) as exc:
        raise ContractViolation("INVALID_AI_PREREVIEW_CSV") from exc
    if not rows and not allow_empty:
        raise ContractViolation("EMPTY_AI_PREREVIEW_CSV")
    if any(None in row or set(row) != set(fields) for row in rows):
        raise ContractViolation("AI_PREREVIEW_CSV_ROW_SCHEMA_MISMATCH")
    return rows


def _logical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_csv_sources(
    inputs: Mapping[str, Any],
    *,
    scope_root: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], str]:
    loaded: dict[str, bytes] = {}
    for name in sorted(REQUIRED_INPUT_NAMES):
        _path, loaded[name] = _read_artifact_ref(inputs[name], scope_root=scope_root)

    terms = _read_csv(loaded["terms_long"], TERM_FIELDS)
    pairs = _read_csv(loaded["pair_selection_sheet"], PAIR_FIELDS)
    screen = _read_csv(loaded["untranslated_screen"], SCREEN_FIELDS, allow_empty=True)

    if any(any(row[field] for field in TERM_HUMAN_FIELDS) for row in terms):
        raise ContractViolation("AI_PLAN_REQUIRES_BLANK_HUMAN_TERM_FIELDS")
    if any(any(row[field] for field in PAIR_HUMAN_FIELDS) for row in pairs):
        raise ContractViolation("AI_PLAN_REQUIRES_BLANK_HUMAN_PAIR_FIELDS")

    terms_by_id: dict[str, dict[str, str]] = {}
    source_identity: set[tuple[str, str]] = set()
    for row in terms:
        if row["schema_version"] != VERSION:
            raise ContractViolation("AI_PLAN_SOURCE_VERSION_MISMATCH")
        term_id = row["term_id"]
        if not term_id or term_id in terms_by_id:
            raise ContractViolation("AI_PLAN_DUPLICATE_TERM_ID")
        if row["lang"] not in LANGUAGES or not term_id.startswith(
            f"{row['candidate_id']}|{row['lang']}|"
        ):
            raise ContractViolation("AI_PLAN_INVALID_TERM_ID")
        require_sha256(row["candidate_sha256"], "AI_PLAN_INVALID_CANDIDATE_HASH")
        require_sha256(row["term_sha256"], "AI_PLAN_INVALID_TERM_HASH")
        require_sha256(row["merge_sha256"], "AI_PLAN_INVALID_MERGE_HASH")
        try:
            start = int(row["source_span_start"])
            end = int(row["source_span_end"])
        except ValueError as exc:
            raise ContractViolation("AI_PLAN_INVALID_SOURCE_SPAN") from exc
        if start < 0 or end <= start or end > len(row["source_word_raw"]):
            raise ContractViolation("AI_PLAN_INVALID_SOURCE_SPAN")
        core = {
            field: row[field]
            for field in TERM_IMMUTABLE_FIELDS
            if field != "term_sha256"
        }
        if _logical_hash(core) != row["term_sha256"]:
            raise ContractViolation("AI_PLAN_TERM_HASH_MISMATCH")
        if canonical(row["term_canonical"]) != row["term_canonical"]:
            raise ContractViolation("AI_PLAN_NONCANONICAL_TERM")
        source_identity.add((row["snapshot_id"], row["merge_sha256"]))
        terms_by_id[term_id] = row

    pairs_by_id: dict[str, dict[str, str]] = {}
    review_orders: set[int] = set()
    for row in pairs:
        if row["schema_version"] != VERSION:
            raise ContractViolation("AI_PLAN_SOURCE_VERSION_MISMATCH")
        candidate_id = row["candidate_id"]
        try:
            review_order = int(row["review_order"])
        except ValueError as exc:
            raise ContractViolation("AI_PLAN_INVALID_REVIEW_ORDER") from exc
        if (
            not candidate_id
            or candidate_id in pairs_by_id
            or review_order <= 0
            or review_order in review_orders
        ):
            raise ContractViolation("AI_PLAN_DUPLICATE_CANDIDATE_OR_ORDER")
        require_sha256(row["candidate_sha256"], "AI_PLAN_INVALID_CANDIDATE_HASH")
        require_sha256(row["merge_sha256"], "AI_PLAN_INVALID_MERGE_HASH")
        source_identity.add((row["snapshot_id"], row["merge_sha256"]))
        review_orders.add(review_order)
        pairs_by_id[candidate_id] = row

    if len(source_identity) != 1:
        raise ContractViolation("AI_PLAN_SOURCE_IDENTITY_MISMATCH")
    snapshot_id, _merge_sha256 = next(iter(source_identity))
    if not snapshot_id:
        raise ContractViolation("AI_PLAN_SOURCE_IDENTITY_MISMATCH")

    for term in terms:
        pair = pairs_by_id.get(term["candidate_id"])
        if (
            pair is None
            or pair["candidate_sha256"] != term["candidate_sha256"]
            or pair["review_order"] != term["review_order"]
        ):
            raise ContractViolation("AI_PLAN_TERM_CANDIDATE_SOURCE_MISMATCH")

    for row in screen:
        if (
            row["schema_version"] != VERSION
            or row["snapshot_id"] != snapshot_id
            or row["merge_sha256"] != _merge_sha256
            or row["advisory_only"] != "TRUE"
        ):
            raise ContractViolation("AI_PLAN_INVALID_SCREEN_ROW")
        term = terms_by_id.get(row["term_id"])
        pair = pairs_by_id.get(row["candidate_id"])
        if (
            term is None
            or pair is None
            or term["candidate_id"] != row["candidate_id"]
            or term["candidate_sha256"] != row["candidate_sha256"]
            or term["lang"] != row["lang"]
            or term["term_canonical"] != row["term_canonical"]
            or pair["candidate_sha256"] != row["candidate_sha256"]
            or pair["review_order"] != row["review_order"]
        ):
            raise ContractViolation("AI_PLAN_SCREEN_SOURCE_MISMATCH")
    return terms_by_id, pairs_by_id, snapshot_id


def _validate_created_at(value: Any) -> None:
    text = _safe_text(value, "AI_PLAN_INVALID_CREATED_AT", limit=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractViolation("AI_PLAN_INVALID_CREATED_AT") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractViolation("AI_PLAN_INVALID_CREATED_AT")


def validate_ai_prereview_selection_plan(
    plan: Mapping[str, Any],
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Validate an exact, pre-model-results 60-item selection commitment."""
    if not isinstance(plan, Mapping) or set(plan) != _PLAN_FIELDS:
        raise ContractViolation("AI_PLAN_SCHEMA_MISMATCH")
    if plan.get("schema_version") != SELECTION_PLAN_SCHEMA:
        raise ContractViolation("AI_PLAN_VERSION_MISMATCH")
    _exact_bool(plan.get("advisory_only"), True, "AI_PLAN_NOT_ADVISORY_ONLY")
    _exact_bool(plan.get("training_eligible"), False, "AI_PLAN_TRAINING_FORBIDDEN")
    _exact_bool(
        plan.get("requires_human_review"), True, "AI_PLAN_HUMAN_REVIEW_REQUIRED"
    )
    _exact_bool(
        plan.get("model_results_seen"), False, "AI_PLAN_MODEL_RESULTS_FORBIDDEN"
    )
    _exact_bool(
        plan.get("human_approval_claimed"), False, "AI_PLAN_HUMAN_APPROVAL_FORBIDDEN"
    )
    _exact_bool(
        plan.get("selection_locked_before_evidence_review"),
        True,
        "AI_PLAN_SELECTION_NOT_LOCKED",
    )
    _exact_int(plan.get("selected_count"), SELECTION_COUNT, "AI_PLAN_COUNT_MISMATCH")
    if plan.get("coverage_targets_not_claims") != ANTICIPATED_QUOTA_TARGETS:
        raise ContractViolation("AI_PLAN_QUOTA_TARGET_MISMATCH")
    if plan.get("replacement_policy") != REPLACEMENT_POLICY:
        raise ContractViolation("AI_PLAN_AUTOMATIC_REPLACEMENT_FORBIDDEN")
    if plan.get("selection_method") != SELECTION_METHOD:
        raise ContractViolation("AI_PLAN_SELECTION_METHOD_MISMATCH")
    if plan.get("forbidden_outputs") != list(FORBIDDEN_OUTPUTS):
        raise ContractViolation("AI_PLAN_FORBIDDEN_OUTPUT_CONTRACT_MISMATCH")
    _safe_text(plan.get("selection_rationale"), "AI_PLAN_RATIONALE_REQUIRED")
    _validate_created_at(plan.get("created_at_utc"))
    commit = plan.get("implementation_git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ContractViolation("AI_PLAN_INVALID_IMPLEMENTATION_COMMIT")
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != REQUIRED_INPUT_NAMES:
        raise ContractViolation("AI_PLAN_INPUT_SET_MISMATCH")
    terms_by_id, pairs_by_id, snapshot_id = _validate_csv_sources(
        inputs, scope_root=scope_root
    )
    if plan.get("snapshot_id") != snapshot_id:
        raise ContractViolation("AI_PLAN_SNAPSHOT_MISMATCH")

    selected = plan.get("selected")
    if not isinstance(selected, list) or len(selected) != SELECTION_COUNT:
        raise ContractViolation("AI_PLAN_COUNT_MISMATCH")
    candidate_ids: set[str] = set()
    selected_term_ids: set[str] = set()
    pairwise_count = 0
    for ordinal, item in enumerate(selected, 1):
        if not isinstance(item, Mapping) or set(item) != _SELECTION_FIELDS:
            raise ContractViolation("AI_PLAN_SELECTION_SCHEMA_MISMATCH")
        _exact_int(
            item.get("selection_ordinal"), ordinal, "AI_PLAN_SELECTION_ORDER_MISMATCH"
        )
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in candidate_ids:
            raise ContractViolation("AI_PLAN_DUPLICATE_CANDIDATE")
        pair = pairs_by_id.get(candidate_id)
        if pair is None:
            raise ContractViolation("AI_PLAN_UNKNOWN_CANDIDATE")
        if item.get("candidate_sha256") != pair["candidate_sha256"]:
            raise ContractViolation("AI_PLAN_CANDIDATE_HASH_MISMATCH")
        try:
            pair_order = int(pair["review_order"])
        except ValueError as exc:  # defensive; checked above
            raise ContractViolation("AI_PLAN_INVALID_REVIEW_ORDER") from exc
        _exact_int(item.get("review_order"), pair_order, "AI_PLAN_REVIEW_ORDER_MISMATCH")
        selected_terms = item.get("terms")
        if not isinstance(selected_terms, Mapping) or set(selected_terms) != set(LANGUAGES):
            raise ContractViolation("AI_PLAN_FOUR_TERMS_REQUIRED")
        canonical_values: list[str] = []
        for language in LANGUAGES:
            selected_term = selected_terms.get(language)
            if (
                not isinstance(selected_term, Mapping)
                or set(selected_term) != _SELECTED_TERM_FIELDS
            ):
                raise ContractViolation("AI_PLAN_SELECTED_TERM_SCHEMA_MISMATCH")
            term_id = selected_term.get("term_id")
            if not isinstance(term_id, str) or term_id in selected_term_ids:
                raise ContractViolation("AI_PLAN_DUPLICATE_SELECTED_TERM")
            source = terms_by_id.get(term_id)
            try:
                source_span_start = int(source["source_span_start"]) if source else None
                source_span_end = int(source["source_span_end"]) if source else None
            except (TypeError, ValueError) as exc:
                raise ContractViolation("AI_PLAN_SELECTED_TERM_SOURCE_MISMATCH") from exc
            selected_span_start = selected_term.get("source_span_start")
            selected_span_end = selected_term.get("source_span_end")
            if (
                source is None
                or not isinstance(selected_span_start, int)
                or isinstance(selected_span_start, bool)
                or not isinstance(selected_span_end, int)
                or isinstance(selected_span_end, bool)
                or source["candidate_id"] != candidate_id
                or source["lang"] != language
                or selected_term.get("term_sha256") != source["term_sha256"]
                or selected_term.get("canonical") != source["term_canonical"]
                or selected_term.get("source_gloss") != source["source_gloss"]
                or selected_term.get("source_option_id") != source["source_option_id"]
                or selected_span_start != source_span_start
                or selected_span_end != source_span_end
            ):
                raise ContractViolation("AI_PLAN_SELECTED_TERM_SOURCE_MISMATCH")
            canonical_values.append(source["term_canonical"])
            selected_term_ids.add(term_id)
        pairwise = len(set(canonical_values)) == len(LANGUAGES)
        _exact_bool(
            item.get("pairwise_distinct_canonical"),
            pairwise,
            "AI_PLAN_PAIRWISE_DISTINCT_MISMATCH",
        )
        pairwise_count += int(pairwise)
        candidate_ids.add(candidate_id)

    _exact_int(
        plan.get("pairwise_distinct_precheck_count"),
        pairwise_count,
        "AI_PLAN_PAIRWISE_COUNT_MISMATCH",
    )
    if pairwise_count < ANTICIPATED_QUOTA_TARGETS["minimum_identifiable"]:
        raise ContractViolation("AI_PLAN_PAIRWISE_COVERAGE_TOO_LOW")
    return {
        "status": "PASS_ADVISORY_SELECTION_PLAN",
        "schema_version": SELECTION_PLAN_SCHEMA,
        "spec_version": VERSION,
        "advisory_only": True,
        "training_eligible": False,
        "requires_human_review": True,
        "model_results_seen": False,
        "selection_count": SELECTION_COUNT,
        "pairwise_distinct_count": pairwise_count,
        "anticipated_quota_targets": dict(ANTICIPATED_QUOTA_TARGETS),
        "replacement_policy": REPLACEMENT_POLICY,
    }


def load_and_validate_ai_prereview_selection_plan(
    path: Path,
    *,
    scope_root: Path = WORK_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = require_relative_to(
        Path(path), scope_root, "AI_PREREVIEW_PLAN_OUTSIDE_SCOPE"
    )
    raw = read_regular_file_bytes(resolved)
    plan = _loads_object(raw, "INVALID_AI_PREREVIEW_PLAN_JSON")
    summary = validate_ai_prereview_selection_plan(plan, scope_root=scope_root)
    return plan, {**_artifact_ref(resolved, raw), "summary": summary}


def publish_ai_prereview_selection_plan(
    draft_path: Path,
    output_path: Path,
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Validate and publish a plan once; never edits its source CSVs."""
    plan, source = load_and_validate_ai_prereview_selection_plan(
        draft_path, scope_root=scope_root
    )
    output = require_relative_to(
        Path(output_path), scope_root, "AI_PREREVIEW_PLAN_OUTPUT_OUTSIDE_SCOPE"
    )
    ref = publish_bytes_once(output, canonical_json_bytes(plan))
    loaded, _published = load_and_validate_ai_prereview_selection_plan(
        output, scope_root=scope_root
    )
    if loaded != plan:
        raise ContractViolation("AI_PREREVIEW_PLAN_PUBLICATION_MISMATCH")
    return {
        "status": "PASS_ADVISORY_SELECTION_PLAN_PUBLISHED",
        "advisory_only": True,
        "training_eligible": False,
        "requires_human_review": True,
        "source_plan": {key: source[key] for key in ("path", "sha256", "bytes")},
        "plan_artifact": ref,
        "selection_count": SELECTION_COUNT,
    }


def _https_url(value: Any) -> str:
    url = _safe_text(value, "AI_ADVISORY_INVALID_EVIDENCE_URL", limit=2048)
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ContractViolation("AI_ADVISORY_INVALID_EVIDENCE_URL") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(char.isspace() for char in url)
    ):
        raise ContractViolation("AI_ADVISORY_INVALID_EVIDENCE_URL")
    return url


def _validate_evidence_candidate(
    value: Any,
    *,
    scope_root: Path,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_CANDIDATE_FIELDS:
        raise ContractViolation("AI_ADVISORY_EVIDENCE_CANDIDATE_SCHEMA_MISMATCH")
    checked: dict[str, Any] = {
        "source_url": _https_url(value.get("source_url")),
        "locator": _safe_text(
            value.get("locator"), "AI_ADVISORY_EVIDENCE_LOCATOR_REQUIRED", limit=1000
        ),
        "summary": _safe_text(
            value.get("summary"), "AI_ADVISORY_EVIDENCE_SUMMARY_REQUIRED", limit=4000
        ),
        "capture_artifact": None,
    }
    capture = value.get("capture_artifact")
    if capture is not None:
        path, raw = _read_artifact_ref(capture, scope_root=scope_root)
        checked["capture_artifact"] = _artifact_ref(path, raw)
    return checked, capture is not None


def _validate_advisory_rows(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    scope_root: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    if len(rows) != SELECTION_COUNT:
        raise ContractViolation("AI_ADVISORY_COUNT_MISMATCH")
    checked_rows: list[dict[str, Any]] = []
    evidence_count = 0
    captured_count = 0
    for ordinal, (row, selection) in enumerate(zip(rows, plan["selected"]), 1):
        if not isinstance(row, Mapping) or set(row) != _ADVISORY_FIELDS:
            raise ContractViolation("AI_ADVISORY_ROW_SCHEMA_MISMATCH")
        if row.get("schema_version") != ADVISORY_ROW_SCHEMA:
            raise ContractViolation("AI_ADVISORY_ROW_VERSION_MISMATCH")
        _exact_int(
            row.get("selection_ordinal"), ordinal, "AI_ADVISORY_ORDER_MISMATCH"
        )
        if row.get("candidate_id") != selection["candidate_id"]:
            raise ContractViolation("AI_ADVISORY_CANDIDATE_ORDER_MISMATCH")
        _exact_bool(row.get("advisory_only"), True, "AI_ADVISORY_ONLY_REQUIRED")
        _exact_bool(
            row.get("requires_human_review"),
            True,
            "AI_ADVISORY_HUMAN_REVIEW_REQUIRED",
        )
        meaning = row.get("suggested_meaning_alignment")
        subtype = row.get("suggested_etymology_subtype")
        direction = row.get("suggested_relation_direction")
        shared_source = _safe_text(
            row.get("suggested_shared_source"),
            "AI_ADVISORY_INVALID_SHARED_SOURCE",
            allow_empty=True,
            limit=1000,
        )
        confidence = row.get("suggested_confidence")
        if meaning not in MEANING_SUGGESTIONS:
            raise ContractViolation("AI_ADVISORY_INVALID_MEANING_SUGGESTION")
        if subtype not in ETYMOLOGY_6_TO_4:
            raise ContractViolation("AI_ADVISORY_INVALID_ETYMOLOGY_SUGGESTION")
        if direction not in _DIRECTIONS[str(subtype)]:
            raise ContractViolation("AI_ADVISORY_INVALID_DIRECTION_SUGGESTION")
        shared_subtypes = {
            "BORROWING_PARALLEL",
            "NEOCLASSICAL_SHARED",
            "COGNATE_INHERITED",
        }
        if subtype in shared_subtypes and not shared_source.strip():
            raise ContractViolation("AI_ADVISORY_SHARED_SOURCE_REQUIRED")
        if subtype not in shared_subtypes and shared_source:
            raise ContractViolation("AI_ADVISORY_SHARED_SOURCE_NOT_ALLOWED")
        if confidence not in CONFIDENCE_SUGGESTIONS or (
            subtype == "INDETERMINATE" and confidence == "HIGH"
        ):
            raise ContractViolation("AI_ADVISORY_INVALID_CONFIDENCE_SUGGESTION")
        quality = row.get("suggested_term_quality")
        if (
            not isinstance(quality, Mapping)
            or set(quality) != set(LANGUAGES)
            or any(value not in TERM_QUALITY_SUGGESTIONS for value in quality.values())
        ):
            raise ContractViolation("AI_ADVISORY_INVALID_TERM_QUALITY_SUGGESTION")
        evidence = row.get("evidence_candidates")
        if not isinstance(evidence, list) or len(evidence) > 25:
            raise ContractViolation("AI_ADVISORY_INVALID_EVIDENCE_CANDIDATES")
        checked_evidence: list[dict[str, Any]] = []
        urls: set[str] = set()
        for item in evidence:
            checked, captured = _validate_evidence_candidate(
                item, scope_root=scope_root
            )
            if checked["source_url"] in urls:
                raise ContractViolation("AI_ADVISORY_DUPLICATE_EVIDENCE_URL")
            urls.add(checked["source_url"])
            checked_evidence.append(checked)
            evidence_count += 1
            captured_count += int(captured)
        notes = _safe_text(
            row.get("notes"), "AI_ADVISORY_INVALID_NOTES", allow_empty=True
        )
        if "APPROVED_BY_RESEARCHER" in notes or re.search(
            r"reviewer\s*=\s*human", notes, flags=re.IGNORECASE
        ):
            raise ContractViolation("AI_ADVISORY_HUMAN_APPROVAL_CLAIM_FORBIDDEN")
        checked_rows.append(
            {
                "schema_version": ADVISORY_ROW_SCHEMA,
                "selection_ordinal": ordinal,
                "candidate_id": selection["candidate_id"],
                "advisory_only": True,
                "requires_human_review": True,
                "suggested_meaning_alignment": meaning,
                "suggested_term_quality": {
                    language: quality[language] for language in LANGUAGES
                },
                "suggested_etymology_subtype": subtype,
                "suggested_relation_direction": direction,
                "suggested_shared_source": shared_source,
                "suggested_confidence": confidence,
                "evidence_candidates": checked_evidence,
                "notes": notes,
            }
        )
    return checked_rows, evidence_count, captured_count


def _read_advisory_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = read_regular_file_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractViolation("INVALID_AI_ADVISORY_JSONL") from exc
    lines = text.splitlines()
    if len(lines) != SELECTION_COUNT or any(not line.strip() for line in lines):
        raise ContractViolation("AI_ADVISORY_COUNT_MISMATCH")
    rows = [
        _loads_object(line.encode("utf-8"), "INVALID_AI_ADVISORY_JSONL")
        for line in lines
    ]
    return rows, raw


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _bundle_manifest(
    *,
    source_plan_ref: Mapping[str, Any],
    plan_ref: Mapping[str, Any],
    advisory_ref: Mapping[str, Any],
    pairwise_count: int,
    evidence_count: int,
    captured_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": ADVISORY_MANIFEST_SCHEMA,
        "spec_version": VERSION,
        "status": "AI_ADVISORY_ONLY_REQUIRES_HUMAN_REVIEW",
        "advisory_only": True,
        "training_eligible": False,
        "requires_human_review": True,
        "model_results_seen": False,
        "human_approval_claimed": False,
        "automatic_approval_count": 0,
        "selection_count": SELECTION_COUNT,
        "pairwise_distinct_count": pairwise_count,
        "anticipated_quota_targets": dict(ANTICIPATED_QUOTA_TARGETS),
        "selection_commitment_source": dict(source_plan_ref),
        "selection_plan_artifact": dict(plan_ref),
        "advisory_artifact": dict(advisory_ref),
        "evidence_candidate_count": evidence_count,
        "captured_candidate_count": captured_count,
        "evidence_candidates_are_production_evidence": False,
        "capture_artifacts_are_production_evidence": False,
        "review_csvs_modified": False,
        "network_requests_by_this_tool": 0,
        "notice": _NOTICE,
    }


def _cleanup_private_stage(stage: Path) -> None:
    for name in (
        "ai_prereview_manifest.json",
        "ai_prereview.jsonl",
        "selection_plan.json",
    ):
        try:
            (stage / name).unlink()
        except FileNotFoundError:
            pass
    try:
        stage.rmdir()
    except FileNotFoundError:
        pass


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Reuse the repository's Linux atomic no-replace directory publisher."""
    # Late import keeps ordinary advisory validation dependency-light and
    # avoids duplicating the security-critical renameat2 wrapper.
    from .prepare_data import _rename_directory_noreplace

    try:
        _rename_directory_noreplace(source, destination)
    except ContractViolation as exc:
        mapped = {
            "ANNOTATION_FREEZE_OUTPUT_EXISTS": "AI_ADVISORY_OUTPUT_EXISTS",
            "ANNOTATION_FREEZE_ATOMIC_NOREPLACE_UNSUPPORTED": (
                "AI_ADVISORY_ATOMIC_NOREPLACE_UNSUPPORTED"
            ),
            "ANNOTATION_FREEZE_ATOMIC_PUBLISH_FAILED": (
                "AI_ADVISORY_ATOMIC_PUBLISH_FAILED"
            ),
            "ANNOTATION_FREEZE_PARENT_FSYNC_FAILED": (
                "AI_ADVISORY_PARENT_FSYNC_FAILED"
            ),
        }.get(exc.code, "AI_ADVISORY_ATOMIC_PUBLISH_FAILED")
        raise ContractViolation(mapped) from exc


def _require_read_only_bundle(directory: Path) -> None:
    try:
        directory_mode = stat.S_IMODE(directory.stat(follow_symlinks=False).st_mode)
        file_modes = [
            stat.S_IMODE((directory / name).stat(follow_symlinks=False).st_mode)
            for name in (
                "selection_plan.json",
                "ai_prereview.jsonl",
                "ai_prereview_manifest.json",
            )
        ]
    except OSError as exc:
        raise ContractViolation("AI_ADVISORY_READ_ONLY_BUNDLE_REQUIRED") from exc
    if directory_mode & 0o222 or any(mode & 0o222 for mode in file_modes):
        raise ContractViolation("AI_ADVISORY_READ_ONLY_BUNDLE_REQUIRED")


def publish_ai_prereview_bundle(
    selection_plan_path: Path,
    advisory_draft_path: Path,
    *,
    output_dir: Path,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Atomically publish a read-only advisory bundle without touching CSVs."""
    plan, source_info = load_and_validate_ai_prereview_selection_plan(
        selection_plan_path, scope_root=scope_root
    )
    draft = require_relative_to(
        Path(advisory_draft_path), scope_root, "AI_ADVISORY_DRAFT_OUTSIDE_SCOPE"
    )
    rows, _draft_raw = _read_advisory_jsonl(draft)
    checked_rows, evidence_count, captured_count = _validate_advisory_rows(
        rows, plan, scope_root=scope_root
    )
    destination = require_relative_to(
        Path(output_dir), scope_root, "AI_ADVISORY_OUTPUT_OUTSIDE_SCOPE"
    )
    if destination.exists() or destination.is_symlink():
        raise ContractViolation("AI_ADVISORY_OUTPUT_EXISTS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    stage.mkdir(mode=0o700)
    plan_bytes = canonical_json_bytes(plan)
    advisory_bytes = _canonical_jsonl(checked_rows)
    final_plan = destination / "selection_plan.json"
    final_advisory = destination / "ai_prereview.jsonl"
    plan_ref = _artifact_ref(final_plan, plan_bytes)
    advisory_ref = _artifact_ref(final_advisory, advisory_bytes)
    source_ref = {
        key: source_info[key] for key in ("path", "sha256", "bytes")
    }
    manifest = _bundle_manifest(
        source_plan_ref=source_ref,
        plan_ref=plan_ref,
        advisory_ref=advisory_ref,
        pairwise_count=int(plan["pairwise_distinct_precheck_count"]),
        evidence_count=evidence_count,
        captured_count=captured_count,
    )
    try:
        publish_bytes_once(stage / "selection_plan.json", plan_bytes)
        publish_bytes_once(stage / "ai_prereview.jsonl", advisory_bytes)
        publish_bytes_once(
            stage / "ai_prereview_manifest.json", canonical_json_bytes(manifest)
        )
        # Re-open the original plan and every bound source before committing
        # the directory.  A changed review CSV therefore leaves no final bundle.
        current, _current_ref = load_and_validate_ai_prereview_selection_plan(
            selection_plan_path, scope_root=scope_root
        )
        if current != plan:
            raise ContractViolation("AI_PREREVIEW_SOURCE_CHANGED_DURING_PUBLICATION")
        os.chmod(stage, 0o555)
        _publish_directory_noreplace(stage, destination)
    except Exception:
        try:
            os.chmod(stage, 0o700)
        except FileNotFoundError:
            pass
        _cleanup_private_stage(stage)
        raise
    return {
        "status": "AI_ADVISORY_ONLY_REQUIRES_HUMAN_REVIEW",
        "advisory_only": True,
        "training_eligible": False,
        "requires_human_review": True,
        "model_results_seen": False,
        "selection_count": SELECTION_COUNT,
        "pairwise_distinct_count": plan["pairwise_distinct_precheck_count"],
        "evidence_candidate_count": evidence_count,
        "captured_candidate_count": captured_count,
        "manifest_artifact": {
            "path": str(destination / "ai_prereview_manifest.json"),
            "sha256": sha256_bytes(canonical_json_bytes(manifest)),
            "bytes": len(canonical_json_bytes(manifest)),
        },
        "notice": _NOTICE,
    }


def audit_ai_prereview_bundle(
    manifest_path: Path,
    *,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Re-open and validate every source-bound advisory bundle artifact."""
    manifest_resolved = require_relative_to(
        Path(manifest_path), scope_root, "AI_ADVISORY_MANIFEST_OUTSIDE_SCOPE"
    )
    manifest_raw = read_regular_file_bytes(manifest_resolved)
    manifest = _loads_object(manifest_raw, "INVALID_AI_ADVISORY_MANIFEST")
    if manifest_raw != canonical_json_bytes(manifest):
        raise ContractViolation("AI_ADVISORY_NONCANONICAL_MANIFEST")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ContractViolation("AI_ADVISORY_MANIFEST_SCHEMA_MISMATCH")
    plan_path, plan_raw = _read_artifact_ref(
        manifest.get("selection_plan_artifact"), scope_root=scope_root
    )
    advisory_path, _advisory_raw = _read_artifact_ref(
        manifest.get("advisory_artifact"), scope_root=scope_root
    )
    if (
        manifest_resolved.name != "ai_prereview_manifest.json"
        or plan_path != manifest_resolved.parent / "selection_plan.json"
        or advisory_path != manifest_resolved.parent / "ai_prereview.jsonl"
    ):
        raise ContractViolation("AI_ADVISORY_BUNDLE_PATH_MISMATCH")
    _require_read_only_bundle(manifest_resolved.parent)
    plan = _loads_object(plan_raw, "INVALID_AI_PREREVIEW_PLAN_JSON")
    plan_summary = validate_ai_prereview_selection_plan(plan, scope_root=scope_root)
    source_path, source_raw = _read_artifact_ref(
        manifest.get("selection_commitment_source"), scope_root=scope_root
    )
    source_plan = _loads_object(source_raw, "INVALID_AI_PREREVIEW_PLAN_JSON")
    if source_plan != plan:
        raise ContractViolation("AI_ADVISORY_SELECTION_COMMITMENT_MISMATCH")
    # Also validate the source path independently; the local variable makes the
    # required, exact source commitment explicit in audit results.
    if not source_path.is_file():  # pragma: no cover - descriptor read above
        raise ContractViolation("AI_ADVISORY_SELECTION_COMMITMENT_MISMATCH")
    rows, _raw = _read_advisory_jsonl(advisory_path)
    checked_rows, evidence_count, captured_count = _validate_advisory_rows(
        rows, plan, scope_root=scope_root
    )
    if _canonical_jsonl(checked_rows) != _advisory_raw:
        raise ContractViolation("AI_ADVISORY_NONCANONICAL_ARTIFACT")
    expected = _bundle_manifest(
        source_plan_ref=manifest["selection_commitment_source"],
        plan_ref=manifest["selection_plan_artifact"],
        advisory_ref=manifest["advisory_artifact"],
        pairwise_count=plan_summary["pairwise_distinct_count"],
        evidence_count=evidence_count,
        captured_count=captured_count,
    )
    if manifest != expected:
        raise ContractViolation("AI_ADVISORY_MANIFEST_CONTENT_MISMATCH")
    return {
        "status": "PASS_AI_ADVISORY_ONLY",
        "advisory_only": True,
        "training_eligible": False,
        "requires_human_review": True,
        "model_results_seen": False,
        "human_approval_claimed": False,
        "selection_count": SELECTION_COUNT,
        "pairwise_distinct_count": plan_summary["pairwise_distinct_count"],
        "evidence_candidate_count": evidence_count,
        "captured_candidate_count": captured_count,
        "evidence_candidates_are_production_evidence": False,
        "capture_artifacts_are_production_evidence": False,
        "manifest_sha256": sha256_file(manifest_resolved),
        "notice": _NOTICE,
    }


__all__ = [
    "ADVISORY_MANIFEST_SCHEMA",
    "ADVISORY_ROW_SCHEMA",
    "ANTICIPATED_QUOTA_TARGETS",
    "REPLACEMENT_POLICY",
    "SELECTION_PLAN_SCHEMA",
    "audit_ai_prereview_bundle",
    "load_and_validate_ai_prereview_selection_plan",
    "publish_ai_prereview_bundle",
    "publish_ai_prereview_selection_plan",
    "validate_ai_prereview_selection_plan",
]
