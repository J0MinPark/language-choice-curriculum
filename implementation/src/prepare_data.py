"""Strict source, review, and freeze preparation for the v4 pilot.

The functions in this module deliberately separate four trust boundaries:

* a KRDICT response is immutable source material, not an annotation;
* a three-language merge is a pending candidate, not approved data;
* a recorded review is useful only when it is bound to the exact candidate;
* trainers consume a verified freeze manifest, never loose JSON/JSONL files.

Network access is confined to :func:`collect_krdict`.  Its transport is
injectable so all tests can be completely offline.  No function accepts an API
key as a command-line-shaped argument and no result contains the key or a full
request URL.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import http.client
import json
import os
import re
import shutil
import sqlite3
import ssl
import stat
import tempfile
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from freshstart.core import membership

from .artifacts import (
    publish_bytes_once,
    publish_json_once,
    publish_verified_file_once,
    read_regular_file_bytes,
    read_regular_file_bytes_exact,
    read_verified_json,
    read_verified_json_with_sha256,
    sha256_file as artifact_sha256_file,
)
from .contracts import (
    IMPLEMENTATION_REVISION,
    LANGUAGES,
    PROJECT_ROOT,
    VERSION,
    WORK_ROOT,
    ContractViolation,
    canonical_json_bytes,
    load_json,
    load_pilot_config,
    require_relative_to,
    require_sha256,
    sha256_bytes,
)
from .trust import require_trusted_artifact_anchor


KRDICT_HOST = "krdict.korean.go.kr"
KRDICT_PATH = "/api/search"
KRDICT_ENDPOINT = f"https://{KRDICT_HOST}{KRDICT_PATH}"
KRDICT_LANG_CODE = {"en": "1", "fr": "3", "zh": "11"}
KRDICT_LANG_NAME = {"en": "영어", "fr": "프랑스어", "zh": "중국어"}
TARGET_LANGUAGES = ("en", "zh", "fr")
ETYMOLOGY_LABELS = {
    "BORROWING_DOCUMENTED",
    "SHARED_SOURCE_DOCUMENTED",
    "DISTINCT_ROUTES_REVIEWED",
    "UNRESOLVED",
}
ETYMOLOGY_SUBTYPE_TO_PRIMARY = {
    "BORROWING_DIRECT": "BORROWING_DOCUMENTED",
    "BORROWING_PARALLEL": "SHARED_SOURCE_DOCUMENTED",
    "NEOCLASSICAL_SHARED": "SHARED_SOURCE_DOCUMENTED",
    "COGNATE_INHERITED": "SHARED_SOURCE_DOCUMENTED",
    "DISTINCT_ROUTES_REVIEWED": "DISTINCT_ROUTES_REVIEWED",
    "INDETERMINATE": "UNRESOLVED",
}
ETYMOLOGY_SUBTYPE_DIRECTIONS = {
    "BORROWING_DIRECT": {"EN_TO_FR", "FR_TO_EN"},
    "BORROWING_PARALLEL": {"COMMON_SOURCE_TO_BOTH"},
    "NEOCLASSICAL_SHARED": {"COMMON_SOURCE_TO_BOTH"},
    "COGNATE_INHERITED": {"COMMON_ANCESTOR_TO_BOTH"},
    "DISTINCT_ROUTES_REVIEWED": {"NONE"},
    "INDETERMINATE": {"UNKNOWN"},
}
EVIDENCE_RECORD_FIELDS = {
    "evidence_id",
    "subject_kind",
    "subject_id",
    "supports_label",
    "source_url",
    "evidence_origin",
    "source_name",
    "source_version",
    "source_license",
    "source_license_url",
    "sense_locator",
    "payload_path",
    "payload_sha256",
    "payload_bytes",
    "retrieved_at",
    "conflicts_with",
    "record_sha256",
}
TERM_QUALITY_REVIEW_INPUT_FIELDS = {
    "term_id",
    "term_sha256",
    "segmentation_decision",
    "translation_quality",
    "is_transliteration",
    "quality_note",
    "quality_evidence_ids",
    "evidence",
    "qa",
}
TERM_QUALITY_REVIEW_FROZEN_FIELDS = TERM_QUALITY_REVIEW_INPUT_FIELDS | {"answer"}
TERM_QUALITY_QA_FIELDS = {"status", "reviewer", "review_date"}
FROZEN_CONCEPT_FIELDS = {
    "schema_version",
    "implementation_revision",
    "project_id",
    "status",
    "review_validation_mode",
    "synthetic_fixture",
    "concept_id",
    "snapshot_id",
    "source_candidate_sha256",
    "source_record_hashes",
    "source_refs",
    "source_urls",
    "answers",
    "glosses",
    "selected_option_ids",
    "selected_source_spans",
    "memberships",
    "identifiable",
    "synonym_cluster_id",
    "qa",
    "etymology_family_id",
    "analysis_component_id",
    "concept_record_sha256",
}
FROZEN_CONCEPT_OPTIONAL_FIELDS = {
    "meaning_alignment_decision",
    "selected_term_quality",
}
FROZEN_CONCEPT_QA_FIELDS = {
    "status",
    "reviewer",
    "review_date",
    "source_alignment_checked",
    "answer_copy_checked",
    "meaning_alignment_note",
    "review_candidate_sha256",
}
FROZEN_ETYMOLOGY_FIELDS = {
    "schema_version",
    "implementation_revision",
    "project_id",
    "status",
    "review_validation_mode",
    "synthetic_fixture",
    "concept_id",
    "concept_record_sha256",
    "answer_pair_sha256",
    "pair",
    "relation",
    "evidence",
    "evidence_search_note",
    "sense_alignment_note",
    "historical_scope",
    "family_id",
    "qa",
    "etymology_record_sha256",
}
FROZEN_ETYMOLOGY_SUBTYPE_FIELDS = {
    "relation_subtype",
    "relation_direction",
    "shared_source",
    "confidence",
    "evidence_ids",
    "evidence_subject_sha256",
}
FROZEN_ETYMOLOGY_QA_FIELDS = {
    "status",
    "reviewer",
    "review_date",
    "source_alignment_checked",
    "answer_copy_checked",
}
REAL_KRDICT = "REAL_KRDICT_API"
REAL_WIKI40B = "REAL_WIKI40B_TFDS"
SYNTHETIC = "SYNTHETIC_TEST_FIXTURE"
PRODUCTION_REVIEW_MODE = "PRODUCTION"
MAX_XML_BYTES = 8 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
KEY_RE = re.compile(r"[0-9a-fA-F]{32}")
# The v4r1 campaign spent all 600 allowed calls (3 probes + 597 responses).
# A later collection requires a new policy/code revision, never a runtime flag.
CAMPAIGN_NETWORK_CLOSED = True


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_text(value: Any, code: str) -> str:
    if not isinstance(value, str):
        raise ContractViolation(code)
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized or "\x00" in normalized:
        raise ContractViolation(code)
    return normalized


def _single_line(value: Any, code: str) -> str:
    if not isinstance(value, str) or any(ch in value for ch in "\r\n\x00"):
        raise ContractViolation(code)
    return _normalized_text(value, code)


def _logical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def _validate_embedded_hash(row: Mapping[str, Any], field: str, code: str) -> None:
    claimed = row.get(field)
    require_sha256(claimed, code)
    if claimed != _logical_hash(_without(row, field)):
        raise ContractViolation(code)


def _artifact_ref_for_manifest(ref: Mapping[str, Any], base: Path) -> dict[str, Any]:
    """Use a relative path when an artifact is below ``base``."""
    raw_path = Path(str(ref["path"]))
    try:
        path = str(raw_path.resolve().relative_to(base.resolve()))
    except ValueError:
        path = str(raw_path.resolve())
    return {
        "path": path,
        "sha256": require_sha256(str(ref["sha256"])),
        "bytes": int(ref["bytes"]),
    }


def _fsync_directory(path: Path) -> None:
    """Durably publish a directory-entry change in ``path``."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractViolation("ANNOTATION_FREEZE_PARENT_FSYNC_FAILED") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ContractViolation("ANNOTATION_FREEZE_PARENT_FSYNC_FAILED")
        os.fsync(descriptor)
    except OSError as exc:
        raise ContractViolation("ANNOTATION_FREEZE_PARENT_FSYNC_FAILED") from exc
    finally:
        os.close(descriptor)


def _renameat2_noreplace(source: Path, destination: Path) -> None:
    """Invoke Linux ``renameat2(..., RENAME_NOREPLACE)`` or raise ``OSError``."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS)) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD; both paths are absolute and share one parent.
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a complete directory without replacing any inode.

    Python's :func:`os.rename` may replace an existing empty directory on
    POSIX.  The freeze boundary needs the stronger Linux ``renameat2``
    ``RENAME_NOREPLACE`` contract.  A platform/filesystem without that primitive
    fails closed rather than silently weakening write-once publication.
    """
    try:
        _renameat2_noreplace(source, destination)
    except OSError as exc:
        error = exc.errno
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ContractViolation("ANNOTATION_FREEZE_OUTPUT_EXISTS") from exc
        if error in {
            errno.ENOSYS,
            errno.EINVAL,
            errno.EOPNOTSUPP,
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
        }:
            raise ContractViolation(
                "ANNOTATION_FREEZE_ATOMIC_NOREPLACE_UNSUPPORTED"
            ) from exc
        raise ContractViolation("ANNOTATION_FREEZE_ATOMIC_PUBLISH_FAILED") from exc
    _fsync_directory(destination.parent)


def _resolve_artifact_ref(
    ref: Mapping[str, Any],
    *,
    base: Path,
    production: bool,
    scope_root: Path | None = None,
) -> Path:
    required = {"path", "sha256", "bytes"}
    if not isinstance(ref, Mapping) or not required.issubset(ref):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    require_sha256(str(ref["sha256"]), "INVALID_ARTIFACT_SHA256")
    if not isinstance(ref["bytes"], int) or isinstance(ref["bytes"], bool) or ref["bytes"] < 0:
        raise ContractViolation("INVALID_ARTIFACT_SIZE")
    raw = Path(str(ref["path"]))
    unresolved = raw if raw.is_absolute() else base / raw
    if unresolved.is_symlink():
        raise ContractViolation("ARTIFACT_NOT_REGULAR_FILE")
    candidate = unresolved.resolve(strict=False)
    if production:
        require_relative_to(candidate, scope_root or WORK_ROOT)
    if candidate.is_symlink() or not candidate.is_file():
        raise ContractViolation("ARTIFACT_NOT_REGULAR_FILE")
    if candidate.stat().st_size != ref["bytes"]:
        raise ContractViolation("ARTIFACT_SIZE_MISMATCH")
    if artifact_sha256_file(candidate) != ref["sha256"]:
        raise ContractViolation("ARTIFACT_HASH_MISMATCH")
    return candidate


def _publish_existing_file_once(temporary: Path, destination: Path) -> dict[str, Any]:
    """Publish a completed local file without loading a large corpus into RAM."""
    if temporary.is_symlink() or not temporary.is_file():
        raise ContractViolation("ARTIFACT_NOT_REGULAR_FILE")
    if destination.is_symlink() or destination.exists():
        raise ContractViolation("ARTIFACT_EXISTS")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise ContractViolation("ARTIFACT_EXISTS") from exc
    directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "path": str(destination),
        "sha256": artifact_sha256_file(destination),
        "bytes": destination.stat().st_size,
        "write_semantics": "ATOMIC_WRITE_ONCE",
    }


def _load_mapping(source: Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None, str]:
    if isinstance(source, Mapping):
        value = dict(source)
        return value, None, _logical_hash(value)
    path = Path(source)
    value, source_sha256 = read_verified_json_with_sha256(path)
    return value, path, source_sha256


def _read_jsonl(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        raw = (
            read_regular_file_bytes_exact(path, expected_bytes=expected_bytes)
            if expected_bytes is not None
            else read_regular_file_bytes(path)
        )
        if expected_bytes is not None and len(raw) != expected_bytes:
            raise ContractViolation("ARTIFACT_SIZE_MISMATCH")
        if expected_sha256 is not None and sha256_bytes(raw) != expected_sha256:
            raise ContractViolation("ARTIFACT_HASH_MISMATCH")
        for raw_line in raw.decode("utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ContractViolation("JSONL_ROW_NOT_OBJECT")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation("INVALID_JSONL") from exc
    return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def load_campaign_policy(path: Path | None = None) -> dict[str, Any]:
    policy = load_json(path or PROJECT_ROOT / "implementation/config/campaign_policy.json")
    if policy.get("version") != IMPLEMENTATION_REVISION:
        raise ContractViolation("BLOCKED_IMPLEMENTATION_REVISION")
    budget = policy.get("budget")
    revision = policy.get("collection_revision")
    if not isinstance(budget, dict) or not isinstance(revision, dict):
        raise ContractViolation("INVALID_CAMPAIGN_POLICY")
    cap = budget.get("api_requests_cap")
    prior = budget.get("api_requests_already_attempted")
    planned = revision.get("planned_requests")
    selected = revision.get("selected_query_count")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in (cap, prior, planned, selected)):
        raise ContractViolation("INVALID_API_BUDGET")
    if prior + planned > cap or planned != selected * len(TARGET_LANGUAGES):
        raise ContractViolation("INVALID_API_BUDGET")
    return policy


def _parse_declared_queries(source: Path | Sequence[str]) -> tuple[list[str], str | None]:
    source_hash: str | None = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ContractViolation("INVALID_QUERY_FILE")
        source_hash = artifact_sha256_file(path)
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    else:
        raw_lines = list(source)
    queries: list[str] = []
    for raw in raw_lines:
        if not isinstance(raw, str):
            raise ContractViolation("INVALID_QUERY")
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized = _single_line(stripped, "INVALID_QUERY")
        if normalized != stripped:
            raise ContractViolation("NONCANONICAL_QUERY")
        queries.append(normalized)
    if not queries:
        raise ContractViolation("EMPTY_QUERY_PLAN")
    if len(set(queries)) != len(queries):
        raise ContractViolation("DUPLICATE_QUERY")
    return queries, source_hash


def build_krdict_collection_plan(
    queries: Path | Sequence[str],
    *,
    campaign_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the frozen first-199 plan without credentials or network access."""
    policy = dict(campaign_policy or load_campaign_policy())
    budget = policy["budget"]
    revision = policy["collection_revision"]
    declared, source_hash = _parse_declared_queries(queries)
    selected_count = int(revision["selected_query_count"])
    prior = int(budget["api_requests_already_attempted"])
    cap = int(budget["api_requests_cap"])
    remaining = cap - prior
    if selected_count != remaining // len(TARGET_LANGUAGES):
        raise ContractViolation("COLLECTION_SELECTION_NOT_MAX_COMPLETE_TRIPLETS")
    if len(declared) < selected_count:
        raise ContractViolation("INSUFFICIENT_DECLARED_QUERIES")
    selected = declared[:selected_count]
    excluded = declared[selected_count:]
    requests: list[dict[str, Any]] = []
    for query_index, query in enumerate(selected):
        for language in TARGET_LANGUAGES:
            params = {
                "q": query,
                "translated": "y",
                "trans_lang": KRDICT_LANG_CODE[language],
                "advanced": "y",
                "method": "exact",
                "pos": "1",
                "part": "word",
                "num": "100",
                "start": "1",
            }
            request_core = {
                "query_index": query_index,
                "query": query,
                "language": language,
                "endpoint": KRDICT_ENDPOINT,
                "params_without_key": params,
            }
            requests.append({**request_core, "request_id": _logical_hash(request_core)})
    if len(requests) != int(revision["planned_requests"]):
        raise ContractViolation("PLANNED_REQUEST_COUNT_MISMATCH")
    core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "READY_FOR_COLLECTION",
        "selection_rule": revision["selection_rule"],
        "query_source_sha256": source_hash,
        "declared_query_count": len(declared),
        "selected_query_count": len(selected),
        "excluded_queries": [
            {"declared_index": selected_count + i, "query": query}
            for i, query in enumerate(excluded)
        ],
        "campaign_cap": cap,
        "consumed_before_plan": prior,
        "planned_requests": len(requests),
        "remaining_after_plan": remaining - len(requests),
        "requests": requests,
    }
    return {**core, "plan_sha256": _logical_hash(core)}


def _ledger_record(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**core, "record_hash": _logical_hash(core)}


def _verify_ledger_rows(rows: Sequence[Mapping[str, Any]], campaign_cap: int) -> dict[str, Any]:
    if not rows:
        raise ContractViolation("EMPTY_API_LEDGER")
    previous = "0" * 64
    consumed = 0
    attempted: set[str] = set()
    completed: set[str] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        claimed = row.pop("record_hash", None)
        require_sha256(claimed, "API_LEDGER_HASH_INVALID")
        if claimed != _logical_hash(row) or row.get("previous_hash") != previous:
            raise ContractViolation("API_LEDGER_HASH_INVALID")
        if row.get("sequence") != index:
            raise ContractViolation("API_LEDGER_SEQUENCE_INVALID")
        event = row.get("event")
        count = row.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ContractViolation("API_LEDGER_COUNT_INVALID")
        if event == "PRIOR_USAGE":
            if index != 0:
                raise ContractViolation("API_LEDGER_PRIOR_POSITION")
        elif event == "REQUEST_RESERVED":
            request_id = row.get("request_id")
            require_sha256(request_id, "API_LEDGER_REQUEST_INVALID")
            if count != 1 or request_id in attempted:
                raise ContractViolation("API_REQUEST_ALREADY_ATTEMPTED")
            attempted.add(request_id)
        elif event == "REQUEST_RESULT":
            request_id = row.get("request_id")
            require_sha256(request_id, "API_LEDGER_REQUEST_INVALID")
            if count != 0 or request_id not in attempted or request_id in completed:
                raise ContractViolation("API_LEDGER_RESULT_INVALID")
            completed.add(request_id)
        else:
            raise ContractViolation("API_LEDGER_EVENT_INVALID")
        consumed += count
        if consumed > campaign_cap:
            raise ContractViolation("API_REQUEST_BUDGET_EXHAUSTED")
        previous = claimed
    return {
        "consumed": consumed,
        "attempted": attempted,
        "completed": completed,
        "tail_hash": previous,
        "next_sequence": len(rows),
    }


@contextlib.contextmanager
def _locked_ledger(path: Path) -> Iterator[tuple[Any, list[dict[str, Any]]]]:
    if path.is_symlink():
        raise ContractViolation("API_LEDGER_SYMLINK")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows: list[dict[str, Any]] = []
        for line in handle:
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractViolation("INVALID_API_LEDGER") from exc
                if not isinstance(row, dict):
                    raise ContractViolation("INVALID_API_LEDGER")
                rows.append(row)
        yield handle, rows
    finally:
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def initialize_api_ledger(
    path: Path,
    *,
    prior_attempts: int,
    campaign_cap: int,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    path = require_relative_to(Path(path), scope_root, "API_LEDGER_OUTSIDE_SCOPE")
    with _locked_ledger(path) as (handle, rows):
        if not rows:
            core = {
                "sequence": 0,
                "previous_hash": "0" * 64,
                "event": "PRIOR_USAGE",
                "count": prior_attempts,
                "campaign_cap": campaign_cap,
                "recorded_at": _now_utc(),
            }
            record = _ledger_record(core)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            rows.append(record)
        state = _verify_ledger_rows(rows, campaign_cap)
        first = rows[0]
        if first.get("count") != prior_attempts or first.get("campaign_cap") != campaign_cap:
            raise ContractViolation("API_LEDGER_GENESIS_MISMATCH")
        return state


def _append_ledger_event(
    path: Path,
    event: Mapping[str, Any],
    *,
    prior_attempts: int,
    campaign_cap: int,
    scope_root: Path,
) -> dict[str, Any]:
    path = require_relative_to(Path(path), scope_root, "API_LEDGER_OUTSIDE_SCOPE")
    initialize_api_ledger(
        path,
        prior_attempts=prior_attempts,
        campaign_cap=campaign_cap,
        scope_root=scope_root,
    )
    with _locked_ledger(path) as (handle, rows):
        state = _verify_ledger_rows(rows, campaign_cap)
        request_id = event.get("request_id")
        if event.get("event") == "REQUEST_RESERVED":
            if request_id in state["attempted"]:
                raise ContractViolation("API_REQUEST_ALREADY_ATTEMPTED")
            if state["consumed"] + 1 > campaign_cap:
                raise ContractViolation("API_REQUEST_BUDGET_EXHAUSTED")
        core = {
            "sequence": state["next_sequence"],
            "previous_hash": state["tail_hash"],
            **dict(event),
            "recorded_at": _now_utc(),
        }
        record = _ledger_record(core)
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        rows.append(record)
        return _verify_ledger_rows(rows, campaign_cap)


@dataclass(frozen=True)
class TransportResponse:
    status: int
    content_type: str
    body: bytes


Transport = Callable[[Mapping[str, str], str, float, int], TransportResponse]


def _https_transport(
    params_without_key: Mapping[str, str],
    credential: str,
    timeout_seconds: float,
    max_response_bytes: int,
) -> TransportResponse:
    """Issue one direct TLS request.  The credential never leaves this scope."""
    params = {**dict(params_without_key), "key": credential}
    target = KRDICT_PATH + "?" + urllib.parse.urlencode(params)
    connection = http.client.HTTPSConnection(
        KRDICT_HOST,
        port=443,
        timeout=timeout_seconds,
        context=ssl.create_default_context(),
    )
    try:
        connection.request(
            "GET",
            target,
            headers={
                "User-Agent": "LexicalChoiceResearch/4.0 (bounded exact-query collector)",
                "Accept": "application/xml,text/xml",
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        content_length = response.getheader("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > max_response_bytes:
                    raise ContractViolation("RESPONSE_SIZE_LIMIT")
            except ValueError as exc:
                raise ContractViolation("INVALID_CONTENT_LENGTH") from exc
        body = response.read(max_response_bytes + 1)
        return TransportResponse(
            status=int(response.status),
            content_type=response.getheader("Content-Type", ""),
            body=body,
        )
    finally:
        connection.close()


def parse_krdict_xml(
    payload: bytes,
    *,
    expected_language: str,
    expected_query: str,
    max_response_bytes: int = MAX_XML_BYTES,
) -> dict[str, Any]:
    if expected_language not in TARGET_LANGUAGES:
        raise ContractViolation("UNKNOWN_API_LANGUAGE")
    query = _single_line(expected_query, "INVALID_QUERY")
    if not isinstance(payload, bytes) or len(payload) > max_response_bytes:
        raise ContractViolation("RESPONSE_SIZE_LIMIT")
    upper = payload.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ContractViolation("UNSAFE_XML")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ContractViolation("MALFORMED_XML") from exc
    if root.tag == "error":
        error_code = root.findtext("error_code", "UNKNOWN")
        safe = error_code if re.fullmatch(r"\d{3}", error_code or "") else "UNKNOWN"
        raise ContractViolation("KRDICT_API_ERROR_" + safe)
    if root.tag != "channel":
        raise ContractViolation("UNEXPECTED_XML_ROOT")

    def required_int(name: str) -> int:
        raw = root.findtext(name)
        try:
            value = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ContractViolation("INVALID_RESPONSE_" + name.upper()) from exc
        if value < 0:
            raise ContractViolation("INVALID_RESPONSE_" + name.upper())
        return value

    total = required_int("total")
    start = required_int("start")
    num = required_int("num")
    if start != 1 or not 10 <= num <= 100:
        raise ContractViolation("RESPONSE_PAGINATION_MISMATCH")
    items = root.findall("item")
    if len(items) > 100 or len(items) > num:
        raise ContractViolation("RESPONSE_ITEM_COUNT_INVALID")
    senses: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for item_index, item in enumerate(items):
        target_code_raw = item.findtext("target_code")
        word_raw = item.findtext("word")
        pos_raw = item.findtext("pos")
        entry_url_raw = item.findtext("link")
        target_code = (target_code_raw or "").strip()
        pos = _normalized_text(pos_raw, "UNEXPECTED_POS")
        entry_url = (entry_url_raw or "").strip()
        if not target_code.isdigit() or not word_raw:
            raise ContractViolation("MISSING_ENTRY_ID")
        parsed_entry_url = urllib.parse.urlparse(entry_url or "")
        if (
            parsed_entry_url.scheme != "https"
            or parsed_entry_url.hostname != KRDICT_HOST
            or parsed_entry_url.username
            or parsed_entry_url.password
        ):
            raise ContractViolation("INVALID_KRDICT_ENTRY_URL")
        if pos != "명사":
            raise ContractViolation("UNEXPECTED_POS")
        if _normalized_text(word_raw, "EMPTY_KO_WORD") != query:
            raise ContractViolation("NONEXACT_QUERY_RESULT")
        item_senses = item.findall("sense")
        if not item_senses:
            raise ContractViolation("MISSING_SENSE")
        for sense_index, sense in enumerate(item_senses):
            sense_order = (sense.findtext("sense_order") or "").strip()
            ko_definition = sense.findtext("definition")
            if not sense_order or not sense_order.isdigit() or not ko_definition:
                raise ContractViolation("MISSING_SENSE")
            key = (target_code, sense_order)
            if key in seen_keys:
                raise ContractViolation("DUPLICATE_SENSE_KEY_IN_RESPONSE")
            seen_keys.add(key)
            translations: list[dict[str, Any]] = []
            for translation_index, translation in enumerate(sense.findall("translation")):
                trans_lang_raw = translation.findtext("trans_lang")
                trans_lang = _normalized_text(
                    trans_lang_raw, "UNEXPECTED_TRANSLATION_LANGUAGE"
                )
                allowed = {
                    expected_language,
                    KRDICT_LANG_CODE[expected_language],
                    KRDICT_LANG_NAME[expected_language],
                }
                if trans_lang not in allowed:
                    raise ContractViolation("UNEXPECTED_TRANSLATION_LANGUAGE")
                translations.append(
                    {
                        "translation_index": translation_index,
                        "trans_lang_raw": trans_lang_raw,
                        "trans_lang": trans_lang,
                        "word_raw": translation.findtext("trans_word"),
                        "definition_raw": translation.findtext("trans_dfn"),
                    }
                )
            senses.append(
                {
                    "item_index": item_index,
                    "sense_index": sense_index,
                    "target_code": target_code,
                    "sense_order": sense_order,
                    "ko_word_raw": word_raw,
                    "ko_definition_raw": ko_definition,
                    "pos": pos,
                    "origin_raw": item.findtext("origin"),
                    "entry_url": entry_url,
                    "target_language": expected_language,
                    "translations": translations,
                }
            )
    return {
        "total_entries": total,
        "returned_entries": len(items),
        "start": start,
        "num": num,
        "all_exact_entries_returned": len(items) >= total,
        "candidate_senses": senses,
    }


def _safe_transport_response(response: Any, max_response_bytes: int) -> TransportResponse:
    if not isinstance(response, TransportResponse):
        raise ContractViolation("INVALID_TRANSPORT_RESPONSE")
    if response.status != 200:
        raise ContractViolation("HTTP_STATUS_" + str(response.status))
    media_type = response.content_type.split(";", 1)[0].strip().lower()
    if media_type not in {"application/xml", "text/xml", "application/rss+xml"}:
        raise ContractViolation("UNEXPECTED_CONTENT_TYPE")
    if len(response.body) > max_response_bytes:
        raise ContractViolation("RESPONSE_SIZE_LIMIT")
    return response


def collect_krdict(
    plan: Mapping[str, Any],
    *,
    output_dir: Path,
    ledger_path: Path,
    credential_provider: Callable[[], str | None],
    transport: Transport = _https_transport,
    campaign_policy: Mapping[str, Any] | None = None,
    scope_root: Path = WORK_ROOT,
    data_kind: str = REAL_KRDICT,
    timeout_seconds: float = 20.0,
    max_response_bytes: int = MAX_XML_BYTES,
    synthetic_transport_authorized: bool = False,
) -> dict[str, Any]:
    """Execute a request plan once, stopping at the first safe failure.

    Tests must pass an injected transport.  Production callers normally use
    the direct HTTPS transport above.  Reusing a request ID is prohibited by
    the campaign ledger, so this function never retries automatically.
    """
    if data_kind == REAL_KRDICT and CAMPAIGN_NETWORK_CLOSED:
        return {
            "schema_version": VERSION,
            "status": "BLOCKED_CAMPAIGN_EXHAUSTED",
            "requests_made": 0,
            "network_requests": 0,
            "reason": "The v4r1 campaign is closed at 3 + 597 = 600 requests; use the offline importer.",
        }
    if (
        data_kind != SYNTHETIC
        or not synthetic_transport_authorized
        or transport is _https_transport
    ):
        return {
            "schema_version": VERSION,
            "status": "BLOCKED_NETWORK_NOT_AUTHORIZED",
            "requests_made": 0,
            "network_requests": 0,
            "reason": "Only an explicitly injected synthetic test transport is permitted in v4r1.",
        }
    credential = credential_provider()
    if not isinstance(credential, str) or not KEY_RE.fullmatch(credential):
        return {
            "schema_version": VERSION,
            "status": "BLOCKED_CREDENTIALS",
            "requests_made": 0,
            "network_requests": 0,
            "reason": "Export a valid KRDICT key privately in the invoking terminal.",
        }
    plan_dict = dict(plan)
    _validate_embedded_hash(plan_dict, "plan_sha256", "COLLECTION_PLAN_HASH_MISMATCH")
    policy = dict(campaign_policy or load_campaign_policy())
    budget = policy["budget"]
    prior = int(budget["api_requests_already_attempted"])
    cap = int(budget["api_requests_cap"])
    requests = plan_dict.get("requests")
    if not isinstance(requests, list) or len(requests) != plan_dict.get("planned_requests"):
        raise ContractViolation("INVALID_COLLECTION_PLAN")
    if plan_dict.get("consumed_before_plan") != prior or prior + len(requests) > cap:
        raise ContractViolation("INVALID_COLLECTION_PLAN_BUDGET")
    output_dir = require_relative_to(Path(output_dir), scope_root, "SOURCE_OUTPUT_OUTSIDE_SCOPE")
    ledger_path = require_relative_to(Path(ledger_path), scope_root, "API_LEDGER_OUTSIDE_SCOPE")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    initialize_api_ledger(
        ledger_path,
        prior_attempts=prior,
        campaign_cap=cap,
        scope_root=scope_root,
    )
    records: list[dict[str, Any]] = []
    incomplete = False
    blocked_code: str | None = None
    network_requests = 0
    for ordinal, request in enumerate(requests):
        try:
            if not isinstance(request, dict):
                raise ContractViolation("INVALID_COLLECTION_REQUEST")
            request_id = str(request.get("request_id", ""))
            require_sha256(request_id, "INVALID_REQUEST_ID")
            request_core = _without(request, "request_id")
            if request_id != _logical_hash(request_core):
                raise ContractViolation("REQUEST_ID_MISMATCH")
            language = request.get("language")
            query = request.get("query")
            params = request.get("params_without_key")
            if language not in TARGET_LANGUAGES or not isinstance(params, dict):
                raise ContractViolation("INVALID_COLLECTION_REQUEST")
            if params.get("q") != query or "key" in params:
                raise ContractViolation("INVALID_COLLECTION_REQUEST")
            _append_ledger_event(
                ledger_path,
                {
                    "event": "REQUEST_RESERVED",
                    "count": 1,
                    "request_id": request_id,
                    "query_index": request.get("query_index"),
                    "language": language,
                },
                prior_attempts=prior,
                campaign_cap=cap,
                scope_root=scope_root,
            )
            network_requests += 1
            response = _safe_transport_response(
                transport(params, credential, timeout_seconds, max_response_bytes),
                max_response_bytes,
            )
            if credential.encode("ascii") in response.body:
                raise ContractViolation("CREDENTIAL_ECHO_IN_RESPONSE")
            parsed = parse_krdict_xml(
                response.body,
                expected_language=language,
                expected_query=str(query),
                max_response_bytes=max_response_bytes,
            )
            filename = f"{ordinal:04d}_{language}.xml"
            raw_ref = publish_bytes_once(raw_dir / filename, response.body, mode=0o444)
            record = {
                "ordinal": ordinal,
                "request_id": request_id,
                "query_index": request["query_index"],
                "query": query,
                "language": language,
                "endpoint": KRDICT_ENDPOINT,
                "params_without_key": params,
                "status": "PAYLOAD_SCHEMA_CHECKED",
                "content_type": response.content_type,
                "raw_artifact": _artifact_ref_for_manifest(raw_ref, output_dir),
                "parsed_sha256": _logical_hash(parsed),
                "summary": {
                    "total_entries": parsed["total_entries"],
                    "returned_entries": parsed["returned_entries"],
                    "candidate_sense_count": len(parsed["candidate_senses"]),
                    "all_exact_entries_returned": parsed["all_exact_entries_returned"],
                },
            }
            records.append(record)
            incomplete = incomplete or not parsed["all_exact_entries_returned"]
            _append_ledger_event(
                ledger_path,
                {
                    "event": "REQUEST_RESULT",
                    "count": 0,
                    "request_id": request_id,
                    "result": "PAYLOAD_SCHEMA_CHECKED",
                    "response_sha256": raw_ref["sha256"],
                },
                prior_attempts=prior,
                campaign_cap=cap,
                scope_root=scope_root,
            )
        except Exception as exc:  # Never stringify exceptions: URLs may contain credentials.
            blocked_code = exc.code if isinstance(exc, ContractViolation) else "TRANSPORT_OR_IO_FAILURE"
            records.append(
                {
                    "ordinal": ordinal,
                    "request_id": request.get("request_id") if isinstance(request, dict) else None,
                    "query_index": request.get("query_index") if isinstance(request, dict) else None,
                    "language": request.get("language") if isinstance(request, dict) else None,
                    "status": "BLOCKED_SOURCE",
                    "safe_error_code": blocked_code,
                    "error_type": type(exc).__name__,
                }
            )
            request_id = request.get("request_id") if isinstance(request, dict) else None
            if isinstance(request_id, str) and SHA256_RE.fullmatch(request_id):
                try:
                    _append_ledger_event(
                        ledger_path,
                        {
                            "event": "REQUEST_RESULT",
                            "count": 0,
                            "request_id": request_id,
                            "result": "BLOCKED_SOURCE",
                            "safe_error_code": blocked_code,
                        },
                        prior_attempts=prior,
                        campaign_cap=cap,
                        scope_root=scope_root,
                    )
                except Exception:
                    pass
            break
    ledger_state = initialize_api_ledger(
        ledger_path,
        prior_attempts=prior,
        campaign_cap=cap,
        scope_root=scope_root,
    )
    if blocked_code:
        status = "BLOCKED_SOURCE"
    elif incomplete:
        status = "BLOCKED_TRUNCATED_EXACT_QUERY"
    elif len(records) != len(requests):
        status = "BLOCKED_SOURCE"
    else:
        status = "COLLECTED_UNREVIEWED"
    source_core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": status,
        "data_kind": data_kind,
        "source": "KRDICT",
        "endpoint": KRDICT_ENDPOINT,
        "plan_sha256": plan_dict["plan_sha256"],
        "query_source_sha256": plan_dict.get("query_source_sha256"),
        "campaign_cap": cap,
        "prior_requests_charged": prior,
        "requests_planned": len(requests),
        "requests_made_this_call": network_requests,
        "network_requests": network_requests,
        "campaign_requests_consumed": ledger_state["consumed"],
        "ledger_tail_hash": ledger_state["tail_hash"],
        "records": records,
        "scope": "Exact declared queries; source candidates only; no human review or training approval.",
    }
    source_set_sha256 = _logical_hash(source_core)
    manifest = {
        **source_core,
        "source_set_sha256": source_set_sha256,
        "snapshot_id": "krdict-" + source_set_sha256[:16],
    }
    manifest_ref = publish_json_once(output_dir / "source_manifest.json", manifest)
    return {**manifest, "manifest_artifact": _artifact_ref_for_manifest(manifest_ref, output_dir)}


def import_existing_krdict_collection(
    legacy_manifest_path: Path,
    plan: Mapping[str, Any],
    *,
    output_dir: Path,
    campaign_policy: Mapping[str, Any] | None = None,
    production: bool = True,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Bind a completed reference-collector snapshot to the hardened plan.

    The adapter is intentionally offline: it accepts no credential or
    transport, does not copy or mutate the source XML, and records zero new
    network requests.  Every raw file is hashed and reparsed before the new
    manifest is published.
    """
    legacy_path = require_relative_to(
        Path(legacy_manifest_path), scope_root, "LEGACY_SOURCE_OUTSIDE_SCOPE"
    )
    if legacy_path.is_symlink() or not legacy_path.is_file():
        raise ContractViolation("LEGACY_MANIFEST_NOT_REGULAR_FILE")
    legacy = read_verified_json(legacy_path)
    legacy_records = legacy.get("records")
    if (
        legacy.get("status") != "COLLECTED_UNREVIEWED"
        or not isinstance(legacy_records, list)
        or legacy.get("requests_made") != len(legacy_records)
    ):
        raise ContractViolation("LEGACY_COLLECTION_INCOMPLETE")

    plan_dict = dict(plan)
    _validate_embedded_hash(plan_dict, "plan_sha256", "COLLECTION_PLAN_HASH_MISMATCH")
    requests = plan_dict.get("requests")
    if not isinstance(requests, list) or len(requests) != plan_dict.get("planned_requests"):
        raise ContractViolation("INVALID_COLLECTION_PLAN")
    if len(legacy_records) != len(requests):
        raise ContractViolation("LEGACY_PLAN_COUNT_MISMATCH")
    policy = dict(campaign_policy or load_campaign_policy())
    budget = policy.get("budget", {})
    prior = int(budget.get("api_requests_already_attempted", -1))
    cap = int(budget.get("api_requests_cap", -1))
    if prior < 0 or cap < 0 or prior + len(requests) != cap:
        raise ContractViolation("CAMPAIGN_ACCOUNTING_MISMATCH")
    if production and (prior, len(requests), cap) != (3, 597, 600):
        raise ContractViolation("CAMPAIGN_ACCOUNTING_MISMATCH")

    hardened_records: list[dict[str, Any]] = []
    incomplete = False
    for ordinal, (request, legacy_record) in enumerate(zip(requests, legacy_records, strict=True)):
        if not isinstance(request, Mapping) or not isinstance(legacy_record, Mapping):
            raise ContractViolation("LEGACY_RECORD_INVALID")
        request_id = require_sha256(
            str(request.get("request_id", "")), "INVALID_REQUEST_ID"
        )
        request_core = _without(request, "request_id")
        if request_id != _logical_hash(request_core):
            raise ContractViolation("REQUEST_ID_MISMATCH")
        expected_legacy = {
            "query_index": request.get("query_index"),
            "language": request.get("language"),
            "endpoint": request.get("endpoint"),
            "params_without_key": request.get("params_without_key"),
        }
        if any(legacy_record.get(key) != value for key, value in expected_legacy.items()):
            raise ContractViolation("LEGACY_PLAN_BINDING_MISMATCH")
        if legacy_record.get("status") != "PAYLOAD_SCHEMA_CHECKED":
            raise ContractViolation("LEGACY_RECORD_NOT_COMPLETE")
        raw_name = legacy_record.get("raw_file")
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or Path(raw_name).name != raw_name
        ):
            raise ContractViolation("LEGACY_RAW_PATH_INVALID")
        expected_name = f"{int(request['query_index']):04d}_{request['language']}.xml"
        if raw_name != expected_name:
            raise ContractViolation("LEGACY_RAW_NAME_MISMATCH")
        raw_path = require_relative_to(
            legacy_path.parent / raw_name,
            legacy_path.parent,
            "LEGACY_RAW_PATH_INVALID",
        )
        if raw_path.is_symlink() or not raw_path.is_file():
            raise ContractViolation("LEGACY_RAW_NOT_REGULAR_FILE")
        raw_bytes = read_regular_file_bytes(raw_path)
        raw_size = len(raw_bytes)
        raw_sha = sha256_bytes(raw_bytes)
        if raw_size != legacy_record.get("bytes") or raw_sha != legacy_record.get("sha256"):
            raise ContractViolation("LEGACY_RAW_HASH_MISMATCH")
        parsed = parse_krdict_xml(
            raw_bytes,
            expected_language=str(request["language"]),
            expected_query=str(request["query"]),
        )
        incomplete = incomplete or not parsed["all_exact_entries_returned"]
        hardened_records.append(
            {
                "ordinal": ordinal,
                "request_id": request_id,
                "query_index": request["query_index"],
                "query": request["query"],
                "language": request["language"],
                "endpoint": KRDICT_ENDPOINT,
                "params_without_key": request["params_without_key"],
                "status": "PAYLOAD_SCHEMA_CHECKED",
                "content_type": "application/xml",
                "raw_artifact": {
                    "path": str(raw_path),
                    "sha256": raw_sha,
                    "bytes": raw_size,
                },
                "parsed_sha256": _logical_hash(parsed),
                "summary": {
                    "total_entries": parsed["total_entries"],
                    "returned_entries": parsed["returned_entries"],
                    "candidate_sense_count": len(parsed["candidate_senses"]),
                    "all_exact_entries_returned": parsed["all_exact_entries_returned"],
                },
            }
        )
    if incomplete:
        raise ContractViolation("TRUNCATED_SOURCE")

    output_dir = require_relative_to(
        Path(output_dir), scope_root, "SOURCE_OUTPUT_OUTSIDE_SCOPE"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    legacy_ref = {
        "path": str(legacy_path),
        "sha256": artifact_sha256_file(legacy_path),
        "bytes": legacy_path.stat().st_size,
    }
    source_core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "COLLECTED_UNREVIEWED",
        "data_kind": REAL_KRDICT if production else SYNTHETIC,
        "source": "KRDICT",
        "endpoint": KRDICT_ENDPOINT,
        "collection_method": "OFFLINE_IMPORT_EXISTING_IMMUTABLE_RAW",
        "legacy_manifest": legacy_ref,
        "plan_sha256": plan_dict["plan_sha256"],
        "query_source_sha256": plan_dict.get("query_source_sha256"),
        "campaign_cap": cap,
        "prior_requests_charged": prior,
        "requests_planned": len(requests),
        "requests_previously_made_for_snapshot": len(requests),
        "requests_made_this_call": 0,
        "network_requests": 0,
        "campaign_requests_consumed": prior + len(requests),
        "records": hardened_records,
        "scope": "Exact declared queries; source candidates only; no human review or training approval.",
    }
    source_set_sha256 = _logical_hash(source_core)
    manifest = {
        **source_core,
        "source_set_sha256": source_set_sha256,
        "snapshot_id": "krdict-" + source_set_sha256[:16],
    }
    manifest_ref = publish_json_once(output_dir / "source_manifest.json", manifest)
    return {
        **manifest,
        "manifest_artifact": _artifact_ref_for_manifest(manifest_ref, output_dir),
    }


def verify_krdict_collection(
    manifest_source: Path | Mapping[str, Any],
    *,
    production: bool = True,
) -> dict[str, Any]:
    manifest, manifest_path, manifest_sha = _load_mapping(manifest_source)
    if manifest.get("schema_version") != VERSION or manifest.get("source") != "KRDICT":
        raise ContractViolation("INVALID_KRDICT_MANIFEST")
    if manifest.get("status") != "COLLECTED_UNREVIEWED":
        raise ContractViolation("KRDICT_COLLECTION_NOT_COMPLETE")
    if production and manifest.get("data_kind") != REAL_KRDICT:
        raise ContractViolation("SYNTHETIC_SOURCE_NOT_PRODUCTION")
    source_core = _without(manifest, "source_set_sha256", "snapshot_id")
    claimed = manifest.get("source_set_sha256")
    require_sha256(claimed, "SOURCE_SET_HASH_INVALID")
    if claimed != _logical_hash(source_core) or manifest.get("snapshot_id") != "krdict-" + claimed[:16]:
        raise ContractViolation("SOURCE_SET_HASH_INVALID")
    base = manifest_path.parent if manifest_path is not None else None
    if base is None:
        raise ContractViolation("KRDICT_MANIFEST_PATH_REQUIRED")
    expected_requests: list[Mapping[str, Any]] | None = None
    if production:
        policy = load_campaign_policy()
        revision = policy["collection_revision"]
        query_source = require_relative_to(
            PROJECT_ROOT / str(revision.get("source_query_file", "")),
            PROJECT_ROOT,
            "PRODUCTION_QUERY_SOURCE_OUTSIDE_PROJECT",
        )
        expected_plan = build_krdict_collection_plan(
            query_source, campaign_policy=policy
        )
        expected_requests = list(expected_plan["requests"])
        production_metadata = {
            "implementation_revision": IMPLEMENTATION_REVISION,
            "collection_method": "OFFLINE_IMPORT_EXISTING_IMMUTABLE_RAW",
            "data_kind": REAL_KRDICT,
            "endpoint": KRDICT_ENDPOINT,
            "plan_sha256": expected_plan["plan_sha256"],
            "query_source_sha256": expected_plan["query_source_sha256"],
            "requests_planned": len(expected_requests),
            "source_set_sha256": claimed,
        }
        if any(manifest.get(key) != value for key, value in production_metadata.items()):
            raise ContractViolation("PRODUCTION_SOURCE_PLAN_BINDING_MISMATCH")
        if (
            manifest.get("requests_previously_made_for_snapshot")
            != len(expected_requests)
            or manifest.get("requests_made_this_call") != 0
            or manifest.get("network_requests") != 0
        ):
            raise ContractViolation("PRODUCTION_SOURCE_ACQUISITION_MODE_MISMATCH")
        require_trusted_artifact_anchor(
            artifact_kind="KRDICT_SOURCE_COLLECTION",
            artifact_id=str(manifest.get("snapshot_id", "")),
            manifest_sha256=manifest_sha,
            bindings=production_metadata,
        )
        # The source manifest is the retained normalized snapshot, while this
        # reference preserves the exact pre-normalization acquisition ledger.
        _resolve_artifact_ref(
            manifest.get("legacy_manifest", {}),
            base=base,
            production=True,
            scope_root=WORK_ROOT,
        )
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != manifest.get("requests_planned"):
        raise ContractViolation("KRDICT_RECORD_SET_INCOMPLETE")
    seen: set[str] = set()
    verified_records: list[dict[str, Any]] = []
    if production and (
        manifest.get("requests_planned") != 597
        or manifest.get("campaign_cap") != 600
        or manifest.get("prior_requests_charged") != 3
        or manifest.get("campaign_requests_consumed") != 600
    ):
        raise ContractViolation("CAMPAIGN_ACCOUNTING_MISMATCH")
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict) or record.get("status") != "PAYLOAD_SCHEMA_CHECKED":
            raise ContractViolation("KRDICT_RECORD_NOT_VERIFIED")
        if record.get("ordinal") != ordinal:
            raise ContractViolation("KRDICT_RECORD_ORDER_MISMATCH")
        request_id = record.get("request_id")
        require_sha256(request_id, "INVALID_REQUEST_ID")
        request_core = {
            "query_index": record.get("query_index"),
            "query": record.get("query"),
            "language": record.get("language"),
            "endpoint": record.get("endpoint"),
            "params_without_key": record.get("params_without_key"),
        }
        if request_id != _logical_hash(request_core):
            raise ContractViolation("REQUEST_ID_MISMATCH")
        if expected_requests is not None:
            expected_request = expected_requests[ordinal]
            if (
                request_id != expected_request.get("request_id")
                or request_core != _without(expected_request, "request_id")
                or record.get("content_type") != "application/xml"
            ):
                raise ContractViolation("PRODUCTION_SOURCE_REQUEST_PLAN_MISMATCH")
        if request_id in seen:
            raise ContractViolation("DUPLICATE_REQUEST_ID")
        seen.add(request_id)
        raw_path = _resolve_artifact_ref(
            record.get("raw_artifact", {}),
            base=base,
            production=production,
            scope_root=WORK_ROOT,
        )
        raw_bytes = read_regular_file_bytes(raw_path)
        raw_ref = record["raw_artifact"]
        if len(raw_bytes) != raw_ref["bytes"]:
            raise ContractViolation("ARTIFACT_SIZE_MISMATCH")
        if sha256_bytes(raw_bytes) != raw_ref["sha256"]:
            raise ContractViolation("ARTIFACT_HASH_MISMATCH")
        parsed = parse_krdict_xml(
            raw_bytes,
            expected_language=record.get("language"),
            expected_query=record.get("query"),
        )
        if _logical_hash(parsed) != record.get("parsed_sha256"):
            raise ContractViolation("PARSED_SOURCE_HASH_MISMATCH")
        if not parsed["all_exact_entries_returned"]:
            raise ContractViolation("TRUNCATED_SOURCE")
        expected_summary = {
            "total_entries": parsed["total_entries"],
            "returned_entries": parsed["returned_entries"],
            "candidate_sense_count": len(parsed["candidate_senses"]),
            "all_exact_entries_returned": parsed["all_exact_entries_returned"],
        }
        if record.get("summary") != expected_summary:
            raise ContractViolation("KRDICT_RECORD_SUMMARY_MISMATCH")
        verified_records.append({**record, "raw_path": str(raw_path), "parsed": parsed})
    triplets: dict[int, list[str]] = defaultdict(list)
    for record in records:
        triplets[record["query_index"]].append(record["language"])
    if any(languages != list(TARGET_LANGUAGES) for languages in triplets.values()):
        raise ContractViolation("KRDICT_LANGUAGE_TRIPLET_INCOMPLETE")
    if sorted(triplets) != list(range(len(triplets))):
        raise ContractViolation("KRDICT_QUERY_INDEX_GAP")
    return {
        **manifest,
        "manifest_sha256": manifest_sha,
        "manifest_path": str(manifest_path),
        "verified_records": verified_records,
    }


def _option_id(language: str, answer: str, gloss: str, sources: Sequence[Mapping[str, Any]]) -> str:
    return _logical_hash(
        {
            "language": language,
            "answer": answer,
            "gloss": gloss,
            "sources": list(sources),
        }
    )


def merge_krdict_snapshot(
    manifest_source: Path | Mapping[str, Any],
    *,
    production: bool = True,
) -> dict[str, Any]:
    """Merge EN/ZH/FR calls by the official target/sense key."""
    verified = verify_krdict_collection(manifest_source, production=production)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in verified["verified_records"]:
        raw_sha = record["raw_artifact"]["sha256"]
        for sense in record["parsed"]["candidate_senses"]:
            source_ref = {
                "request_id": record["request_id"],
                "query_index": record["query_index"],
                "query": record["query"],
                "language": record["language"],
                "raw_sha256": raw_sha,
                "item_index": sense["item_index"],
                "sense_index": sense["sense_index"],
            }
            grouped[(sense["target_code"], sense["sense_order"])].append(
                {**sense, "source_ref": source_ref}
            )
    eligible: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for (target_code, sense_order), observations in sorted(grouped.items()):
        reasons: list[str] = []
        language_set = {row["target_language"] for row in observations}
        for language in TARGET_LANGUAGES:
            if language not in language_set:
                reasons.append("MISSING_LANGUAGE:" + language)
        try:
            ko_words = {
                _normalized_text(row["ko_word_raw"], "EMPTY_KO_WORD") for row in observations
            }
            ko_glosses = {
                _normalized_text(row["ko_definition_raw"], "EMPTY_KO_GLOSS")
                for row in observations
            }
        except ContractViolation as exc:
            reasons.append(exc.code)
            ko_words, ko_glosses = set(), set()
        if len(ko_words) != 1:
            reasons.append("KO_WORD_MISMATCH")
        if len(ko_glosses) != 1:
            reasons.append("KO_DEFINITION_MISMATCH")
        if {row["pos"] for row in observations} != {"명사"}:
            reasons.append("POS_MISMATCH")

        options: dict[str, list[dict[str, Any]]] = {language: [] for language in LANGUAGES}
        if len(ko_words) == 1 and len(ko_glosses) == 1:
            ko_sources = sorted(
                (row["source_ref"] for row in observations),
                key=lambda row: (row["query_index"], row["language"], row["item_index"], row["sense_index"]),
            )
            answer = next(iter(ko_words))
            gloss = next(iter(ko_glosses))
            options["ko"].append(
                {
                    "option_id": _option_id("ko", answer, gloss, ko_sources),
                    "answer": answer,
                    "gloss": gloss,
                    "word_raw": observations[0]["ko_word_raw"],
                    "definition_raw": observations[0]["ko_definition_raw"],
                    "source_refs": ko_sources,
                }
            )
        for language in TARGET_LANGUAGES:
            seen_options: dict[tuple[str, str], dict[str, Any]] = {}
            for row in observations:
                if row["target_language"] != language:
                    continue
                for translation in row["translations"]:
                    raw_word = translation.get("word_raw")
                    raw_gloss = translation.get("definition_raw")
                    try:
                        answer = _normalized_text(raw_word, "EMPTY_TRANSLATION_WORD")
                        gloss = _normalized_text(raw_gloss, "EMPTY_TRANSLATION_DEFINITION")
                    except ContractViolation:
                        continue
                    source_ref = {
                        **row["source_ref"],
                        "translation_index": translation["translation_index"],
                    }
                    key = (answer, gloss)
                    if key not in seen_options:
                        seen_options[key] = {
                            "answer": answer,
                            "gloss": gloss,
                            "word_raw": raw_word,
                            "definition_raw": raw_gloss,
                            "source_refs": [],
                        }
                    seen_options[key]["source_refs"].append(source_ref)
            for answer_gloss in sorted(seen_options):
                option = seen_options[answer_gloss]
                option["source_refs"] = sorted(
                    option["source_refs"],
                    key=lambda row: (
                        row["query_index"],
                        row["item_index"],
                        row["sense_index"],
                        row["translation_index"],
                    ),
                )
                option["option_id"] = _option_id(
                    language,
                    option["answer"],
                    option["gloss"],
                    option["source_refs"],
                )
                options[language].append(option)
            if not options[language]:
                reasons.append("MISSING_COMPLETE_TRANSLATION:" + language)
        base = {
            "schema_version": VERSION,
            "data_kind": verified["data_kind"],
            "snapshot_id": verified["snapshot_id"],
            "target_code": target_code,
            "sense_order": sense_order,
            "candidate_id": f"{verified['snapshot_id']}:{target_code}:{sense_order}",
            "queries": sorted({row["source_ref"]["query"] for row in observations}),
            "options": options,
            "source_record_hashes": sorted(
                {row["source_ref"]["raw_sha256"] for row in observations}
            ),
            "entry_urls": sorted(
                {row["entry_url"] for row in observations if row.get("entry_url")}
            ),
        }
        if reasons:
            quarantine.append(
                {
                    **base,
                    "machine_status": "QUARANTINED",
                    "reasons": sorted(set(reasons)),
                }
            )
        else:
            candidate_core = {**base, "machine_status": "PENDING_HUMAN_REVIEW"}
            eligible.append(
                {
                    **candidate_core,
                    "candidate_sha256": _logical_hash(candidate_core),
                }
            )
    core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "MERGED_PENDING_REVIEW",
        "data_kind": verified["data_kind"],
        "snapshot_id": verified["snapshot_id"],
        "source_manifest_sha256": verified["manifest_sha256"],
        "source_set_sha256": verified["source_set_sha256"],
        "eligible_candidates": eligible,
        "quarantine": quarantine,
        "summary": {
            "source_sense_keys": len(grouped),
            "eligible": len(eligible),
            "quarantined": len(quarantine),
            "quarantine_reasons": dict(
                Counter(reason for row in quarantine for reason in row["reasons"])
            ),
        },
    }
    return {**core, "merge_sha256": _logical_hash(core)}


def export_pending_review(merge_result: Mapping[str, Any]) -> dict[str, Any]:
    merge = dict(merge_result)
    _validate_embedded_hash(merge, "merge_sha256", "MERGE_HASH_MISMATCH")
    tasks = []
    for candidate in merge.get("eligible_candidates", []):
        _validate_embedded_hash(candidate, "candidate_sha256", "CANDIDATE_HASH_MISMATCH")
        tasks.append(
            {
                "task_type": "CONCEPT_AND_EN_FR_ETYMOLOGY",
                "candidate_id": candidate["candidate_id"],
                "candidate_sha256": candidate["candidate_sha256"],
                "status": "PENDING",
                "answer_options": candidate["options"],
                "required_human_checks": [
                    "same_KRDICT_sense_across_languages",
                    "one_registered_expression_per_language",
                    "answer_can_be_copied_exactly",
                    "EN_FR_etymology_is_sense_aligned",
                    "synonym_and_etymology_family_assignment",
                ],
            }
        )
    return {
        "schema_version": VERSION,
        "status": "PENDING_HUMAN_REVIEW",
        "data_kind": merge["data_kind"],
        "snapshot_id": merge["snapshot_id"],
        "merge_sha256": merge["merge_sha256"],
        "pending_review": tasks,
        "summary": {
            "pending": len(tasks),
            "automatically_approved": 0,
            "human_signoff_required": True,
        },
    }


def _validate_basic_review_identity(review: Mapping[str, Any]) -> tuple[str, str]:
    reviewer = _single_line(review.get("reviewer"), "REVIEWER_REQUIRED")
    review_date = _single_line(review.get("review_date"), "REVIEW_DATE_REQUIRED")
    try:
        parsed_date = date.fromisoformat(review_date)
    except ValueError as exc:
        raise ContractViolation("INVALID_REVIEW_DATE") from exc
    if parsed_date > date.today():
        raise ContractViolation("FUTURE_REVIEW_DATE")
    if review.get("status") != "APPROVED_BY_RESEARCHER":
        raise ContractViolation("DATA_QA_PENDING")
    return reviewer, review_date


def _validate_review_identity(review: Mapping[str, Any]) -> tuple[str, str]:
    reviewer, review_date = _validate_basic_review_identity(review)
    if review.get("source_alignment_checked") is not True:
        raise ContractViolation("SOURCE_ALIGNMENT_NOT_CHECKED")
    if review.get("answer_copy_checked") is not True:
        raise ContractViolation("ANSWER_COPY_NOT_CHECKED")
    return reviewer, review_date


def _select_option(
    candidate: Mapping[str, Any],
    language: str,
    selection: Mapping[str, Any],
) -> tuple[str, str, str, list[dict[str, Any]], list[int] | None]:
    options = candidate["options"][language]
    by_id = {option["option_id"]: option for option in options}
    option = by_id.get(selection.get("option_id"))
    if option is None:
        raise ContractViolation("SELECTION_NOT_IN_SOURCE:" + language)
    answer = _normalized_text(selection.get("answer", option["answer"]), "EMPTY_ANSWER:" + language)
    source_span = selection.get("source_span")
    if source_span is None:
        if answer != option["answer"]:
            raise ContractViolation("ANSWER_NOT_EXACT_SOURCE_OPTION:" + language)
    else:
        if (
            not isinstance(source_span, list)
            or len(source_span) != 2
            or any(not isinstance(x, int) or isinstance(x, bool) for x in source_span)
        ):
            raise ContractViolation("INVALID_SOURCE_SPAN:" + language)
        start, end = source_span
        raw = option["word_raw"]
        if start < 0 or end <= start or end > len(raw):
            raise ContractViolation("INVALID_SOURCE_SPAN:" + language)
        if _normalized_text(raw[start:end], "INVALID_SOURCE_SPAN:" + language) != answer:
            raise ContractViolation("SOURCE_SPAN_MISMATCH:" + language)
        _single_line(selection.get("selection_rationale"), "SELECTION_RATIONALE_REQUIRED:" + language)
    source_refs = option.get("source_refs")
    if not isinstance(source_refs, list) or not source_refs:
        raise ContractViolation("SELECTION_SOURCE_REFS_MISSING:" + language)
    return (
        answer,
        option["gloss"],
        option["option_id"],
        [dict(row) for row in source_refs],
        list(source_span) if source_span is not None else None,
    )


def _validate_evidence_url(value: Any) -> str:
    url = _single_line(value, "INVALID_EVIDENCE_URL")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ContractViolation("INVALID_EVIDENCE_URL")
    return url


def _validate_bound_evidence(
    evidence: Any,
    *,
    evidence_record_hashes: set[str] | None,
    production: bool,
    empty_allowed: bool,
    invalid_code: str,
) -> list[dict[str, Any]]:
    """Validate captured evidence records, never URL/hash assertions alone."""
    if not isinstance(evidence, list):
        raise ContractViolation(invalid_code)
    if not evidence and not empty_allowed:
        raise ContractViolation(invalid_code)
    checked: list[dict[str, Any]] = []
    from .review_import import validate_evidence_origin

    for item in evidence:
        if not isinstance(item, Mapping):
            raise ContractViolation(invalid_code)
        record_hash = require_sha256(
            str(item.get("record_sha256", "")), "INVALID_EVIDENCE_SHA256"
        )
        if production and (
            evidence_record_hashes is None or record_hash not in evidence_record_hashes
        ):
            raise ContractViolation("UNBOUND_REVIEW_EVIDENCE")
        origin = item.get("evidence_origin")
        validate_evidence_origin(origin, production=production)
        _validate_evidence_url(item.get("source_url"))
        _single_line(item.get("source_name"), "EVIDENCE_SOURCE_NAME_REQUIRED")
        _single_line(item.get("source_version"), "EVIDENCE_SOURCE_VERSION_REQUIRED")
        _single_line(item.get("source_license"), "EVIDENCE_SOURCE_LICENSE_REQUIRED")
        _validate_evidence_url(item.get("source_license_url"))
        _single_line(item.get("sense_locator"), "EVIDENCE_SENSE_LOCATOR_REQUIRED")
        if production or "evidence_id" in item:
            if set(item) != EVIDENCE_RECORD_FIELDS:
                raise ContractViolation("INVALID_EVIDENCE_RECORD_SCHEMA")
            _single_line(item.get("evidence_id"), "INVALID_EVIDENCE_ID")
            if item.get("subject_kind") not in {"TERM", "EN_FR_PAIR"}:
                raise ContractViolation("INVALID_EVIDENCE_SUBJECT_KIND")
            _single_line(item.get("subject_id"), "INVALID_EVIDENCE_SUBJECT_ID")
            _single_line(
                item.get("supports_label"), "INVALID_EVIDENCE_SUPPORT_LABEL"
            )
            _single_line(item.get("payload_path"), "INVALID_EVIDENCE_PAYLOAD_PATH")
            require_sha256(
                str(item.get("payload_sha256", "")),
                "INVALID_EVIDENCE_PAYLOAD_SHA256",
            )
            payload_bytes = item.get("payload_bytes")
            if (
                not isinstance(payload_bytes, int)
                or isinstance(payload_bytes, bool)
                or payload_bytes < 1
            ):
                raise ContractViolation("INVALID_EVIDENCE_PAYLOAD_SIZE")
            retrieved_at = _single_line(
                item.get("retrieved_at"), "INVALID_EVIDENCE_RETRIEVED_AT"
            )
            try:
                retrieved_time = datetime.fromisoformat(
                    retrieved_at.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise ContractViolation("INVALID_EVIDENCE_RETRIEVED_AT") from exc
            if (
                retrieved_time.tzinfo is None
                or retrieved_time > datetime.now(timezone.utc)
            ):
                raise ContractViolation("INVALID_EVIDENCE_RETRIEVED_AT")
            conflicts = item.get("conflicts_with")
            if (
                not isinstance(conflicts, list)
                or any(not isinstance(value, str) or not value for value in conflicts)
                or len(conflicts) != len(set(conflicts))
                or item["evidence_id"] in conflicts
            ):
                raise ContractViolation("INVALID_EVIDENCE_CONFLICTS")
            if record_hash != _logical_hash(_without(item, "record_sha256")):
                raise ContractViolation("EVIDENCE_RECORD_HASH_MISMATCH")
            checked.append(dict(item))
        else:
            # Backward-compatible synthetic fixtures retain the smaller
            # evidence shape; production never accepts it.
            checked.append(
                {
                    "source_url": item["source_url"],
                    "record_sha256": record_hash,
                    "evidence_origin": item["evidence_origin"],
                    "source_name": item["source_name"],
                    "source_version": item["source_version"],
                    "source_license": item["source_license"],
                    "source_license_url": item["source_license_url"],
                    "sense_locator": item["sense_locator"],
                }
            )
    return checked


def _validate_selected_term_quality(
    candidate_id: str,
    selected_answers: Mapping[str, str],
    quality_reviews: Any,
    *,
    evidence_record_hashes: set[str] | None,
    production: bool,
    frozen_record: bool = False,
) -> dict[str, dict[str, Any]] | None:
    """Validate the two selected translation terms as an independent QA layer."""
    if quality_reviews is None:
        if production:
            raise ContractViolation("SELECTED_TERM_QUALITY_REVIEWS_REQUIRED")
        return None
    if not isinstance(quality_reviews, Mapping) or set(quality_reviews) != {"en", "fr"}:
        raise ContractViolation("INVALID_SELECTED_TERM_QUALITY_REVIEWS")
    checked: dict[str, dict[str, Any]] = {}
    for language in ("en", "fr"):
        review = quality_reviews[language]
        if not isinstance(review, Mapping):
            raise ContractViolation("INVALID_SELECTED_TERM_QUALITY_REVIEW")
        expected_fields = (
            TERM_QUALITY_REVIEW_FROZEN_FIELDS
            if frozen_record
            else TERM_QUALITY_REVIEW_INPUT_FIELDS
        )
        if set(review) != expected_fields:
            raise ContractViolation("INVALID_SELECTED_TERM_QUALITY_REVIEW_SCHEMA")
        if frozen_record and review.get("answer") != selected_answers[language]:
            raise ContractViolation("TERM_QUALITY_ANSWER_MISMATCH")
        term_id = _single_line(review.get("term_id"), "TERM_ID_REQUIRED")
        if not term_id.startswith(f"{candidate_id}|{language}|"):
            raise ContractViolation("TERM_ID_SELECTION_MISMATCH")
        term_hash = require_sha256(
            str(review.get("term_sha256", "")), "INVALID_TERM_SHA256"
        )
        if review.get("segmentation_decision") != "APPROVED":
            raise ContractViolation("SELECTED_TERM_SEGMENTATION_NOT_APPROVED")
        if review.get("translation_quality") != "ATTESTED_SAME_SENSE":
            raise ContractViolation("SELECTED_TERM_QUALITY_NOT_ATTESTED")
        if not isinstance(review.get("is_transliteration"), bool):
            raise ContractViolation("INVALID_TRANSLITERATION_VALUE")
        quality_note_raw = review.get("quality_note")
        quality_note = (
            _single_line(quality_note_raw, "TERM_QUALITY_NOTE_REQUIRED")
            if quality_note_raw not in (None, "")
            else None
        )
        evidence_ids = review.get("quality_evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(not isinstance(value, str) or not value for value in evidence_ids)
            or len(evidence_ids) != len(set(evidence_ids))
        ):
            raise ContractViolation("INVALID_TERM_QUALITY_EVIDENCE_IDS")
        checked_evidence = _validate_bound_evidence(
            review.get("evidence"),
            evidence_record_hashes=evidence_record_hashes,
            production=production,
            empty_allowed=False,
            invalid_code="INVALID_TERM_QUALITY_EVIDENCE",
        )
        if len(checked_evidence) != len(evidence_ids):
            raise ContractViolation("TERM_QUALITY_EVIDENCE_COUNT_MISMATCH")
        if production or any("evidence_id" in item for item in checked_evidence):
            if [item.get("evidence_id") for item in checked_evidence] != evidence_ids:
                raise ContractViolation("TERM_QUALITY_EVIDENCE_ID_MISMATCH")
            if any(
                item.get("subject_kind") != "TERM"
                or item.get("subject_id") != term_id
                or item.get("supports_label") != "ATTESTED_SAME_SENSE"
                for item in checked_evidence
            ):
                raise ContractViolation("TERM_QUALITY_EVIDENCE_SUBJECT_MISMATCH")
        qa = review.get("qa")
        if not isinstance(qa, Mapping) or set(qa) != TERM_QUALITY_QA_FIELDS:
            raise ContractViolation("INVALID_TERM_QUALITY_QA")
        term_reviewer, term_review_date = _validate_basic_review_identity(qa)
        checked[language] = {
            "term_id": term_id,
            "term_sha256": term_hash,
            "answer": selected_answers[language],
            "segmentation_decision": "APPROVED",
            "translation_quality": "ATTESTED_SAME_SENSE",
            "is_transliteration": review["is_transliteration"],
            "quality_note": quality_note,
            "quality_evidence_ids": list(evidence_ids),
            "evidence": checked_evidence,
            "qa": {
                "status": "APPROVED_BY_RESEARCHER",
                "reviewer": term_reviewer,
                "review_date": term_review_date,
            },
        }
    return checked


def _union_find_components(
    concept_ids: Sequence[str],
    synonym_clusters: Mapping[str, str],
    family_ids: Mapping[str, str],
) -> dict[str, str]:
    parent = {concept_id: concept_id for concept_id in concept_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for grouping in (synonym_clusters, family_ids):
        buckets: dict[str, list[str]] = defaultdict(list)
        for concept_id, group_id in grouping.items():
            buckets[group_id].append(concept_id)
        for members in buckets.values():
            for other in members[1:]:
                union(members[0], other)
    components: dict[str, list[str]] = defaultdict(list)
    for concept_id in concept_ids:
        components[find(concept_id)].append(concept_id)
    result: dict[str, str] = {}
    for members in components.values():
        component_id = "component-" + _logical_hash(sorted(members))[:16]
        for concept_id in members:
            result[concept_id] = component_id
    return result


def validate_review_bundle(
    merge_result: Mapping[str, Any],
    concept_reviews: Sequence[Mapping[str, Any]],
    etymology_reviews: Sequence[Mapping[str, Any]],
    *,
    cohort_ids: Sequence[str] | None = None,
    requirements: Mapping[str, Any] | None = None,
    evidence_record_hashes: set[str] | None = None,
    production: bool = True,
) -> dict[str, Any]:
    merge = dict(merge_result)
    project_id = load_pilot_config()["project_id"]
    review_validation_mode = PRODUCTION_REVIEW_MODE if production else SYNTHETIC
    synthetic_fixture = not production
    _validate_embedded_hash(merge, "merge_sha256", "MERGE_HASH_MISMATCH")
    if merge.get("status") != "MERGED_PENDING_REVIEW":
        raise ContractViolation("INVALID_MERGE_STATUS")
    if production and merge.get("data_kind") != REAL_KRDICT:
        raise ContractViolation("SYNTHETIC_SOURCE_NOT_PRODUCTION")
    candidates = {row["candidate_id"]: row for row in merge["eligible_candidates"]}
    for candidate in candidates.values():
        _validate_embedded_hash(candidate, "candidate_sha256", "CANDIDATE_HASH_MISMATCH")
    review_map: dict[str, Mapping[str, Any]] = {}
    for review in concept_reviews:
        candidate_id = review.get("candidate_id")
        if candidate_id in review_map:
            raise ContractViolation("DUPLICATE_CONCEPT_REVIEW")
        review_map[candidate_id] = review
    selected_ids = list(cohort_ids) if cohort_ids is not None else sorted(review_map)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ContractViolation("INVALID_COHORT_IDS")
    if any(candidate_id not in candidates for candidate_id in selected_ids):
        raise ContractViolation("COHORT_CANDIDATE_MISSING")

    concepts: list[dict[str, Any]] = []
    synonym_clusters: dict[str, str] = {}
    selected_option_ids: dict[str, dict[str, str]] = {}
    for candidate_id in sorted(selected_ids):
        candidate = candidates[candidate_id]
        review = review_map.get(candidate_id)
        if not isinstance(review, Mapping):
            raise ContractViolation("MISSING_CONCEPT_REVIEW")
        if review.get("candidate_sha256") != candidate["candidate_sha256"]:
            raise ContractViolation("REVIEW_CANDIDATE_HASH_MISMATCH")
        if review.get("synthetic_fixture") is not synthetic_fixture:
            raise ContractViolation("CONCEPT_REVIEW_MODE_MISMATCH")
        qa = review.get("qa")
        if not isinstance(qa, Mapping):
            raise ContractViolation("INVALID_QA_RECORD")
        reviewer, review_date = _validate_review_identity(qa)
        _single_line(qa.get("meaning_alignment_note"), "MEANING_ALIGNMENT_NOTE_REQUIRED")
        meaning_alignment = review.get("meaning_alignment_decision")
        if meaning_alignment is None:
            if production:
                raise ContractViolation("MEANING_ALIGNMENT_DECISION_REQUIRED")
        elif meaning_alignment not in {"ALIGNED", "PARTIAL"}:
            raise ContractViolation("MEANING_ALIGNMENT_NOT_ELIGIBLE")
        selections = review.get("selections")
        if not isinstance(selections, Mapping) or set(selections) != set(LANGUAGES):
            raise ContractViolation("FOUR_SELECTIONS_REQUIRED")
        answers: dict[str, str] = {}
        glosses: dict[str, str] = {}
        option_ids: dict[str, str] = {}
        selected_source_refs: dict[str, list[dict[str, Any]]] = {}
        selected_source_spans: dict[str, list[int] | None] = {}
        for language in LANGUAGES:
            if not isinstance(selections[language], Mapping):
                raise ContractViolation("INVALID_SELECTION:" + language)
            answer, gloss, option_id, source_refs, source_span = _select_option(
                candidate, language, selections[language]
            )
            answers[language] = answer
            glosses[language] = gloss
            option_ids[language] = option_id
            selected_source_refs[language] = source_refs
            selected_source_spans[language] = source_span
        synonym_cluster = _single_line(
            review.get("synonym_cluster_id"), "SYNONYM_CLUSTER_REQUIRED"
        )
        synonym_clusters[candidate_id] = synonym_cluster
        member_map = membership(answers)
        selected_term_quality = _validate_selected_term_quality(
            candidate_id,
            answers,
            review.get("term_quality_reviews"),
            evidence_record_hashes=evidence_record_hashes,
            production=production,
        )
        concept_core = {
            "schema_version": VERSION,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "project_id": project_id,
            "status": "PASS",
            "review_validation_mode": review_validation_mode,
            "synthetic_fixture": synthetic_fixture,
            "concept_id": candidate_id,
            "snapshot_id": candidate["snapshot_id"],
            "source_candidate_sha256": candidate["candidate_sha256"],
            "source_record_hashes": candidate["source_record_hashes"],
            "source_refs": selected_source_refs,
            "source_urls": candidate["entry_urls"],
            "answers": answers,
            "glosses": glosses,
            "selected_option_ids": option_ids,
            "selected_source_spans": selected_source_spans,
            "memberships": {word: list(langs) for word, langs in member_map.items()},
            "identifiable": all(len(langs) == 1 for langs in member_map.values()),
            "synonym_cluster_id": synonym_cluster,
            "qa": {
                "status": "APPROVED_BY_RESEARCHER",
                "reviewer": reviewer,
                "review_date": review_date,
                "source_alignment_checked": True,
                "answer_copy_checked": True,
                "meaning_alignment_note": qa["meaning_alignment_note"],
                "review_candidate_sha256": candidate["candidate_sha256"],
            },
        }
        if meaning_alignment is not None:
            concept_core["meaning_alignment_decision"] = meaning_alignment
        if selected_term_quality is not None:
            concept_core["selected_term_quality"] = selected_term_quality
        concepts.append(
            {**concept_core, "concept_record_sha256": _logical_hash(concept_core)}
        )
        selected_option_ids[candidate_id] = option_ids

    etym_map: dict[str, Mapping[str, Any]] = {}
    for review in etymology_reviews:
        candidate_id = review.get("candidate_id")
        if candidate_id in etym_map:
            raise ContractViolation("DUPLICATE_ETYMOLOGY_REVIEW")
        etym_map[candidate_id] = review
    if set(etym_map) != set(selected_ids):
        raise ContractViolation("ETYMOLOGY_ROW_SET_MISMATCH")
    family_ids: dict[str, str] = {}
    etymology: list[dict[str, Any]] = []
    concept_by_id = {row["concept_id"]: row for row in concepts}
    for candidate_id in sorted(selected_ids):
        review = etym_map[candidate_id]
        concept = concept_by_id[candidate_id]
        if review.get("candidate_sha256") != concept["source_candidate_sha256"]:
            raise ContractViolation("ETYMOLOGY_CANDIDATE_HASH_MISMATCH")
        if review.get("synthetic_fixture") is not synthetic_fixture:
            raise ContractViolation("ETYMOLOGY_REVIEW_MODE_MISMATCH")
        if review.get("pair") != ["en", "fr"]:
            raise ContractViolation("PRIMARY_ETYMOLOGY_PAIR_MUST_BE_EN_FR")
        relation = review.get("relation")
        if relation not in ETYMOLOGY_LABELS:
            raise ContractViolation("UNKNOWN_ETYMOLOGY_LABEL")
        subtype_raw = review.get("relation_subtype")
        subtype_details: dict[str, Any] | None = None
        evidence_ids: list[str] | None = None
        if subtype_raw is None:
            if production:
                raise ContractViolation("ETYMOLOGY_SUBTYPE_REQUIRED")
        else:
            if not isinstance(subtype_raw, str) or subtype_raw not in ETYMOLOGY_SUBTYPE_TO_PRIMARY:
                raise ContractViolation("UNKNOWN_ETYMOLOGY_SUBTYPE")
            if ETYMOLOGY_SUBTYPE_TO_PRIMARY[subtype_raw] != relation:
                raise ContractViolation("ETYMOLOGY_SUBTYPE_PRIMARY_MISMATCH")
            direction = review.get("relation_direction")
            if direction not in ETYMOLOGY_SUBTYPE_DIRECTIONS[subtype_raw]:
                raise ContractViolation("INVALID_ETYMOLOGY_DIRECTION")
            shared_raw = review.get("shared_source")
            shared_subtypes = {
                "BORROWING_PARALLEL",
                "NEOCLASSICAL_SHARED",
                "COGNATE_INHERITED",
            }
            if subtype_raw in shared_subtypes:
                shared_source = _single_line(
                    shared_raw, "ETYMOLOGY_SHARED_SOURCE_REQUIRED"
                )
            else:
                if shared_raw not in (None, ""):
                    raise ContractViolation("ETYMOLOGY_SHARED_SOURCE_NOT_ALLOWED")
                shared_source = None
            confidence = review.get("confidence")
            if confidence not in {"HIGH", "MEDIUM", "LOW"}:
                raise ContractViolation("INVALID_ETYMOLOGY_CONFIDENCE")
            if subtype_raw == "INDETERMINATE" and confidence == "HIGH":
                raise ContractViolation("INDETERMINATE_HIGH_CONFIDENCE_FORBIDDEN")
            evidence_ids_raw = review.get("evidence_ids")
            if (
                not isinstance(evidence_ids_raw, list)
                or (not evidence_ids_raw and subtype_raw != "INDETERMINATE")
                or any(
                    not isinstance(value, str) or not value
                    for value in evidence_ids_raw
                )
                or len(evidence_ids_raw) != len(set(evidence_ids_raw))
            ):
                raise ContractViolation("INVALID_ETYMOLOGY_EVIDENCE_IDS")
            evidence_ids = list(evidence_ids_raw)
            subtype_details = {
                "relation_subtype": subtype_raw,
                "relation_direction": direction,
                "shared_source": shared_source,
                "confidence": confidence,
                "evidence_ids": evidence_ids,
            }
        qa = review.get("qa")
        if not isinstance(qa, Mapping):
            raise ContractViolation("INVALID_ETYMOLOGY_QA")
        reviewer, review_date = _validate_review_identity(qa)
        sense_note = _single_line(
            review.get("sense_alignment_note"), "ETYMOLOGY_SENSE_NOTE_REQUIRED"
        )
        historical_scope = _single_line(
            review.get("historical_scope"), "ETYMOLOGY_SCOPE_REQUIRED"
        )
        family_id = _single_line(review.get("family_id"), "ETYMOLOGY_FAMILY_REQUIRED")
        family_ids[candidate_id] = family_id
        pair_hash = _logical_hash(
            {"en": concept["answers"]["en"], "fr": concept["answers"]["fr"]}
        )
        if review.get("answer_pair_sha256") != pair_hash:
            raise ContractViolation("ETYMOLOGY_ANSWER_PAIR_HASH_MISMATCH")
        evidence = review.get("evidence")
        checked_evidence = _validate_bound_evidence(
            evidence,
            evidence_record_hashes=evidence_record_hashes,
            production=production,
            empty_allowed=relation == "UNRESOLVED",
            invalid_code="INVALID_ETYMOLOGY_EVIDENCE",
        )
        if evidence_ids is not None and len(evidence_ids) != len(checked_evidence):
            raise ContractViolation("ETYMOLOGY_EVIDENCE_COUNT_MISMATCH")
        if subtype_details is not None:
            if [item.get("evidence_id") for item in checked_evidence] != evidence_ids:
                raise ContractViolation("ETYMOLOGY_EVIDENCE_ID_MISMATCH")
            quality = concept.get("selected_term_quality")
            if not isinstance(quality, Mapping):
                raise ContractViolation("SELECTED_TERM_QUALITY_REVIEWS_REQUIRED")
            from .review_import import (
                en_fr_pair_evidence_subject_sha256,
                validate_pair_evidence_set,
            )

            evidence_subject = en_fr_pair_evidence_subject_sha256(
                candidate_id=candidate_id,
                en_term_id=str(quality["en"]["term_id"]),
                en_term_sha256=str(quality["en"]["term_sha256"]),
                en_canonical_answer=str(concept["answers"]["en"]),
                fr_term_id=str(quality["fr"]["term_id"]),
                fr_term_sha256=str(quality["fr"]["term_sha256"]),
                fr_canonical_answer=str(concept["answers"]["fr"]),
            )
            if review.get("evidence_subject_sha256") != evidence_subject:
                raise ContractViolation("ETYMOLOGY_EVIDENCE_SUBJECT_HASH_MISMATCH")
            checked_evidence = validate_pair_evidence_set(
                checked_evidence,
                subject_id=evidence_subject,
                selected_subtype=subtype_raw,
                confidence=confidence,
            )
            subtype_details["evidence_subject_sha256"] = evidence_subject
        if relation == "UNRESOLVED":
            search_note = _single_line(
                review.get("evidence_search_note"), "UNRESOLVED_SEARCH_NOTE_REQUIRED"
            )
        else:
            if not checked_evidence:
                raise ContractViolation("ETYMOLOGY_EVIDENCE_REQUIRED")
            search_note = review.get("evidence_search_note")
            if search_note is not None:
                search_note = _single_line(search_note, "INVALID_EVIDENCE_SEARCH_NOTE")
        etym_core = {
            "schema_version": VERSION,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "project_id": project_id,
            "status": "PASS",
            "review_validation_mode": review_validation_mode,
            "synthetic_fixture": synthetic_fixture,
            "concept_id": candidate_id,
            "concept_record_sha256": concept["concept_record_sha256"],
            "answer_pair_sha256": pair_hash,
            "pair": ["en", "fr"],
            "relation": relation,
            "evidence": checked_evidence,
            "evidence_search_note": search_note,
            "sense_alignment_note": sense_note,
            "historical_scope": historical_scope,
            "family_id": family_id,
            "qa": {
                "status": "APPROVED_BY_RESEARCHER",
                "reviewer": reviewer,
                "review_date": review_date,
                "source_alignment_checked": True,
                "answer_copy_checked": True,
            },
        }
        if subtype_details is not None:
            etym_core.update(subtype_details)
        etymology.append(
            {**etym_core, "etymology_record_sha256": _logical_hash(etym_core)}
        )

    components = _union_find_components(selected_ids, synonym_clusters, family_ids)
    for concept in concepts:
        concept["etymology_family_id"] = family_ids[concept["concept_id"]]
        concept["analysis_component_id"] = components[concept["concept_id"]]
        core = _without(concept, "concept_record_sha256")
        concept["concept_record_sha256"] = _logical_hash(core)
    # Component fields alter the concept hash; bind etymology to the final hash.
    final_concepts = {row["concept_id"]: row for row in concepts}
    for row in etymology:
        concept_hash = final_concepts[row["concept_id"]]["concept_record_sha256"]
        row["concept_record_sha256"] = concept_hash
        core = _without(row, "etymology_record_sha256")
        row["etymology_record_sha256"] = _logical_hash(core)

    req = dict(requirements or load_pilot_config()["data"])
    reasons: list[str] = []
    identifiable = [row for row in concepts if row["identifiable"]]
    relation_by_id = {row["concept_id"]: row["relation"] for row in etymology}
    counts = Counter(relation_by_id[row["concept_id"]] for row in identifiable)
    related = counts["BORROWING_DOCUMENTED"] + counts["SHARED_SOURCE_DOCUMENTED"]
    distinct = counts["DISTINCT_ROUTES_REVIEWED"]
    if len(concepts) < int(req["min_total"]):
        reasons.append("TOTAL_COVERAGE")
    requested_size = req.get("requested_pilot_cohort_size")
    if production and requested_size is not None and len(concepts) != int(requested_size):
        reasons.append("PILOT_COHORT_SIZE")
    if len(concepts) % 2:
        reasons.append("EVEN_COHORT_REQUIRED_FOR_PAIRS")
    if len(identifiable) < int(req["min_identifiable"]):
        reasons.append("IDENTIFIABLE_COVERAGE")
    if related < int(req["min_related"]):
        reasons.append("RELATED_COVERAGE")
    if distinct < int(req["min_distinct_routes"]):
        reasons.append("REVIEWED_CONTROL_COVERAGE")
    result_core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "project_id": project_id,
        "review_validation_mode": review_validation_mode,
        "synthetic_fixture": synthetic_fixture,
        "status": "PASS" if not reasons else "BLOCKED_DATA_COVERAGE",
        "data_kind": merge["data_kind"],
        "snapshot_id": merge["snapshot_id"],
        "source_manifest_sha256": merge["source_manifest_sha256"],
        "source_set_sha256": merge["source_set_sha256"],
        "merge_sha256": merge["merge_sha256"],
        "concepts": sorted(concepts, key=lambda row: row["concept_id"]),
        "etymology": sorted(etymology, key=lambda row: row["concept_id"]),
        "cohort": {
            "concept_ids": sorted(selected_ids),
            "identifiable_concept_ids": sorted(row["concept_id"] for row in identifiable),
        },
        "coverage": {
            "n_total": len(concepts),
            "n_identifiable": len(identifiable),
            "n_shared": len(concepts) - len(identifiable),
            "n_related_identifiable": related,
            "n_distinct_routes_identifiable": distinct,
            "relation_counts_identifiable": dict(counts),
            "requirements": req,
        },
        "reasons": reasons,
        "limits": "Recorded, hash-bound review; software cannot independently authenticate reviewer identity or prove etymological truth.",
    }
    return {**result_core, "review_bundle_sha256": _logical_hash(result_core)}


def freeze_reviewed_dataset(
    collection_manifest: Path,
    concept_reviews: Sequence[Mapping[str, Any]],
    etymology_reviews: Sequence[Mapping[str, Any]],
    *,
    cohort_ids: Sequence[str],
    output_dir: Path,
    requirements: Mapping[str, Any] | None = None,
    evidence_record_hashes: set[str] | None = None,
    evidence_records: Sequence[Mapping[str, Any]] | None = None,
    production: bool = True,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Reparse source, validate reviews, and publish an immutable annotation freeze."""
    scope_root = Path(scope_root).resolve(strict=True)
    output_dir = require_relative_to(Path(output_dir), scope_root, "FREEZE_OUTPUT_OUTSIDE_SCOPE")
    indexed_evidence: dict[str, Mapping[str, Any]] = {}
    evidence_payload_sources: dict[str, Path] = {}
    if evidence_records is None:
        if production:
            raise ContractViolation("EVIDENCE_RECORDS_REQUIRED")
    else:
        from .review_import import index_evidence_records, verify_evidence_payloads

        indexed_evidence = index_evidence_records(
            evidence_records, production=production
        )
        verified_payload_hashes = verify_evidence_payloads(indexed_evidence, scope_root)
        indexed_record_hashes = {
            str(record["record_sha256"]) for record in indexed_evidence.values()
        }
        if evidence_record_hashes != indexed_record_hashes:
            raise ContractViolation("EVIDENCE_RECORD_HASH_SET_MISMATCH")
        for record in indexed_evidence.values():
            payload_hash = str(record["payload_sha256"])
            if payload_hash not in verified_payload_hashes:
                raise ContractViolation("EVIDENCE_PAYLOAD_NOT_VERIFIED")
            raw_path = Path(str(record["payload_path"]))
            unresolved = raw_path if raw_path.is_absolute() else scope_root / raw_path
            evidence_payload_sources[payload_hash] = require_relative_to(
                unresolved, scope_root, "EVIDENCE_PAYLOAD_OUTSIDE_SCOPE"
            )
    merge = merge_krdict_snapshot(collection_manifest, production=production)
    reviewed = validate_review_bundle(
        merge,
        concept_reviews,
        etymology_reviews,
        cohort_ids=cohort_ids,
        requirements=requirements,
        evidence_record_hashes=evidence_record_hashes,
        production=production,
    )
    if reviewed["status"] != "PASS":
        raise ContractViolation("BLOCKED_DATA_COVERAGE")
    if production and reviewed["data_kind"] != REAL_KRDICT:
        raise ContractViolation("SYNTHETIC_SOURCE_NOT_PRODUCTION")
    if indexed_evidence:
        referenced_evidence_ids: set[str] = set()
        for concept in reviewed["concepts"]:
            quality = concept.get("selected_term_quality")
            if isinstance(quality, Mapping):
                for language in ("en", "fr"):
                    referenced_evidence_ids.update(
                        quality[language]["quality_evidence_ids"]
                    )
        for etymology_row in reviewed["etymology"]:
            referenced_evidence_ids.update(etymology_row.get("evidence_ids", []))
        if referenced_evidence_ids != set(indexed_evidence):
            raise ContractViolation("EVIDENCE_RECORD_SET_MISMATCH")
    if output_dir.exists():
        raise ContractViolation("ANNOTATION_FREEZE_OUTPUT_EXISTS")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    publication_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    concepts_ref = publish_bytes_once(
        publication_dir / "concepts.jsonl", _jsonl_bytes(reviewed["concepts"])
    )
    etym_ref = publish_bytes_once(
        publication_dir / "etymology_en_fr.jsonl", _jsonl_bytes(reviewed["etymology"])
    )
    evidence_records_ref: dict[str, Any] | None = None
    evidence_payload_refs: dict[str, dict[str, Any]] = {}
    if indexed_evidence:
        evidence_records_ref = publish_bytes_once(
            publication_dir / "evidence_records.jsonl",
            _jsonl_bytes(indexed_evidence[key] for key in sorted(indexed_evidence)),
        )
        for payload_hash, payload_source in sorted(evidence_payload_sources.items()):
            matching_records = [
                record
                for record in indexed_evidence.values()
                if record["payload_sha256"] == payload_hash
            ]
            expected_sizes = {record["payload_bytes"] for record in matching_records}
            if len(expected_sizes) != 1:
                raise ContractViolation("EVIDENCE_PAYLOAD_SIZE_MISMATCH")
            payload_ref = publish_verified_file_once(
                payload_source,
                publication_dir / "evidence_payloads" / f"{payload_hash}.payload",
                expected_sha256=payload_hash,
                expected_bytes=int(next(iter(expected_sizes))),
            )
            if payload_ref.get("sha256") != payload_hash:
                raise ContractViolation("EVIDENCE_PAYLOAD_CHANGED_DURING_FREEZE")
            evidence_payload_refs[payload_hash] = _artifact_ref_for_manifest(
                payload_ref, publication_dir
            )
    cohort_ref = publish_json_once(publication_dir / "cohort.json", reviewed["cohort"])
    ordered_ids = sorted(reviewed["cohort"]["concept_ids"])
    if len(ordered_ids) % 2:
        raise ContractViolation("EVEN_COHORT_REQUIRED_FOR_PAIRS")
    history_pair_rows: list[dict[str, Any]] = []
    for pair_index in range(0, len(ordered_ids), 2):
        pair_core = {
            "pair_id": f"B36-pair-{pair_index // 2:02d}",
            "pair_index": pair_index // 2,
            "concept_ids": ordered_ids[pair_index : pair_index + 2],
        }
        history_pair_rows.append(
            {**pair_core, "pair_sha256": _logical_hash(pair_core)}
        )
    history_pairs_core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "project_id": load_pilot_config()["project_id"],
        "review_validation_mode": PRODUCTION_REVIEW_MODE if production else SYNTHETIC,
        "synthetic_fixture": not production,
        "status": "PASS",
        "history_family": load_pilot_config()["preparation"]["history_family"],
        "pairing_rule": "lexicographically sort frozen concept_id and pair adjacent positions (0,1),(2,3),...",
        "concept_order": ordered_ids,
        "pairs": history_pair_rows,
    }
    history_pairs = {
        **history_pairs_core,
        "history_pair_set_sha256": _logical_hash(history_pairs_core),
    }
    history_pairs_ref = publish_json_once(
        publication_dir / "history_pairs.json", history_pairs
    )
    reviews_ref = publish_json_once(
        publication_dir / "review_audit.json",
        {
            key: value
            for key, value in reviewed.items()
            if key not in {"concepts", "etymology"}
        },
    )
    collection_path = Path(collection_manifest).resolve()
    collection_ref = {
        "path": str(collection_path),
        "sha256": artifact_sha256_file(collection_path),
        "bytes": collection_path.stat().st_size,
    }
    annotation_artifacts = {
        "concepts": _artifact_ref_for_manifest(concepts_ref, publication_dir),
        "etymology": _artifact_ref_for_manifest(etym_ref, publication_dir),
        "cohort": _artifact_ref_for_manifest(cohort_ref, publication_dir),
        "review_audit": _artifact_ref_for_manifest(reviews_ref, publication_dir),
        "history_pairs": _artifact_ref_for_manifest(history_pairs_ref, publication_dir),
    }
    annotation_gates = {
        "source": "PASS",
        "human_review_recorded": "PASS",
        "data_coverage": "PASS",
    }
    if evidence_records_ref is not None:
        annotation_artifacts["evidence_records"] = _artifact_ref_for_manifest(
            evidence_records_ref, publication_dir
        )
        annotation_gates["evidence_payloads"] = "PASS"
    core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "project_id": load_pilot_config()["project_id"],
        "review_validation_mode": PRODUCTION_REVIEW_MODE if production else SYNTHETIC,
        "synthetic_fixture": not production,
        "status": "PASS",
        "data_kind": reviewed["data_kind"],
        "freeze_kind": "ANNOTATION_FREEZE",
        "direct_trainer_input_allowed": False,
        "source_collection": collection_ref,
        "source_set_sha256": reviewed["source_set_sha256"],
        "source_collection_manifest_sha256": reviewed["source_manifest_sha256"],
        "merge_sha256": reviewed["merge_sha256"],
        "review_bundle_sha256": reviewed["review_bundle_sha256"],
        "artifacts": annotation_artifacts,
        "evidence_payloads": evidence_payload_refs,
        "gates": annotation_gates,
    }
    freeze_id = "annotation-" + _logical_hash(core)[:20]
    manifest = {**core, "freeze_id": freeze_id}
    # The manifest is deliberately published last inside a private staging
    # directory.  Only a complete bundle is atomically renamed to the requested
    # final path, so a validation or I/O failure cannot leave a plausible
    # production freeze at that path.
    manifest_ref = publish_json_once(
        publication_dir / "annotation_freeze_manifest.json", manifest
    )
    manifest_artifact = _artifact_ref_for_manifest(manifest_ref, publication_dir)
    _rename_directory_noreplace(publication_dir, output_dir)
    return {
        **manifest,
        "manifest_artifact": manifest_artifact,
    }


def _verify_concept_source_binding(
    concept: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    """Rebind a frozen concept to an exact source option and optional span."""
    if (
        concept.get("snapshot_id") != candidate.get("snapshot_id")
        or concept.get("source_candidate_sha256") != candidate.get("candidate_sha256")
        or concept.get("source_record_hashes") != candidate.get("source_record_hashes")
        or concept.get("source_urls") != candidate.get("entry_urls")
    ):
        raise ContractViolation("FROZEN_CONCEPT_SOURCE_MISMATCH")
    answers = concept.get("answers")
    glosses = concept.get("glosses")
    option_ids = concept.get("selected_option_ids")
    spans = concept.get("selected_source_spans")
    source_refs = concept.get("source_refs")
    if any(
        not isinstance(value, Mapping) or set(value) != set(LANGUAGES)
        for value in (answers, glosses, option_ids, spans, source_refs)
    ):
        raise ContractViolation("FROZEN_SELECTION_SET_MISMATCH")
    for language in LANGUAGES:
        options = candidate["options"][language]
        option = next(
            (
                value
                for value in options
                if value.get("option_id") == option_ids[language]
            ),
            None,
        )
        if not isinstance(option, Mapping):
            raise ContractViolation("FROZEN_SELECTION_NOT_IN_SOURCE")
        source_span = spans[language]
        if source_span is None:
            expected_answer = option.get("answer")
        else:
            if (
                not isinstance(source_span, list)
                or len(source_span) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in source_span
                )
            ):
                raise ContractViolation("INVALID_FROZEN_SOURCE_SPAN")
            start, end = source_span
            raw = option.get("word_raw")
            if (
                not isinstance(raw, str)
                or start < 0
                or end <= start
                or end > len(raw)
            ):
                raise ContractViolation("INVALID_FROZEN_SOURCE_SPAN")
            expected_answer = _normalized_text(
                raw[start:end], "INVALID_FROZEN_SOURCE_SPAN"
            )
        if (
            answers[language] != expected_answer
            or glosses[language] != option.get("gloss")
            or source_refs[language] != option.get("source_refs")
        ):
            raise ContractViolation("FROZEN_SELECTION_SOURCE_MISMATCH")


def _recompute_frozen_coverage(
    concepts: Sequence[Mapping[str, Any]],
    etymology: Sequence[Mapping[str, Any]],
    requirements: Mapping[str, Any],
    *,
    production: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    identifiable_ids = sorted(
        str(row["concept_id"]) for row in concepts if row.get("identifiable") is True
    )
    relation_by_id = {str(row["concept_id"]): row.get("relation") for row in etymology}
    counts = Counter(relation_by_id[value] for value in identifiable_ids)
    related = counts["BORROWING_DOCUMENTED"] + counts["SHARED_SOURCE_DOCUMENTED"]
    distinct = counts["DISTINCT_ROUTES_REVIEWED"]
    reasons: list[str] = []
    if len(concepts) < int(requirements["min_total"]):
        reasons.append("TOTAL_COVERAGE")
    requested = requirements.get("requested_pilot_cohort_size")
    if production and requested is not None and len(concepts) != int(requested):
        reasons.append("PILOT_COHORT_SIZE")
    if len(concepts) % 2:
        reasons.append("EVEN_COHORT_REQUIRED_FOR_PAIRS")
    if len(identifiable_ids) < int(requirements["min_identifiable"]):
        reasons.append("IDENTIFIABLE_COVERAGE")
    if related < int(requirements["min_related"]):
        reasons.append("RELATED_COVERAGE")
    if distinct < int(requirements["min_distinct_routes"]):
        reasons.append("REVIEWED_CONTROL_COVERAGE")
    coverage = {
        "n_total": len(concepts),
        "n_identifiable": len(identifiable_ids),
        "n_shared": len(concepts) - len(identifiable_ids),
        "n_related_identifiable": related,
        "n_distinct_routes_identifiable": distinct,
        "relation_counts_identifiable": dict(counts),
        "requirements": dict(requirements),
    }
    return coverage, reasons, identifiable_ids


def verify_annotation_freeze(
    manifest_source: Path | Mapping[str, Any],
    *,
    production: bool = True,
) -> dict[str, Any]:
    manifest, manifest_path, manifest_sha = _load_mapping(manifest_source)
    if manifest_path is None:
        raise ContractViolation("ANNOTATION_FREEZE_PATH_REQUIRED")
    project_id = load_pilot_config()["project_id"]
    expected_review_mode = PRODUCTION_REVIEW_MODE if production else SYNTHETIC
    expected_synthetic_fixture = not production
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("implementation_revision") != IMPLEMENTATION_REVISION
        or manifest.get("project_id") != project_id
        or manifest.get("freeze_kind") != "ANNOTATION_FREEZE"
        or manifest.get("status") != "PASS"
        or manifest.get("direct_trainer_input_allowed") is not False
    ):
        raise ContractViolation("INVALID_ANNOTATION_FREEZE")
    if (
        manifest.get("review_validation_mode") != expected_review_mode
        or manifest.get("synthetic_fixture") is not expected_synthetic_fixture
    ):
        raise ContractViolation(
            "SYNTHETIC_FREEZE_NOT_PRODUCTION"
            if production
            else "ANNOTATION_REVIEW_MODE_MISMATCH"
        )
    if production and manifest.get("data_kind") != REAL_KRDICT:
        raise ContractViolation("SYNTHETIC_FREEZE_NOT_PRODUCTION")
    core = _without(manifest, "freeze_id")
    expected_id = "annotation-" + _logical_hash(core)[:20]
    if manifest.get("freeze_id") != expected_id:
        raise ContractViolation("ANNOTATION_FREEZE_ID_MISMATCH")
    if production:
        require_trusted_artifact_anchor(
            artifact_kind="ANNOTATION_FREEZE",
            artifact_id=expected_id,
            manifest_sha256=manifest_sha,
            bindings={
                "data_kind": manifest.get("data_kind"),
                "freeze_kind": "ANNOTATION_FREEZE",
                "implementation_revision": IMPLEMENTATION_REVISION,
                "merge_sha256": manifest.get("merge_sha256"),
                "review_bundle_sha256": manifest.get("review_bundle_sha256"),
                "source_collection_manifest_sha256": manifest.get(
                    "source_collection_manifest_sha256"
                ),
                "source_set_sha256": manifest.get("source_set_sha256"),
            },
        )
    artifacts = manifest.get("artifacts")
    base_artifacts = {
        "concepts",
        "etymology",
        "cohort",
        "review_audit",
        "history_pairs",
    }
    if not isinstance(artifacts, Mapping) or frozenset(artifacts) not in {
        frozenset(base_artifacts),
        frozenset(base_artifacts | {"evidence_records"}),
    }:
        raise ContractViolation("ANNOTATION_ARTIFACT_SET_INVALID")
    has_evidence_artifact = "evidence_records" in artifacts
    if production and not has_evidence_artifact:
        raise ContractViolation("ANNOTATION_EVIDENCE_ARTIFACT_REQUIRED")
    resolved = {
        name: _resolve_artifact_ref(
            ref,
            base=manifest_path.parent,
            production=production,
            scope_root=WORK_ROOT,
        )
        for name, ref in artifacts.items()
    }
    concepts = _read_jsonl(
        resolved["concepts"],
        expected_sha256=str(artifacts["concepts"]["sha256"]),
        expected_bytes=int(artifacts["concepts"]["bytes"]),
    )
    etymology = _read_jsonl(
        resolved["etymology"],
        expected_sha256=str(artifacts["etymology"]["sha256"]),
        expected_bytes=int(artifacts["etymology"]["bytes"]),
    )
    cohort = read_verified_json(
        resolved["cohort"], expected_sha256=str(artifacts["cohort"]["sha256"])
    )
    review_audit = read_verified_json(
        resolved["review_audit"],
        expected_sha256=str(artifacts["review_audit"]["sha256"]),
    )
    history_pairs = read_verified_json(
        resolved["history_pairs"],
        expected_sha256=str(artifacts["history_pairs"]["sha256"]),
    )
    frozen_evidence_records: list[dict[str, Any]] = []
    indexed_frozen_evidence: dict[str, Mapping[str, Any]] = {}
    if has_evidence_artifact:
        frozen_evidence_records = _read_jsonl(
            resolved["evidence_records"],
            expected_sha256=str(artifacts["evidence_records"]["sha256"]),
            expected_bytes=int(artifacts["evidence_records"]["bytes"]),
        )
        from .review_import import index_evidence_records

        indexed_frozen_evidence = index_evidence_records(
            frozen_evidence_records, production=production
        )
    payload_refs = manifest.get("evidence_payloads")
    if not isinstance(payload_refs, Mapping):
        raise ContractViolation("INVALID_EVIDENCE_PAYLOAD_ARTIFACTS")
    resolved_evidence_payloads: dict[str, Path] = {}
    verified_evidence_payload_sizes: dict[str, int] = {}
    for payload_hash, payload_ref in payload_refs.items():
        checked_payload_hash = require_sha256(
            str(payload_hash), "INVALID_EVIDENCE_PAYLOAD_SHA256"
        )
        if (
            not isinstance(payload_ref, Mapping)
            or payload_ref.get("sha256") != checked_payload_hash
        ):
            raise ContractViolation("EVIDENCE_PAYLOAD_SHA256_MISMATCH")
        payload_path = _resolve_artifact_ref(
            payload_ref,
            base=manifest_path.parent,
            production=production,
            scope_root=WORK_ROOT,
        )
        payload_bytes = read_regular_file_bytes_exact(
            payload_path,
            expected_bytes=int(payload_ref.get("bytes")),
        )
        if len(payload_bytes) != payload_ref.get("bytes"):
            raise ContractViolation("EVIDENCE_PAYLOAD_SIZE_MISMATCH")
        if sha256_bytes(payload_bytes) != checked_payload_hash:
            raise ContractViolation("EVIDENCE_PAYLOAD_SHA256_MISMATCH")
        resolved_evidence_payloads[checked_payload_hash] = payload_path
        verified_evidence_payload_sizes[checked_payload_hash] = len(payload_bytes)
    if has_evidence_artifact:
        expected_payload_hashes = {
            str(record["payload_sha256"])
            for record in indexed_frozen_evidence.values()
        }
        if set(resolved_evidence_payloads) != expected_payload_hashes:
            raise ContractViolation("EVIDENCE_PAYLOAD_ARTIFACT_SET_MISMATCH")
        for record in indexed_frozen_evidence.values():
            payload_hash = str(record["payload_sha256"])
            if verified_evidence_payload_sizes[payload_hash] != record["payload_bytes"]:
                raise ContractViolation("EVIDENCE_PAYLOAD_SIZE_MISMATCH")
    elif payload_refs:
        raise ContractViolation("ORPHAN_EVIDENCE_PAYLOAD_ARTIFACT")
    if any(
        not isinstance(value, Mapping)
        for value in (cohort, review_audit, history_pairs)
    ):
        raise ContractViolation("INVALID_ANNOTATION_METADATA")
    if (
        review_audit.get("schema_version") != VERSION
        or review_audit.get("implementation_revision") != IMPLEMENTATION_REVISION
        or review_audit.get("project_id") != project_id
        or review_audit.get("review_validation_mode") != expected_review_mode
        or review_audit.get("synthetic_fixture") is not expected_synthetic_fixture
        or review_audit.get("status") != "PASS"
    ):
        raise ContractViolation("INVALID_REVIEW_AUDIT_METADATA")
    reconstructed_review = {
        **_without(review_audit, "review_bundle_sha256"),
        "concepts": concepts,
        "etymology": etymology,
    }
    if review_audit.get("review_bundle_sha256") != _logical_hash(reconstructed_review):
        raise ContractViolation("REVIEW_BUNDLE_HASH_MISMATCH")
    if review_audit.get("review_bundle_sha256") != manifest.get("review_bundle_sha256"):
        raise ContractViolation("REVIEW_BUNDLE_HASH_MISMATCH")
    ids = [row.get("concept_id") for row in concepts]
    if (
        not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
        or not isinstance(cohort.get("concept_ids"), list)
        or set(ids) != set(cohort.get("concept_ids", []))
    ):
        raise ContractViolation("ANNOTATION_COHORT_MISMATCH")
    _validate_embedded_hash(
        history_pairs,
        "history_pair_set_sha256",
        "HISTORY_PAIR_SET_HASH_MISMATCH",
    )
    if (
        history_pairs.get("schema_version") != VERSION
        or history_pairs.get("implementation_revision") != IMPLEMENTATION_REVISION
        or history_pairs.get("project_id") != project_id
        or history_pairs.get("review_validation_mode") != expected_review_mode
        or history_pairs.get("synthetic_fixture") is not expected_synthetic_fixture
        or history_pairs.get("status") != "PASS"
    ):
        raise ContractViolation("INVALID_HISTORY_PAIR_METADATA")
    ordered_ids = sorted(ids)
    if (
        history_pairs.get("history_family")
        != load_pilot_config()["preparation"]["history_family"]
        or history_pairs.get("pairing_rule")
        != "lexicographically sort frozen concept_id and pair adjacent positions (0,1),(2,3),..."
        or history_pairs.get("concept_order") != ordered_ids
    ):
        raise ContractViolation("HISTORY_PAIR_RULE_MISMATCH")
    pairs = history_pairs.get("pairs")
    if not isinstance(pairs, list) or len(pairs) * 2 != len(ordered_ids):
        raise ContractViolation("HISTORY_PAIR_COVERAGE_MISMATCH")
    flattened: list[str] = []
    for pair_index, pair in enumerate(pairs):
        _validate_embedded_hash(pair, "pair_sha256", "HISTORY_PAIR_HASH_MISMATCH")
        expected_pair_ids = ordered_ids[pair_index * 2 : pair_index * 2 + 2]
        if (
            pair.get("pair_id") != f"B36-pair-{pair_index:02d}"
            or pair.get("pair_index") != pair_index
            or pair.get("concept_ids") != expected_pair_ids
        ):
            raise ContractViolation("HISTORY_PAIR_RULE_MISMATCH")
        flattened.extend(pair["concept_ids"])
    if flattened != ordered_ids:
        raise ContractViolation("HISTORY_PAIR_COVERAGE_MISMATCH")
    etymology_ids = [row.get("concept_id") for row in etymology]
    if (
        any(not isinstance(value, str) or not value for value in etymology_ids)
        or len(etymology_ids) != len(set(etymology_ids))
        or set(etymology_ids) != set(ids)
    ):
        raise ContractViolation("ETYMOLOGY_ROW_SET_MISMATCH")
    for concept in concepts:
        _validate_embedded_hash(concept, "concept_record_sha256", "CONCEPT_RECORD_HASH_MISMATCH")
        expected_concept_fields = set(FROZEN_CONCEPT_FIELDS)
        expected_concept_fields.update(
            field for field in FROZEN_CONCEPT_OPTIONAL_FIELDS if field in concept
        )
        if set(concept) != expected_concept_fields:
            raise ContractViolation("CONCEPT_RECORD_SCHEMA_MISMATCH")
        if (
            concept.get("schema_version") != VERSION
            or concept.get("implementation_revision") != IMPLEMENTATION_REVISION
            or concept.get("project_id") != project_id
            or concept.get("review_validation_mode") != expected_review_mode
            or concept.get("synthetic_fixture") is not expected_synthetic_fixture
            or concept.get("status") != "PASS"
        ):
            raise ContractViolation("INVALID_CONCEPT_METADATA")
        if set(concept.get("answers", {})) != set(LANGUAGES):
            raise ContractViolation("FOUR_ANSWERS_REQUIRED")
        expected_membership = {
            word: list(langs) for word, langs in membership(concept["answers"]).items()
        }
        if concept.get("memberships") != expected_membership:
            raise ContractViolation("MEMBERSHIP_MISMATCH")
        expected_identifiable = all(
            len(languages) == 1 for languages in expected_membership.values()
        )
        if concept.get("identifiable") is not expected_identifiable:
            raise ContractViolation("IDENTIFIABILITY_MISMATCH")
        concept_qa = concept.get("qa")
        if (
            not isinstance(concept_qa, Mapping)
            or set(concept_qa) != FROZEN_CONCEPT_QA_FIELDS
        ):
            raise ContractViolation("INVALID_QA_RECORD")
        _validate_review_identity(concept_qa)
        _single_line(
            concept_qa.get("meaning_alignment_note"),
            "MEANING_ALIGNMENT_NOTE_REQUIRED",
        )
        _single_line(
            concept.get("synonym_cluster_id"), "SYNONYM_CLUSTER_REQUIRED"
        )
        meaning_alignment = concept.get("meaning_alignment_decision")
        if production or meaning_alignment is not None:
            if meaning_alignment not in {"ALIGNED", "PARTIAL"}:
                raise ContractViolation("MEANING_ALIGNMENT_NOT_ELIGIBLE")
        if concept_qa.get("review_candidate_sha256") != concept.get(
            "source_candidate_sha256"
        ):
            raise ContractViolation("REVIEW_CANDIDATE_HASH_MISMATCH")
        if production or concept.get("selected_term_quality") is not None:
            _validate_selected_term_quality(
                str(concept.get("concept_id", "")),
                concept["answers"],
                concept.get("selected_term_quality"),
                evidence_record_hashes=None,
                production=False,
                frozen_record=True,
            )
    concepts_by_id = {row["concept_id"]: row for row in concepts}
    for row in etymology:
        _validate_embedded_hash(
            row, "etymology_record_sha256", "ETYMOLOGY_RECORD_HASH_MISMATCH"
        )
        expected_etymology_fields = set(FROZEN_ETYMOLOGY_FIELDS)
        if "relation_subtype" in row:
            expected_etymology_fields.update(FROZEN_ETYMOLOGY_SUBTYPE_FIELDS)
        if set(row) != expected_etymology_fields:
            raise ContractViolation("ETYMOLOGY_RECORD_SCHEMA_MISMATCH")
        if (
            row.get("schema_version") != VERSION
            or row.get("implementation_revision") != IMPLEMENTATION_REVISION
            or row.get("project_id") != project_id
            or row.get("review_validation_mode") != expected_review_mode
            or row.get("synthetic_fixture") is not expected_synthetic_fixture
            or row.get("status") != "PASS"
        ):
            raise ContractViolation("INVALID_ETYMOLOGY_METADATA")
        concept = concepts_by_id[row["concept_id"]]
        if row.get("concept_record_sha256") != concept.get("concept_record_sha256"):
            raise ContractViolation("ETYMOLOGY_CONCEPT_HASH_MISMATCH")
        if row.get("pair") != ["en", "fr"]:
            raise ContractViolation("PRIMARY_ETYMOLOGY_PAIR_MUST_BE_EN_FR")
        relation = row.get("relation")
        if relation not in ETYMOLOGY_LABELS:
            raise ContractViolation("UNKNOWN_ETYMOLOGY_LABEL")
        expected_pair_hash = _logical_hash(
            {"en": concept["answers"]["en"], "fr": concept["answers"]["fr"]}
        )
        if row.get("answer_pair_sha256") != expected_pair_hash:
            raise ContractViolation("ETYMOLOGY_ANSWER_PAIR_HASH_MISMATCH")
        etymology_qa = row.get("qa")
        if (
            not isinstance(etymology_qa, Mapping)
            or set(etymology_qa) != FROZEN_ETYMOLOGY_QA_FIELDS
        ):
            raise ContractViolation("INVALID_ETYMOLOGY_QA")
        _validate_review_identity(etymology_qa)
        _single_line(
            row.get("sense_alignment_note"), "ETYMOLOGY_SENSE_NOTE_REQUIRED"
        )
        _single_line(row.get("historical_scope"), "ETYMOLOGY_SCOPE_REQUIRED")
        _single_line(row.get("family_id"), "ETYMOLOGY_FAMILY_REQUIRED")
        _validate_bound_evidence(
            row.get("evidence"),
            evidence_record_hashes=None,
            production=False,
            empty_allowed=relation == "UNRESOLVED",
            invalid_code="INVALID_ETYMOLOGY_EVIDENCE",
        )
        if relation == "UNRESOLVED":
            _single_line(
                row.get("evidence_search_note"), "UNRESOLVED_SEARCH_NOTE_REQUIRED"
            )
        elif row.get("evidence_search_note") is not None:
            _single_line(
                row.get("evidence_search_note"), "INVALID_EVIDENCE_SEARCH_NOTE"
            )
        if production or row.get("relation_subtype") is not None:
            subtype = row.get("relation_subtype")
            if (
                not isinstance(subtype, str)
                or subtype not in ETYMOLOGY_SUBTYPE_TO_PRIMARY
            ):
                raise ContractViolation("UNKNOWN_ETYMOLOGY_SUBTYPE")
            if ETYMOLOGY_SUBTYPE_TO_PRIMARY[subtype] != row.get("relation"):
                raise ContractViolation("ETYMOLOGY_SUBTYPE_PRIMARY_MISMATCH")
            if row.get("relation_direction") not in ETYMOLOGY_SUBTYPE_DIRECTIONS[subtype]:
                raise ContractViolation("INVALID_ETYMOLOGY_DIRECTION")
            shared_source = row.get("shared_source")
            if subtype in {
                "BORROWING_PARALLEL",
                "NEOCLASSICAL_SHARED",
                "COGNATE_INHERITED",
            }:
                _single_line(shared_source, "ETYMOLOGY_SHARED_SOURCE_REQUIRED")
            elif shared_source not in (None, ""):
                raise ContractViolation("ETYMOLOGY_SHARED_SOURCE_NOT_ALLOWED")
            if row.get("confidence") not in {"HIGH", "MEDIUM", "LOW"}:
                raise ContractViolation("INVALID_ETYMOLOGY_CONFIDENCE")
            if subtype == "INDETERMINATE" and row.get("confidence") == "HIGH":
                raise ContractViolation("INDETERMINATE_HIGH_CONFIDENCE_FORBIDDEN")
            evidence_ids = row.get("evidence_ids")
            frozen_evidence = row.get("evidence")
            if (
                not isinstance(evidence_ids, list)
                or (not evidence_ids and subtype != "INDETERMINATE")
                or not isinstance(frozen_evidence, list)
                or any(
                    not isinstance(value, str) or not value for value in evidence_ids
                )
                or len(evidence_ids) != len(set(evidence_ids))
                or len(evidence_ids) != len(frozen_evidence)
            ):
                raise ContractViolation("ETYMOLOGY_EVIDENCE_COUNT_MISMATCH")
            if [item.get("evidence_id") for item in frozen_evidence] != evidence_ids:
                raise ContractViolation("ETYMOLOGY_EVIDENCE_ID_MISMATCH")
            selected_quality = concept.get("selected_term_quality")
            if not isinstance(selected_quality, Mapping):
                raise ContractViolation("SELECTED_TERM_QUALITY_REVIEWS_REQUIRED")
            from .review_import import (
                en_fr_pair_evidence_subject_sha256,
                validate_pair_evidence_set,
            )

            evidence_subject = en_fr_pair_evidence_subject_sha256(
                candidate_id=str(concept["concept_id"]),
                en_term_id=str(selected_quality["en"]["term_id"]),
                en_term_sha256=str(selected_quality["en"]["term_sha256"]),
                en_canonical_answer=str(concept["answers"]["en"]),
                fr_term_id=str(selected_quality["fr"]["term_id"]),
                fr_term_sha256=str(selected_quality["fr"]["term_sha256"]),
                fr_canonical_answer=str(concept["answers"]["fr"]),
            )
            if row.get("evidence_subject_sha256") != evidence_subject:
                raise ContractViolation("ETYMOLOGY_EVIDENCE_SUBJECT_HASH_MISMATCH")
            validate_pair_evidence_set(
                frozen_evidence,
                subject_id=evidence_subject,
                selected_subtype=subtype,
                confidence=str(row.get("confidence")),
            )
    if has_evidence_artifact:
        referenced_evidence_ids: set[str] = set()
        for concept in concepts:
            selected_quality = concept.get("selected_term_quality")
            if not isinstance(selected_quality, Mapping):
                if production:
                    raise ContractViolation("SELECTED_TERM_QUALITY_REVIEWS_REQUIRED")
                continue
            for language in ("en", "fr"):
                quality = selected_quality[language]
                for evidence_id, evidence_item in zip(
                    quality["quality_evidence_ids"], quality["evidence"]
                ):
                    if indexed_frozen_evidence.get(evidence_id) != evidence_item:
                        raise ContractViolation("FROZEN_EVIDENCE_RECORD_MISMATCH")
                    referenced_evidence_ids.add(evidence_id)
        for row in etymology:
            for evidence_id, evidence_item in zip(
                row["evidence_ids"], row["evidence"]
            ):
                if indexed_frozen_evidence.get(evidence_id) != evidence_item:
                    raise ContractViolation("FROZEN_EVIDENCE_RECORD_MISMATCH")
                referenced_evidence_ids.add(evidence_id)
        if referenced_evidence_ids != set(indexed_frozen_evidence):
            raise ContractViolation("EVIDENCE_RECORD_SET_MISMATCH")
    source_ref = manifest.get("source_collection")
    if not isinstance(source_ref, Mapping):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    source_ref_sha256 = require_sha256(
        str(source_ref.get("sha256", "")), "INVALID_ARTIFACT_SHA256"
    )
    source_path = _resolve_artifact_ref(
        source_ref,
        base=manifest_path.parent,
        production=production,
        scope_root=WORK_ROOT,
    )
    source = verify_krdict_collection(source_path, production=production)
    if (
        source.get("manifest_sha256") != source_ref_sha256
        or source.get("source_set_sha256") != manifest.get("source_set_sha256")
        or source.get("manifest_sha256")
        != manifest.get("source_collection_manifest_sha256")
    ):
        raise ContractViolation("ANNOTATION_SOURCE_BINDING_MISMATCH")
    rebuilt_merge = merge_krdict_snapshot(source_path, production=production)
    if (
        rebuilt_merge.get("source_manifest_sha256") != source.get("manifest_sha256")
        or rebuilt_merge.get("source_set_sha256") != source.get("source_set_sha256")
        or rebuilt_merge.get("snapshot_id") != source.get("snapshot_id")
    ):
        raise ContractViolation("ANNOTATION_SOURCE_BINDING_MISMATCH")
    if rebuilt_merge.get("merge_sha256") != manifest.get("merge_sha256"):
        raise ContractViolation("ANNOTATION_MERGE_BINDING_MISMATCH")
    candidates = {
        row["candidate_id"]: row for row in rebuilt_merge["eligible_candidates"]
    }
    for concept in concepts:
        candidate = candidates.get(concept["concept_id"])
        if not isinstance(candidate, Mapping):
            raise ContractViolation("FROZEN_CONCEPT_NOT_IN_SOURCE")
        _verify_concept_source_binding(concept, candidate)
    if production:
        # Rebuild the deterministic term catalog so term IDs/hashes cannot be
        # asserted independently of the selected source option and span.
        from .review_import import build_review_tables

        source_terms = {
            row["term_id"]: row for row in build_review_tables(rebuilt_merge).terms
        }
        for concept in concepts:
            for language in ("en", "fr"):
                quality = concept["selected_term_quality"][language]
                term = source_terms.get(quality["term_id"])
                if (
                    term is None
                    or quality["term_sha256"] != term["term_sha256"]
                    or term["candidate_id"] != concept["concept_id"]
                    or term["lang"] != language
                    or term["term_canonical"] != concept["answers"][language]
                    or term["source_option_id"]
                    != concept["selected_option_ids"][language]
                    or [
                        int(term["source_span_start"]),
                        int(term["source_span_end"]),
                    ]
                    != concept["selected_source_spans"][language]
                ):
                    raise ContractViolation("FROZEN_TERM_SOURCE_MISMATCH")

    etymology_by_id = {row["concept_id"]: row for row in etymology}
    expected_components = _union_find_components(
        [str(value) for value in ids],
        {
            row["concept_id"]: str(row["synonym_cluster_id"])
            for row in concepts
        },
        {
            row["concept_id"]: str(etymology_by_id[row["concept_id"]]["family_id"])
            for row in concepts
        },
    )
    for concept in concepts:
        concept_id = concept["concept_id"]
        if (
            concept.get("etymology_family_id")
            != etymology_by_id[concept_id].get("family_id")
            or concept.get("analysis_component_id")
            != expected_components[concept_id]
        ):
            raise ContractViolation("ANALYSIS_COMPONENT_MISMATCH")

    audit_coverage = review_audit.get("coverage")
    if not isinstance(audit_coverage, Mapping):
        raise ContractViolation("INVALID_REVIEW_COVERAGE")
    requirements = audit_coverage.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ContractViolation("INVALID_REVIEW_REQUIREMENTS")
    if production and dict(requirements) != dict(load_pilot_config()["data"]):
        raise ContractViolation("PILOT_REQUIREMENTS_MISMATCH")
    expected_coverage, coverage_reasons, identifiable_ids = _recompute_frozen_coverage(
        concepts,
        etymology,
        requirements,
        production=production,
    )
    expected_cohort = {
        "concept_ids": sorted(str(value) for value in ids),
        "identifiable_concept_ids": identifiable_ids,
    }
    if cohort != expected_cohort or review_audit.get("cohort") != expected_cohort:
        raise ContractViolation("ANNOTATION_COHORT_MISMATCH")
    if (
        coverage_reasons
        or dict(audit_coverage) != expected_coverage
        or review_audit.get("status") != "PASS"
        or review_audit.get("reasons") != []
        or review_audit.get("data_kind") != manifest.get("data_kind")
        or review_audit.get("snapshot_id") != rebuilt_merge.get("snapshot_id")
        or review_audit.get("source_manifest_sha256")
        != source.get("manifest_sha256")
        or review_audit.get("source_set_sha256") != source.get("source_set_sha256")
        or review_audit.get("merge_sha256") != rebuilt_merge.get("merge_sha256")
    ):
        raise ContractViolation("ANNOTATION_REVIEW_AUDIT_MISMATCH")
    expected_gates = {
        "source": "PASS",
        "human_review_recorded": "PASS",
        "data_coverage": "PASS",
    }
    if has_evidence_artifact:
        expected_gates["evidence_payloads"] = "PASS"
    if manifest.get("gates") != expected_gates:
        raise ContractViolation("ANNOTATION_GATE_MISMATCH")
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "resolved_artifacts": {key: str(value) for key, value in resolved.items()},
        "concepts": concepts,
        "etymology": etymology,
        "cohort": cohort,
        "review_audit": review_audit,
        "history_pairs": history_pairs,
        "evidence_records": frozen_evidence_records,
        "resolved_evidence_payloads": {
            key: str(value) for key, value in resolved_evidence_payloads.items()
        },
    }


def iter_wiki40b_jsonl_export(
    path: Path,
    *,
    expected_sha256: str,
    maximum_line_bytes: int = 32 * 1024 * 1024,
) -> Iterator[dict[str, Any]]:
    """Yield a hash-pinned, newline-delimited offline TFDS export."""
    source = Path(path)
    require_sha256(expected_sha256, "WIKI40B_EXPORT_SHA256_INVALID")
    if source.is_symlink() or not source.is_file():
        raise ContractViolation("WIKI40B_EXPORT_NOT_REGULAR_FILE")
    if artifact_sha256_file(source) != expected_sha256:
        raise ContractViolation("WIKI40B_EXPORT_HASH_MISMATCH")
    with source.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line) > maximum_line_bytes:
                raise ContractViolation("WIKI40B_EXPORT_LINE_TOO_LARGE")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractViolation("WIKI40B_EXPORT_INVALID_JSON") from exc
            if not isinstance(value, dict):
                raise ContractViolation("WIKI40B_EXPORT_ROW_NOT_OBJECT")
            yield value


def stage_wiki40b_documents(
    examples: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    output_dir: Path,
    production: bool = True,
    scope_root: Path = WORK_ROOT,
    download_bytes_cap: int | None = None,
) -> dict[str, Any]:
    """Stage the deterministic frozen subset from an offline Wiki40B iterator.

    Source text is preserved byte-for-byte after strict UTF-8 decoding: no NFC,
    whitespace collapse, or Wiki40B marker removal occurs here.  All examples
    are counted and externally ordered in a temporary SQLite database, while
    only the frozen first-N/byte-capped subset is materialized.
    """
    output_dir = require_relative_to(
        Path(output_dir), scope_root, "CORPUS_OUTPUT_OUTSIDE_SCOPE"
    )
    data_kind = metadata.get("data_kind")
    if production and data_kind != REAL_WIKI40B:
        raise ContractViolation("SYNTHETIC_CORPUS_NOT_PRODUCTION")
    expected = {
        "dataset_name": "wiki40b",
        "config_name": "ko",
        "version": "1.3.0",
        "split": "train",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ContractViolation("WIKI40B_IDENTITY_MISMATCH")
    features = metadata.get("features")
    if set(features or []) != {"text", "version_id", "wikidata_id"}:
        raise ContractViolation("WIKI40B_FEATURE_MISMATCH")
    expected_count = metadata.get("num_examples")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
    ):
        raise ContractViolation("WIKI40B_COUNT_INVALID")
    if production and expected_count != 194_977:
        raise ContractViolation("WIKI40B_COUNT_MISMATCH")
    license_text = metadata.get("license")
    if production and (not isinstance(license_text, str) or not license_text.strip()):
        raise ContractViolation("WIKI40B_LICENSE_UNRECORDED")

    campaign_policy = load_campaign_policy()
    budget = campaign_policy["budget"]
    cap = int(
        download_bytes_cap
        if download_bytes_cap is not None
        else budget["automatic_download_bytes_cap"]
    )
    recorded_download = metadata.get("downloaded_bytes")
    if production and (
        not isinstance(recorded_download, int)
        or isinstance(recorded_download, bool)
        or recorded_download < 0
        or recorded_download > cap
    ):
        raise ContractViolation("DOWNLOAD_BUDGET_UNVERIFIED")
    source_files = metadata.get("source_files", [])
    if production and not source_files:
        raise ContractViolation("WIKI40B_SOURCE_FILES_UNRECORDED")
    for ref in source_files:
        _resolve_artifact_ref(
            ref,
            base=PROJECT_ROOT,
            production=production,
            scope_root=WORK_ROOT,
        )

    evaluation_plan = load_json(PROJECT_ROOT / "implementation/config/evaluation_plan.json")
    if evaluation_plan.get("version") != IMPLEMENTATION_REVISION:
        raise ContractViolation("BLOCKED_IMPLEMENTATION_REVISION")
    tokenizer_policy = evaluation_plan.get("tokenizer", {})
    document_cap = int(tokenizer_policy.get("training_document_cap", 0))
    utf8_cap = int(tokenizer_policy.get("training_utf8_byte_cap", 0))
    if document_cap < 1 or utf8_cap < 1:
        raise ContractViolation("INVALID_TOKENIZER_SUBSET_POLICY")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if production:
        generated_hint = metadata.get("generated_bytes", recorded_download)
        if not isinstance(generated_hint, int) or generated_hint < 1:
            raise ContractViolation("WIKI40B_GENERATED_SIZE_UNRECORDED")
        emergency = int(budget["minimum_emergency_free_bytes"])
        temporary_reservation = max(1024**3, generated_hint * 4)
        if shutil.disk_usage(output_dir.parent).free < emergency + temporary_reservation:
            raise ContractViolation("INSUFFICIENT_DISK_FOR_WIKI40B_STAGE")

    output_dir.mkdir(parents=True, exist_ok=False)
    database_path = output_dir / ".wiki40b-order.sqlite3"
    document_temp = output_dir / ".documents.bin.tmp"
    index_temp = output_dir / ".documents.index.jsonl.tmp"
    connection: sqlite3.Connection | None = None
    source_count = 0
    try:
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """CREATE TABLE documents (
                   order_key TEXT NOT NULL,
                   wikidata_id TEXT NOT NULL,
                   version_id TEXT NOT NULL,
                   text_sha256 TEXT NOT NULL,
                   text BLOB NOT NULL
               )"""
        )
        for row in examples:
            if not isinstance(row, Mapping) or set(row) != {
                "text",
                "version_id",
                "wikidata_id",
            }:
                raise ContractViolation("WIKI40B_ROW_SCHEMA_MISMATCH")
            text_value = row["text"]
            if isinstance(text_value, bytes):
                try:
                    text_bytes = bytes(text_value)
                    text_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ContractViolation("WIKI40B_INVALID_UTF8") from exc
            elif isinstance(text_value, str):
                try:
                    text_bytes = text_value.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ContractViolation("WIKI40B_INVALID_UTF8") from exc
            else:
                raise ContractViolation("WIKI40B_INVALID_TEXT")
            if not text_bytes:
                raise ContractViolation("WIKI40B_EMPTY_TEXT")
            version_raw = row["version_id"]
            wikidata_raw = row["wikidata_id"]
            if isinstance(version_raw, bytes):
                try:
                    version_raw = version_raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ContractViolation("WIKI40B_VERSION_ID_INVALID") from exc
            if isinstance(wikidata_raw, bytes):
                try:
                    wikidata_raw = wikidata_raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ContractViolation("WIKI40B_ID_INVALID") from exc
            version_id = _single_line(version_raw, "WIKI40B_VERSION_ID_INVALID")
            wikidata_id = _single_line(wikidata_raw, "WIKI40B_ID_INVALID")
            text_sha = sha256_bytes(text_bytes)
            order_key = sha256_bytes(
                (
                    "lexical-freshstart-v4/wiki40b-ko|"
                    + wikidata_id
                    + "|"
                    + version_id
                    + "|"
                    + text_sha
                ).encode("utf-8")
            )
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (order_key, wikidata_id, version_id, text_sha, text_bytes),
            )
            source_count += 1
            if source_count % 10_000 == 0:
                connection.commit()
        connection.commit()
        if source_count != expected_count:
            raise ContractViolation("WIKI40B_COUNT_MISMATCH")

        duplicate_counter: Counter[tuple[str, str, str]] = Counter()
        order_keys: list[str] = []
        selected_count = 0
        selected_utf8_bytes = 0
        stream_offset = 0
        with document_temp.open("xb") as documents_handle, index_temp.open("xb") as index_handle:
            cursor = connection.execute(
                """SELECT order_key, wikidata_id, version_id, text_sha256, text
                   FROM documents
                   ORDER BY order_key, wikidata_id, version_id, text_sha256, hex(text)"""
            )
            for order_key, wikidata_id, version_id, text_sha, text_blob in cursor:
                text_bytes = bytes(text_blob)
                if selected_count >= document_cap or selected_utf8_bytes + len(text_bytes) > utf8_cap:
                    break
                identity = (wikidata_id, version_id, text_sha)
                occurrence = duplicate_counter[identity]
                duplicate_counter[identity] += 1
                documents_handle.write(len(text_bytes).to_bytes(8, "big"))
                documents_handle.write(text_bytes)
                index_handle.write(
                    canonical_json_bytes(
                        {
                            "ordinal": selected_count,
                            "order_key": order_key,
                            "wikidata_id": wikidata_id,
                            "version_id": version_id,
                            "text_sha256": text_sha,
                            "duplicate_occurrence": occurrence,
                            "offset": stream_offset,
                            "utf8_bytes": len(text_bytes),
                        }
                    )
                )
                order_keys.append(order_key)
                stream_offset += 8 + len(text_bytes)
                selected_utf8_bytes += len(text_bytes)
                selected_count += 1
            documents_handle.flush()
            index_handle.flush()
            os.fsync(documents_handle.fileno())
            os.fsync(index_handle.fileno())
        if selected_count < 1:
            raise ContractViolation("WIKI40B_EMPTY_FROZEN_SUBSET")

        documents_ref = _publish_existing_file_once(
            document_temp, output_dir / "documents.bin"
        )
        index_ref = _publish_existing_file_once(
            index_temp, output_dir / "documents.index.jsonl"
        )
        core = {
            "schema_version": VERSION,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "status": "PASS",
            "data_kind": data_kind,
            "source": "TFDS",
            "dataset": expected,
            "official_num_examples": source_count,
            "staged_num_examples": selected_count,
            "staged_utf8_bytes": selected_utf8_bytes,
            "features": sorted(features),
            "license": license_text,
            "citation": metadata.get("citation"),
            "homepage": metadata.get("homepage"),
            "tfds_version": metadata.get("tfds_version"),
            "tensorflow_version": metadata.get("tensorflow_version"),
            "downloaded_bytes": recorded_download,
            "generated_bytes": metadata.get("generated_bytes"),
            "download_bytes_cap": cap,
            "source_files": source_files,
            "source_text_normalization": "NONE_EXACT_UTF8_PRESERVED_INCLUDING_WIKI40B_MARKERS",
            "selection": {
                "kind": "SORTED_PREFIX_WHOLE_DOCUMENTS",
                "document_cap": document_cap,
                "utf8_byte_cap": utf8_cap,
                "partial_document": False,
            },
            "document_order": "sha256(domain|wikidata_id|version_id|exact_text_sha256), ascending",
            "document_order_sha256": _logical_hash(order_keys),
            "duplicates_preserved": sum(value - 1 for value in duplicate_counter.values()),
            "artifacts": {
                "documents": _artifact_ref_for_manifest(documents_ref, output_dir),
                "index": _artifact_ref_for_manifest(index_ref, output_dir),
            },
        }
        manifest = {**core, "source_snapshot_sha256": _logical_hash(core)}
        manifest_ref = publish_json_once(output_dir / "source_manifest.json", manifest)
        return {
            **manifest,
            "manifest_artifact": _artifact_ref_for_manifest(manifest_ref, output_dir),
        }
    finally:
        if connection is not None:
            connection.close()
        for temporary_path in (database_path, document_temp, index_temp):
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def verify_wiki40b_snapshot(
    manifest_source: Path | Mapping[str, Any],
    *,
    production: bool = True,
) -> dict[str, Any]:
    manifest, path, manifest_sha = _load_mapping(manifest_source)
    if path is None:
        raise ContractViolation("WIKI40B_MANIFEST_PATH_REQUIRED")
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("implementation_revision") != IMPLEMENTATION_REVISION
        or manifest.get("status") != "PASS"
        or manifest.get("source") != "TFDS"
    ):
        raise ContractViolation("INVALID_WIKI40B_MANIFEST")
    if production and manifest.get("data_kind") != REAL_WIKI40B:
        raise ContractViolation("SYNTHETIC_CORPUS_NOT_PRODUCTION")
    _validate_embedded_hash(
        manifest, "source_snapshot_sha256", "WIKI40B_SNAPSHOT_HASH_MISMATCH"
    )
    expected = {
        "dataset_name": "wiki40b",
        "config_name": "ko",
        "version": "1.3.0",
        "split": "train",
    }
    if manifest.get("dataset") != expected:
        raise ContractViolation("WIKI40B_IDENTITY_MISMATCH")
    if set(manifest.get("features", [])) != {"text", "version_id", "wikidata_id"}:
        raise ContractViolation("WIKI40B_FEATURE_MISMATCH")
    source_count = manifest.get("official_num_examples")
    staged_count = manifest.get("staged_num_examples")
    if (
        not isinstance(source_count, int)
        or isinstance(source_count, bool)
        or not isinstance(staged_count, int)
        or isinstance(staged_count, bool)
        or staged_count < 1
        or staged_count > source_count
    ):
        raise ContractViolation("WIKI40B_COUNT_INVALID")
    if production and source_count != 194_977:
        raise ContractViolation("WIKI40B_COUNT_MISMATCH")
    if manifest.get("source_text_normalization") != "NONE_EXACT_UTF8_PRESERVED_INCLUDING_WIKI40B_MARKERS":
        raise ContractViolation("WIKI40B_SOURCE_TEXT_WAS_NORMALIZED")
    selection = manifest.get("selection")
    expected_policy = load_json(
        PROJECT_ROOT / "implementation/config/evaluation_plan.json"
    )["tokenizer"]
    if (
        not isinstance(selection, Mapping)
        or selection.get("kind") != "SORTED_PREFIX_WHOLE_DOCUMENTS"
        or selection.get("document_cap") != expected_policy["training_document_cap"]
        or selection.get("utf8_byte_cap") != expected_policy["training_utf8_byte_cap"]
        or selection.get("partial_document") is not False
    ):
        raise ContractViolation("WIKI40B_SELECTION_POLICY_MISMATCH")
    if staged_count > selection["document_cap"]:
        raise ContractViolation("WIKI40B_SELECTION_CAP_EXCEEDED")
    staged_bytes = manifest.get("staged_utf8_bytes")
    if (
        not isinstance(staged_bytes, int)
        or isinstance(staged_bytes, bool)
        or staged_bytes < 1
        or staged_bytes > selection["utf8_byte_cap"]
    ):
        raise ContractViolation("WIKI40B_SELECTION_CAP_EXCEEDED")
    if production:
        if not isinstance(manifest.get("license"), str) or not manifest["license"].strip():
            raise ContractViolation("WIKI40B_LICENSE_UNRECORDED")
        downloaded = manifest.get("downloaded_bytes")
        if (
            not isinstance(downloaded, int)
            or isinstance(downloaded, bool)
            or downloaded < 0
            or downloaded > manifest.get("download_bytes_cap", -1)
        ):
            raise ContractViolation("DOWNLOAD_BUDGET_UNVERIFIED")
    source_files = manifest.get("source_files")
    if production and (not isinstance(source_files, list) or not source_files):
        raise ContractViolation("WIKI40B_SOURCE_FILES_UNRECORDED")
    for ref in source_files or []:
        _resolve_artifact_ref(
            ref,
            base=PROJECT_ROOT,
            production=production,
            scope_root=WORK_ROOT,
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"documents", "index"}:
        raise ContractViolation("WIKI40B_ARTIFACT_SET_INVALID")
    documents_path = _resolve_artifact_ref(
        artifacts["documents"], base=path.parent, production=production, scope_root=WORK_ROOT
    )
    index_path = _resolve_artifact_ref(
        artifacts["index"], base=path.parent, production=production, scope_root=WORK_ROOT
    )
    index = _read_jsonl(index_path)
    if len(index) != staged_count:
        raise ContractViolation("WIKI40B_COUNT_MISMATCH")
    previous_key: tuple[Any, ...] | None = None
    order_keys: list[str] = []
    duplicate_counter: Counter[tuple[str, str, str]] = Counter()
    total_utf8_bytes = 0
    with documents_path.open("rb") as handle:
        for ordinal, row in enumerate(index):
            if set(row) != {
                "ordinal",
                "order_key",
                "wikidata_id",
                "version_id",
                "text_sha256",
                "duplicate_occurrence",
                "offset",
                "utf8_bytes",
            }:
                raise ContractViolation("WIKI40B_INDEX_SCHEMA_MISMATCH")
            key = (
                row.get("order_key"),
                row.get("wikidata_id"),
                row.get("version_id"),
                row.get("text_sha256"),
            )
            if previous_key is not None and key < previous_key:
                raise ContractViolation("WIKI40B_ORDER_MISMATCH")
            previous_key = key
            if row.get("ordinal") != ordinal or row.get("offset") != handle.tell():
                raise ContractViolation("WIKI40B_INDEX_MISMATCH")
            size_bytes = handle.read(8)
            if len(size_bytes) != 8:
                raise ContractViolation("WIKI40B_DOCUMENT_TRUNCATED")
            size = int.from_bytes(size_bytes, "big")
            payload = handle.read(size)
            if len(payload) != size or size != row.get("utf8_bytes"):
                raise ContractViolation("WIKI40B_DOCUMENT_TRUNCATED")
            if sha256_bytes(payload) != row.get("text_sha256"):
                raise ContractViolation("WIKI40B_DOCUMENT_HASH_MISMATCH")
            expected_order_key = sha256_bytes(
                (
                    "lexical-freshstart-v4/wiki40b-ko|"
                    + row["wikidata_id"]
                    + "|"
                    + row["version_id"]
                    + "|"
                    + row["text_sha256"]
                ).encode("utf-8")
            )
            if row["order_key"] != expected_order_key:
                raise ContractViolation("WIKI40B_ORDER_KEY_MISMATCH")
            identity = (row["wikidata_id"], row["version_id"], row["text_sha256"])
            if row["duplicate_occurrence"] != duplicate_counter[identity]:
                raise ContractViolation("WIKI40B_DUPLICATE_INDEX_MISMATCH")
            duplicate_counter[identity] += 1
            order_keys.append(row["order_key"])
            total_utf8_bytes += size
        if handle.read(1):
            raise ContractViolation("WIKI40B_DOCUMENT_TRAILING_BYTES")
    if total_utf8_bytes != staged_bytes:
        raise ContractViolation("WIKI40B_BYTE_COUNT_MISMATCH")
    if _logical_hash(order_keys) != manifest.get("document_order_sha256"):
        raise ContractViolation("WIKI40B_ORDER_HASH_MISMATCH")
    if sum(value - 1 for value in duplicate_counter.values()) != manifest.get(
        "duplicates_preserved"
    ):
        raise ContractViolation("WIKI40B_DUPLICATE_COUNT_MISMATCH")
    return {
        **manifest,
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha,
        "documents_path": str(documents_path),
        "index_path": str(index_path),
        "index": index,
    }


def publish_experiment_freeze(
    annotation_manifest: Path,
    tokenizer_manifest: Path,
    corpus_manifest: Path,
    *,
    boundary_audit: Mapping[str, Any],
    output_path: Path,
    production: bool = True,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Publish the sole manifest accepted by model training and scoring."""
    output_path = require_relative_to(
        Path(output_path), scope_root, "EXPERIMENT_FREEZE_OUTSIDE_SCOPE"
    )
    annotation = verify_annotation_freeze(annotation_manifest, production=production)
    from .build_tokenizer import verify_corpus_manifest, verify_tokenizer_manifest

    tokenizer = verify_tokenizer_manifest(tokenizer_manifest, production=production)
    corpus = verify_corpus_manifest(corpus_manifest, production=production)
    if boundary_audit.get("status") != "PASS":
        raise ContractViolation("TOKEN_BOUNDARY_AUDIT_NOT_PASS")
    _validate_embedded_hash(
        boundary_audit, "audit_sha256", "TOKEN_BOUNDARY_AUDIT_HASH_MISMATCH"
    )
    expected_concept_hashes = sorted(
        row["concept_record_sha256"] for row in annotation["concepts"]
    )
    if (
        boundary_audit.get("data_kind") != ("REAL" if production else SYNTHETIC)
        or boundary_audit.get("concept_record_hashes") != expected_concept_hashes
        or boundary_audit.get("tokenizer_file_sha256")
        != tokenizer.get("tokenizer_file_sha256")
        or boundary_audit.get("evaluation_plan_sha256")
        != tokenizer.get("evaluation_plan_sha256")
        or boundary_audit.get("contexts_checked") != len(expected_concept_hashes) * 120
        or boundary_audit.get("contexts_passed")
        != boundary_audit.get("contexts_checked")
        or boundary_audit.get("contexts_failed") != 0
    ):
        raise ContractViolation("TOKEN_BOUNDARY_AUDIT_BINDING_MISMATCH")
    if corpus.get("tokenizer_file_sha256") != tokenizer.get("tokenizer_file_sha256"):
        raise ContractViolation("CORPUS_TOKENIZER_MISMATCH")
    data_kind = "REAL" if production else SYNTHETIC
    boundary_ref = publish_json_once(
        output_path.with_name(output_path.stem + ".token_boundary_audit.json"),
        dict(boundary_audit),
    )
    refs: dict[str, dict[str, Any]] = {
        "boundary_audit": {
            "path": str(Path(str(boundary_ref["path"])).resolve()),
            "sha256": boundary_ref["sha256"],
            "bytes": boundary_ref["bytes"],
        }
    }
    for name, source in {
        "annotation": Path(annotation_manifest),
        "tokenizer": Path(tokenizer_manifest),
        "corpus": Path(corpus_manifest),
        "spec": PROJECT_ROOT / "spec/RESEARCH_SPEC_V4_KO.md",
        "pilot_config": PROJECT_ROOT / "spec/pilot.json",
        "campaign_policy": PROJECT_ROOT / "implementation/config/campaign_policy.json",
        "evaluation_plan": Path(tokenizer["evaluation_plan_path"]),
    }.items():
        refs[name] = {
            "path": str(source.resolve()),
            "sha256": artifact_sha256_file(source),
            "bytes": source.stat().st_size,
        }
    core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "project_id": load_pilot_config()["project_id"],
        "freeze_kind": "EXPERIMENT_FREEZE",
        "status": "PASS",
        "freeze_gate_status": "PASS",
        "data_kind": data_kind,
        "artifacts": refs,
        "annotation_freeze_id": annotation["freeze_id"],
        "tokenizer_file_sha256": tokenizer["tokenizer_file_sha256"],
        "corpus_token_count": corpus["materialized_token_count"],
        "boundary_audit_sha256": _logical_hash(boundary_audit),
        "evaluation_record_hash_schema": "score-record-v1",
        "evaluation_record_hash_projection": [
            "record_id",
            "concept_id",
            "input_language",
            "format",
            "wrapper",
            "split",
            "mode",
            "prefix",
            "answers",
            "requested_language-if-present",
        ],
        "gates": {
            "source": "PASS",
            "human_review_recorded": "PASS",
            "data_coverage": "PASS",
            "tokenizer": "PASS",
            "corpus_materialization": "PASS",
            "token_boundaries_and_lengths": "PASS",
        },
    }
    freeze_id = "experiment-" + _logical_hash(core)[:20]
    manifest = {**core, "freeze_id": freeze_id}
    ref = publish_json_once(output_path, manifest)
    return {**manifest, "manifest_artifact": dict(ref)}


def verify_experiment_freeze(
    manifest_source: Path,
    *,
    production: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_source)
    manifest, manifest_sha256 = read_verified_json_with_sha256(path)
    if manifest.get("schema_version") == "pilot-execution-freeze-v1":
        from .pilot_execution_freeze import verify_execution_freeze

        return verify_execution_freeze(path, production=production)
    if manifest.get("schema_version") == "semantic-experiment-freeze-v4.1.2":
        from .semantic_experiment_freeze import verify_semantic_experiment_freeze

        return verify_semantic_experiment_freeze(path, production=production)
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("implementation_revision") != IMPLEMENTATION_REVISION
        or manifest.get("freeze_kind") != "EXPERIMENT_FREEZE"
        or manifest.get("status") != "PASS"
        or manifest.get("freeze_gate_status") != "PASS"
    ):
        raise ContractViolation("INVALID_EXPERIMENT_FREEZE")
    expected_kind = "REAL" if production else SYNTHETIC
    if manifest.get("data_kind") != expected_kind:
        raise ContractViolation("SYNTHETIC_FREEZE_NOT_PRODUCTION")
    core = _without(manifest, "freeze_id")
    if manifest.get("freeze_id") != "experiment-" + _logical_hash(core)[:20]:
        raise ContractViolation("EXPERIMENT_FREEZE_ID_MISMATCH")
    if production:
        require_trusted_artifact_anchor(
            artifact_kind="EXPERIMENT_FREEZE",
            artifact_id=str(manifest.get("freeze_id", "")),
            manifest_sha256=manifest_sha256,
            bindings={
                "annotation_freeze_id": manifest.get("annotation_freeze_id"),
                "boundary_audit_sha256": manifest.get("boundary_audit_sha256"),
                "corpus_token_count": manifest.get("corpus_token_count"),
                "data_kind": manifest.get("data_kind"),
                "freeze_kind": "EXPERIMENT_FREEZE",
                "implementation_revision": IMPLEMENTATION_REVISION,
                "tokenizer_file_sha256": manifest.get("tokenizer_file_sha256"),
            },
        )
    expected_names = {
        "annotation",
        "tokenizer",
        "corpus",
        "spec",
        "pilot_config",
        "campaign_policy",
        "evaluation_plan",
        "boundary_audit",
    }
    refs = manifest.get("artifacts")
    if not isinstance(refs, Mapping) or set(refs) != expected_names:
        raise ContractViolation("EXPERIMENT_ARTIFACT_SET_INVALID")
    resolved = {
        name: _resolve_artifact_ref(
            ref,
            base=path.parent,
            production=production,
            scope_root=PROJECT_ROOT if name in {"spec", "pilot_config", "campaign_policy", "evaluation_plan"} else WORK_ROOT,
        )
        for name, ref in refs.items()
    }
    immutable_paths = {
        "spec": PROJECT_ROOT / "spec/RESEARCH_SPEC_V4_KO.md",
        "pilot_config": PROJECT_ROOT / "spec/pilot.json",
        "campaign_policy": PROJECT_ROOT / "implementation/config/campaign_policy.json",
        "evaluation_plan": PROJECT_ROOT / "implementation/config/evaluation_plan.json",
    }
    paths_to_pin = immutable_paths if production else {
        name: value for name, value in immutable_paths.items() if name != "evaluation_plan"
    }
    if any(resolved[name] != expected.resolve() for name, expected in paths_to_pin.items()):
        raise ContractViolation("EXPERIMENT_CONFIG_PATH_MISMATCH")
    boundary_audit = read_verified_json(
        resolved["boundary_audit"],
        expected_sha256=str(refs["boundary_audit"]["sha256"]),
    )
    _validate_embedded_hash(
        boundary_audit, "audit_sha256", "TOKEN_BOUNDARY_AUDIT_HASH_MISMATCH"
    )
    if (
        boundary_audit.get("status") != "PASS"
        or _logical_hash(boundary_audit) != manifest.get("boundary_audit_sha256")
    ):
        raise ContractViolation("TOKEN_BOUNDARY_AUDIT_NOT_PASS")
    annotation = verify_annotation_freeze(resolved["annotation"], production=production)
    from .build_tokenizer import verify_corpus_manifest, verify_tokenizer_manifest

    tokenizer = verify_tokenizer_manifest(resolved["tokenizer"], production=production)
    corpus = verify_corpus_manifest(resolved["corpus"], production=production)
    if (
        annotation.get("manifest_sha256") != refs["annotation"]["sha256"]
        or annotation.get("freeze_id") != manifest.get("annotation_freeze_id")
        or tokenizer.get("manifest_sha256") != refs["tokenizer"]["sha256"]
        or corpus.get("manifest_sha256") != refs["corpus"]["sha256"]
    ):
        raise ContractViolation("EXPERIMENT_ARTIFACT_IDENTITY_MISMATCH")
    expected_concept_hashes = sorted(
        row["concept_record_sha256"] for row in annotation["concepts"]
    )
    if (
        boundary_audit.get("data_kind") != expected_kind
        or boundary_audit.get("concept_record_hashes") != expected_concept_hashes
        or boundary_audit.get("tokenizer_file_sha256")
        != tokenizer.get("tokenizer_file_sha256")
        or boundary_audit.get("evaluation_plan_sha256")
        != refs["evaluation_plan"]["sha256"]
        or boundary_audit.get("contexts_checked") != len(expected_concept_hashes) * 120
        or boundary_audit.get("contexts_passed")
        != boundary_audit.get("contexts_checked")
        or boundary_audit.get("contexts_failed") != 0
    ):
        raise ContractViolation("TOKEN_BOUNDARY_AUDIT_BINDING_MISMATCH")
    if tokenizer["tokenizer_file_sha256"] != manifest.get("tokenizer_file_sha256"):
        raise ContractViolation("EXPERIMENT_TOKENIZER_HASH_MISMATCH")
    if corpus["materialized_token_count"] != manifest.get("corpus_token_count"):
        raise ContractViolation("EXPERIMENT_CORPUS_COUNT_MISMATCH")
    if corpus["tokenizer_file_sha256"] != tokenizer["tokenizer_file_sha256"]:
        raise ContractViolation("CORPUS_TOKENIZER_MISMATCH")
    if annotation["status"] != "PASS":
        raise ContractViolation("ANNOTATION_GATE_NOT_PASS")
    if any(value != "PASS" for value in manifest.get("gates", {}).values()):
        raise ContractViolation("EXPERIMENT_GATE_NOT_PASS")
    expected_projection = [
        "record_id",
        "concept_id",
        "input_language",
        "format",
        "wrapper",
        "split",
        "mode",
        "prefix",
        "answers",
        "requested_language-if-present",
    ]
    if (
        manifest.get("evaluation_record_hash_schema") != "score-record-v1"
        or manifest.get("evaluation_record_hash_projection") != expected_projection
    ):
        raise ContractViolation("EVALUATION_RECORD_HASH_SCHEMA_MISMATCH")
    return {
        **manifest,
        "freeze_sha256": manifest_sha256,
        "manifest_path": str(path.resolve()),
        "resolved_artifacts": {name: str(value) for name, value in resolved.items()},
        "concepts": annotation["concepts"],
        "etymology": annotation["etymology"],
        "cohort": annotation["cohort"],
        "history_pairs": annotation["history_pairs"],
        "boundary_audit": boundary_audit,
        "verified": True,
    }


__all__ = [
    "REAL_KRDICT",
    "REAL_WIKI40B",
    "SYNTHETIC",
    "TransportResponse",
    "build_krdict_collection_plan",
    "collect_krdict",
    "export_pending_review",
    "freeze_reviewed_dataset",
    "import_existing_krdict_collection",
    "initialize_api_ledger",
    "iter_wiki40b_jsonl_export",
    "load_campaign_policy",
    "merge_krdict_snapshot",
    "parse_krdict_xml",
    "publish_experiment_freeze",
    "stage_wiki40b_documents",
    "validate_review_bundle",
    "verify_annotation_freeze",
    "verify_experiment_freeze",
    "verify_krdict_collection",
    "verify_wiki40b_snapshot",
]
