"""Strict, non-approving CSV workflow for the v4 lexical review.

The spreadsheet files produced here are review interfaces, never trainer
inputs.  They are derived from a hash-verified KRDICT merge and all source
columns are checked again when a filled worksheet is imported.  Only
``compile_filled_review_bundle`` can turn completed human decisions into the
JSON records accepted by :mod:`implementation.src.prepare_data`.

No function in this module performs network or GPU work.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from freshstart.core import ContractError as ReferenceContractError
from freshstart.core import canonical as _scorer_canonical

from .artifacts import read_regular_file_bytes, sha256_file
from .contracts import VERSION, ContractViolation, canonical_json_bytes, require_sha256
from .contracts import require_relative_to


LANGUAGES = ("ko", "en", "zh", "fr")
ETYMOLOGY_6_TO_4 = {
    "BORROWING_DIRECT": "BORROWING_DOCUMENTED",
    "BORROWING_PARALLEL": "SHARED_SOURCE_DOCUMENTED",
    "NEOCLASSICAL_SHARED": "SHARED_SOURCE_DOCUMENTED",
    "COGNATE_INHERITED": "SHARED_SOURCE_DOCUMENTED",
    "DISTINCT_ROUTES_REVIEWED": "DISTINCT_ROUTES_REVIEWED",
    "INDETERMINATE": "UNRESOLVED",
}
_DIRECTIONS = {
    "BORROWING_DIRECT": {"EN_TO_FR", "FR_TO_EN"},
    "BORROWING_PARALLEL": {"COMMON_SOURCE_TO_BOTH"},
    "NEOCLASSICAL_SHARED": {"COMMON_SOURCE_TO_BOTH"},
    "COGNATE_INHERITED": {"COMMON_ANCESTOR_TO_BOTH"},
    "DISTINCT_ROUTES_REVIEWED": {"NONE"},
    "INDETERMINATE": {"UNKNOWN"},
}
_HASH_RE = re.compile(r"[0-9a-f]{64}")

EVIDENCE_FIELDS = (
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
)

HUMAN_REVIEWED_EVIDENCE_ORIGINS = frozenset(
    {
        "human-reviewed-dictionary-record",
        "human-reviewed-corpus-record",
        "human-reviewed-academic-publication",
        "human-reviewed-official-reference",
        "human-reviewed-archival-record",
    }
)
SYNTHETIC_EVIDENCE_ORIGIN = "synthetic-human-reviewed-fixture"


TERM_IMMUTABLE_FIELDS = (
    "schema_version",
    "snapshot_id",
    "merge_sha256",
    "review_order",
    "candidate_id",
    "candidate_sha256",
    "lang",
    "term_index",
    "source_option_index",
    "source_option_id",
    "source_word_raw",
    "source_definition_raw",
    "source_gloss",
    "source_refs_json",
    "source_word_raw_sha256",
    "source_span_start",
    "source_span_end",
    "source_text_exact",
    "term_canonical",
    "term_id",
    "term_sha256",
    "term_flags_json",
)
TERM_HUMAN_FIELDS = (
    "segmentation_decision",
    "translation_quality",
    "is_transliteration",
    "quality_evidence_ids_json",
    "quality_note",
    "term_review_status",
    "term_reviewer",
    "term_review_date",
)
TERM_FIELDS = TERM_IMMUTABLE_FIELDS + TERM_HUMAN_FIELDS

PAIR_IMMUTABLE_FIELDS = (
    "schema_version",
    "snapshot_id",
    "merge_sha256",
    "review_order",
    "candidate_id",
    "candidate_sha256",
    "ko_expression",
    "ko_definition",
    "zh_expression",
    "zh_definition",
    "source_record_hashes_json",
)
PAIR_HUMAN_FIELDS = (
    "candidate_decision",
    "selected_ko_term_id",
    "selected_en_term_id",
    "selected_zh_term_id",
    "selected_fr_term_id",
    "selection_rationale",
    "meaning_alignment_decision",
    "sense_alignment_note",
    "source_alignment_checked",
    "answer_copy_checked",
    "synonym_cluster_id",
    "pilot_selected",
    "etymology_subtype",
    "etymology_primary",
    "relation_direction",
    "shared_source",
    "historical_scope",
    "confidence",
    "etymology_evidence_ids_json",
    "evidence_search_note",
    "family_id",
    "concept_review_status",
    "concept_reviewer",
    "concept_review_date",
    "etymology_review_status",
    "etymology_reviewer",
    "etymology_review_date",
)
PAIR_FIELDS = PAIR_IMMUTABLE_FIELDS + PAIR_HUMAN_FIELDS

SCREEN_FIELDS = (
    "schema_version",
    "snapshot_id",
    "merge_sha256",
    "review_order",
    "candidate_id",
    "candidate_sha256",
    "term_id",
    "lang",
    "ko_expression",
    "ko_romanizations_json",
    "term_canonical",
    "screen_flags_json",
    "advisory_only",
)


@dataclass(frozen=True)
class ReviewTables:
    terms: tuple[dict[str, str], ...]
    pairs: tuple[dict[str, str], ...]
    screen: tuple[dict[str, str], ...]


def canonical(text: str) -> str:
    """Use the exact registered-expression canonicalization used by scorer."""
    try:
        return _scorer_canonical(text)
    except ReferenceContractError as exc:
        raise ContractViolation("INVALID_REGISTERED_EXPRESSION") from exc


def _source_normalized(text: str) -> str:
    """Normalize source whitespace before checking a scorer-safe expression."""
    if not isinstance(text, str) or not text or "\x00" in text:
        raise ContractViolation("INVALID_SOURCE_EXPRESSION")
    value = " ".join(unicodedata.normalize("NFC", text).split())
    if not value:
        raise ContractViolation("INVALID_SOURCE_EXPRESSION")
    return canonical(value)


def _logical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def map_etymology_label(subtype: str) -> str:
    try:
        return ETYMOLOGY_6_TO_4[subtype]
    except KeyError as exc:
        raise ContractViolation("UNKNOWN_ETYMOLOGY_SUBTYPE") from exc


def answer_pair_sha256(en_answer: str, fr_answer: str) -> str:
    """Stable surface-form fingerprint for one selected EN--FR answer pair."""
    return _logical_hash({"en": canonical(en_answer), "fr": canonical(fr_answer)})


def en_fr_pair_evidence_subject_sha256(
    *,
    candidate_id: str,
    en_term_id: str,
    en_term_sha256: str,
    en_canonical_answer: str,
    fr_term_id: str,
    fr_term_sha256: str,
    fr_canonical_answer: str,
) -> str:
    """Bind pair evidence to one source sense and two selected term records."""
    candidate = _single_line_text(candidate_id, "INVALID_EVIDENCE_CANDIDATE_ID")
    selected_terms: dict[str, dict[str, str]] = {}
    for language, term_id, term_sha256, answer in (
        ("en", en_term_id, en_term_sha256, en_canonical_answer),
        ("fr", fr_term_id, fr_term_sha256, fr_canonical_answer),
    ):
        checked_term_id = _single_line_text(term_id, "INVALID_EVIDENCE_TERM_ID")
        if not checked_term_id.startswith(f"{candidate}|{language}|"):
            raise ContractViolation("EVIDENCE_TERM_ID_CANDIDATE_MISMATCH")
        checked_answer = canonical(answer)
        if checked_answer != answer:
            raise ContractViolation("NONCANONICAL_EVIDENCE_ANSWER")
        selected_terms[language] = {
            "term_id": checked_term_id,
            "term_sha256": require_sha256(
                term_sha256, "INVALID_EVIDENCE_TERM_SHA256"
            ),
            "canonical_answer": checked_answer,
        }
    return _logical_hash(
        {
            "subject_schema": "en-fr-selected-term-pair-v1",
            "subject_kind": "EN_FR_PAIR",
            "candidate_id": candidate,
            "selected_terms": selected_terms,
        }
    )


def validate_relation_direction(subtype: str, direction: str, shared_source: str) -> None:
    map_etymology_label(subtype)
    if direction not in _DIRECTIONS[subtype]:
        raise ContractViolation("INVALID_ETYMOLOGY_DIRECTION")
    if subtype in {
        "BORROWING_PARALLEL",
        "NEOCLASSICAL_SHARED",
        "COGNATE_INHERITED",
    } and not shared_source.strip():
        raise ContractViolation("SHARED_SOURCE_REQUIRED")
    if subtype not in {
        "BORROWING_PARALLEL",
        "NEOCLASSICAL_SHARED",
        "COGNATE_INHERITED",
    } and shared_source.strip():
        raise ContractViolation("SHARED_SOURCE_NOT_ALLOWED")


def evidence_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash every evidence-record field except the self-authenticating hash."""
    if not isinstance(record, Mapping):
        raise ContractViolation("INVALID_EVIDENCE_RECORD")
    return _logical_hash(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def _single_line_text(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(char in value for char in "\r\n\x00")
    ):
        raise ContractViolation(code)
    return value


def validate_evidence_origin(value: Any, *, production: bool) -> str:
    """Accept only explicit human-reviewed provenance classes.

    A closed vocabulary prevents spelling or branding variants such as
    ``model-generated`` or ``ChatGPT`` from bypassing a blacklist.  The
    synthetic value exists only for bounded fixture tests and is forbidden in
    production.
    """
    origin = _single_line_text(value, "INVALID_EVIDENCE_ORIGIN")
    allowed = HUMAN_REVIEWED_EVIDENCE_ORIGINS
    if not production:
        allowed = allowed | {SYNTHETIC_EVIDENCE_ORIGIN}
    if origin not in allowed:
        raise ContractViolation("INVALID_EVIDENCE_ORIGIN")
    return origin


def index_evidence_records(
    evidence_records: Sequence[Mapping[str, Any]],
    *,
    production: bool = True,
) -> dict[str, Mapping[str, Any]]:
    """Validate strict evidence metadata and return it indexed by evidence ID.

    Payload bytes are deliberately verified by :mod:`review_freeze`, where a
    concrete filesystem scope is available.  This function verifies the
    record hash and all metadata needed to perform that filesystem check.
    """
    if not isinstance(evidence_records, Sequence) or isinstance(
        evidence_records, (str, bytes, bytearray)
    ):
        raise ContractViolation("INVALID_EVIDENCE_RECORDS")
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in evidence_records:
        if not isinstance(record, Mapping) or set(record) != set(EVIDENCE_FIELDS):
            raise ContractViolation("INVALID_EVIDENCE_RECORD_SCHEMA")
        evidence_id = _single_line_text(
            record.get("evidence_id"), "INVALID_EVIDENCE_ID"
        )
        if evidence_id in indexed:
            raise ContractViolation("DUPLICATE_EVIDENCE_ID")
        subject_kind = record.get("subject_kind")
        if subject_kind not in {"TERM", "EN_FR_PAIR"}:
            raise ContractViolation("INVALID_EVIDENCE_SUBJECT_KIND")
        _single_line_text(record.get("subject_id"), "INVALID_EVIDENCE_SUBJECT_ID")
        support = _single_line_text(
            record.get("supports_label"), "INVALID_EVIDENCE_SUPPORT_LABEL"
        )
        if subject_kind == "TERM":
            if support != "ATTESTED_SAME_SENSE":
                raise ContractViolation("TERM_EVIDENCE_LABEL_MISMATCH")
        elif support not in ETYMOLOGY_6_TO_4:
            raise ContractViolation("PAIR_EVIDENCE_LABEL_MISMATCH")
        url = _single_line_text(record.get("source_url"), "INVALID_EVIDENCE_URL")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ContractViolation("INVALID_EVIDENCE_URL")
        validate_evidence_origin(record.get("evidence_origin"), production=production)
        _single_line_text(record.get("source_name"), "INVALID_EVIDENCE_SOURCE_NAME")
        _single_line_text(
            record.get("source_version"), "INVALID_EVIDENCE_SOURCE_VERSION"
        )
        _single_line_text(
            record.get("source_license"), "INVALID_EVIDENCE_SOURCE_LICENSE"
        )
        license_url = _single_line_text(
            record.get("source_license_url"),
            "INVALID_EVIDENCE_SOURCE_LICENSE_URL",
        )
        parsed_license = urllib.parse.urlparse(license_url)
        if (
            parsed_license.scheme != "https"
            or not parsed_license.hostname
            or parsed_license.username
            or parsed_license.password
        ):
            raise ContractViolation("INVALID_EVIDENCE_SOURCE_LICENSE_URL")
        _single_line_text(record.get("sense_locator"), "INVALID_EVIDENCE_SENSE_LOCATOR")
        payload_path = _single_line_text(
            record.get("payload_path"), "INVALID_EVIDENCE_PAYLOAD_PATH"
        )
        if Path(payload_path).name in {"", ".", ".."}:
            raise ContractViolation("INVALID_EVIDENCE_PAYLOAD_PATH")
        require_sha256(
            str(record.get("payload_sha256", "")), "INVALID_EVIDENCE_PAYLOAD_SHA256"
        )
        payload_bytes = record.get("payload_bytes")
        if (
            not isinstance(payload_bytes, int)
            or isinstance(payload_bytes, bool)
            or payload_bytes < 1
        ):
            raise ContractViolation("INVALID_EVIDENCE_PAYLOAD_SIZE")
        retrieved_at = _single_line_text(
            record.get("retrieved_at"), "INVALID_EVIDENCE_RETRIEVED_AT"
        )
        try:
            parsed_time = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractViolation("INVALID_EVIDENCE_RETRIEVED_AT") from exc
        if parsed_time.tzinfo is None or parsed_time > datetime.now(timezone.utc):
            raise ContractViolation("INVALID_EVIDENCE_RETRIEVED_AT")
        conflicts = record.get("conflicts_with")
        if (
            not isinstance(conflicts, list)
            or any(not isinstance(value, str) or not value for value in conflicts)
            or len(conflicts) != len(set(conflicts))
            or evidence_id in conflicts
        ):
            raise ContractViolation("INVALID_EVIDENCE_CONFLICTS")
        claimed = require_sha256(
            str(record.get("record_sha256", "")), "INVALID_EVIDENCE_RECORD_SHA256"
        )
        if claimed != evidence_record_sha256(record):
            raise ContractViolation("EVIDENCE_RECORD_HASH_MISMATCH")
        indexed[evidence_id] = dict(record)
    if not indexed:
        raise ContractViolation("EMPTY_EVIDENCE_RECORDS")
    for evidence_id, record in indexed.items():
        for conflict_id in record["conflicts_with"]:
            conflict = indexed.get(conflict_id)
            if conflict is None:
                raise ContractViolation("EVIDENCE_CONFLICT_TARGET_MISSING")
            if (
                conflict["subject_kind"] != record["subject_kind"]
                or conflict["subject_id"] != record["subject_id"]
                or evidence_id not in conflict["conflicts_with"]
            ):
                raise ContractViolation("EVIDENCE_CONFLICT_NOT_SYMMETRIC_OR_SAME_SUBJECT")
    return indexed


def verify_evidence_payloads(
    evidence_records: Mapping[str, Mapping[str, Any]],
    scope_root: Path,
) -> set[str]:
    """Open, size-check, and hash every evidence payload inside ``scope_root``."""
    root = Path(scope_root).resolve(strict=True)
    verified: set[str] = set()
    for record in evidence_records.values():
        raw_path = Path(str(record["payload_path"]))
        unresolved = raw_path if raw_path.is_absolute() else root / raw_path
        if unresolved.is_symlink():
            raise ContractViolation("EVIDENCE_PAYLOAD_NOT_REGULAR_FILE")
        path = require_relative_to(unresolved, root, "EVIDENCE_PAYLOAD_OUTSIDE_SCOPE")
        if path.is_symlink() or not path.is_file():
            raise ContractViolation("EVIDENCE_PAYLOAD_NOT_REGULAR_FILE")
        if path.stat().st_size != record["payload_bytes"]:
            raise ContractViolation("EVIDENCE_PAYLOAD_SIZE_MISMATCH")
        actual = sha256_file(path)
        if actual != record["payload_sha256"]:
            raise ContractViolation("EVIDENCE_PAYLOAD_SHA256_MISMATCH")
        verified.add(actual)
    return verified


def _verified_candidates(merge: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(merge, Mapping) or merge.get("status") != "MERGED_PENDING_REVIEW":
        raise ContractViolation("INVALID_MERGE_STATUS")
    claimed = str(merge.get("merge_sha256", ""))
    require_sha256(claimed, "INVALID_MERGE_SHA256")
    core = {key: value for key, value in merge.items() if key != "merge_sha256"}
    if _logical_hash(core) != claimed:
        raise ContractViolation("MERGE_HASH_MISMATCH")
    snapshot_id = merge.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ContractViolation("INVALID_SNAPSHOT_ID")
    rows = merge.get("eligible_candidates")
    if not isinstance(rows, list) or not rows:
        raise ContractViolation("EMPTY_CANDIDATE_SET")
    seen: set[str] = set()
    result: list[Mapping[str, Any]] = []
    for candidate in rows:
        if not isinstance(candidate, Mapping):
            raise ContractViolation("INVALID_CANDIDATE")
        candidate_id = candidate.get("candidate_id")
        candidate_hash = str(candidate.get("candidate_sha256", ""))
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ContractViolation("DUPLICATE_OR_INVALID_CANDIDATE_ID")
        seen.add(candidate_id)
        require_sha256(candidate_hash, "INVALID_CANDIDATE_SHA256")
        candidate_core = {
            key: value for key, value in candidate.items() if key != "candidate_sha256"
        }
        if _logical_hash(candidate_core) != candidate_hash:
            raise ContractViolation("CANDIDATE_HASH_MISMATCH")
        if candidate.get("snapshot_id") != snapshot_id:
            raise ContractViolation("CANDIDATE_SNAPSHOT_MISMATCH")
        options = candidate.get("options")
        if not isinstance(options, Mapping) or set(options) != set(LANGUAGES):
            raise ContractViolation("FOUR_LANGUAGE_OPTIONS_REQUIRED")
        result.append(candidate)
    return result


def _delimiter_positions(raw: str, delimiter: str) -> list[int]:
    """Return top-level delimiter positions, preserving punctuation in brackets."""
    positions: list[int] = []
    depth = 0
    opening = {"(": ")", "[": "]", "{": "}", "（": "）", "［": "］"}
    closing = set(opening.values())
    stack: list[str] = []
    for index, char in enumerate(raw):
        if char in opening:
            stack.append(opening[char])
            depth += 1
        elif char in closing and stack and char == stack[-1]:
            stack.pop()
            depth -= 1
        elif char == delimiter and depth == 0:
            positions.append(index)
    return positions


def _trimmed_segments(raw: str, language: str) -> list[tuple[int, int, int]]:
    delimiter = (
        ";" if language == "en" else "," if language == "fr" else "，" if language == "zh" else None
    )
    cuts: list[tuple[int, int]] = []
    start = 0
    if delimiter is not None:
        for position in _delimiter_positions(raw, delimiter):
            cuts.append((start, position))
            start = position + len(delimiter)
    cuts.append((start, len(raw)))
    segments: list[tuple[int, int, int]] = []
    for segment_index, (left, right) in enumerate(cuts, 1):
        while left < right and raw[left].isspace():
            left += 1
        while right > left and raw[right - 1].isspace():
            right -= 1
        if left == right:
            raise ContractViolation("EMPTY_SOURCE_SEGMENT")
        segments.append((segment_index, left, right))
    return segments


def _format_flags(raw: str, exact: str, language: str) -> set[str]:
    flags: set[str] = set()
    if "\n" in exact or "\r" in exact:
        flags.add("MULTILINE_SOURCE")
    if raw != raw.strip():
        flags.add("OUTER_WHITESPACE")
    if re.search(r"\([^)]*\)", exact):
        flags.add("PAREN_TAG_OR_VARIANT")
    if '"' in exact or "“" in exact or "”" in exact:
        flags.add("QUOTED_TEXT")
    if language == "en" and "," in raw:
        flags.add("UNEXPECTED_COMMA")
    if language == "fr" and ";" in raw:
        flags.add("UNEXPECTED_SEMICOLON")
    if re.search(r"\s+[;,]", raw) or re.search(r"[;,](?!\s|$)", raw):
        flags.add("IRREGULAR_DELIMITER_WHITESPACE")
    return flags


_INITIAL = (
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "",
    "j", "jj", "ch", "k", "t", "p", "h",
)
_VOWEL = (
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
    "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
)
_FINAL = (
    "", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "p", "l",
    "l", "p", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t",
)


def _romanize_basic(text: str) -> str:
    pieces: list[str] = []
    for char in unicodedata.normalize("NFC", text):
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            value = code - 0xAC00
            initial = value // 588
            vowel = (value % 588) // 28
            final = value % 28
            pieces.append(_INITIAL[initial] + _VOWEL[vowel] + _FINAL[final])
        elif char.isascii() and char.isalnum():
            pieces.append(char.lower())
        elif char.isspace() or char in "-_":
            pieces.append(" ")
    return " ".join("".join(pieces).split())


def korean_romanization_variants(text: str) -> tuple[str, ...]:
    basic = _romanize_basic(text)
    if not basic:
        return ()
    variants = {basic}
    # Some source romanizations voice a batchim immediately before the next
    # consonant (notably 접시 -> jeobsi).  This is advisory screening only.
    variants.add(re.sub(r"p(?=[bcdfghjklmnpqrstvwxyz])", "b", basic))
    variants.add(re.sub(r"t(?=[bcdfghjklmnpqrstvwxyz])", "d", basic))
    variants.add(re.sub(r"k(?=[bcdfghjklmnpqrstvwxyz])", "g", basic))
    return tuple(sorted(variants))


def _romanization_flags(term: str, variants: Sequence[str]) -> set[str]:
    lowered = term.casefold()
    flags: set[str] = set()
    for variant in variants:
        if lowered == variant.casefold():
            flags.add("ROMANIZED_KO_EXACT")
        elif re.search(r"(?<![a-z])" + re.escape(variant.casefold()) + r"(?![a-z])", lowered):
            flags.add("ROMANIZED_KO_EMBEDDED")
    return flags


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def _option(option: Any) -> Mapping[str, Any]:
    if not isinstance(option, Mapping):
        raise ContractViolation("INVALID_SOURCE_OPTION")
    option_id = str(option.get("option_id", ""))
    require_sha256(option_id, "INVALID_SOURCE_OPTION_ID")
    raw = option.get("word_raw")
    answer = option.get("answer")
    if not isinstance(raw, str) or not isinstance(answer, str):
        raise ContractViolation("INVALID_SOURCE_OPTION")
    if _source_normalized(raw) != canonical(answer):
        raise ContractViolation("SOURCE_OPTION_ANSWER_MISMATCH")
    refs = option.get("source_refs")
    if not isinstance(refs, list) or not refs:
        raise ContractViolation("SOURCE_OPTION_REFS_REQUIRED")
    for ref in refs:
        if not isinstance(ref, Mapping):
            raise ContractViolation("INVALID_SOURCE_OPTION_REF")
        require_sha256(str(ref.get("raw_sha256", "")), "INVALID_SOURCE_OPTION_REF_HASH")
    return option


def build_review_tables(merge: Mapping[str, Any]) -> ReviewTables:
    """Build blank, non-approving review tables from a verified merge."""
    candidates = _verified_candidates(merge)
    snapshot_id = str(merge["snapshot_id"])
    merge_hash = str(merge["merge_sha256"])
    terms: list[dict[str, str]] = []
    pairs: list[dict[str, str]] = []
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}

    for review_order, candidate in enumerate(candidates, 1):
        candidate_id = str(candidate["candidate_id"])
        per_language_index = {language: 0 for language in LANGUAGES}
        for language in LANGUAGES:
            options = candidate["options"][language]
            if not isinstance(options, list) or not options:
                raise ContractViolation("EMPTY_LANGUAGE_OPTIONS")
            for option_index, raw_option in enumerate(options, 1):
                option = _option(raw_option)
                raw = str(option["word_raw"])
                for _segment_index, start, end in _trimmed_segments(raw, language):
                    per_language_index[language] += 1
                    term_index = per_language_index[language]
                    exact = raw[start:end]
                    registered = _source_normalized(exact)
                    term_id = f"{candidate_id}|{language}|{term_index}"
                    row: dict[str, str] = {
                        "schema_version": VERSION,
                        "snapshot_id": snapshot_id,
                        "merge_sha256": merge_hash,
                        "review_order": str(review_order),
                        "candidate_id": candidate_id,
                        "candidate_sha256": str(candidate["candidate_sha256"]),
                        "lang": language,
                        "term_index": str(term_index),
                        "source_option_index": str(option_index),
                        "source_option_id": str(option["option_id"]),
                        "source_word_raw": raw,
                        "source_definition_raw": str(option.get("definition_raw", "")),
                        "source_gloss": str(option.get("gloss", "")),
                        "source_refs_json": json.dumps(
                            option.get("source_refs", []),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "source_word_raw_sha256": _sha_text(raw),
                        "source_span_start": str(start),
                        "source_span_end": str(end),
                        "source_text_exact": exact,
                        "term_canonical": registered,
                        "term_id": term_id,
                        "term_sha256": "",
                        "term_flags_json": "",
                        **{field: "" for field in TERM_HUMAN_FIELDS},
                    }
                    row["term_flags_json"] = _json_list(_format_flags(raw, exact, language))
                    terms.append(row)

        def sole(language: str) -> Mapping[str, Any]:
            values = candidate["options"][language]
            if len(values) != 1:
                raise ContractViolation("PAIR_DISPLAY_REQUIRES_ONE_SOURCE_OPTION")
            return _option(values[0])

        ko, zh = sole("ko"), sole("zh")
        pairs.append(
            {
                "schema_version": VERSION,
                "snapshot_id": snapshot_id,
                "merge_sha256": merge_hash,
                "review_order": str(review_order),
                "candidate_id": candidate_id,
                "candidate_sha256": str(candidate["candidate_sha256"]),
                "ko_expression": str(ko["answer"]),
                "ko_definition": str(ko.get("gloss", ko.get("definition_raw", ""))),
                "zh_expression": str(zh["answer"]),
                "zh_definition": str(zh.get("gloss", zh.get("definition_raw", ""))),
                "source_record_hashes_json": _json_list(
                    str(value) for value in candidate.get("source_record_hashes", [])
                ),
                **{field: "" for field in PAIR_HUMAN_FIELDS},
            }
        )

    # Add cross-term flags only after the complete catalog exists.
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    by_candidate_language: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in terms:
        groups.setdefault(
            (row["candidate_id"], row["lang"], row["term_canonical"]), []
        ).append(row)
        by_candidate_language.setdefault((row["candidate_id"], row["lang"]), []).append(row)
    for duplicate_rows in groups.values():
        if len(duplicate_rows) > 1:
            for row in duplicate_rows:
                flags = set(json.loads(row["term_flags_json"]))
                flags.add("DUPLICATE_CANONICAL")
                row["term_flags_json"] = _json_list(flags)
    for candidate_id in candidate_by_id:
        en_rows = by_candidate_language.get((candidate_id, "en"), [])
        fr_rows = by_candidate_language.get((candidate_id, "fr"), [])
        for en_row in en_rows:
            for fr_row in fr_rows:
                if en_row["term_canonical"] == fr_row["term_canonical"]:
                    for row in (en_row, fr_row):
                        flags = set(json.loads(row["term_flags_json"]))
                        flags.add("EXACT_EN_FR")
                        row["term_flags_json"] = _json_list(flags)
                elif en_row["term_canonical"].casefold() == fr_row[
                    "term_canonical"
                ].casefold():
                    for row in (en_row, fr_row):
                        flags = set(json.loads(row["term_flags_json"]))
                        flags.add("CASE_VARIANT_EN_FR")
                        row["term_flags_json"] = _json_list(flags)
    for row in terms:
        core = {field: row[field] for field in TERM_IMMUTABLE_FIELDS if field != "term_sha256"}
        row["term_sha256"] = _logical_hash(core)

    screen: list[dict[str, str]] = []
    term_by_candidate_language = by_candidate_language
    for pair in pairs:
        candidate_id = pair["candidate_id"]
        ko_terms = term_by_candidate_language[(candidate_id, "ko")]
        if len(ko_terms) != 1:
            raise ContractViolation("SCREEN_REQUIRES_ONE_KO_TERM")
        ko = ko_terms[0]["term_canonical"]
        romanizations = korean_romanization_variants(ko)
        for language in ("en", "fr"):
            for term in term_by_candidate_language[(candidate_id, language)]:
                flags = set(json.loads(term["term_flags_json"]))
                screen_flags = _romanization_flags(term["term_canonical"], romanizations)
                screen_flags.update(flag for flag in flags if flag != "OUTER_WHITESPACE")
                if not screen_flags:
                    continue
                screen.append(
                    {
                        "schema_version": VERSION,
                        "snapshot_id": snapshot_id,
                        "merge_sha256": merge_hash,
                        "review_order": pair["review_order"],
                        "candidate_id": candidate_id,
                        "candidate_sha256": pair["candidate_sha256"],
                        "term_id": term["term_id"],
                        "lang": language,
                        "ko_expression": ko,
                        "ko_romanizations_json": _json_list(romanizations),
                        "term_canonical": term["term_canonical"],
                        "screen_flags_json": _json_list(screen_flags),
                        "advisory_only": "TRUE",
                    }
                )
    return ReviewTables(tuple(terms), tuple(pairs), tuple(screen))


def _render(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise ContractViolation("CSV_ROW_SCHEMA_MISMATCH")
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _screen_summary(tables: ReviewTables) -> dict[str, int]:
    exact_candidates: set[str] = set()
    case_candidates: set[str] = set()
    exact_terms: set[str] = set()
    case_terms: set[str] = set()
    romanized_terms: set[str] = set()
    priority_terms: set[str] = set()
    for row in tables.screen:
        flags = set(json.loads(row["screen_flags_json"]))
        if "EXACT_EN_FR" in flags:
            exact_candidates.add(row["candidate_id"])
            exact_terms.add(row["term_id"])
        if "CASE_VARIANT_EN_FR" in flags:
            case_candidates.add(row["candidate_id"])
            case_terms.add(row["term_id"])
        if flags & {"ROMANIZED_KO_EXACT", "ROMANIZED_KO_EMBEDDED"}:
            romanized_terms.add(row["term_id"])
        if flags & {
            "EXACT_EN_FR",
            "CASE_VARIANT_EN_FR",
            "ROMANIZED_KO_EXACT",
            "ROMANIZED_KO_EMBEDDED",
        }:
            priority_terms.add(row["term_id"])
    return {
        "candidate_count": len(tables.pairs),
        "term_count": len(tables.terms),
        "screen_count": len(tables.screen),
        "exact_en_fr_pair_count": len(exact_candidates),
        "exact_en_fr_term_count": len(exact_terms),
        "case_variant_pair_count": len(case_candidates),
        "case_variant_term_count": len(case_terms),
        "romanized_term_count": len(romanized_terms),
        "priority_union_term_count": len(priority_terms),
    }


def render_review_csv_bundle(merge: Mapping[str, Any]) -> dict[str, Any]:
    tables = build_review_tables(merge)
    return {
        "status": "PENDING_HUMAN_REVIEW",
        "training_eligible": False,
        "summary": {
            **_screen_summary(tables),
            "automatically_approved": 0,
            "human_signoff_required": True,
        },
        "files": {
            "terms_long.csv": _render(tables.terms, TERM_FIELDS),
            "pair_selection_sheet.csv": _render(tables.pairs, PAIR_FIELDS),
            "untranslated_screen.csv": _render(tables.screen, SCREEN_FIELDS),
        },
    }


def _read_csv(
    source: Path | bytes | str,
    fields: Sequence[str],
    *,
    allow_empty: bool = False,
) -> tuple[dict[str, str], ...]:
    try:
        if isinstance(source, Path):
            try:
                text = read_regular_file_bytes(source).decode("utf-8")
            except ContractViolation as exc:
                raise ContractViolation("CSV_NOT_REGULAR_FILE") from exc
        elif isinstance(source, bytes):
            text = source.decode("utf-8")
        elif isinstance(source, str):
            text = source
        else:
            raise ContractViolation("INVALID_CSV_SOURCE")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None or tuple(reader.fieldnames) != tuple(fields):
            raise ContractViolation("CSV_HEADER_MISMATCH")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ContractViolation("CSV_DUPLICATE_HEADER")
        rows = tuple(dict(row) for row in reader)
    except ContractViolation:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractViolation("INVALID_CSV") from exc
    if not rows and not allow_empty:
        raise ContractViolation("EMPTY_CSV")
    if any(None in row or set(row) != set(fields) for row in rows):
        raise ContractViolation("CSV_ROW_SCHEMA_MISMATCH")
    return rows


def _compare_source_rows(
    actual: Sequence[Mapping[str, str]],
    expected: Sequence[Mapping[str, str]],
    *,
    key: str,
    immutable: Sequence[str],
) -> None:
    actual_ids = [row.get(key) for row in actual]
    expected_ids = [row[key] for row in expected]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise ContractViolation("CSV_ROW_SET_OR_ORDER_MISMATCH")
    for actual_row, expected_row in zip(actual, expected):
        if set(actual_row) != set(expected_row):
            raise ContractViolation("CSV_ROW_SCHEMA_MISMATCH")
        for field in immutable:
            if actual_row.get(field) != expected_row[field]:
                raise ContractViolation("CSV_SOURCE_COLUMN_MISMATCH")


def _parse_string_list(value: str, code: str, *, allow_empty: bool = True) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ContractViolation(code) from exc
    if (
        not isinstance(parsed, list)
        or any(not isinstance(item, str) or not item for item in parsed)
        or len(parsed) != len(set(parsed))
    ):
        raise ContractViolation(code)
    if not allow_empty and not parsed:
        raise ContractViolation(code)
    return parsed


def _review_identity(status: str, reviewer: str, review_date: str, code: str) -> None:
    if status != "APPROVED_BY_RESEARCHER":
        raise ContractViolation(code)
    if not reviewer.strip() or any(char in reviewer for char in "\r\n\x00"):
        raise ContractViolation(code)
    try:
        parsed = date.fromisoformat(review_date)
    except ValueError as exc:
        raise ContractViolation(code) from exc
    if parsed > date.today():
        raise ContractViolation(code)


def _validate_term_human(row: Mapping[str, str], *, selected: bool) -> None:
    if not any(row[field] for field in TERM_HUMAN_FIELDS):
        if selected:
            raise ContractViolation("SELECTED_TERM_QA_PENDING")
        return
    if row["segmentation_decision"] not in {"APPROVED", "REJECTED", "UNRESOLVED"}:
        raise ContractViolation("INVALID_SEGMENTATION_DECISION")
    if row["translation_quality"] not in {
        "ATTESTED_SAME_SENSE",
        "TRANSLATION_MISSING",
        "UNRESOLVED",
    }:
        raise ContractViolation("INVALID_TRANSLATION_QUALITY")
    if row["is_transliteration"] not in {"TRUE", "FALSE"}:
        raise ContractViolation("INVALID_TRANSLITERATION_VALUE")
    _parse_string_list(
        row["quality_evidence_ids_json"], "INVALID_TERM_EVIDENCE_IDS", allow_empty=False
    )
    _review_identity(
        row["term_review_status"],
        row["term_reviewer"],
        row["term_review_date"],
        "TERM_REVIEW_NOT_APPROVED",
    )
    if selected and (
        row["segmentation_decision"] != "APPROVED"
        or row["translation_quality"] != "ATTESTED_SAME_SENSE"
    ):
        raise ContractViolation("SELECTED_TERM_QA_FAILED")


def _validate_pair_relation(row: Mapping[str, str]) -> None:
    subtype = row["etymology_subtype"]
    primary = map_etymology_label(subtype)
    if row["etymology_primary"] != primary:
        raise ContractViolation("ETYMOLOGY_MAPPING_MISMATCH")
    validate_relation_direction(subtype, row["relation_direction"], row["shared_source"])
    evidence = _parse_string_list(
        row["etymology_evidence_ids_json"],
        "INVALID_ETYMOLOGY_EVIDENCE_IDS",
        allow_empty=subtype == "INDETERMINATE",
    )
    if subtype == "INDETERMINATE" and not row["evidence_search_note"].strip():
        raise ContractViolation("UNRESOLVED_SEARCH_NOTE_REQUIRED")
    if not row["historical_scope"].strip() or not row["family_id"].strip():
        raise ContractViolation("ETYMOLOGY_SCOPE_OR_FAMILY_REQUIRED")
    if row["confidence"] not in {"HIGH", "MEDIUM", "LOW"}:
        raise ContractViolation("INVALID_ETYMOLOGY_CONFIDENCE")
    if subtype == "INDETERMINATE" and row["confidence"] == "HIGH":
        raise ContractViolation("INDETERMINATE_HIGH_CONFIDENCE_FORBIDDEN")


def validate_review_tables(
    merge: Mapping[str, Any],
    tables: ReviewTables,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    expected = build_review_tables(merge)
    _compare_source_rows(
        tables.terms,
        expected.terms,
        key="term_id",
        immutable=TERM_IMMUTABLE_FIELDS,
    )
    _compare_source_rows(
        tables.pairs,
        expected.pairs,
        key="candidate_id",
        immutable=PAIR_IMMUTABLE_FIELDS,
    )
    _compare_source_rows(
        tables.screen,
        expected.screen,
        key="term_id",
        immutable=SCREEN_FIELDS,
    )
    terms_by_id = {row["term_id"]: row for row in tables.terms}
    selected_term_ids: set[str] = set()
    pilot_rows: list[Mapping[str, str]] = []
    for row in tables.pairs:
        selected = [row[f"selected_{language}_term_id"] for language in LANGUAGES]
        if any(selected):
            if row["candidate_decision"] != "SELECT_PAIR" or not all(selected):
                raise ContractViolation("INCOMPLETE_TERM_SELECTION")
            for language, term_id in zip(LANGUAGES, selected):
                term = terms_by_id.get(term_id)
                if (
                    term is None
                    or term["candidate_id"] != row["candidate_id"]
                    or term["lang"] != language
                ):
                    raise ContractViolation("SELECTED_TERM_FK_MISMATCH")
                selected_term_ids.add(term_id)
        elif row["candidate_decision"] not in {"", "EXCLUDE_NO_VALID_PAIR", "UNRESOLVED"}:
            raise ContractViolation("INVALID_CANDIDATE_DECISION")
        if row["pilot_selected"] not in {"", "TRUE", "FALSE"}:
            raise ContractViolation("INVALID_PILOT_SELECTED")
        if row["pilot_selected"] == "TRUE":
            pilot_rows.append(row)
        relation_fields = (
            "etymology_subtype",
            "etymology_primary",
            "relation_direction",
            "shared_source",
            "historical_scope",
            "confidence",
            "etymology_evidence_ids_json",
            "evidence_search_note",
            "family_id",
        )
        if any(row[field] for field in relation_fields):
            if not all(
                row[field]
                for field in (
                    "etymology_subtype",
                    "etymology_primary",
                    "relation_direction",
                    "historical_scope",
                    "confidence",
                    "family_id",
                )
            ):
                raise ContractViolation("INCOMPLETE_ETYMOLOGY_DECISION")
            _validate_pair_relation(row)

    for term in tables.terms:
        _validate_term_human(
            term,
            selected=term["term_id"] in selected_term_ids
            and term["lang"] in {"en", "fr"},
        )

    if require_complete:
        if len(pilot_rows) != 60:
            raise ContractViolation("PILOT_COHORT_MUST_BE_EXACTLY_60")
        identifiable = 0
        related = 0
        distinct = 0
        for row in pilot_rows:
            if row["candidate_decision"] != "SELECT_PAIR":
                raise ContractViolation("PILOT_ROW_NOT_SELECTED")
            if row["meaning_alignment_decision"] not in {"ALIGNED", "PARTIAL"}:
                raise ContractViolation("PILOT_MEANING_NOT_ALIGNED")
            if not row["sense_alignment_note"].strip():
                raise ContractViolation("SENSE_ALIGNMENT_NOTE_REQUIRED")
            if row["source_alignment_checked"] != "TRUE" or row[
                "answer_copy_checked"
            ] != "TRUE":
                raise ContractViolation("CONCEPT_QA_CHECKS_REQUIRED")
            if not row["selection_rationale"].strip() or not row[
                "synonym_cluster_id"
            ].strip():
                raise ContractViolation("SELECTION_RATIONALE_OR_CLUSTER_REQUIRED")
            _review_identity(
                row["concept_review_status"],
                row["concept_reviewer"],
                row["concept_review_date"],
                "CONCEPT_REVIEW_NOT_APPROVED",
            )
            _validate_pair_relation(row)
            _review_identity(
                row["etymology_review_status"],
                row["etymology_reviewer"],
                row["etymology_review_date"],
                "ETYMOLOGY_REVIEW_NOT_APPROVED",
            )
            answers = [
                terms_by_id[row[f"selected_{language}_term_id"]]["term_canonical"]
                for language in LANGUAGES
            ]
            if len(set(answers)) == 4:
                identifiable += 1
                if row["etymology_primary"] in {
                    "BORROWING_DOCUMENTED",
                    "SHARED_SOURCE_DOCUMENTED",
                }:
                    related += 1
                if row["etymology_primary"] == "DISTINCT_ROUTES_REVIEWED":
                    distinct += 1
        if identifiable < 40 or related < 12 or distinct < 12:
            raise ContractViolation("BLOCKED_DATA_COVERAGE")
        status = "STRUCTURE_PASS_PENDING_EVIDENCE"
    else:
        identifiable = related = distinct = 0
        status = "BLOCKED_DATA_QA"
    summary = {
        **_screen_summary(tables),
        "pilot_count": len(pilot_rows),
        "n_identifiable": identifiable,
        "n_related_identifiable": related,
        "n_distinct_routes_identifiable": distinct,
    }
    return {
        "status": status,
        "training_eligible": False,
        "automatically_approved": 0,
        "term_count": len(tables.terms),
        "candidate_count": len(tables.pairs),
        "screen_count": len(tables.screen),
        "pilot_count": len(pilot_rows),
        "n_identifiable": identifiable,
        "n_related_identifiable": related,
        "n_distinct_routes_identifiable": distinct,
        "summary": summary,
    }


def load_and_validate_review_csv_bundle(
    merge: Mapping[str, Any],
    terms_csv: Path | bytes | str,
    pair_csv: Path | bytes | str,
    screen_csv: Path | bytes | str,
    *,
    require_complete: bool = False,
) -> ReviewTables:
    tables = ReviewTables(
        _read_csv(terms_csv, TERM_FIELDS),
        _read_csv(pair_csv, PAIR_FIELDS),
        _read_csv(screen_csv, SCREEN_FIELDS, allow_empty=True),
    )
    validate_review_tables(merge, tables, require_complete=require_complete)
    return tables


def _evidence_record(
    evidence_id: str,
    evidence_records: Mapping[str, Mapping[str, Any]],
    evidence_payload_hashes: set[str],
    *,
    subject_kind: str,
    subject_id: str,
    supports_label: str | None,
) -> dict[str, Any]:
    record = evidence_records.get(evidence_id)
    if not isinstance(record, Mapping):
        raise ContractViolation("EVIDENCE_RECORD_MISSING")
    if (
        record.get("subject_kind") != subject_kind
        or record.get("subject_id") != subject_id
    ):
        raise ContractViolation("EVIDENCE_SUBJECT_MISMATCH")
    if supports_label is None:
        if record.get("supports_label") not in ETYMOLOGY_6_TO_4:
            raise ContractViolation("PAIR_EVIDENCE_LABEL_MISMATCH")
    elif record.get("supports_label") != supports_label:
        raise ContractViolation("EVIDENCE_SUPPORT_LABEL_MISMATCH")
    payload_hash = require_sha256(
        str(record.get("payload_sha256", "")), "INVALID_EVIDENCE_PAYLOAD_SHA256"
    )
    if payload_hash not in evidence_payload_hashes:
        raise ContractViolation("UNBOUND_REVIEW_EVIDENCE_PAYLOAD")
    return dict(record)


def validate_pair_evidence_set(
    evidence: Sequence[Mapping[str, Any]],
    *,
    subject_id: str,
    selected_subtype: str,
    confidence: str,
) -> list[dict[str, Any]]:
    """Validate evidence for a selected subtype without discarding conflicts.

    At least one record must support the selected six-way subtype.  Records
    supporting another subtype are preserved, but each must have a direct,
    symmetric conflict edge to selected-subtype evidence.  Alternative
    evidence never votes or silently changes the researcher's selected label.
    """
    map_etymology_label(selected_subtype)
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        raise ContractViolation("INVALID_ETYMOLOGY_CONFIDENCE")
    if not isinstance(evidence, Sequence) or isinstance(
        evidence, (str, bytes, bytearray)
    ):
        raise ContractViolation("INVALID_ETYMOLOGY_EVIDENCE")
    checked = [dict(item) for item in evidence if isinstance(item, Mapping)]
    if len(checked) != len(evidence):
        raise ContractViolation("INVALID_ETYMOLOGY_EVIDENCE")
    if not checked:
        if selected_subtype == "INDETERMINATE":
            return []
        raise ContractViolation("INVALID_ETYMOLOGY_EVIDENCE")
    by_id: dict[str, dict[str, Any]] = {}
    for item in checked:
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in by_id:
            raise ContractViolation("INVALID_ETYMOLOGY_EVIDENCE_IDS")
        if (
            item.get("subject_kind") != "EN_FR_PAIR"
            or item.get("subject_id") != subject_id
        ):
            raise ContractViolation("ETYMOLOGY_EVIDENCE_SUBJECT_MISMATCH")
        if item.get("supports_label") not in ETYMOLOGY_6_TO_4:
            raise ContractViolation("PAIR_EVIDENCE_LABEL_MISMATCH")
        conflicts = item.get("conflicts_with")
        if not isinstance(conflicts, list):
            raise ContractViolation("INVALID_EVIDENCE_CONFLICTS")
        by_id[evidence_id] = item
    selected_ids = {
        evidence_id
        for evidence_id, item in by_id.items()
        if item["supports_label"] == selected_subtype
    }
    if not selected_ids:
        raise ContractViolation("SELECTED_ETYMOLOGY_SUPPORT_REQUIRED")
    for evidence_id, item in by_id.items():
        for conflict_id in item["conflicts_with"]:
            conflict = by_id.get(conflict_id)
            if conflict is None or evidence_id not in conflict.get("conflicts_with", []):
                raise ContractViolation("EVIDENCE_CONFLICT_NOT_SYMMETRIC_OR_SAME_SUBJECT")
        if item["supports_label"] != selected_subtype and not (
            set(item["conflicts_with"]) & selected_ids
        ):
            raise ContractViolation("ALTERNATIVE_ETYMOLOGY_CONFLICT_REQUIRED")
    if confidence == "HIGH" and any(item["conflicts_with"] for item in checked):
        raise ContractViolation("CONFLICTING_EVIDENCE_HIGH_CONFIDENCE_FORBIDDEN")
    return checked


def compile_filled_review_bundle(
    merge: Mapping[str, Any],
    tables: ReviewTables,
    evidence_records: Mapping[str, Mapping[str, Any]],
    evidence_payload_hashes: set[str],
    production: bool = True,
    evidence_scope_root: Path | None = None,
) -> dict[str, Any]:
    """Compile completed CSV decisions and run the production review validator."""
    audit = validate_review_tables(merge, tables, require_complete=True)
    if not isinstance(evidence_records, Mapping):
        raise ContractViolation("INVALID_EVIDENCE_RECORDS")
    indexed_evidence = index_evidence_records(
        list(evidence_records.values()), production=production
    )
    if set(indexed_evidence) != set(evidence_records) or any(
        indexed_evidence[key] != evidence_records[key] for key in indexed_evidence
    ):
        raise ContractViolation("EVIDENCE_INDEX_MISMATCH")
    if not isinstance(evidence_payload_hashes, set):
        raise ContractViolation("EVIDENCE_PAYLOAD_HASH_SET_REQUIRED")
    asserted_payload_hashes = {
        require_sha256(value, "INVALID_EVIDENCE_PAYLOAD_SHA256")
        for value in evidence_payload_hashes
    }
    if production:
        if evidence_scope_root is None:
            raise ContractViolation("PRODUCTION_EVIDENCE_SCOPE_REQUIRED")
        verified_payload_hashes = verify_evidence_payloads(
            indexed_evidence, evidence_scope_root
        )
        if asserted_payload_hashes != verified_payload_hashes:
            raise ContractViolation("EVIDENCE_PAYLOAD_HASH_ASSERTION_MISMATCH")
    else:
        verified_payload_hashes = asserted_payload_hashes
    term_by_id = {row["term_id"]: row for row in tables.terms}
    candidates = {str(row["candidate_id"]): row for row in _verified_candidates(merge)}
    pilot_rows = [row for row in tables.pairs if row["pilot_selected"] == "TRUE"]
    concept_reviews: list[dict[str, Any]] = []
    etymology_reviews: list[dict[str, Any]] = []
    referenced_record_hashes: set[str] = set()
    referenced_payload_hashes: set[str] = set()
    referenced_evidence_ids: set[str] = set()
    for row in pilot_rows:
        candidate = candidates[row["candidate_id"]]
        selections: dict[str, dict[str, Any]] = {}
        for language in LANGUAGES:
            term = term_by_id[row[f"selected_{language}_term_id"]]
            selections[language] = {
                "option_id": term["source_option_id"],
                "answer": term["term_canonical"],
                "source_span": [
                    int(term["source_span_start"]),
                    int(term["source_span_end"]),
                ],
                "selection_rationale": row["selection_rationale"],
            }
        term_quality_reviews: dict[str, dict[str, Any]] = {}
        for language in ("en", "fr"):
            term = term_by_id[row[f"selected_{language}_term_id"]]
            quality_evidence_ids = _parse_string_list(
                term["quality_evidence_ids_json"],
                "INVALID_TERM_EVIDENCE_IDS",
                allow_empty=False,
            )
            quality_evidence = [
                _evidence_record(
                    evidence_id,
                    indexed_evidence,
                    verified_payload_hashes,
                    subject_kind="TERM",
                    subject_id=term["term_id"],
                    supports_label="ATTESTED_SAME_SENSE",
                )
                for evidence_id in quality_evidence_ids
            ]
            if referenced_evidence_ids.intersection(quality_evidence_ids):
                raise ContractViolation("EVIDENCE_REUSED_ACROSS_SUBJECTS")
            referenced_evidence_ids.update(quality_evidence_ids)
            referenced_record_hashes.update(
                item["record_sha256"] for item in quality_evidence
            )
            referenced_payload_hashes.update(
                item["payload_sha256"] for item in quality_evidence
            )
            term_quality_reviews[language] = {
                "term_id": term["term_id"],
                "term_sha256": term["term_sha256"],
                "segmentation_decision": term["segmentation_decision"],
                "translation_quality": term["translation_quality"],
                "is_transliteration": term["is_transliteration"] == "TRUE",
                "quality_note": term["quality_note"] or None,
                "quality_evidence_ids": quality_evidence_ids,
                "evidence": quality_evidence,
                "qa": {
                    "status": term["term_review_status"],
                    "reviewer": term["term_reviewer"],
                    "review_date": term["term_review_date"],
                },
            }
        concept_reviews.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_sha256": row["candidate_sha256"],
                "synthetic_fixture": not production,
                "selections": selections,
                "synonym_cluster_id": row["synonym_cluster_id"],
                "term_quality_reviews": term_quality_reviews,
                "meaning_alignment_decision": row["meaning_alignment_decision"],
                "qa": {
                    "status": row["concept_review_status"],
                    "reviewer": row["concept_reviewer"],
                    "review_date": row["concept_review_date"],
                    "source_alignment_checked": True,
                    "answer_copy_checked": True,
                    "meaning_alignment_note": row["sense_alignment_note"],
                },
            }
        )
        evidence_ids = _parse_string_list(
            row["etymology_evidence_ids_json"],
            "INVALID_ETYMOLOGY_EVIDENCE_IDS",
            allow_empty=row["etymology_subtype"] == "INDETERMINATE",
        )
        selected_pair_hash = answer_pair_sha256(
            selections["en"]["answer"], selections["fr"]["answer"]
        )
        selected_en_term = term_by_id[row["selected_en_term_id"]]
        selected_fr_term = term_by_id[row["selected_fr_term_id"]]
        evidence_subject_hash = en_fr_pair_evidence_subject_sha256(
            candidate_id=row["candidate_id"],
            en_term_id=selected_en_term["term_id"],
            en_term_sha256=selected_en_term["term_sha256"],
            en_canonical_answer=selected_en_term["term_canonical"],
            fr_term_id=selected_fr_term["term_id"],
            fr_term_sha256=selected_fr_term["term_sha256"],
            fr_canonical_answer=selected_fr_term["term_canonical"],
        )
        evidence = [
            _evidence_record(
                value,
                indexed_evidence,
                verified_payload_hashes,
                subject_kind="EN_FR_PAIR",
                subject_id=evidence_subject_hash,
                supports_label=None,
            )
            for value in evidence_ids
        ]
        evidence = validate_pair_evidence_set(
            evidence,
            subject_id=evidence_subject_hash,
            selected_subtype=row["etymology_subtype"],
            confidence=row["confidence"],
        )
        if referenced_evidence_ids.intersection(evidence_ids):
            raise ContractViolation("EVIDENCE_REUSED_ACROSS_SUBJECTS")
        referenced_evidence_ids.update(evidence_ids)
        referenced_record_hashes.update(item["record_sha256"] for item in evidence)
        referenced_payload_hashes.update(item["payload_sha256"] for item in evidence)
        en_answer = selections["en"]["answer"]
        fr_answer = selections["fr"]["answer"]
        etymology_reviews.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_sha256": row["candidate_sha256"],
                "synthetic_fixture": not production,
                "answer_pair_sha256": selected_pair_hash,
                "evidence_subject_sha256": evidence_subject_hash,
                "pair": ["en", "fr"],
                "relation": row["etymology_primary"],
                "relation_subtype": row["etymology_subtype"],
                "relation_direction": row["relation_direction"],
                "shared_source": row["shared_source"] or None,
                "confidence": row["confidence"],
                "evidence_ids": evidence_ids,
                "family_id": row["family_id"],
                "sense_alignment_note": row["sense_alignment_note"],
                "historical_scope": row["historical_scope"],
                "evidence": evidence,
                "evidence_search_note": row["evidence_search_note"] or None,
                "qa": {
                    "status": row["etymology_review_status"],
                    "reviewer": row["etymology_reviewer"],
                    "review_date": row["etymology_review_date"],
                    "source_alignment_checked": True,
                    "answer_copy_checked": True,
                },
            }
        )
    if set(indexed_evidence) != referenced_evidence_ids:
        raise ContractViolation("EVIDENCE_RECORD_SET_MISMATCH")

    from .prepare_data import validate_review_bundle

    cohort_ids = [row["candidate_id"] for row in pilot_rows]
    validated = validate_review_bundle(
        merge,
        concept_reviews,
        etymology_reviews,
        cohort_ids=cohort_ids,
        evidence_record_hashes=referenced_record_hashes,
        production=production,
    )
    if validated.get("status") != "PASS":
        raise ContractViolation("BLOCKED_DATA_COVERAGE")
    return {
        "status": "PASS",
        "training_eligible": False,
        "direct_trainer_input_allowed": False,
        "annotation_freeze_eligible": bool(
            production and validated.get("data_kind") == "REAL_KRDICT_API"
        ),
        "audit": audit,
        "cohort_ids": cohort_ids,
        "concept_reviews": concept_reviews,
        "etymology_reviews": etymology_reviews,
        "evidence_records": [indexed_evidence[value] for value in sorted(referenced_evidence_ids)],
        "evidence_record_hashes": referenced_record_hashes,
        "evidence_payload_hashes": referenced_payload_hashes,
        "validated": validated,
    }


__all__ = [
    "HUMAN_REVIEWED_EVIDENCE_ORIGINS",
    "SYNTHETIC_EVIDENCE_ORIGIN",
    "EVIDENCE_FIELDS",
    "ETYMOLOGY_6_TO_4",
    "PAIR_FIELDS",
    "SCREEN_FIELDS",
    "TERM_FIELDS",
    "ReviewTables",
    "answer_pair_sha256",
    "build_review_tables",
    "canonical",
    "compile_filled_review_bundle",
    "en_fr_pair_evidence_subject_sha256",
    "evidence_record_sha256",
    "korean_romanization_variants",
    "index_evidence_records",
    "load_and_validate_review_csv_bundle",
    "map_etymology_label",
    "render_review_csv_bundle",
    "validate_relation_direction",
    "validate_evidence_origin",
    "validate_pair_evidence_set",
    "validate_review_tables",
    "verify_evidence_payloads",
]
