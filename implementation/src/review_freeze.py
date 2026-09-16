"""Compile completed review CSVs into the existing annotation freeze boundary.

The CSV files are human-review interchange artifacts, not trainer inputs.  This
module deliberately performs no CSV interpretation itself and never invents a
selection, evidence hash, or reviewer sign-off.  It reparses the pinned source,
delegates strict CSV loading/compilation to :mod:`review_import`, and gives the
result to :func:`prepare_data.freeze_reviewed_dataset`, whose existing review
and coverage gates remain authoritative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import sha256_file
from .contracts import (
    WORK_ROOT,
    ContractViolation,
    require_relative_to,
    require_sha256,
)
from .prepare_data import freeze_reviewed_dataset, merge_krdict_snapshot


ReviewLoader = Callable[[Mapping[str, Any], Path, Path, Path], Mapping[str, Any]]
ReviewCompiler = Callable[..., Mapping[str, Any]]


def _load_review_import_api() -> tuple[ReviewLoader, ReviewCompiler]:
    """Load the review adapter lazily so this boundary has one dependency edge."""

    try:
        from .review_import import (  # type: ignore[import-not-found]
            compile_filled_review_bundle,
            load_and_validate_review_csv_bundle,
        )
    except (ImportError, AttributeError) as exc:
        raise ContractViolation("NOT_IMPLEMENTED_REVIEW_IMPORT") from exc
    if not callable(load_and_validate_review_csv_bundle) or not callable(
        compile_filled_review_bundle
    ):
        raise ContractViolation("NOT_IMPLEMENTED_REVIEW_IMPORT")
    return load_and_validate_review_csv_bundle, compile_filled_review_bundle


def _checked_input(path: Path, scope_root: Path, code: str) -> Path:
    checked = require_relative_to(Path(path), scope_root, code)
    if checked.is_symlink() or not checked.is_file():
        raise ContractViolation(code)
    return checked


def _compiled_sequence(value: Any, code: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractViolation(code)
    rows = list(value)
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ContractViolation(code)
    return rows


def _evidence_ids(value: Any, code: str) -> set[str]:
    if value is None or value == "":
        return set()
    if not isinstance(value, str):
        raise ContractViolation(code)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ContractViolation(code) from exc
    if (
        not isinstance(parsed, list)
        or any(not isinstance(item, str) or not item for item in parsed)
        or len(parsed) != len(set(parsed))
    ):
        raise ContractViolation(code)
    return set(parsed)


def _referenced_evidence_bindings(
    tables: Any,
) -> dict[str, tuple[str, str, str]]:
    """Bind every pilot evidence ID to one subject and one exact label."""

    from .review_import import en_fr_pair_evidence_subject_sha256

    terms = getattr(tables, "terms", None)
    pairs = getattr(tables, "pairs", None)
    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes, bytearray)):
        raise ContractViolation("INVALID_REVIEW_TABLE_BUNDLE")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes, bytearray)):
        raise ContractViolation("INVALID_REVIEW_TABLE_BUNDLE")
    term_by_id: dict[str, Mapping[str, Any]] = {}
    for row in terms:
        if not isinstance(row, Mapping):
            raise ContractViolation("INVALID_REVIEW_TABLE_BUNDLE")
        term_id = row.get("term_id")
        if not isinstance(term_id, str) or not term_id or term_id in term_by_id:
            raise ContractViolation("INVALID_REVIEW_TABLE_BUNDLE")
        term_by_id[term_id] = row

    bindings: dict[str, tuple[str, str, str]] = {}

    def add(evidence_id: str, binding: tuple[str, str, str]) -> None:
        previous = bindings.get(evidence_id)
        if previous is not None and previous != binding:
            raise ContractViolation("EVIDENCE_REUSED_ACROSS_SUBJECTS")
        bindings[evidence_id] = binding

    for row in pairs:
        if not isinstance(row, Mapping):
            raise ContractViolation("INVALID_REVIEW_TABLE_BUNDLE")
        if row.get("pilot_selected") != "TRUE":
            continue
        candidate_id = row.get("candidate_id")
        subtype = row.get("etymology_subtype")
        if not isinstance(candidate_id, str) or not candidate_id or not isinstance(
            subtype, str
        ):
            raise ContractViolation("INVALID_REVIEW_TABLE_BUNDLE")
        selected_terms: dict[str, Mapping[str, Any]] = {}
        for language in ("en", "fr"):
            term_id = row.get(f"selected_{language}_term_id")
            term = term_by_id.get(term_id) if isinstance(term_id, str) else None
            if term is None:
                raise ContractViolation("SELECTED_TERM_FK_MISMATCH")
            selected_terms[language] = term
            ids = _evidence_ids(
                term.get("quality_evidence_ids_json"), "INVALID_TERM_EVIDENCE_IDS"
            )
            if not ids:
                raise ContractViolation("TERM_EVIDENCE_REQUIRED")
            for evidence_id in ids:
                add(evidence_id, ("TERM", term_id, "ATTESTED_SAME_SENSE"))
        pair_ids = _evidence_ids(
            row.get("etymology_evidence_ids_json"),
            "INVALID_ETYMOLOGY_EVIDENCE_IDS",
        )
        if not pair_ids and subtype != "INDETERMINATE":
            raise ContractViolation("ETYMOLOGY_EVIDENCE_REQUIRED")
        pair_subject = en_fr_pair_evidence_subject_sha256(
            candidate_id=candidate_id,
            en_term_id=str(selected_terms["en"].get("term_id", "")),
            en_term_sha256=str(selected_terms["en"].get("term_sha256", "")),
            en_canonical_answer=str(
                selected_terms["en"].get("term_canonical", "")
            ),
            fr_term_id=str(selected_terms["fr"].get("term_id", "")),
            fr_term_sha256=str(selected_terms["fr"].get("term_sha256", "")),
            fr_canonical_answer=str(
                selected_terms["fr"].get("term_canonical", "")
            ),
        )
        for evidence_id in pair_ids:
            add(evidence_id, ("EN_FR_PAIR", pair_subject, subtype))
    return bindings


def _index_and_bind_evidence(
    evidence_records: Sequence[Mapping[str, Any]],
    scope_root: Path,
    referenced: Mapping[str, tuple[str, str, str]],
    *,
    production: bool,
) -> tuple[dict[str, Mapping[str, Any]], set[str], set[str]]:
    """Validate record metadata and hash the actual payload files in scope."""

    from .review_import import index_evidence_records

    by_id = index_evidence_records(evidence_records, production=production)
    if set(by_id) != set(referenced):
        raise ContractViolation("EVIDENCE_RECORD_SET_MISMATCH")
    payload_hashes: set[str] = set()
    record_hashes: set[str] = set()
    for evidence_id, expected in referenced.items():
        record = by_id[evidence_id]
        actual_kind = str(record["subject_kind"])
        actual_subject = str(record["subject_id"])
        actual_label = str(record["supports_label"])
        expected_kind, expected_subject, expected_label = expected
        if (
            actual_kind != expected_kind
            or actual_subject != expected_subject
            or (
                expected_kind == "TERM"
                and actual_label != expected_label
            )
        ):
            raise ContractViolation("EVIDENCE_SUBJECT_OR_LABEL_MISMATCH")
        raw_path = Path(str(record["payload_path"]))
        unresolved = raw_path if raw_path.is_absolute() else scope_root / raw_path
        if unresolved.is_symlink():
            raise ContractViolation("EVIDENCE_PAYLOAD_NOT_REGULAR_FILE")
        payload_path = require_relative_to(
            unresolved, scope_root, "EVIDENCE_PAYLOAD_OUTSIDE_SCOPE"
        )
        if payload_path.is_symlink() or not payload_path.is_file():
            raise ContractViolation("EVIDENCE_PAYLOAD_NOT_REGULAR_FILE")
        if payload_path.stat().st_size != record["payload_bytes"]:
            raise ContractViolation("EVIDENCE_PAYLOAD_SIZE_MISMATCH")
        actual_hash = sha256_file(payload_path)
        if actual_hash != record["payload_sha256"]:
            raise ContractViolation("EVIDENCE_PAYLOAD_SHA256_MISMATCH")
        payload_hashes.add(actual_hash)
        record_hashes.add(str(record["record_sha256"]))
    return by_id, payload_hashes, record_hashes


def freeze_completed_review_csv_bundle(
    collection_manifest: Path,
    terms_csv: Path,
    pair_csv: Path,
    screen_csv: Path,
    *,
    evidence_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    evidence_payload_hashes: set[str] | frozenset[str] | None = None,
    requirements: Mapping[str, Any] | None = None,
    production: bool = True,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Create an annotation freeze without exposing CSVs to model code.

    ``review_import.compile_filled_review_bundle`` must return a mapping with
    ``concept_reviews``, ``etymology_reviews``, and ``cohort_ids``.  Those are
    passed unchanged to the established freeze function.  Every referenced
    payload is opened and hashed here.  ``evidence_payload_hashes`` is only an
    optional consistency assertion retained for callers of the earlier API;
    it is never a source of trust.

    Production always uses the preregistered requirements.  A requirements
    override exists only for bounded synthetic fixture tests.
    """

    root = Path(scope_root).resolve(strict=True)
    collection_path = _checked_input(
        collection_manifest, root, "COLLECTION_MANIFEST_OUTSIDE_OR_INVALID"
    )
    terms_path = _checked_input(terms_csv, root, "TERMS_CSV_OUTSIDE_OR_INVALID")
    pair_path = _checked_input(pair_csv, root, "PAIR_CSV_OUTSIDE_OR_INVALID")
    screen_path = _checked_input(screen_csv, root, "SCREEN_CSV_OUTSIDE_OR_INVALID")
    output_path = require_relative_to(
        Path(output_dir), root, "ANNOTATION_FREEZE_OUTSIDE_SCOPE"
    )
    if output_path.exists():
        raise ContractViolation("ANNOTATION_FREEZE_OUTPUT_EXISTS")
    if production and requirements is not None:
        raise ContractViolation("PRODUCTION_REQUIREMENTS_OVERRIDE_FORBIDDEN")

    if not isinstance(evidence_records, Sequence) or isinstance(
        evidence_records, (str, bytes, bytearray)
    ):
        raise ContractViolation("INVALID_EVIDENCE_RECORDS")
    records = list(evidence_records)
    if any(not isinstance(record, Mapping) for record in records):
        raise ContractViolation("INVALID_EVIDENCE_RECORDS")
    asserted_payload_hashes: set[str] | None = None
    if evidence_payload_hashes is not None:
        if not isinstance(evidence_payload_hashes, (set, frozenset)):
            raise ContractViolation("INVALID_EVIDENCE_PAYLOAD_HASH_ASSERTION")
        asserted_payload_hashes = {
            require_sha256(value, "INVALID_EVIDENCE_PAYLOAD_HASH")
            for value in evidence_payload_hashes
        }

    # First reparse: bind CSV identities and source selections to current raw
    # material.  freeze_reviewed_dataset reparses once more immediately before
    # publication, making source drift between compilation and freeze fail.
    merge = merge_krdict_snapshot(collection_path, production=production)
    load_bundle, compile_bundle = _load_review_import_api()
    tables = load_bundle(merge, terms_path, pair_path, screen_path)
    referenced = _referenced_evidence_bindings(tables)
    evidence_by_id, bound_payload_hashes, bound_record_hashes = _index_and_bind_evidence(
        records, root, referenced, production=production
    )
    if (
        asserted_payload_hashes is not None
        and asserted_payload_hashes != bound_payload_hashes
    ):
        raise ContractViolation("EVIDENCE_PAYLOAD_HASH_ASSERTION_MISMATCH")
    compiled = compile_bundle(
        merge,
        tables,
        evidence_records=evidence_by_id,
        evidence_payload_hashes=bound_payload_hashes,
        production=production,
        evidence_scope_root=root,
    )
    if not isinstance(compiled, Mapping):
        raise ContractViolation("INVALID_COMPILED_REVIEW_BUNDLE")
    if compiled.get("status") != "PASS":
        raise ContractViolation("COMPILED_REVIEW_NOT_PASS")
    validated = compiled.get("validated")
    if not isinstance(validated, Mapping) or validated.get("status") != "PASS":
        raise ContractViolation("COMPILED_REVIEW_VALIDATION_NOT_PASS")
    compiled_hashes_raw = compiled.get("evidence_record_hashes")
    if not isinstance(compiled_hashes_raw, (set, frozenset)):
        raise ContractViolation("INVALID_COMPILED_EVIDENCE_HASHES")
    compiled_hashes = {
        require_sha256(value, "INVALID_COMPILED_EVIDENCE_HASH")
        for value in compiled_hashes_raw
    }
    if compiled_hashes != bound_record_hashes:
        raise ContractViolation("COMPILED_EVIDENCE_NOT_BOUND")
    compiled_payload_hashes_raw = compiled.get("evidence_payload_hashes")
    if not isinstance(compiled_payload_hashes_raw, (set, frozenset)):
        raise ContractViolation("INVALID_COMPILED_EVIDENCE_PAYLOAD_HASHES")
    compiled_payload_hashes = {
        require_sha256(value, "INVALID_COMPILED_EVIDENCE_PAYLOAD_HASH")
        for value in compiled_payload_hashes_raw
    }
    if compiled_payload_hashes != bound_payload_hashes:
        raise ContractViolation("COMPILED_EVIDENCE_PAYLOAD_NOT_BOUND")

    concept_reviews = _compiled_sequence(
        compiled.get("concept_reviews"), "INVALID_COMPILED_CONCEPT_REVIEWS"
    )
    etymology_reviews = _compiled_sequence(
        compiled.get("etymology_reviews"), "INVALID_COMPILED_ETYMOLOGY_REVIEWS"
    )
    cohort_ids_raw = compiled.get("cohort_ids")
    if not isinstance(cohort_ids_raw, Sequence) or isinstance(
        cohort_ids_raw, (str, bytes, bytearray)
    ):
        raise ContractViolation("INVALID_COMPILED_COHORT_IDS")
    cohort_ids = list(cohort_ids_raw)
    if (
        not cohort_ids
        or any(not isinstance(value, str) or not value for value in cohort_ids)
        or len(cohort_ids) != len(set(cohort_ids))
    ):
        raise ContractViolation("INVALID_COMPILED_COHORT_IDS")

    return freeze_reviewed_dataset(
        collection_path,
        concept_reviews,
        etymology_reviews,
        cohort_ids=cohort_ids,
        output_dir=output_path,
        requirements=requirements,
        evidence_record_hashes=bound_record_hashes,
        evidence_records=[evidence_by_id[key] for key in sorted(evidence_by_id)],
        production=production,
        scope_root=root,
    )


__all__ = ["freeze_completed_review_csv_bundle"]
