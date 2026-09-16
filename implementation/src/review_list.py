"""Produce a human-readable, non-approving etymology review worksheet."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import publish_bytes_once, publish_json_once, read_verified_json, sha256_file
from .contracts import (
    VERSION,
    WORK_ROOT,
    ContractViolation,
    require_relative_to,
    require_sha256,
)


LANGUAGES = ("ko", "en", "zh", "fr")
RELATION_LABELS = (
    "BORROWING_DOCUMENTED",
    "SHARED_SOURCE_DOCUMENTED",
    "DISTINCT_ROUTES_REVIEWED",
    "UNRESOLVED",
)
CSV_FIELDS = (
    "review_order",
    "candidate_id",
    "candidate_sha256",
    "ko_expression",
    "ko_definition",
    "en_expression",
    "en_definition",
    "zh_expression",
    "zh_definition",
    "fr_expression",
    "fr_definition",
    "ko_option_id",
    "en_option_id",
    "zh_option_id",
    "fr_option_id",
    "en_source_hashes",
    "fr_source_hashes",
    "pilot_selected",
    "meaning_alignment_decision",
    "en_fr_relation",
    "evidence_urls",
    "evidence_record_hashes",
    "sense_alignment_note",
    "historical_scope",
    "family_id",
    "synonym_cluster_id",
    "reviewer",
    "review_date",
)


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ContractViolation("PENDING_REVIEW_NOT_REGULAR_FILE")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ContractViolation("INVALID_PENDING_REVIEW_ROW")
                rows.append(value)
    except ContractViolation:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation("INVALID_PENDING_REVIEW_JSONL") from exc
    if not rows:
        raise ContractViolation("EMPTY_PENDING_REVIEW")
    return rows


def _one_option(task: Mapping[str, Any], language: str) -> Mapping[str, Any]:
    options = task.get("answer_options")
    if not isinstance(options, Mapping) or set(options) != set(LANGUAGES):
        raise ContractViolation("FOUR_LANGUAGE_OPTIONS_REQUIRED")
    values = options.get(language)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], Mapping):
        raise ContractViolation("WORKSHEET_REQUIRES_ONE_OPTION_PER_LANGUAGE")
    option = values[0]
    for field in ("answer", "definition_raw", "option_id"):
        if not isinstance(option.get(field), str) or not option[field].strip():
            raise ContractViolation("INVALID_REVIEW_OPTION")
    require_sha256(option["option_id"], "INVALID_REVIEW_OPTION_ID")
    refs = option.get("source_refs")
    if not isinstance(refs, list) or not refs:
        raise ContractViolation("REVIEW_OPTION_SOURCE_REQUIRED")
    for ref in refs:
        if not isinstance(ref, Mapping):
            raise ContractViolation("INVALID_REVIEW_SOURCE_REF")
        require_sha256(str(ref.get("raw_sha256", "")), "INVALID_REVIEW_SOURCE_HASH")
    return option


def _source_hashes(option: Mapping[str, Any]) -> str:
    return ";".join(
        sorted({str(ref["raw_sha256"]) for ref in option["source_refs"]})
    )


def export_etymology_review_list(
    pending_jsonl: Path,
    pending_summary_json: Path,
    *,
    output_dir: Path,
    production: bool = True,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Publish a worksheet that cannot itself satisfy the review gate."""
    pending_path = require_relative_to(
        Path(pending_jsonl), scope_root, "PENDING_REVIEW_OUTSIDE_SCOPE"
    )
    summary_path = require_relative_to(
        Path(pending_summary_json), scope_root, "PENDING_SUMMARY_OUTSIDE_SCOPE"
    )
    destination = require_relative_to(
        Path(output_dir), scope_root, "REVIEW_LIST_OUTPUT_OUTSIDE_SCOPE"
    )
    summary = read_verified_json(summary_path)
    if (
        summary.get("schema_version") != VERSION
        or summary.get("status") != "PENDING_HUMAN_REVIEW"
        or not isinstance(summary.get("summary"), Mapping)
        or summary["summary"].get("automatically_approved") != 0
        or summary["summary"].get("human_signoff_required") is not True
    ):
        raise ContractViolation("INVALID_PENDING_REVIEW_SUMMARY")
    if production and summary.get("data_kind") != "REAL_KRDICT_API":
        raise ContractViolation("SYNTHETIC_REVIEW_LIST_NOT_PRODUCTION")
    tasks = _read_tasks(pending_path)
    if summary["summary"].get("pending") != len(tasks):
        raise ContractViolation("PENDING_REVIEW_COUNT_MISMATCH")
    seen: set[str] = set()
    worksheet_rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        candidate_id = task.get("candidate_id")
        candidate_hash = task.get("candidate_sha256")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen
            or task.get("task_type") != "CONCEPT_AND_EN_FR_ETYMOLOGY"
            or task.get("status") != "PENDING"
        ):
            raise ContractViolation("INVALID_PENDING_REVIEW_TASK")
        seen.add(candidate_id)
        require_sha256(str(candidate_hash or ""), "INVALID_CANDIDATE_HASH")
        options = {language: _one_option(task, language) for language in LANGUAGES}
        row = {
            "review_order": index,
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_hash,
        }
        for language in LANGUAGES:
            row[f"{language}_expression"] = options[language]["answer"]
            row[f"{language}_definition"] = options[language]["definition_raw"]
            row[f"{language}_option_id"] = options[language]["option_id"]
        row["en_source_hashes"] = _source_hashes(options["en"])
        row["fr_source_hashes"] = _source_hashes(options["fr"])
        for field in CSV_FIELDS:
            row.setdefault(field, "")
        worksheet_rows.append(row)

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(worksheet_rows)
    csv_ref = publish_bytes_once(
        destination / "etymology_review_list.csv",
        stream.getvalue().encode("utf-8"),
    )
    manifest = {
        "schema_version": "etymology-review-list-v1",
        "status": "PENDING_HUMAN_REVIEW",
        "training_eligible": False,
        "automatically_approved": 0,
        "row_count": len(worksheet_rows),
        "source": {
            "pending_review": {
                "path": str(pending_path),
                "sha256": sha256_file(pending_path),
                "bytes": pending_path.stat().st_size,
            },
            "pending_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
                "bytes": summary_path.stat().st_size,
            },
        },
        "worksheet": csv_ref,
        "allowed_en_fr_relation_labels": list(RELATION_LABELS),
        "pilot_requirements": {
            "reviewed_concepts": 60,
            "identifiable_concepts_min": 40,
            "related_en_fr_min": 12,
            "distinct_routes_en_fr_min": 12,
        },
        "notice": "This CSV is a review worksheet only. Blank cells are not decisions, and the file cannot pass the annotation or training gate.",
    }
    manifest_ref = publish_json_once(
        destination / "etymology_review_list_manifest.json", manifest
    )
    return {**manifest, "manifest_artifact": manifest_ref}


__all__ = ["CSV_FIELDS", "RELATION_LABELS", "export_etymology_review_list"]
