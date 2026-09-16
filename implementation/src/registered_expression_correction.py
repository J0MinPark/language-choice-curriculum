"""Source-bound, non-production selection correction authorized by jm02.

The source candidate hash continues to identify the original KRDICT record.
A separate selection hash identifies the corrected four-language selection.
This artifact is not an annotation freeze or a trainer input.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .artifacts import publish_json_once, read_regular_file_bytes_exact
from .contracts import PROJECT_ROOT, WORK_ROOT, ContractViolation, canonical_json_bytes, sha256_bytes
from .design_revision import audit_v4_1_design_amendment
from .oewn_evidence import audit_oewn_evidence_bundle
from .semantic_review_projection import build_semantic_review_projection

AMENDMENT = "implementation/protocol/AMENDMENT_V4_1_1_EGG.json"
AMENDMENT_SHA256 = "ca56f6aaed26619ee10ee49f852ae0e5e015fb5fc1348dc695fcf584768d5cfd"
AMENDMENT_BYTES = 7209
SCHEMA = "registered-expression-correction-v1"


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _read_bound(ref: dict, root: Path) -> tuple[dict, bytes]:
    declared = Path(ref["path"])
    if not declared.is_absolute():
        declared = root / declared
    path = declared.resolve(strict=True)
    if path != declared.absolute() or not path.is_relative_to(root.resolve()):
        raise ContractViolation("CORRECTION_ARTIFACT_PATH_MISMATCH")
    raw = read_regular_file_bytes_exact(path, expected_bytes=ref["bytes"])
    if sha256_bytes(raw) != ref["sha256"]:
        raise ContractViolation("CORRECTION_ARTIFACT_HASH_MISMATCH")
    return {"path": str(path), "sha256": ref["sha256"], "bytes": len(raw)}, raw


def build_registered_expression_correction(evidence_manifest_path: Path) -> dict:
    """Re-derive exactly one authorized correction from verified inputs."""
    audit_v4_1_design_amendment()
    amendment_ref, raw = _read_bound(
        {"path": AMENDMENT, "sha256": AMENDMENT_SHA256, "bytes": AMENDMENT_BYTES},
        PROJECT_ROOT,
    )
    amendment = json.loads(raw)
    refs, loaded = {}, {}
    for name, binding in amendment["artifact_bindings"].items():
        refs[name], loaded[name] = _read_bound(binding, PROJECT_ROOT)
    evidence = audit_oewn_evidence_bundle(evidence_manifest_path)
    projection = build_semantic_review_projection(
        Path(refs["old_review_feedback_intake_manifest"]["path"])
    )
    plan = json.loads(loaded["old_selection_plan"])
    selected = copy.deepcopy(plan["selected"])
    if len(selected) != 60 or [r["selection_ordinal"] for r in selected] != list(range(1, 61)):
        raise ContractViolation("CORRECTION_SELECTION_MISMATCH")
    correction = amendment["correction"]
    row = selected[37]
    old = copy.deepcopy(row["terms"]["en"])
    if (row["candidate_id"] != correction["candidate_id"]
        or row["candidate_sha256"] != correction["candidate_sha256"]
        or old["term_sha256"] != correction["from"]["term_sha256"]
        or old["canonical"] != "hen's egg"):
        raise ContractViolation("CORRECTION_SUBJECT_MISMATCH")
    # The new record explicitly identifies both the original selection and
    # dictionary evidence; it cannot be mistaken for an original CSV term.
    new_term = {
        "schema_version": SCHEMA,
        "term_id": old["term_id"] + "|correction-v4.1.1",
        "canonical": "egg",
        "source_gloss": old["source_gloss"],
        "source_option_id": old["source_option_id"],
        "source_span_start": 6,
        "source_span_end": 9,
        "source_text_exact": "egg",
        "source_term": old,
        "amendment_artifact": amendment_ref,
        "evidence_manifest_artifact": evidence["manifest_artifact"],
    }
    if old["canonical"][6:9] != new_term["canonical"]:
        raise ContractViolation("CORRECTION_SOURCE_SPAN_MISMATCH")
    new_term["term_sha256"] = _hash(new_term)
    row["terms"]["en"] = new_term
    # Preserve the immutable source candidate identity; hash the selection
    # separately, because changing an answer does not change the raw source.
    row["selection_sha256"] = _hash(row)
    core = {
        "schema_version": SCHEMA,
        "protocol_version": "4.1.1",
        "status": "CORRECTED_SELECTION_PENDING_FORMAL_FREEZE",
        "amendment_artifact": amendment_ref,
        "source_artifacts": refs,
        "evidence_manifest_artifact": evidence["manifest_artifact"],
        "reviewer_id": "jm02",
        "explicit_user_correction_recorded": True,
        "identity_authentication_claimed": False,
        "formal_evidence_verified": False,
        "training_eligible": False,
        "direct_trainer_input_allowed": False,
        "annotation_freeze_created": False,
        "main_enabled": False,
        "correction_count": 1,
        "selected_count": 60,
        "selected": selected,
        "cohort_sha256": _hash(selected),
        "prior_core_review_projection_sha256": projection["projection_sha256"],
        "source_review_counts": projection["decision_counts"],
        "correction_decision": {
            "selection_ordinal": 38, "language": "en",
            "from": "hen's egg", "to": "egg",
            "status": "USER_CORRECTION_RECORDED_WITH_DICTIONARY_SUPPORT",
            "requires_formal_freeze": True,
        },
        "optional_etymology": projection["optional_etymology"],
        "remaining_gates": [
            "V4_1_1_FORMAL_REVIEW_AND_ANNOTATION_FREEZE",
            "RECOMPUTE_SURFACE_AND_TOKEN_FEATURES",
            "EXPERIMENT_FREEZE_AND_GPU_REPLAY",
            "PRODUCTION_PILOT_RUNNER_IMPLEMENTATION",
        ],
    }
    return {**core, "correction_sha256": _hash(core)}


def validate_registered_expression_correction(payload: dict) -> dict:
    """Rebuild from the original bytes, so rehashing a changed answer fails."""
    try:
        evidence_path = Path(payload["evidence_manifest_artifact"]["path"])
        expected = build_registered_expression_correction(evidence_path)
    except (KeyError, TypeError) as exc:
        raise ContractViolation("CORRECTION_INVALID_PAYLOAD") from exc
    if canonical_json_bytes(payload) != canonical_json_bytes(expected):
        raise ContractViolation("CORRECTION_CONTENT_MISMATCH")
    return {"status": "PASS_REGISTERED_EXPRESSION_CORRECTION", "cohort_sha256": expected["cohort_sha256"], "training_eligible": False}


def publish_registered_expression_correction(evidence_manifest_path: Path, output_path: Path) -> dict:
    path = output_path.absolute()
    if path != path.resolve() or not path.is_relative_to(WORK_ROOT):
        raise ContractViolation("CORRECTION_OUTPUT_OUTSIDE_SCOPE")
    payload = build_registered_expression_correction(evidence_manifest_path)
    validate_registered_expression_correction(payload)
    artifact = publish_json_once(path, payload)
    raw = read_regular_file_bytes_exact(path, expected_bytes=artifact["bytes"])
    if sha256_bytes(raw) != artifact["sha256"]:
        raise ContractViolation("CORRECTION_PUBLICATION_HASH_MISMATCH")
    audit = validate_registered_expression_correction(json.loads(raw))
    return {**audit, "artifact": artifact}
