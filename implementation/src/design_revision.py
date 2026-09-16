"""Fail-closed audit for the prospective v4.1 research-design amendment.

The delivered v4.0 specification remains immutable.  This module validates one
additive, Git-tracked amendment and the revised protocol/configuration that it
binds.  It deliberately does not activate a data pipeline, reinterpret a v4.0
freeze, authorize training, or claim that the remaining human-review gate has
passed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .artifacts import read_regular_file_bytes, read_regular_file_bytes_exact
from .contracts import PROJECT_ROOT, ContractViolation, require_relative_to, sha256_bytes


SCHEMA_VERSION = "protocol-amendment-v1"
AMENDMENT_STATUS = "APPROVED_DESIGN_PENDING_IMPLEMENTATION"
AUDIT_STATUS = "PASS_V4_1_PROSPECTIVE_DESIGN_AMENDMENT"
AMENDMENT_RELATIVE_PATH = "implementation/protocol/AMENDMENT_V4_1.json"
REVISED_SPEC_RELATIVE_PATH = "implementation/protocol/RESEARCH_SPEC_V4_1_KO.md"
REVISED_CONFIG_RELATIVE_PATH = "implementation/config/pilot_v4_1.json"
BASE_SPEC_RELATIVE_PATH = "spec/RESEARCH_SPEC_V4_KO.md"
BASE_CONFIG_RELATIVE_PATH = "spec/pilot.json"

# These constants make the first amendment immutable in its own validator.
# Any later scientific change must receive a new amendment/version instead of
# silently rewriting v4.1 in place.
EXPECTED_AMENDMENT_SHA256 = (
    "b0fa7066f1dc401f6a3f72178e639cb809190563d78a328d5481ff8abc19f013"
)
EXPECTED_AMENDMENT_BYTES = 4529
EXPECTED_BINDINGS = {
    "base_spec": {
        "path": BASE_SPEC_RELATIVE_PATH,
        "sha256": "cbab5968ede0c897cce684927a6eaa4b9ebef87fccb5cda44f43d02ee636e105",
        "bytes": 24144,
        "role": "IMMUTABLE_DELIVERED_BASELINE",
    },
    "base_config": {
        "path": BASE_CONFIG_RELATIVE_PATH,
        "sha256": "08a3eddd9b161d563b35d5842555ed7c309033099362aeb96a81bd94b163f76c",
        "bytes": 2066,
        "role": "IMMUTABLE_DELIVERED_BASELINE",
    },
    "revised_spec": {
        "path": REVISED_SPEC_RELATIVE_PATH,
        "sha256": "7d0d27602aa3616d7f23601c6989d0462321ce85642d9714e27aa96c918e83c4",
        "bytes": 14121,
        "role": "ACTIVE_V4_1_PROTOCOL",
    },
    "revised_config": {
        "path": REVISED_CONFIG_RELATIVE_PATH,
        "sha256": "22dcb12ba2a1976810fb55722a6cfcc8bac43c54f916e133bb8433c0720eb2e5",
        "bytes": 5284,
        "role": "ACTIVE_V4_1_CONFIG",
    },
}

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "amendment_id",
        "amendment_version",
        "implementation_revision_target",
        "project_id",
        "status",
        "recorded_at_utc",
        "hash_algorithm",
        "approval",
        "timing",
        "artifact_bindings",
        "integrity_model",
        "scientific_changes",
        "current_primary_data_state",
        "not_claimed",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "version",
        "implementation_revision",
        "project_id",
        "design_status",
        "languages",
        "roots",
        "policy",
        "budget",
        "data",
        "surface_features",
        "leakage_control",
        "model",
        "preparation",
        "measurement",
        "prediction",
        "exploratory",
        "proposed_main_criteria_NOT_AUTHORIZED",
    }
)
_LANGUAGES = ["ko", "en", "zh", "fr"]
_LANGUAGE_PAIRS = [
    ["ko", "en"],
    ["ko", "zh"],
    ["ko", "fr"],
    ["en", "zh"],
    ["en", "fr"],
    ["zh", "fr"],
]
_TERM_BLOCKER = {
    "selection_ordinal": 38,
    "review_order": 195,
    "candidate_id": "krdict-b100064abad6c836:60487:1",
    "language": "en",
    "registered_expression": "hen's egg",
    "status": "BLOCKED_TERM_QUALITY",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation("DESIGN_REVISION_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_token: str) -> None:
    raise ContractViolation("DESIGN_REVISION_NONFINITE_JSON_NUMBER")


def _load_json_bytes(raw: bytes, code: str) -> dict[str, Any]:
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


def _exact_mapping(value: Any, fields: set[str] | frozenset[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ContractViolation(code)
    return value


def _read_bound_file(root: Path, binding: Mapping[str, Any]) -> tuple[Path, bytes]:
    raw_path = binding.get("path")
    raw_size = binding.get("bytes")
    raw_hash = binding.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or Path(raw_path).is_absolute()
        or not isinstance(raw_size, int)
        or isinstance(raw_size, bool)
        or raw_size < 0
        or not isinstance(raw_hash, str)
    ):
        raise ContractViolation("DESIGN_REVISION_INVALID_ARTIFACT_BINDING")
    declared = root / raw_path
    try:
        # ``require_relative_to`` resolves symlinks for containment.  Compare
        # the result with the declared absolute path as well so neither a
        # final symlink nor a symlinked parent can alias a different inode.
        path = require_relative_to(declared, root, "DESIGN_REVISION_PATH_OUTSIDE_ROOT")
    except OSError as exc:
        raise ContractViolation("DESIGN_REVISION_PATH_OUTSIDE_ROOT") from exc
    if path != declared.absolute() or declared.is_symlink():
        raise ContractViolation("DESIGN_REVISION_ARTIFACT_NOT_REGULAR_FILE")
    try:
        raw = read_regular_file_bytes_exact(path, expected_bytes=raw_size)
    except ContractViolation as exc:
        if exc.code == "ARTIFACT_SIZE_MISMATCH":
            raise ContractViolation("DESIGN_REVISION_ARTIFACT_SIZE_MISMATCH") from exc
        if exc.code == "ARTIFACT_NOT_REGULAR_FILE":
            raise ContractViolation("DESIGN_REVISION_ARTIFACT_NOT_REGULAR_FILE") from exc
        raise
    if sha256_bytes(raw) != raw_hash:
        raise ContractViolation("DESIGN_REVISION_ARTIFACT_HASH_MISMATCH")
    return path, raw


def _validate_config(config: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    _exact_mapping(config, _CONFIG_FIELDS, "DESIGN_REVISION_CONFIG_SCHEMA_MISMATCH")
    if (
        config.get("version") != "4.1.0"
        or config.get("implementation_revision") != "4.1.0-r1"
        or config.get("project_id") != "lexical-freshstart-4lang-history-v4_1"
        or config.get("design_status") != AMENDMENT_STATUS
        or config.get("languages") != _LANGUAGES
        or config.get("roots") != [4101, 4102, 4103, 4104]
    ):
        raise ContractViolation("DESIGN_REVISION_CONFIG_IDENTITY_MISMATCH")

    # Scientific invariants not changed by this prospective amendment.
    for field in ("languages", "roots", "budget", "model", "preparation"):
        if config.get(field) != base.get(field):
            raise ContractViolation("DESIGN_REVISION_UNAUTHORIZED_BASE_CHANGE:" + field)

    policy = config.get("policy")
    if not isinstance(policy, Mapping) or (
        policy.get("main_enabled") is not False
        or policy.get("human_training_enabled") is not False
        or policy.get("correction_policy_enabled") is not False
        or policy.get("legacy_weights_allowed") is not False
        or policy.get("old_project_search_allowed") is not False
        or policy.get("model_results_observed_before_change") is not False
        or policy.get("identity_authentication_claimed") is not False
        or policy.get("legacy_freeze_reinterpretation_allowed") is not False
        or policy.get("upstream_compatibility") != "UPSTREAM_EXACT_HASH_ONLY"
    ):
        raise ContractViolation("DESIGN_REVISION_POLICY_MISMATCH")

    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ContractViolation("DESIGN_REVISION_DATA_SCHEMA_MISMATCH")
    if "min_related" in data or "min_distinct_routes" in data:
        raise ContractViolation("DESIGN_REVISION_ETYMOLOGY_QUOTA_FORBIDDEN")
    if (
        data.get("min_total") != 60
        or data.get("min_primary_approved") != 60
        or data.get("min_identifiable") != 40
        or data.get("requested_pilot_cohort_size") != 60
        or data.get("require_meaning_alignment_for_every_selected_concept") is not True
        or data.get("require_term_quality_for_every_selected_expression") is not True
        or data.get("require_source_bound_selections") is not True
        or data.get("required_registered_languages") != _LANGUAGES
        or data.get("etymology_required") is not False
        or data.get("current_term_quality_blockers") != [_TERM_BLOCKER]
    ):
        raise ContractViolation("DESIGN_REVISION_DATA_REQUIREMENTS_MISMATCH")

    surface = config.get("surface_features")
    if not isinstance(surface, Mapping) or (
        surface.get("required_for_core") is not True
        or surface.get("freeze_before_model_results") is not True
        or surface.get("language_pairs") != _LANGUAGE_PAIRS
        or surface.get("normalization") != "NFC_AND_WHITESPACE_ONLY"
        or surface.get("character_metric")
        != "ONE_MINUS_LEVENSHTEIN_OVER_MAX_UNICODE_CODEPOINT_LENGTH"
        or surface.get("token_metric") != "EXPRESSION_ONLY_TOKEN_ID_SET_JACCARD"
        or surface.get("include_character_length_per_language") is not True
        or surface.get("include_token_length_per_language") is not True
        or surface.get("tokenizer") != "ACTUAL_KO_ONLY_FROZEN_TOKENIZER"
        or surface.get("historical_relation_claimed") is not False
        or set(surface.get("excluded_tokens", []))
        != {"PROMPT", "INSTRUCTION", "PADDING", "SPECIAL", "ANSWER_TERMINATOR_NEWLINE"}
    ):
        raise ContractViolation("DESIGN_REVISION_SURFACE_FEATURE_MISMATCH")

    leakage = config.get("leakage_control")
    if not isinstance(leakage, Mapping) or (
        leakage.get("component_field") != "leakage_component_id"
        or leakage.get("etymology_family_required") is not False
        or leakage.get("required_for_current_s_h_pilot") is not False
        or leakage.get("required_before_any_predictor_fit") is not True
        or leakage.get("incomplete_predictor_action") != "BLOCKED_PREDICTOR_SPLIT"
    ):
        raise ContractViolation("DESIGN_REVISION_LEAKAGE_POLICY_MISMATCH")

    prediction = config.get("prediction")
    if not isinstance(prediction, Mapping) or (
        prediction.get("enabled_for_current_pilot") is not False
        or prediction.get("status_when_disabled") != "NOT_RUN_PREDICTION_BY_DESIGN"
        or prediction.get("heldout_unit") != "leakage_component_id"
        or prediction.get("heldout_concepts_are_lm_unseen") is not False
        or prediction.get("requires_new_preregistration") is not True
    ):
        raise ContractViolation("DESIGN_REVISION_PREDICTION_POLICY_MISMATCH")

    exploratory = config.get("exploratory")
    relation = exploratory.get("documented_lexical_relation") if isinstance(exploratory, Mapping) else None
    if not isinstance(relation, Mapping) or (
        relation.get("enabled_for_current_pilot") is not False
        or relation.get("required_for_core_gates") is not False
        or relation.get("status_when_disabled")
        != "NOT_RUN_EXPLORATORY_LEXICAL_RELATION_BY_DESIGN"
        or relation.get("unresolved_as_distinct_forbidden") is not True
        or relation.get("requires_preoutcome_freeze_and_new_preregistration") is not True
    ):
        raise ContractViolation("DESIGN_REVISION_EXPLORATORY_POLICY_MISMATCH")


def audit_v4_1_design_amendment(
    amendment_path: Path | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify the exact v4.1 design bundle and report its non-execution state."""
    root = Path(project_root).resolve(strict=True)
    expected_declared_path = root / AMENDMENT_RELATIVE_PATH
    expected_amendment_path = require_relative_to(
        expected_declared_path, root, "DESIGN_REVISION_PATH_OUTSIDE_ROOT"
    )
    declared_candidate = (
        expected_declared_path
        if amendment_path is None
        else (
            Path(amendment_path)
            if Path(amendment_path).is_absolute()
            else root / Path(amendment_path)
        )
    )
    candidate = require_relative_to(
        declared_candidate, root, "DESIGN_REVISION_PATH_OUTSIDE_ROOT"
    )
    if (
        candidate != expected_amendment_path
        or declared_candidate.absolute() != expected_declared_path.absolute()
    ):
        raise ContractViolation("DESIGN_REVISION_AMENDMENT_PATH_MISMATCH")
    if candidate != declared_candidate.absolute() or declared_candidate.is_symlink():
        raise ContractViolation("DESIGN_REVISION_AMENDMENT_NOT_REGULAR_FILE")
    amendment_raw = read_regular_file_bytes(candidate)
    if len(amendment_raw) != EXPECTED_AMENDMENT_BYTES:
        raise ContractViolation("DESIGN_REVISION_AMENDMENT_SIZE_MISMATCH")
    if sha256_bytes(amendment_raw) != EXPECTED_AMENDMENT_SHA256:
        raise ContractViolation("DESIGN_REVISION_AMENDMENT_HASH_MISMATCH")
    amendment = _load_json_bytes(amendment_raw, "DESIGN_REVISION_INVALID_AMENDMENT_JSON")
    _exact_mapping(amendment, _TOP_LEVEL_FIELDS, "DESIGN_REVISION_AMENDMENT_SCHEMA_MISMATCH")
    if (
        amendment.get("schema_version") != SCHEMA_VERSION
        or amendment.get("amendment_version") != "4.1.0"
        or amendment.get("implementation_revision_target") != "4.1.0-r1"
        or amendment.get("project_id") != "lexical-freshstart-4lang-history-v4_1"
        or amendment.get("status") != AMENDMENT_STATUS
        or amendment.get("hash_algorithm") != "SHA-256"
    ):
        raise ContractViolation("DESIGN_REVISION_AMENDMENT_IDENTITY_MISMATCH")
    try:
        recorded = datetime.strptime(str(amendment.get("recorded_at_utc")), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ContractViolation("DESIGN_REVISION_INVALID_RECORDED_AT") from exc
    if recorded.year != 2026:
        raise ContractViolation("DESIGN_REVISION_INVALID_RECORDED_AT")

    approval = _exact_mapping(
        amendment.get("approval"),
        {
            "reviewer_id",
            "approval_scope",
            "identity_authentication_claimed",
            "electronic_signature_claimed",
            "data_qa_approval_claimed",
            "implementation_approval_claimed",
        },
        "DESIGN_REVISION_APPROVAL_SCHEMA_MISMATCH",
    )
    if (
        approval.get("reviewer_id") != "jm02"
        or approval.get("approval_scope") != "DESIGN_REVISION_ONLY"
        or any(
            approval.get(field) is not False
            for field in (
                "identity_authentication_claimed",
                "electronic_signature_claimed",
                "data_qa_approval_claimed",
                "implementation_approval_claimed",
            )
        )
    ):
        raise ContractViolation("DESIGN_REVISION_APPROVAL_CLAIM_MISMATCH")

    timing = amendment.get("timing")
    if not isinstance(timing, Mapping) or (
        timing.get("model_results_observed_before_change") is not False
        or timing.get("checkpoint_results_observed_before_change") is not False
        or timing.get("readiness_results_observed_before_change") is not False
        or timing.get("history_experiment_results_observed_before_change") is not False
        or timing.get("classification") != "PROSPECTIVE_PRE_MODEL_RESULTS_AMENDMENT"
        or timing.get("on_contradiction") != "STOP_AND_RECLASSIFY_PROTOCOL_CHANGE"
    ):
        raise ContractViolation("DESIGN_REVISION_TIMING_MISMATCH")

    bindings = amendment.get("artifact_bindings")
    if not isinstance(bindings, Mapping) or dict(bindings) != EXPECTED_BINDINGS:
        raise ContractViolation("DESIGN_REVISION_BINDINGS_MISMATCH")
    loaded: dict[str, bytes] = {}
    for name, expected in EXPECTED_BINDINGS.items():
        _path, raw = _read_bound_file(root, expected)
        loaded[name] = raw

    base_config = _load_json_bytes(
        loaded["base_config"], "DESIGN_REVISION_INVALID_BASE_CONFIG_JSON"
    )
    revised_config = _load_json_bytes(
        loaded["revised_config"], "DESIGN_REVISION_INVALID_REVISED_CONFIG_JSON"
    )
    _validate_config(revised_config, base_config)

    integrity = amendment.get("integrity_model")
    if not isinstance(integrity, Mapping) or (
        integrity.get("amendment_self_reference_present") is not False
        or integrity.get("base_delivery_files_modified") is not False
        or integrity.get("legacy_annotation_freeze_reinterpretation_allowed") is not False
        or integrity.get("legacy_experiment_freeze_reinterpretation_allowed") is not False
        or integrity.get("legacy_pass_status_inherited") is not False
        or integrity.get("new_v4_1_annotation_freeze_required") is not True
        or integrity.get("new_v4_1_experiment_freeze_required") is not True
        or integrity.get("upstream_source_compatibility") != "UPSTREAM_EXACT_HASH_ONLY"
    ):
        raise ContractViolation("DESIGN_REVISION_INTEGRITY_POLICY_MISMATCH")

    changes = amendment.get("scientific_changes")
    if not isinstance(changes, Mapping) or (
        changes.get("core_etymology_removed") is not True
        or changes.get("etymology_required_for_data_gate") is not False
        or changes.get("etymology_required_for_freeze_gate") is not False
        or changes.get("etymology_required_for_training_or_measurement") is not False
        or changes.get("documented_lexical_relation_role") != "OPTIONAL_EXPLORATORY_ONLY"
        or changes.get("surface_and_token_similarity_required") is not True
        or changes.get("surface_and_token_language_pairs") != _LANGUAGE_PAIRS
        or changes.get("prediction_current_status") != "NOT_RUN_PREDICTION_BY_DESIGN"
        or changes.get("etymology_family_removed_from_core_split") is not True
        or changes.get("replacement_split_unit") != "leakage_component_id"
    ):
        raise ContractViolation("DESIGN_REVISION_SCIENTIFIC_CHANGE_MISMATCH")

    data_state = amendment.get("current_primary_data_state")
    expected_state = {
        "etymology_deferrals_block_core": False,
        "etymology_deferral_count_ignored_by_core": 8,
        "term_quality_blocker_retained": True,
        "term_quality_blocker": {**_TERM_BLOCKER, "silent_replacement_allowed": False},
    }
    if data_state != expected_state:
        raise ContractViolation("DESIGN_REVISION_CURRENT_DATA_STATE_MISMATCH")
    not_claimed = amendment.get("not_claimed")
    if not isinstance(not_claimed, list) or set(not_claimed) != {
        "REVIEWER_IDENTITY_AUTHENTICATED",
        "HUMAN_REVIEW_COMPLETE",
        "DATA_QA_PASS",
        "IMPLEMENTATION_COMPLETE",
        "ANNOTATION_FREEZE_COMPLETE",
        "EXPERIMENT_FREEZE_COMPLETE",
        "GPU_REPLAY_PASS",
        "PILOT_COMPLETE",
        "MAIN_AUTHORIZED",
    }:
        raise ContractViolation("DESIGN_REVISION_NOT_CLAIMED_MISMATCH")

    return {
        "schema_version": "design-revision-audit-v1",
        "status": AUDIT_STATUS,
        "amendment_status": AMENDMENT_STATUS,
        "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "amendment_bytes": EXPECTED_AMENDMENT_BYTES,
        "project_id": revised_config["project_id"],
        "version": revised_config["version"],
        "core_etymology_required": False,
        "ignored_core_etymology_deferrals": 8,
        "current_core_term_blockers": [{**_TERM_BLOCKER}],
        "prediction_status": "NOT_RUN_PREDICTION_BY_DESIGN",
        "documented_lexical_relation_status": (
            "NOT_RUN_EXPLORATORY_LEXICAL_RELATION_BY_DESIGN"
        ),
        "implementation_ready": False,
        "training_eligible": False,
        "legacy_freeze_reuse_allowed": False,
        "next_required_gates": [
            "IMPLEMENT_V4_1_REVIEW_AND_FREEZE_SCHEMA",
            "RESOLVE_TERM_QUALITY_ORDINAL_38",
            "SOURCE_BOUND_HUMAN_REVIEW_EVIDENCE",
            "V4_1_ANNOTATION_AND_EXPERIMENT_FREEZES",
            "TOKENIZER_SURFACE_FEATURE_BOUNDARY_AUDIT",
            "CPU_INTEGRATION",
            "GPU2_REPLAY",
        ],
    }


__all__ = [
    "AMENDMENT_RELATIVE_PATH",
    "AMENDMENT_STATUS",
    "AUDIT_STATUS",
    "EXPECTED_AMENDMENT_BYTES",
    "EXPECTED_AMENDMENT_SHA256",
    "SCHEMA_VERSION",
    "audit_v4_1_design_amendment",
]
