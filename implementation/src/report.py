"""Strict readiness/measurement gates and JSON-only report projection."""
from __future__ import annotations

import html
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    canonical_json_bytes,
    publish_json_once,
    read_verified_json,
    read_verified_json_with_sha256,
    replace_derived_text,
    sha256_bytes,
    sha256_file,
)
from .contracts import (
    PROJECT_ROOT,
    ContractViolation,
    LANGUAGES,
    VERSION,
    load_pilot_config,
)


LINEAGE_FIELDS = (
    "root_id",
    "phase",
    "stage",
    "branch",
    "checkpoint_sha256",
    "model_fingerprint",
    "freeze_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "code_sha256",
    "evaluation_plan_sha256",
    "evaluation_record_hash_schema",
    "data_kind",
)


def _expected_ids(plan: Mapping[str, Any]) -> set[str]:
    raw = plan.get("expected_record_ids")
    if not isinstance(raw, list) or not raw:
        raise ContractViolation("EXPECTED_RECORD_IDS_REQUIRED")
    values = [str(value) for value in raw]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ContractViolation("INVALID_EXPECTED_RECORD_IDS")
    return set(values)


def _validate_exact_record_set(
    rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    expected = _expected_ids(plan)
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in by_id:
            raise ContractViolation("DUPLICATE_OR_EMPTY_RESULT_RECORD_ID")
        if record_id not in expected:
            raise ContractViolation("UNEXPECTED_RESULT_RECORD_ID")
        by_id[record_id] = row
    if set(by_id) != expected:
        raise ContractViolation("MISSING_RESULT_RECORD_IDS")
    if plan.get("data_kind") == "REAL":
        hashes = plan.get("expected_record_content_sha256_by_id")
        if not isinstance(hashes, Mapping) or set(str(key) for key in hashes) != expected:
            raise ContractViolation("EXPECTED_RECORD_CONTENT_HASHES_REQUIRED")
        for record_id, row in by_id.items():
            expected_hash = hashes.get(record_id)
            _require_hash(
                expected_hash, "INVALID_EXPECTED_RECORD_CONTENT_HASH"
            )
            if row.get("record_content_sha256") != expected_hash:
                raise ContractViolation("RESULT_RECORD_CONTENT_HASH_MISMATCH")
    return by_id


def _validate_lineage(row: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    for field in LINEAGE_FIELDS:
        if field not in plan:
            raise ContractViolation("PLAN_MISSING_" + field.upper())
        if str(row.get(field)) != str(plan[field]):
            raise ContractViolation("MIXED_" + field.upper())


def _require_fixed_cell(
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    fields: Sequence[str],
) -> None:
    _validate_lineage(row, plan)
    for field in fields:
        if field not in plan:
            raise ContractViolation("PLAN_MISSING_" + field.upper())
        if row.get(field) != plan[field]:
            raise ContractViolation("MIXED_" + field.upper())


def _threshold_fraction(value: Any, code: str) -> Fraction:
    try:
        threshold = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ContractViolation(code) from exc
    if threshold < 0 or threshold > 1:
        raise ContractViolation(code)
    return threshold


def compute_readiness(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    threshold: float | str | None = None,
) -> dict[str, Any]:
    """Gate every frozen wrapper/requested-language cell independently."""
    if plan.get("gate_each_wrapper_and_requested_language") is not True:
        raise ContractViolation("READINESS_MUST_GATE_EACH_WRAPPER_LANGUAGE")
    wrappers = tuple(plan.get("wrappers", ()))
    if not wrappers or len(wrappers) != len(set(wrappers)):
        raise ContractViolation("INVALID_READINESS_WRAPPERS")
    active = tuple(plan.get("active_languages", LANGUAGES))
    from .records import validated_language_order
    order = validated_language_order(plan.get("language_order", LANGUAGES))
    if not active or active != order[: len(active)]:
        raise ContractViolation("INVALID_ACTIVE_LANGUAGE_PREFIX")
    threshold_value = (
        plan.get("minimum_unrounded") if threshold is None else threshold
    )
    threshold_ratio = _threshold_fraction(
        threshold_value, "INVALID_READINESS_THRESHOLD"
    )
    by_id = _validate_exact_record_set(rows, plan)
    counts: dict[tuple[str, str], Counter[str]] = {
        (wrapper, language): Counter()
        for wrapper in wrappers
        for language in active
    }
    for record_id in sorted(by_id):
        row = by_id[record_id]
        _require_fixed_cell(
            row,
            plan,
            fields=("stage", "input_language", "format", "split"),
        )
        if row.get("mode") != "REQUESTED":
            raise ContractViolation("READINESS_REQUIRES_REQUESTED_MODE")
        wrapper = row.get("wrapper")
        language = row.get("requested_language")
        if wrapper not in wrappers or language not in active:
            raise ContractViolation("UNEXPECTED_READINESS_CELL")
        generation = row.get("generation")
        if not isinstance(generation, Mapping):
            raise ContractViolation("MISSING_GENERATION_RESULT")
        compatible = generation.get("request_compatible")
        if not isinstance(compatible, bool):
            raise ContractViolation("INVALID_REQUEST_COMPATIBILITY")
        counts[(wrapper, language)]["denominator"] += 1
        counts[(wrapper, language)]["compatible"] += int(compatible)
        label = generation.get("class")
        allowed_labels = {
            "REGISTERED_COMPATIBLE",
            "REGISTERED_OTHER_LANGUAGE",
            "UNREGISTERED",
            "UNREGISTERED_UNTERMINATED",
            "EMPTY",
        }
        if label not in allowed_labels:
            raise ContractViolation("INVALID_REQUESTED_GENERATION_CLASS")
        if compatible != (label == "REGISTERED_COMPATIBLE"):
            raise ContractViolation("INCONSISTENT_REQUEST_COMPATIBILITY")
        counts[(wrapper, language)]["class:" + label] += 1

    cells: dict[str, Any] = {}
    failures: list[str] = []
    for wrapper in wrappers:
        for language in active:
            count = counts[(wrapper, language)]
            denominator = count["denominator"]
            numerator = count["compatible"]
            if denominator == 0:
                raise ContractViolation("EMPTY_READINESS_CELL")
            passed = (
                numerator * threshold_ratio.denominator
                >= denominator * threshold_ratio.numerator
            )
            key = wrapper + ":" + language
            cells[key] = {
                "wrapper": wrapper,
                "requested_language": language,
                "compatible_numerator": numerator,
                "denominator": denominator,
                "A": numerator / denominator,
                "minimum_unrounded": float(threshold_ratio),
                "passed": passed,
                "class_counts": {
                    label.removeprefix("class:"): value
                    for label, value in sorted(count.items())
                    if label.startswith("class:")
                },
            }
            if not passed:
                failures.append(key + ":BELOW_READINESS_MINIMUM")
    return {
        "schema_version": "readiness.v1",
        "status": "PASS" if not failures else "BLOCKED_READINESS",
        "scope": "exact root/checkpoint, KO input, RD dev; each wrapper and requested language",
        "active_languages": list(active),
        "wrappers": list(wrappers),
        "minimum_unrounded": float(threshold_ratio),
        "cells": cells,
        "failed_cells": failures,
        "overall_average_used_for_gate": False,
    }


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2
        for offset in range(start, stop):
            ranks[order[offset]] = average
        start = stop
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        raise ContractViolation("INVALID_CORRELATION_LENGTH")
    if any(not math.isfinite(value) for value in [*x, *y]):
        raise ContractViolation("NONFINITE_CORRELATION_INPUT")
    rank_x, rank_y = _ranks(x), _ranks(y)
    mean_x, mean_y = statistics.mean(rank_x), statistics.mean(rank_y)
    covariance = math.fsum(
        (left - mean_x) * (right - mean_y)
        for left, right in zip(rank_x, rank_y)
    )
    denominator = math.sqrt(
        math.fsum((value - mean_x) ** 2 for value in rank_x)
        * math.fsum((value - mean_y) ** 2 for value in rank_y)
    )
    return covariance / denominator if denominator else None


def _tie_summary(values: Sequence[float]) -> dict[str, Any]:
    counts = Counter(values)
    groups = sorted((count for count in counts.values() if count > 1), reverse=True)
    return {
        "unique_values": len(counts),
        "n_tied_observations": sum(groups),
        "tie_group_sizes": groups,
        "nan_count": 0,
    }


def _measurement_thresholds(
    plan: Mapping[str, Any], thresholds: Mapping[str, Any] | None
) -> dict[str, float]:
    source = dict(plan)
    source.update(thresholds or {})

    def pick(*names: str) -> float:
        for name in names:
            if name in source:
                value = float(source[name])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ContractViolation("INVALID_MEASUREMENT_THRESHOLD")
                return value
        raise ContractViolation("MISSING_MEASUREMENT_THRESHOLD")

    return {
        "rho_min": pick("rho_min_unrounded", "rho_min"),
        "wrapper_abs_max": pick(
            "mean_abs_wrapper_difference_max_unrounded", "wrapper_abs_max"
        ),
        "z_threshold": pick(
            "low_z_threshold_strict_less_than", "z_threshold"
        ),
        "low_z_fraction_max": pick(
            "low_z_fraction_block_at_or_above", "low_z_fraction_max"
        ),
    }


def _probability_values(row: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    probability = row.get("probability")
    if not isinstance(probability, Mapping):
        raise ContractViolation("MISSING_PROBABILITY_RESULT")
    if probability.get("identifiable") is not True:
        raise ContractViolation("NONIDENTIFIABLE_ROW_IN_PRIMARY_COHORT")
    if probability.get("status") != "DEFINED":
        raise ContractViolation("UNDEFINED_Q_IN_PRIMARY_COHORT")
    if probability.get("logZ_status") != "FINITE":
        raise ContractViolation("NONFINITE_LOGZ_IN_PRIMARY_COHORT")
    try:
        log_z = float(probability["logZ"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractViolation("INVALID_LOGZ") from exc
    if not math.isfinite(log_z) or log_z > 1e-10:
        raise ContractViolation("INVALID_LOGZ")
    q = probability.get("Q_language")
    if not isinstance(q, Mapping) or set(q) != set(LANGUAGES):
        raise ContractViolation("INVALID_Q_LANGUAGE_VECTOR")
    normalized_q = {language: float(q[language]) for language in LANGUAGES}
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in normalized_q.values()
    ):
        raise ContractViolation("INVALID_Q_LANGUAGE_VECTOR")
    if not math.isclose(math.fsum(normalized_q.values()), 1.0, abs_tol=1e-8):
        raise ContractViolation("Q_LANGUAGE_DOES_NOT_SUM_TO_ONE")
    try:
        z_value = float(probability["Z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractViolation("INVALID_Z") from exc
    if not math.isfinite(z_value) or not 0 <= z_value <= 1 + 1e-10:
        raise ContractViolation("INVALID_Z")
    reconstructed_z = math.exp(log_z)
    if reconstructed_z != 0.0 and not math.isclose(
        reconstructed_z, z_value, rel_tol=1e-10, abs_tol=0.0
    ):
        raise ContractViolation("LOGZ_Z_MISMATCH")
    if reconstructed_z == 0.0 and z_value != 0.0:
        raise ContractViolation("LOGZ_Z_MISMATCH")
    return normalized_q, z_value


def compute_measurement(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the sole T3/KO/RD/ANY/dev1-vs-dev2 gate."""
    wrappers = tuple(plan.get("wrappers", ()))
    if len(wrappers) != 2 or len(set(wrappers)) != 2:
        raise ContractViolation("MEASUREMENT_REQUIRES_TWO_WRAPPERS")
    concepts_raw = plan.get("identifiable_concept_ids")
    if not isinstance(concepts_raw, list) or len(concepts_raw) < 2:
        raise ContractViolation("INVALID_IDENTIFIABLE_COHORT")
    concepts = [str(value) for value in concepts_raw]
    if len(concepts) != len(set(concepts)) or any(not value for value in concepts):
        raise ContractViolation("INVALID_IDENTIFIABLE_COHORT")
    threshold_values = _measurement_thresholds(plan, thresholds)
    expected_pairs = {(wrapper, concept) for wrapper in wrappers for concept in concepts}
    by_id = _validate_exact_record_set(rows, plan)
    maps: dict[str, dict[str, tuple[dict[str, float], float]]] = {
        wrapper: {} for wrapper in wrappers
    }
    seen_pairs: set[tuple[str, str]] = set()
    for record_id in sorted(by_id):
        row = by_id[record_id]
        _require_fixed_cell(
            row,
            plan,
            fields=("stage", "input_language", "format", "mode", "split"),
        )
        if row.get("mode") != "ANY" or row.get("requested_language") is not None:
            raise ContractViolation("MEASUREMENT_REQUIRES_ANY_WITHOUT_REQUESTED_LANGUAGE")
        wrapper = row.get("wrapper")
        concept = str(row.get("concept_id", ""))
        pair = (wrapper, concept)
        if pair not in expected_pairs:
            raise ContractViolation("UNEXPECTED_MEASUREMENT_ROW")
        if pair in seen_pairs:
            raise ContractViolation("DUPLICATE_MEASUREMENT_ROW")
        seen_pairs.add(pair)
        maps[wrapper][concept] = _probability_values(row)
    if seen_pairs != expected_pairs:
        raise ContractViolation("MISSING_MEASUREMENT_ROWS")

    ids = sorted(concepts)
    reasons: list[str] = []
    metrics_by_language: dict[str, Any] = {}
    for language in LANGUAGES:
        first = [maps[wrappers[0]][concept][0][language] for concept in ids]
        second = [maps[wrappers[1]][concept][0][language] for concept in ids]
        rho = spearman(first, second)
        mean_absolute = math.fsum(
            abs(left - right) for left, right in zip(first, second)
        ) / len(ids)
        metrics_by_language[language] = {
            "spearman": rho,
            "mean_abs_wrapper_difference": mean_absolute,
            wrappers[0]: {
                "sample_sd_ddof1": statistics.stdev(first),
                "range": [min(first), max(first)],
                **_tie_summary(first),
            },
            wrappers[1]: {
                "sample_sd_ddof1": statistics.stdev(second),
                "range": [min(second), max(second)],
                **_tie_summary(second),
            },
        }
        if rho is None:
            reasons.append(language + ":CORRELATION_UNDEFINED")
        elif rho < threshold_values["rho_min"]:
            reasons.append(language + ":RHO_BELOW_MINIMUM")
        if mean_absolute > threshold_values["wrapper_abs_max"]:
            reasons.append(language + ":WRAPPER_DIFFERENCE_ABOVE_MAXIMUM")

    z_by_wrapper: dict[str, Any] = {}
    fraction_threshold = _threshold_fraction(
        threshold_values["low_z_fraction_max"], "INVALID_LOW_Z_FRACTION_GATE"
    )
    for wrapper in wrappers:
        z_values = [maps[wrapper][concept][1] for concept in ids]
        low_count = sum(
            value < threshold_values["z_threshold"] for value in z_values
        )
        denominator = len(z_values)
        blocked = (
            low_count * fraction_threshold.denominator
            >= denominator * fraction_threshold.numerator
        )
        z_by_wrapper[wrapper] = {
            "min": min(z_values),
            "median": statistics.median(z_values),
            "max": max(z_values),
            "low_Z_strict_numerator": low_count,
            "denominator": denominator,
            "low_Z_fraction": low_count / denominator,
            "blocked_at_or_above": float(fraction_threshold),
        }
        if blocked:
            reasons.append(wrapper + ":LOW_Z_FRACTION_AT_OR_ABOVE_MAXIMUM")
    return {
        "schema_version": "measurement.v1",
        "status": "PASS" if not reasons else "BLOCKED_MEASUREMENT",
        "scope": "one exact root/checkpoint, T3, KO input, RD, ANY, dev1-vs-dev2, frozen identifiable cohort",
        "n_concepts": len(ids),
        "concept_ids_sha256": sha256_bytes(canonical_json_bytes(ids)),
        "metrics_by_language": metrics_by_language,
        "Z_by_wrapper": z_by_wrapper,
        "thresholds": threshold_values,
        "reasons": reasons,
        "rounded_values_used_for_gate": False,
    }


def _status(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("status")
    return str(value)


def authorize_phase(
    target: str,
    base_gates: Mapping[str, Any],
    root_gates: Mapping[str | int, Mapping[str, Any]],
    *,
    planned_roots: Sequence[int] = (4101, 4102, 4103, 4104),
) -> dict[str, Any]:
    """Apply the v4 expansion ordering without collapsing individual gates."""
    if target == "MAIN":
        return {
            "status": "MAIN_NOT_AUTHORIZED",
            "authorized": False,
            "reasons": ["MAIN_ENABLED_FALSE"],
        }
    required_base = (
        "delivery_integrity",
        "data_qa",
        "freeze",
        "tokenizer_boundary",
        "cpu_integration",
        "gpu_replay",
        "budget",
    )
    failed = [
        name
        for name in required_base
        if name not in base_gates or _status(base_gates[name]) != "PASS"
    ]
    failed.extend(
        name
        for name, result in sorted(base_gates.items())
        if name not in required_base and _status(result) != "PASS"
    )
    required_roots: Sequence[int] = ()
    if target == "FIRST_ROOT_S":
        required_roots = ()
    elif target == "REMAINING_ROOTS_S":
        required_roots = planned_roots[:1]
    elif target == "H":
        required_roots = planned_roots
    else:
        raise ContractViolation("UNKNOWN_PHASE_AUTHORIZATION_TARGET")
    for root in required_roots:
        result = root_gates.get(root, root_gates.get(str(root), {}))
        if _status(result.get("readiness")) != "PASS":
            failed.append(f"root_{root}:READINESS")
        if _status(result.get("measurement")) != "PASS":
            failed.append(f"root_{root}:MEASUREMENT")
    return {
        "status": "PASS" if not failed else "BLOCKED_PREREQUISITES",
        "authorized": not failed,
        "target": target,
        "reasons": failed,
        "planned_roots": list(planned_roots),
    }


REQUIRED_SUMMARY_FIELDS = (
    "schema_version",
    "project_id",
    "spec_version",
    "run_id",
    "created_at_utc",
    "status",
    "data_kind",
    "policy",
    "provenance",
    "statuses",
    "gates",
    "budget",
    "roots",
    "artifacts",
    "next_action",
)


RUN_SUMMARY_SCHEMA = "run-summary.v1"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SAFE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
KNOWN_GATE_NAMES = (
    "delivery_integrity",
    "source_collection",
    "source_merge",
    "corpus",
    "tokenizer",
    "data_qa",
    "annotation_freeze",
    "experiment_freeze",
    "tokenizer_boundary",
    "cpu_integration",
    "gpu_replay",
    "budget",
    "pilot_training",
    "readiness",
    "measurement",
    "main_study",
)
KNOWN_STATUS_VALUES = {
    "PASS",
    "NOT_RUN",
    "NOT_IMPLEMENTED",
    "BLOCKED",
    "BLOCKED_DELIVERY_INTEGRITY",
    "BLOCKED_CREDENTIALS",
    "BLOCKED_SOURCE_COLLECTION",
    "BLOCKED_SOURCE_MERGE",
    "BLOCKED_CORPUS",
    "BLOCKED_TOKENIZER",
    "BLOCKED_DATA_QA",
    "BLOCKED_DATA_COVERAGE",
    "BLOCKED_FREEZE",
    "BLOCKED_TOKEN_BOUNDARY",
    "BLOCKED_CPU_CHECKS",
    "BLOCKED_GPU_REPLAY",
    "BLOCKED_BUDGET",
    "BLOCKED_PREREQUISITES",
    "BLOCKED_READINESS",
    "BLOCKED_MEASUREMENT",
    "BLOCKED_PILOT",
}
COMPLETION_REQUIRED_GATES = (
    "delivery_integrity",
    "source_collection",
    "source_merge",
    "corpus",
    "tokenizer",
    "data_qa",
    "annotation_freeze",
    "experiment_freeze",
    "tokenizer_boundary",
    "cpu_integration",
    "gpu_replay",
    "budget",
    "pilot_training",
    "readiness",
    "measurement",
)
COMPLETION_REQUIRED_ARTIFACTS = (
    "implementation_manifest",
    "cpu_checks",
    "experiment_freeze",
    "gpu_replay",
    "evaluation_results",
)


def _require_hash(value: Any, code: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ContractViolation(code)
    return value


def _validate_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z",
        value,
    ):
        raise ContractViolation("INVALID_RUN_CREATED_AT_UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractViolation("INVALID_RUN_CREATED_AT_UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractViolation("INVALID_RUN_CREATED_AT_UTC")


def _artifact_path(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ContractViolation("INVALID_RUN_ARTIFACT_PATH")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(PROJECT_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ContractViolation("RUN_ARTIFACT_OUTSIDE_PROJECT") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise ContractViolation("RUN_ARTIFACT_NOT_REGULAR_FILE")
    return resolved


def _validate_artifact_ref(ref: Mapping[str, Any], *, require_hash: bool) -> None:
    if not isinstance(ref, Mapping) or "path" not in ref:
        raise ContractViolation("INVALID_RUN_ARTIFACT_REF")
    if "sha256" not in ref:
        if require_hash:
            raise ContractViolation("RUN_ARTIFACT_HASH_REQUIRED")
        return
    expected = _require_hash(ref.get("sha256"), "INVALID_RUN_ARTIFACT_HASH")
    path = _artifact_path(ref.get("path"))
    if sha256_file(path) != expected:
        raise ContractViolation("RUN_ARTIFACT_HASH_MISMATCH")
    if "bytes" in ref:
        size = ref["bytes"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or path.stat().st_size != size
        ):
            raise ContractViolation("RUN_ARTIFACT_SIZE_MISMATCH")


def _validate_embedded_artifact_refs(value: Any, *, verify_artifacts: bool) -> None:
    if isinstance(value, Mapping):
        if "path" in value:
            if "sha256" not in value:
                raise ContractViolation("RUN_EVIDENCE_ARTIFACT_HASH_REQUIRED")
            if verify_artifacts:
                _validate_artifact_ref(value, require_hash=True)
        for nested in value.values():
            _validate_embedded_artifact_refs(
                nested, verify_artifacts=verify_artifacts
            )
    elif isinstance(value, list):
        for nested in value:
            _validate_embedded_artifact_refs(
                nested, verify_artifacts=verify_artifacts
            )


def _validate_declared_hashes(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and (key == "sha256" or key.endswith("_sha256")):
                _require_hash(nested, "INVALID_DECLARED_RUN_HASH")
            _validate_declared_hashes(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_declared_hashes(nested)


def _validate_artifacts(
    summary: Mapping[str, Any], *, verify_artifacts: bool
) -> Mapping[str, Any]:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractViolation("INVALID_RUN_ARTIFACTS")
    completion = summary.get("status") == "PILOT_COMPLETE"
    for name, ref in artifacts.items():
        if not isinstance(name, str) or not SAFE_NAME_RE.fullmatch(name):
            raise ContractViolation("INVALID_RUN_ARTIFACT_NAME")
        if not isinstance(ref, Mapping) or "path" not in ref:
            raise ContractViolation("INVALID_RUN_ARTIFACT_REF")
        if (completion or verify_artifacts) and "sha256" not in ref:
            raise ContractViolation("RUN_ARTIFACT_HASH_REQUIRED")
        if verify_artifacts and "sha256" in ref:
            _validate_artifact_ref(ref, require_hash=completion)
    return artifacts


def _required_artifact_path(
    artifacts: Mapping[str, Any], name: str, *, gate: str
) -> Path:
    ref = artifacts.get(name)
    if not isinstance(ref, Mapping):
        raise ContractViolation("RUN_PASS_GATE_EVIDENCE_REQUIRED:" + gate)
    return _artifact_path(ref.get("path"))


def _checkpoint_directory(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(PROJECT_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID") from exc
    if candidate.is_symlink() or not resolved.is_dir():
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    return resolved


def _verify_gpu_replay_evidence(
    replay_path: Path,
    *,
    freeze_sha256: str,
    code_sha256: str,
    cpu_check_sha256: str,
) -> dict[str, Any]:
    replay = read_verified_json(replay_path)
    comparison = replay.get("comparison")
    binding = replay.get("gpu_binding")
    campaign_policy = read_verified_json(
        PROJECT_ROOT / "implementation/config/campaign_policy.json"
    )
    gpu_policy = campaign_policy.get("gpu")
    budget_policy = campaign_policy.get("budget")
    expected_binding_fields = {
        "physical_index",
        "logical_index",
        "uuid",
        "pci_bus_id",
        "name",
        "total_memory_mib",
        "torch_cuda_version",
        "torch_version",
    }
    gpu_seconds = replay.get("gpu_seconds_charged")
    if not isinstance(gpu_policy, Mapping) or not isinstance(budget_policy, Mapping):
        raise ContractViolation("RUN_GPU_REPLAY_EVIDENCE_NOT_PASS")
    if (
        replay.get("schema_version") != "gpu2-replay-v1"
        or replay.get("status") != "PASS"
        or replay.get("scientific_result") is not False
        or replay.get("freeze_sha256") != freeze_sha256
        or replay.get("code_sha256") != code_sha256
        or replay.get("cpu_check_sha256") != cpu_check_sha256
        or not isinstance(comparison, Mapping)
        or comparison.get("continuous_vs_resume_1") is not True
        or comparison.get("resume_1_vs_resume_2") is not True
        or comparison.get("steps_compared") != 10
        or replay.get("optimizer_steps_charged") != 32
        or not isinstance(binding, Mapping)
        or set(binding) != expected_binding_fields
        or binding.get("physical_index") != 2
        or binding.get("logical_index") != 0
        or binding.get("uuid") != gpu_policy.get("expected_uuid")
        or not isinstance(binding.get("pci_bus_id"), str)
        or not binding["pci_bus_id"]
        or not isinstance(binding.get("name"), str)
        or not binding["name"]
        or not isinstance(binding.get("total_memory_mib"), int)
        or isinstance(binding.get("total_memory_mib"), bool)
        or binding["total_memory_mib"] <= 0
        or not isinstance(binding.get("torch_cuda_version"), str)
        or not binding["torch_cuda_version"]
        or not isinstance(binding.get("torch_version"), str)
        or not binding["torch_version"]
        or not isinstance(gpu_seconds, (int, float))
        or isinstance(gpu_seconds, bool)
        or not math.isfinite(float(gpu_seconds))
        or float(gpu_seconds) <= 0
    ):
        raise ContractViolation("RUN_GPU_REPLAY_EVIDENCE_NOT_PASS")

    traces = comparison.get("traces")
    if not isinstance(traces, list) or len(traces) != 10 or any(
        not isinstance(row, Mapping) for row in traces
    ):
        raise ContractViolation("RUN_GPU_REPLAY_TRACE_INVALID")
    trace_fields = {
        "global_step",
        "phase_step",
        "record_ids",
        "content_sha256s",
        "loss_hex",
        "learning_rates",
    }
    previous_global: int | None = None
    previous_phase: int | None = None
    for row in traces:
        record_ids = row.get("record_ids")
        content_hashes = row.get("content_sha256s")
        learning_rates = row.get("learning_rates")
        global_step = row.get("global_step")
        phase_step = row.get("phase_step")
        try:
            loss = float.fromhex(str(row.get("loss_hex", "")))
        except ValueError as exc:
            raise ContractViolation("RUN_GPU_REPLAY_TRACE_INVALID") from exc
        if (
            set(row) != trace_fields
            or not isinstance(global_step, int)
            or isinstance(global_step, bool)
            or global_step < 1
            or not isinstance(phase_step, int)
            or isinstance(phase_step, bool)
            or phase_step < 1
            or (previous_global is not None and global_step != previous_global + 1)
            or (previous_phase is not None and phase_step != previous_phase + 1)
            or not isinstance(record_ids, list)
            or len(record_ids) != 32
            or any(not isinstance(value, str) or not value for value in record_ids)
            or not isinstance(content_hashes, list)
            or len(content_hashes) != len(record_ids)
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in content_hashes
            )
            or not isinstance(learning_rates, list)
            or not learning_rates
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
                for value in learning_rates
            )
            or not math.isfinite(loss)
        ):
            raise ContractViolation("RUN_GPU_REPLAY_TRACE_INVALID")
        previous_global = global_step
        previous_phase = phase_step
    expected_loss_hash = sha256_bytes(
        canonical_json_bytes([row.get("loss_hex") for row in traces])
    )
    expected_record_hash = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "record_ids": row.get("record_ids"),
                    "content_sha256s": row.get("content_sha256s"),
                }
                for row in traces
            ]
        )
    )
    final_fingerprints = comparison.get("final_state_fingerprints")
    if (
        comparison.get("loss_trace_sha256") != expected_loss_hash
        or comparison.get("record_trace_sha256") != expected_record_hash
        or not isinstance(final_fingerprints, Mapping)
        or set(final_fingerprints)
        != {"model", "optimizer", "scheduler", "progress", "rng"}
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in final_fingerprints.values()
        )
        or comparison.get("resume_traces") != [traces, traces]
        or comparison.get("resume_final_state_fingerprints")
        != [final_fingerprints, final_fingerprints]
    ):
        raise ContractViolation("RUN_GPU_REPLAY_TRACE_INVALID")

    budget = replay.get("budget")
    root_usage = budget.get("per_root_gpu_seconds_charged_or_reserved") if isinstance(budget, Mapping) else None
    if (
        not isinstance(budget, Mapping)
        or budget.get("campaign_cap_gpu_hours")
        != budget_policy.get("campaign_gpu_hours_cap")
        or budget.get("prior_gpu_hours_user_reported")
        != budget_policy.get("prior_gpu_hours_user_reported")
        or not isinstance(budget.get("new_gpu_seconds_charged_or_reserved"), (int, float))
        or isinstance(budget.get("new_gpu_seconds_charged_or_reserved"), bool)
        or not math.isfinite(float(budget["new_gpu_seconds_charged_or_reserved"]))
        or budget["new_gpu_seconds_charged_or_reserved"] < gpu_seconds
        or not isinstance(budget.get("campaign_gpu_hours_charged_or_reserved"), (int, float))
        or not isinstance(budget.get("campaign_gpu_hours_remaining"), (int, float))
        or not isinstance(root_usage, Mapping)
        or not isinstance(budget.get("ledger_events"), int)
        or isinstance(budget.get("ledger_events"), bool)
        or budget["ledger_events"] < 2
    ):
        raise ContractViolation("RUN_GPU_REPLAY_BUDGET_INVALID")
    _require_hash(
        budget.get("ledger_tip_sha256"), "RUN_GPU_REPLAY_BUDGET_INVALID"
    )
    used_hours = float(budget["campaign_gpu_hours_charged_or_reserved"])
    remaining_hours = float(budget["campaign_gpu_hours_remaining"])
    cap_hours = float(budget_policy["campaign_gpu_hours_cap"])
    prior_hours = float(budget_policy["prior_gpu_hours_user_reported"])
    new_seconds = float(budget["new_gpu_seconds_charged_or_reserved"])
    if (
        not math.isfinite(used_hours)
        or not math.isfinite(remaining_hours)
        or used_hours < prior_hours
        or used_hours > cap_hours
        or not math.isclose(
            used_hours, prior_hours + new_seconds / 3600.0, abs_tol=1e-12
        )
        or not math.isclose(
            remaining_hours, max(0.0, cap_hours - used_hours), abs_tol=1e-12
        )
    ):
        raise ContractViolation("RUN_GPU_REPLAY_BUDGET_INVALID")
    disk = replay.get("disk_reservation")
    if (
        not isinstance(disk, Mapping)
        or set(disk)
        != {
            "available_bytes",
            "planned_bytes",
            "emergency_free_bytes",
            "projected_free_bytes",
        }
        or any(
            not isinstance(disk.get(field), int)
            or isinstance(disk.get(field), bool)
            or disk[field] < 0
            for field in disk
        )
        or disk.get("emergency_free_bytes")
        != budget_policy.get("minimum_emergency_free_bytes")
        or disk["projected_free_bytes"]
        != disk["available_bytes"] - disk["planned_bytes"]
        or disk["projected_free_bytes"] < disk["emergency_free_bytes"]
    ):
        raise ContractViolation("RUN_GPU_REPLAY_DISK_RESERVATION_INVALID")

    checkpoint = replay.get("checkpoint")
    required_checkpoint_ref = {
        "path",
        "state_sha256",
        "manifest_sha256",
        "model_fingerprint",
        "optimizer_fingerprint",
        "bytes",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != required_checkpoint_ref:
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    directory = _checkpoint_directory(checkpoint.get("path"))
    commit_path = directory / "COMMITTED"
    manifest_path = directory / "manifest.json"
    state_path = directory / "state.pt"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (commit_path, manifest_path, state_path)
    ):
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    try:
        committed = commit_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID") from exc
    _require_hash(committed, "RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    if checkpoint.get("manifest_sha256") != committed:
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    manifest = read_verified_json(manifest_path, expected_sha256=committed)
    required_manifest = {
        "schema_version",
        "state_file",
        "state_sha256",
        "state_bytes",
        "model_fingerprint",
        "optimizer_fingerprint",
        "optimizer_class",
        "scheduler_fingerprint",
        "scaler_fingerprint",
        "rng_fingerprint",
        "progress_fingerprint",
        "lineage",
        "progress",
    }
    if set(manifest) != required_manifest:
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    for field in (
        "state_sha256",
        "model_fingerprint",
        "optimizer_fingerprint",
        "scheduler_fingerprint",
        "scaler_fingerprint",
        "rng_fingerprint",
        "progress_fingerprint",
    ):
        _require_hash(manifest.get(field), "RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    _require_hash(
        checkpoint.get("optimizer_fingerprint"),
        "RUN_GPU_REPLAY_CHECKPOINT_INVALID",
    )
    state_sha256 = sha256_file(state_path)
    if (
        manifest.get("schema_version") != "v4-full-state-1"
        or manifest.get("state_file") != "state.pt"
        or manifest.get("state_sha256") != state_sha256
        or checkpoint.get("state_sha256") != state_sha256
        or manifest.get("state_bytes") != state_path.stat().st_size
        or checkpoint.get("bytes") != state_path.stat().st_size
        or manifest.get("model_fingerprint") != checkpoint.get("model_fingerprint")
        or manifest.get("optimizer_fingerprint")
        != checkpoint.get("optimizer_fingerprint")
        or not isinstance(manifest.get("optimizer_class"), str)
        or not manifest["optimizer_class"]
    ):
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    progress = manifest.get("progress")
    required_progress = {
        "global_step",
        "phase_step",
        "loader_state",
        "accumulation_step",
        "examples_seen",
        "model_tokens_seen",
        "plan_sha256",
    }
    if (
        not isinstance(progress, Mapping)
        or not required_progress.issubset(progress)
        or progress.get("global_step") != 2
        or progress.get("phase_step") != 2
        or progress.get("examples_seen") != 64
        or progress.get("accumulation_step") != 0
        or any(
            not isinstance(progress.get(field), int)
            or isinstance(progress.get(field), bool)
            or progress[field] < 0
            for field in (
                "global_step",
                "phase_step",
                "examples_seen",
                "model_tokens_seen",
            )
        )
        or traces[0].get("global_step") != progress.get("global_step", -1) + 1
        or traces[0].get("phase_step") != progress.get("phase_step", -1) + 1
    ):
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    _require_hash(progress.get("plan_sha256"), "RUN_GPU_REPLAY_CHECKPOINT_INVALID")
    lineage = manifest.get("lineage")
    required_lineage = {
        "root_id",
        "phase",
        "stage",
        "branch",
        "phase_parent_sha256",
        "resume_parent_sha256",
        "initial_model_fingerprint",
        "freeze_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "code_sha256",
        "cpu_check_sha256",
        "evaluation_plan_sha256",
        "data_kind",
        "freeze_gate_status",
        "gpu_binding",
        "budget_lease_id",
        "runtime",
        "scientific_result",
        "model_fingerprint",
    }
    if (
        not isinstance(lineage, Mapping)
        or set(lineage) != required_lineage
        or lineage.get("root_id") != 4101
        or lineage.get("phase") != "GPU_REPLAY"
        or lineage.get("stage") != "GPU_REPLAY"
        or lineage.get("branch") is not None
        or lineage.get("data_kind") != "REAL"
        or lineage.get("freeze_gate_status") != "PASS"
        or lineage.get("scientific_result") is not False
        or lineage.get("freeze_sha256") != freeze_sha256
        or lineage.get("code_sha256") != code_sha256
        or lineage.get("cpu_check_sha256") != cpu_check_sha256
        or lineage.get("gpu_binding") != binding
        or lineage.get("model_fingerprint") != manifest.get("model_fingerprint")
    ):
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_LINEAGE_MISMATCH")
    for field in (
        "initial_model_fingerprint",
        "freeze_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "code_sha256",
        "cpu_check_sha256",
        "evaluation_plan_sha256",
        "model_fingerprint",
    ):
        _require_hash(
            lineage.get(field), "RUN_GPU_REPLAY_CHECKPOINT_LINEAGE_MISMATCH"
        )
    runtime = lineage.get("runtime")
    if (
        not isinstance(lineage.get("budget_lease_id"), str)
        or not lineage["budget_lease_id"]
        or not isinstance(runtime, Mapping)
        or not all(
            runtime.get(field)
            for field in ("python", "torch", "transformers", "torch_cuda", "precision")
        )
    ):
        raise ContractViolation("RUN_GPU_REPLAY_CHECKPOINT_LINEAGE_MISMATCH")
    return replay


def _verify_evaluation_index(
    evaluation_path: Path,
    *,
    freeze: Mapping[str, Any],
    code_sha256: str,
    cpu_check_sha256: str,
) -> dict[str, Any]:
    evaluation = read_verified_json(evaluation_path)
    freeze_artifacts = freeze.get("artifacts")
    if not isinstance(freeze_artifacts, Mapping):
        raise ContractViolation("RUN_EVALUATION_EVIDENCE_NOT_PASS")
    expected = {
        "freeze_sha256": freeze.get("freeze_sha256"),
        "config_sha256": freeze_artifacts.get("pilot_config", {}).get("sha256"),
        "tokenizer_sha256": freeze.get("tokenizer_file_sha256"),
        "code_sha256": code_sha256,
        "cpu_check_sha256": cpu_check_sha256,
        "evaluation_plan_sha256": freeze_artifacts.get("evaluation_plan", {}).get(
            "sha256"
        ),
    }
    expected_roots = {str(value) for value in load_pilot_config()["roots"]}
    if (
        evaluation.get("schema_version") != "pilot-evaluation-index-v1"
        or evaluation.get("status") != "PASS"
        or evaluation.get("data_kind") != "REAL"
        or any(evaluation.get(name) != value for name, value in expected.items())
        or not isinstance(evaluation.get("roots"), Mapping)
        or set(str(key) for key in evaluation["roots"]) != expected_roots
    ):
        raise ContractViolation("RUN_EVALUATION_EVIDENCE_NOT_PASS")
    for result in evaluation["roots"].values():
        if not isinstance(result, Mapping) or any(
            result.get(field) != "PASS"
            for field in ("training", "readiness", "measurement")
        ):
            raise ContractViolation("RUN_EVALUATION_EVIDENCE_NOT_PASS")
    return evaluation


def _validate_completion(
    summary: Mapping[str, Any],
    gates_by_name: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
    *,
    verify_artifacts: bool,
) -> None:
    if summary.get("data_kind") != "REAL":
        raise ContractViolation("PILOT_COMPLETION_REQUIRES_REAL_DATA")
    missing = [
        name
        for name in COMPLETION_REQUIRED_GATES
        if name not in gates_by_name or gates_by_name[name].get("status") != "PASS"
    ]
    if missing:
        raise ContractViolation("PILOT_COMPLETION_GATE_NOT_PASS:" + missing[0])
    if gates_by_name.get("main_study", {}).get("status") == "PASS":
        raise ContractViolation("MAIN_NOT_AUTHORIZED")
    missing_artifacts = [name for name in COMPLETION_REQUIRED_ARTIFACTS if name not in artifacts]
    if missing_artifacts:
        raise ContractViolation(
            "PILOT_COMPLETION_ARTIFACT_REQUIRED:" + missing_artifacts[0]
        )
    roots = summary["roots"]
    planned = {str(value) for value in load_pilot_config()["roots"]}
    if set(str(key) for key in roots) != planned:
        raise ContractViolation("PILOT_COMPLETION_REQUIRES_ALL_ROOTS")
    for root in roots.values():
        if any(root.get(field) != "PASS" for field in ("training", "readiness", "measurement")):
            raise ContractViolation("PILOT_COMPLETION_ROOT_NOT_PASS")
    if not verify_artifacts:
        return

    from .integrity import verify_cpu_check_evidence
    from .prepare_data import verify_experiment_freeze

    freeze = verify_experiment_freeze(
        _artifact_path(artifacts["experiment_freeze"]["path"]), production=True
    )
    freeze_artifacts = freeze.get("artifacts")
    provenance = summary["provenance"]
    if (
        not isinstance(freeze_artifacts, Mapping)
        or freeze_artifacts.get("pilot_config", {}).get("sha256")
        != provenance["config_sha256"]
    ):
        raise ContractViolation("RUN_COMPLETION_FREEZE_PROVENANCE_MISMATCH")
    cpu = verify_cpu_check_evidence(
        _artifact_path(artifacts["cpu_checks"]["path"])
    )
    cpu_implementation = cpu.get("implementation")
    if (
        not isinstance(cpu_implementation, Mapping)
        or cpu_implementation.get("implementation_manifest_sha256")
        != provenance["code_sha256"]
    ):
        raise ContractViolation("RUN_CPU_IMPLEMENTATION_PROVENANCE_MISMATCH")
    cpu_check_sha256 = _require_hash(
        cpu.get("cpu_check_sha256"), "INVALID_CPU_CHECK_HASH"
    )
    _verify_gpu_replay_evidence(
        _artifact_path(artifacts["gpu_replay"]["path"]),
        freeze_sha256=freeze["freeze_sha256"],
        code_sha256=provenance["code_sha256"],
        cpu_check_sha256=cpu_check_sha256,
    )
    evaluation = _verify_evaluation_index(
        _artifact_path(artifacts["evaluation_results"]["path"]),
        freeze=freeze,
        code_sha256=provenance["code_sha256"],
        cpu_check_sha256=cpu_check_sha256,
    )
    if {str(key): value for key, value in evaluation["roots"].items()} != {
        str(key): value for key, value in roots.items()
    }:
        raise ContractViolation("RUN_EVALUATION_ROOT_STATUS_MISMATCH")


def _verify_real_summary_evidence(
    summary: Mapping[str, Any],
    gates_by_name: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck evidence behind every PASS claim in a REAL run summary."""
    provenance = summary["provenance"]
    if sha256_file(PROJECT_ROOT / "spec/RESEARCH_SPEC_V4_KO.md") != provenance[
        "spec_sha256"
    ]:
        raise ContractViolation("RUN_PINNED_PROVENANCE_MISMATCH:spec_sha256")
    if sha256_file(PROJECT_ROOT / "spec/pilot.json") != provenance["config_sha256"]:
        raise ContractViolation("RUN_PINNED_PROVENANCE_MISMATCH:config_sha256")
    implementation_ref = artifacts.get("implementation_manifest")
    if not isinstance(implementation_ref, Mapping):
        raise ContractViolation("RUN_IMPLEMENTATION_EVIDENCE_REQUIRED")
    from .integrity import (
        verify_cpu_check_evidence,
        verify_delivery_manifest,
        verify_implementation_manifest,
    )

    implementation = verify_implementation_manifest(
        _artifact_path(implementation_ref.get("path"))
    )
    if implementation.get("implementation_manifest_sha256") != provenance[
        "code_sha256"
    ]:
        raise ContractViolation("RUN_IMPLEMENTATION_PROVENANCE_MISMATCH")
    pass_gates = {
        name for name, gate in gates_by_name.items() if gate.get("status") == "PASS"
    }
    if summary.get("data_kind") == "REAL_INPUTS_UNREVIEWED":
        allowed = {
            "delivery_integrity",
            "source_collection",
            "source_merge",
            "corpus",
            "tokenizer",
            "cpu_integration",
        }
        unsupported = sorted(pass_gates - allowed)
        if unsupported:
            raise ContractViolation("UNREVIEWED_DATA_PASS_GATE:" + unsupported[0])
    if "main_study" in pass_gates:
        raise ContractViolation("MAIN_NOT_AUTHORIZED")
    unsupported_pass_gates = pass_gates & {
        "budget",
        "pilot_training",
        "readiness",
        "measurement",
    }
    if unsupported_pass_gates:
        raise ContractViolation(
            "RUN_PASS_GATE_EVIDENCE_NOT_IMPLEMENTED:"
            + sorted(unsupported_pass_gates)[0]
        )
    if any(
        value == "PASS"
        for result in summary["roots"].values()
        for value in result.values()
    ):
        raise ContractViolation("RUN_ROOT_PASS_EVIDENCE_NOT_IMPLEMENTED")
    for name, value in summary["statuses"].items():
        if value != "PASS" or name == "implementation":
            continue
        if name not in pass_gates:
            raise ContractViolation("RUN_PASS_STATUS_WITHOUT_GATE:" + name)

    cache: dict[str, Any] = {"implementation": implementation}

    def cpu() -> Mapping[str, Any]:
        if "cpu" not in cache:
            path = _required_artifact_path(
                artifacts, "cpu_checks", gate="cpu_integration"
            )
            cache["cpu"] = verify_cpu_check_evidence(path)
        value = cache["cpu"]
        implementation_value = value.get("implementation")
        if (
            not isinstance(implementation_value, Mapping)
            or implementation_value.get("implementation_manifest_sha256")
            != provenance["code_sha256"]
        ):
            raise ContractViolation("RUN_CPU_IMPLEMENTATION_PROVENANCE_MISMATCH")
        return value

    def freeze() -> Mapping[str, Any]:
        if "freeze" not in cache:
            from .prepare_data import verify_experiment_freeze

            path = _required_artifact_path(
                artifacts, "experiment_freeze", gate="experiment_freeze"
            )
            cache["freeze"] = verify_experiment_freeze(path, production=True)
        return cache["freeze"]

    def replay() -> Mapping[str, Any]:
        if "replay" not in cache:
            cpu_value = cpu()
            freeze_value = freeze()
            cache["replay"] = _verify_gpu_replay_evidence(
                _required_artifact_path(
                    artifacts, "gpu_replay", gate="gpu_replay"
                ),
                freeze_sha256=freeze_value["freeze_sha256"],
                code_sha256=provenance["code_sha256"],
                cpu_check_sha256=cpu_value["cpu_check_sha256"],
            )
        return cache["replay"]

    if "delivery_integrity" in pass_gates:
        cache["delivery"] = verify_delivery_manifest(project_root=PROJECT_ROOT)
    if "source_collection" in pass_gates or "source_merge" in pass_gates:
        from .prepare_data import verify_krdict_collection

        krdict_path = _required_artifact_path(
            artifacts, "krdict_manifest", gate="source_collection"
        )
        cache["krdict"] = verify_krdict_collection(krdict_path, production=True)
    if "source_merge" in pass_gates:
        from .prepare_data import merge_krdict_snapshot

        merge_path = _required_artifact_path(
            artifacts, "merge_summary", gate="source_merge"
        )
        recorded_merge = read_verified_json(merge_path)
        recomputed_merge = merge_krdict_snapshot(
            _required_artifact_path(
                artifacts, "krdict_manifest", gate="source_merge"
            ),
            production=True,
        )
        if recorded_merge != recomputed_merge:
            raise ContractViolation("RUN_SOURCE_MERGE_EVIDENCE_MISMATCH")
        cache["merge"] = recorded_merge
    if "tokenizer" in pass_gates:
        from .build_tokenizer import verify_tokenizer_manifest

        cache["tokenizer"] = verify_tokenizer_manifest(
            _required_artifact_path(
                artifacts, "tokenizer_manifest", gate="tokenizer"
            ),
            production=True,
        )
    if "corpus" in pass_gates:
        from .build_tokenizer import verify_corpus_manifest

        cache["corpus"] = verify_corpus_manifest(
            _required_artifact_path(
                artifacts, "corpus_manifest", gate="corpus"
            ),
            production=True,
        )
    if "data_qa" in pass_gates or "annotation_freeze" in pass_gates:
        from .prepare_data import verify_annotation_freeze

        cache["annotation"] = verify_annotation_freeze(
            _required_artifact_path(
                artifacts, "annotation_freeze", gate="annotation_freeze"
            ),
            production=True,
        )
    if "experiment_freeze" in pass_gates:
        freeze()
    if "tokenizer_boundary" in pass_gates:
        boundary = freeze().get("boundary_audit")
        if not isinstance(boundary, Mapping) or boundary.get("status") != "PASS":
            raise ContractViolation("RUN_TOKEN_BOUNDARY_EVIDENCE_NOT_PASS")
    if "cpu_integration" in pass_gates:
        cpu()
    if "gpu_replay" in pass_gates:
        replay()
    return cache


def validate_run_summary(
    summary: Mapping[str, Any], *, verify_artifacts: bool = True
) -> None:
    if not isinstance(summary, Mapping):
        raise ContractViolation("RUN_SUMMARY_MUST_BE_OBJECT")
    missing = [field for field in REQUIRED_SUMMARY_FIELDS if field not in summary]
    if missing:
        raise ContractViolation("MISSING_RUN_SUMMARY_FIELD:" + missing[0])
    if summary.get("schema_version") != RUN_SUMMARY_SCHEMA:
        raise ContractViolation("RUN_SUMMARY_SCHEMA_MISMATCH")
    pilot = load_pilot_config()
    if summary.get("project_id") != pilot["project_id"]:
        raise ContractViolation("RUN_SUMMARY_PROJECT_MISMATCH")
    if summary.get("spec_version") != VERSION:
        raise ContractViolation("RUN_SUMMARY_SPEC_VERSION_MISMATCH")
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ContractViolation("INVALID_RUN_ID")
    _validate_utc_timestamp(summary.get("created_at_utc"))
    overall_status = summary.get("status")
    if overall_status != "PILOT_COMPLETE" and overall_status not in KNOWN_STATUS_VALUES - {"PASS"}:
        raise ContractViolation("INVALID_RUN_STATUS")
    data_kind = summary.get("data_kind")
    if data_kind not in {
        "REAL",
        "REAL_INPUTS_UNREVIEWED",
        "SYNTHETIC_TEST_FIXTURE",
    }:
        raise ContractViolation("INVALID_RUN_DATA_KIND")
    policy = summary.get("policy")
    if not isinstance(policy, Mapping) or policy.get("main_enabled") is not False:
        raise ContractViolation("MAIN_NOT_AUTHORIZED")
    statuses = summary.get("statuses")
    if not isinstance(statuses, Mapping) or not statuses:
        raise ContractViolation("INVALID_RUN_STATUSES")
    for name, value in statuses.items():
        if not isinstance(name, str) or not SAFE_NAME_RE.fullmatch(name):
            raise ContractViolation("INVALID_RUN_STATUS_NAME")
        if value not in KNOWN_STATUS_VALUES:
            raise ContractViolation("INVALID_RUN_STATUS_VALUE")
    gates = summary.get("gates")
    if not isinstance(gates, list):
        raise ContractViolation("INVALID_RUN_GATES")
    names: set[str] = set()
    for gate in gates:
        if not isinstance(gate, Mapping) or not str(gate.get("name", "")):
            raise ContractViolation("INVALID_RUN_GATE")
        name = str(gate["name"])
        if name not in KNOWN_GATE_NAMES:
            raise ContractViolation("UNKNOWN_RUN_GATE")
        if name in names:
            raise ContractViolation("DUPLICATE_RUN_GATE")
        names.add(name)
        gate_status = gate.get("status")
        if gate_status not in KNOWN_STATUS_VALUES:
            raise ContractViolation("INVALID_RUN_GATE_STATUS")
        reasons = gate.get("reasons")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason or len(reason) > 256
            for reason in reasons
        ):
            raise ContractViolation("INVALID_RUN_GATE_REASONS")
        if gate_status == "PASS" and reasons:
            raise ContractViolation("PASS_GATE_HAS_REASONS")
        if gate_status != "PASS" and not reasons:
            raise ContractViolation("NONPASS_GATE_REQUIRES_REASON")
        evidence = gate.get("evidence", {})
        if not isinstance(evidence, Mapping):
            raise ContractViolation("INVALID_RUN_GATE_EVIDENCE")
        if gate_status == "PASS" and not evidence:
            raise ContractViolation("PASS_GATE_REQUIRES_EVIDENCE")
        _validate_embedded_artifact_refs(
            evidence, verify_artifacts=verify_artifacts
        )
        if name in statuses and statuses[name] != gate_status:
            raise ContractViolation("RUN_STATUS_GATE_MISMATCH")
    provenance = summary.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ContractViolation("INVALID_RUN_PROVENANCE")
    for field in ("spec_sha256", "config_sha256", "code_sha256"):
        _require_hash(
            provenance.get(field), "INVALID_RUN_PROVENANCE_HASH:" + field
        )
    if "code_commit" in provenance and not (
        isinstance(provenance["code_commit"], str)
        and re.fullmatch(r"[0-9a-f]{40,64}", provenance["code_commit"])
    ):
        raise ContractViolation("INVALID_RUN_CODE_COMMIT")
    budget = summary.get("budget")
    if not isinstance(budget, Mapping):
        raise ContractViolation("INVALID_RUN_BUDGET")
    for value in budget.values():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ContractViolation("INVALID_RUN_BUDGET_VALUE")
    roots = summary.get("roots")
    if not isinstance(roots, Mapping):
        raise ContractViolation("INVALID_RUN_ROOTS")
    planned_roots = {str(value) for value in pilot["roots"]}
    for root_id, result in roots.items():
        if str(root_id) not in planned_roots or not isinstance(result, Mapping):
            raise ContractViolation("INVALID_RUN_ROOT")
        if set(result) != {"training", "readiness", "measurement"} or any(
            value not in KNOWN_STATUS_VALUES for value in result.values()
        ):
            raise ContractViolation("INVALID_RUN_ROOT_STATUS")
    next_action = summary.get("next_action")
    if not isinstance(next_action, str) or not next_action.strip() or len(next_action) > 4096:
        raise ContractViolation("INVALID_RUN_NEXT_ACTION")
    _validate_declared_hashes(summary)
    artifacts = _validate_artifacts(summary, verify_artifacts=verify_artifacts)
    if summary.get("data_kind") == "SYNTHETIC_TEST_FIXTURE":
        if (
            summary.get("status") == "PILOT_COMPLETE"
            or any(gate.get("status") == "PASS" for gate in gates)
            or any(value == "PASS" for value in statuses.values())
            or any(
                value == "PASS"
                for result in roots.values()
                for value in result.values()
            )
        ):
            raise ContractViolation("FIXTURE_CANNOT_PASS_PRODUCTION_GATE")
    gates_by_name = {str(gate["name"]): gate for gate in gates}
    if overall_status == "PILOT_COMPLETE":
        _validate_completion(
            summary,
            gates_by_name,
            artifacts,
            verify_artifacts=verify_artifacts,
        )
    else:
        if overall_status not in statuses.values() and overall_status not in {
            gate["status"] for gate in gates
        }:
            raise ContractViolation("RUN_OVERALL_STATUS_INCONSISTENT")
    if verify_artifacts and data_kind in {"REAL", "REAL_INPUTS_UNREVIEWED"}:
        _verify_real_summary_evidence(summary, gates_by_name, artifacts)
    try:
        canonical_json_bytes(dict(summary))
    except ContractViolation:
        raise
    except (TypeError, ValueError) as exc:
        raise ContractViolation("INVALID_RUN_SUMMARY_JSON") from exc


def assemble_run_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach the sole source object without adding hidden results."""
    try:
        detached = json.loads(canonical_json_bytes(dict(payload)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractViolation("INVALID_RUN_SUMMARY_JSON") from exc
    validate_run_summary(detached)
    return detached


def publish_run_summary(
    path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    summary = assemble_run_summary(payload)
    return publish_json_once(path, summary)


def report_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return only already-stored decisions for CLI/table consumers."""
    validate_run_summary(summary, verify_artifacts=False)
    return {
        "run_id": summary["run_id"],
        "status": summary["status"],
        "statuses": summary["statuses"],
        "gates": summary["gates"],
        "next_action": summary["next_action"],
    }


def _safe_html_text(value: Any) -> str:
    return html.escape(str(value), quote=True).replace("\r", " ").replace("\n", " ")


def _table_cell(value: Any) -> str:
    # HTML escaping neutralizes tags; entities avoid Markdown table/control syntax.
    escaped = _safe_html_text(value).replace("|", "&#124;")
    for character in ("\\", "`", "*", "_", "[", "]", "(", ")", "#", ">", "!"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped


def render_markdown(
    json_path: str | Path, markdown_path: str | Path
) -> dict[str, Any]:
    """Render a deterministic projection; never recompute a metric or decision."""
    source_path = Path(json_path)
    summary, source_sha256 = read_verified_json_with_sha256(source_path)
    # Rendering is deliberately a pure projection of the authoritative JSON.
    # The source may have been supplied externally, so reopen and verify every
    # artifact/evidence reference before projecting any of its claims.
    validate_run_summary(summary, verify_artifacts=True)
    lines = [
        "# v4 run summary",
        "",
        f"- Run ID: <code>{_safe_html_text(summary['run_id'])}</code>",
        f"- Status: <strong>{_safe_html_text(summary['status'])}</strong>",
        f"- Source JSON: <code>{_safe_html_text(source_path.name)}</code>",
        f"- Source JSON SHA-256: <code>{source_sha256}</code>",
        f"- Main enabled: `{str(summary['policy']['main_enabled']).lower()}`",
        "",
        "## Gate decisions",
        "",
        "| Gate | Status | Reasons |",
        "|---|---|---|",
    ]
    for gate in summary["gates"]:
        reasons = gate.get("reasons", [])
        reason_text = ", ".join(str(reason) for reason in reasons) if reasons else "—"
        lines.append(
            "| "
            + _table_cell(gate["name"])
            + " | "
            + _table_cell(gate["status"])
            + " | "
            + _table_cell(reason_text)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Next action",
            "",
            _table_cell(summary["next_action"]),
            "",
            "## Exact source values",
            "",
            "The block below is a display of the authoritative JSON; no values are recomputed.",
            "",
            "<pre>",
            html.escape(
                json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            ),
            "</pre>",
            "",
        ]
    )
    return replace_derived_text(markdown_path, "\n".join(lines))
