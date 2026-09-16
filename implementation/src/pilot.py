"""Legacy v4.0 orchestration contract (retained for immutable compatibility).

The executable semantic v4.1.2 implementation is in ``pilot_runtime`` and
``python -m implementation.pilot_cli``. This old state-schema adapter is not
used to launch that runner, and cannot silently upgrade an old freeze.

This module deliberately separates *authorization* from *execution*.  It can
derive the one or more next actions allowed by already verified evidence, but
it does not pretend that the repository currently has an atomic scientific
stage runner.  ``run_production_pilot`` therefore returns ``NOT_IMPLEMENTED``
with the exact missing interfaces before importing Torch or acquiring a GPU.

When those interfaces are implemented, every action with
``requires_gpu_session`` must be executed inside
``train.gpu2_training_session``.  There is no device argument or alternate
learning entry point in this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    ContractViolation,
    PROJECT_ROOT,
    load_json,
    load_pilot_config,
    require_sha256,
)


PILOT_STATE_SCHEMA = "pilot-state-v1"
PILOT_PREFLIGHT_SCHEMA = "pilot-preflight-v1"
PLANNED_ROOTS = (4101, 4102, 4103, 4104)
S_PHASES = ("INIT", "CORPUS", "T0", "T1", "T2", "T3")
LEXICAL_PHASES = ("T0", "T1", "T2", "T3")
H_PHASES = ("H_BASE", "H_A", "H_B")
BASE_GATES = (
    "delivery_integrity",
    "data_qa",
    "freeze",
    "tokenizer_boundary",
    "cpu_integration",
    "gpu_replay",
    "budget",
)
LEXICAL_DIAGNOSTIC_STEPS = tuple(range(200, 3001, 200))
H_DIAGNOSTIC_STEPS = (200, 300)

# These are orchestration-level interfaces, not low-level math or checkpoint
# primitives.  Naming them explicitly prevents a partial stack from being
# reported as a runnable or completed scientific pilot.
MISSING_PRODUCTION_INTERFACES = (
    "atomic_write_once_pilot_state_journal",
    "verified_init_and_corpus_fixed_endpoint_runner",
    "verified_lexical_record_stream_and_3000_update_stage_runner",
    "checkpoint_boundary_resume_chain_coordinator",
    "hash_bound_evaluation_record_and_score_artifact_publisher",
    "lineage_bound_readiness_and_measurement_gate_publisher",
    "single_H_BASE_fork_and_paired_B36_branch_runner",
)


@dataclass(frozen=True)
class PilotAction:
    kind: str
    root_id: int | None
    phase: str | None
    stage: str | None
    branch: str | None
    parent_phase: str | None
    requires_gpu_session: bool
    performs_learning: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PilotDecision:
    status: str
    authorized: bool
    actions: tuple[PilotAction, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "authorized": self.authorized,
            "actions": [action.as_dict() for action in self.actions],
            "reasons": list(self.reasons),
        }


def _hash(value: Any, code: str) -> str:
    try:
        return require_sha256(str(value), code)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ContractViolation):
            raise
        raise ContractViolation(code) from exc


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractViolation(code)
    return value


def _status(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("status")
    return str(value)


def _state_bindings(state: Mapping[str, Any]) -> dict[str, str]:
    allowed = {
        "schema_version",
        "project_id",
        "main_enabled",
        "freeze_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "code_sha256",
        "evaluation_plan_sha256",
        "corpus_token_count",
        "base_gates",
        "roots",
        "history",
    }
    if set(state) != allowed:
        raise ContractViolation("PILOT_STATE_FIELD_SET_MISMATCH")
    if state.get("schema_version") != PILOT_STATE_SCHEMA:
        raise ContractViolation("PILOT_STATE_SCHEMA_MISMATCH")
    pilot = load_pilot_config()
    if state.get("project_id") != pilot["project_id"]:
        raise ContractViolation("PILOT_PROJECT_MISMATCH")
    if state.get("main_enabled") is not False:
        raise ContractViolation("MAIN_NOT_AUTHORIZED")
    bindings = {
        name: _hash(state.get(name), "INVALID_PILOT_" + name.upper())
        for name in (
            "freeze_sha256",
            "config_sha256",
            "tokenizer_sha256",
            "code_sha256",
            "evaluation_plan_sha256",
        )
    }
    token_count = state.get("corpus_token_count")
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count != 2_000_000:
        raise ContractViolation("PILOT_CORPUS_TOKEN_COUNT_MISMATCH")
    return bindings


def _validate_base_gates(
    state: Mapping[str, Any], bindings: Mapping[str, str]
) -> tuple[bool, tuple[str, ...]]:
    gates = _mapping(state.get("base_gates", {}), "INVALID_BASE_GATES")
    unknown = set(gates) - set(BASE_GATES)
    if unknown:
        raise ContractViolation("UNKNOWN_BASE_GATE")
    failed: list[str] = []
    for name in BASE_GATES:
        value = gates.get(name)
        if value is None:
            failed.append(name + ":NOT_RUN")
            continue
        gate = _mapping(value, "INVALID_BASE_GATE")
        if _status(gate) != "PASS":
            failed.append(name + ":" + _status(gate))
            continue
        _hash(gate.get("evidence_sha256"), "INVALID_BASE_GATE_EVIDENCE_HASH")
        if gate.get("freeze_sha256") not in {None, bindings["freeze_sha256"]}:
            raise ContractViolation("BASE_GATE_FREEZE_MISMATCH")
        if gate.get("code_sha256") not in {None, bindings["code_sha256"]}:
            raise ContractViolation("BASE_GATE_CODE_MISMATCH")
        if name == "freeze" and gate.get("freeze_sha256") != bindings["freeze_sha256"]:
            raise ContractViolation("FREEZE_GATE_NOT_BOUND")
        if name == "gpu_replay":
            if (
                gate.get("freeze_sha256") != bindings["freeze_sha256"]
                or gate.get("code_sha256") != bindings["code_sha256"]
                or gate.get("scientific_result") is not False
            ):
                raise ContractViolation("GPU_REPLAY_GATE_NOT_BOUND")
    return not failed, tuple(failed)


def _checkpoint_marker(
    value: Any,
    *,
    root_id: int,
    phase: str,
    parent: Mapping[str, Any] | None,
    bindings: Mapping[str, str],
    corpus_token_count: int,
) -> Mapping[str, Any]:
    marker = _mapping(value, "INVALID_CHECKPOINT_EVIDENCE")
    expected_stage = "H" if phase in H_PHASES else phase
    expected_branch = {"H_A": "A", "H_B": "B"}.get(phase)
    if (
        marker.get("status") != "PASS"
        or marker.get("root_id") != root_id
        or marker.get("phase") != phase
        or marker.get("stage") != expected_stage
        or marker.get("branch") != expected_branch
        or marker.get("data_kind") != "REAL"
        or marker.get("freeze_gate_status") != "PASS"
    ):
        raise ContractViolation("CHECKPOINT_EVIDENCE_LINEAGE_MISMATCH")
    for name, expected in bindings.items():
        if marker.get(name) != expected:
            raise ContractViolation("CHECKPOINT_EVIDENCE_BINDING_MISMATCH")
    for name in (
        "checkpoint_sha256",
        "manifest_sha256",
        "model_fingerprint",
        "initial_model_fingerprint",
        "plan_sha256",
    ):
        _hash(marker.get(name), "INVALID_CHECKPOINT_EVIDENCE_HASH")
    binding = _mapping(marker.get("gpu_binding"), "CHECKPOINT_GPU_BINDING_MISSING")
    policy = load_json(PROJECT_ROOT / "implementation/config/campaign_policy.json")["gpu"]
    if (
        binding.get("physical_index") != 2
        or binding.get("logical_index") != 0
        or binding.get("uuid") != policy["expected_uuid"]
    ):
        raise ContractViolation("CHECKPOINT_NOT_BOUND_TO_GPU2")
    if not isinstance(marker.get("budget_lease_id"), str) or not marker["budget_lease_id"]:
        raise ContractViolation("CHECKPOINT_BUDGET_LEASE_MISSING")
    for name in ("global_step", "phase_step", "model_tokens_seen"):
        if not isinstance(marker.get(name), int) or isinstance(marker.get(name), bool) or marker[name] < 0:
            raise ContractViolation("INVALID_CHECKPOINT_PROGRESS")
    if marker.get("accumulation_step") != 0:
        raise ContractViolation("CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY")
    expected_parent_hash = None if parent is None else parent["checkpoint_sha256"]
    if marker.get("phase_parent_sha256") != expected_parent_hash:
        raise ContractViolation("CHECKPOINT_PHASE_PARENT_MISMATCH")
    resume_parent = marker.get("resume_parent_sha256")
    if resume_parent is not None:
        _hash(resume_parent, "INVALID_RESUME_PARENT_HASH")
        chain = marker.get("verified_resume_chain")
        if not isinstance(chain, Sequence) or isinstance(chain, (str, bytes)) or not chain:
            raise ContractViolation("UNVERIFIED_CHECKPOINT_RESUME_CHAIN")
        if chain[-1] != resume_parent or any(not isinstance(item, str) for item in chain):
            raise ContractViolation("UNVERIFIED_CHECKPOINT_RESUME_CHAIN")
        for item in chain:
            _hash(item, "INVALID_RESUME_PARENT_HASH")
    else:
        chain = marker.get("verified_resume_chain")
        if chain is not None and chain != () and chain != []:
            raise ContractViolation("UNEXPECTED_CHECKPOINT_RESUME_CHAIN")

    if phase == "INIT":
        if marker["global_step"] != 0 or marker["phase_step"] != 0 or marker["model_tokens_seen"] != 0:
            raise ContractViolation("INIT_CHECKPOINT_PROGRESS_MISMATCH")
        if marker["model_fingerprint"] != marker["initial_model_fingerprint"]:
            raise ContractViolation("INIT_FINGERPRINT_MISMATCH")
    elif phase == "CORPUS":
        if marker["phase_step"] <= 0 or marker["model_tokens_seen"] != corpus_token_count:
            raise ContractViolation("CORPUS_FIXED_ENDPOINT_MISMATCH")
        if parent is None or marker["global_step"] <= parent["global_step"]:
            raise ContractViolation("CORPUS_FIXED_ENDPOINT_MISMATCH")
    elif phase in LEXICAL_PHASES:
        if parent is None or marker["phase_step"] != 3000:
            raise ContractViolation("LEXICAL_FIXED_ENDPOINT_MISMATCH")
        if marker["global_step"] != parent["global_step"] + 3000:
            raise ContractViolation("LEXICAL_FIXED_ENDPOINT_MISMATCH")
        if marker["model_tokens_seen"] < parent["model_tokens_seen"]:
            raise ContractViolation("MODEL_TOKEN_COUNTER_REGRESSED")
    elif phase == "H_BASE":
        if (
            parent is None
            or marker["phase_step"] != 0
            or marker["global_step"] != parent["global_step"]
            or marker["model_tokens_seen"] != parent["model_tokens_seen"]
            or marker.get("learning_rate") != 0.0001
        ):
            raise ContractViolation("HISTORY_BASE_TRANSITION_MISMATCH")
    else:
        if parent is None or marker["phase_step"] != 300:
            raise ContractViolation("HISTORY_FIXED_ENDPOINT_MISMATCH")
        if marker["global_step"] != parent["global_step"] + 300:
            raise ContractViolation("HISTORY_FIXED_ENDPOINT_MISMATCH")
    if parent is not None and marker["initial_model_fingerprint"] != parent["initial_model_fingerprint"]:
        raise ContractViolation("ROOT_INITIALIZATION_LINEAGE_MISMATCH")
    return marker


def _evaluation_marker(
    value: Any,
    *,
    checkpoint: Mapping[str, Any],
    expected_steps: Sequence[int],
    bindings: Mapping[str, str],
) -> Mapping[str, Any]:
    marker = _mapping(value, "INVALID_EVALUATION_EVIDENCE")
    if (
        marker.get("status") != "PASS"
        or marker.get("root_id") != checkpoint["root_id"]
        or marker.get("phase") != checkpoint["phase"]
        or marker.get("stage") != checkpoint["stage"]
        or marker.get("branch") != checkpoint["branch"]
        or marker.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
        or marker.get("model_fingerprint") != checkpoint["model_fingerprint"]
        or marker.get("diagnostic_steps") != list(expected_steps)
        or marker.get("rng_unchanged") is not True
        or marker.get("fixed_endpoint_used") is not True
        or marker.get("data_kind") != "REAL"
    ):
        raise ContractViolation("EVALUATION_EVIDENCE_MISMATCH")
    for name, expected in bindings.items():
        if marker.get(name) != expected:
            raise ContractViolation("EVALUATION_EVIDENCE_BINDING_MISMATCH")
    _hash(marker.get("score_artifact_sha256"), "INVALID_SCORE_ARTIFACT_HASH")
    return marker


def _gate_marker(
    value: Any,
    *,
    kind: str,
    checkpoint: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    marker = _mapping(value, "INVALID_" + kind.upper() + "_EVIDENCE")
    status = _status(marker)
    allowed = {"PASS", "BLOCKED_READINESS"} if kind == "readiness" else {"PASS", "BLOCKED_MEASUREMENT"}
    if (
        status not in allowed
        or marker.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
        or marker.get("score_artifact_sha256") != evaluation["score_artifact_sha256"]
        or marker.get("root_id") != checkpoint["root_id"]
    ):
        raise ContractViolation(kind.upper() + "_EVIDENCE_MISMATCH")
    _hash(marker.get("gate_artifact_sha256"), "INVALID_GATE_ARTIFACT_HASH")
    return status


def _action(kind: str, root_id: int, phase: str, parent_phase: str | None) -> PilotAction:
    branch = {"H_A": "A", "H_B": "B"}.get(phase)
    stage = "H" if phase in H_PHASES else phase
    learning = kind in {"TRAIN_CORPUS", "TRAIN_LEXICAL", "TRAIN_HISTORY_BRANCH"}
    requires_gpu_session = kind not in {"COMPUTE_READINESS", "COMPUTE_MEASUREMENT"}
    return PilotAction(
        kind,
        root_id,
        phase,
        stage,
        branch,
        parent_phase,
        requires_gpu_session,
        learning,
    )


def _validate_distinct_root_initializations(
    checkpoints_by_root: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    require_all_roots: bool = False,
) -> None:
    """Reject duplicate initializations before any later root GPU action."""
    roots = set(checkpoints_by_root)
    if not roots.issubset(PLANNED_ROOTS) or (
        require_all_roots and roots != set(PLANNED_ROOTS)
    ):
        raise ContractViolation("INVALID_INITIALIZATION_ROOT_SET")
    fingerprints: list[str] = []
    for root_id in PLANNED_ROOTS:
        if root_id not in checkpoints_by_root:
            continue
        checkpoints = checkpoints_by_root.get(root_id)
        if not isinstance(checkpoints, Mapping):
            raise ContractViolation("INVALID_INITIALIZATION_ROOT_SET")
        init = checkpoints.get("INIT")
        if not isinstance(init, Mapping):
            raise ContractViolation("INVALID_INITIALIZATION_ROOT_SET")
        fingerprints.append(
            _hash(
                init.get("initial_model_fingerprint"),
                "INVALID_CHECKPOINT_EVIDENCE_HASH",
            )
        )
    if len(set(fingerprints)) != len(fingerprints):
        raise ContractViolation("ROOT_INITIALIZATION_NOT_DISTINCT")


def _root_progress(
    root_id: int,
    value: Any,
    *,
    bindings: Mapping[str, str],
    corpus_token_count: int,
) -> tuple[PilotAction | None, bool, tuple[str, ...], Mapping[str, Mapping[str, Any]]]:
    root = _mapping(value, "INVALID_ROOT_STATE")
    if set(root) - {"checkpoints", "evaluations", "readiness", "measurement"}:
        raise ContractViolation("UNKNOWN_ROOT_STATE_FIELD")
    checkpoints = _mapping(root.get("checkpoints", {}), "INVALID_ROOT_CHECKPOINTS")
    evaluations = _mapping(root.get("evaluations", {}), "INVALID_ROOT_EVALUATIONS")
    if set(checkpoints) - set(S_PHASES) or set(evaluations) - set(LEXICAL_PHASES):
        raise ContractViolation("UNKNOWN_ROOT_PHASE_EVIDENCE")
    verified: dict[str, Mapping[str, Any]] = {}
    previous: Mapping[str, Any] | None = None
    for index, phase in enumerate(S_PHASES):
        if phase not in checkpoints:
            if any(later in checkpoints for later in S_PHASES[index + 1 :]):
                raise ContractViolation("SKIPPED_CHECKPOINT_PHASE")
            if any(later in evaluations for later in LEXICAL_PHASES if later not in verified):
                raise ContractViolation("EVALUATION_BEFORE_CHECKPOINT")
            kind = "CREATE_INIT" if phase == "INIT" else (
                "TRAIN_CORPUS" if phase == "CORPUS" else "TRAIN_LEXICAL"
            )
            return _action(kind, root_id, phase, S_PHASES[index - 1] if index else None), False, (), verified
        marker = _checkpoint_marker(
            checkpoints[phase],
            root_id=root_id,
            phase=phase,
            parent=previous,
            bindings=bindings,
            corpus_token_count=corpus_token_count,
        )
        verified[phase] = marker
        previous = marker
        if phase in LEXICAL_PHASES:
            if phase not in evaluations:
                if any(later in checkpoints for later in S_PHASES[index + 1 :]):
                    raise ContractViolation("NEXT_STAGE_BEFORE_EVALUATION")
                return _action("EVALUATE_STAGE", root_id, phase, phase), False, (), verified
            _evaluation_marker(
                evaluations[phase],
                checkpoint=marker,
                expected_steps=LEXICAL_DIAGNOSTIC_STEPS,
                bindings=bindings,
            )
    t3 = verified["T3"]
    t3_evaluation = _mapping(evaluations["T3"], "INVALID_EVALUATION_EVIDENCE")
    if "readiness" not in root:
        return _action("COMPUTE_READINESS", root_id, "T3", "T3"), False, (), verified
    readiness = _gate_marker(
        root["readiness"], kind="readiness", checkpoint=t3, evaluation=t3_evaluation
    )
    if readiness != "PASS":
        if "measurement" in root:
            raise ContractViolation("MEASUREMENT_AFTER_BLOCKED_READINESS")
        return None, False, (f"root_{root_id}:BLOCKED_READINESS",), verified
    if "measurement" not in root:
        return _action("COMPUTE_MEASUREMENT", root_id, "T3", "T3"), False, (), verified
    measurement = _gate_marker(
        root["measurement"], kind="measurement", checkpoint=t3, evaluation=t3_evaluation
    )
    if measurement != "PASS":
        return None, False, (f"root_{root_id}:BLOCKED_MEASUREMENT",), verified
    return None, True, (), verified


def _history_progress(
    root_id: int,
    value: Any,
    *,
    t3: Mapping[str, Any],
    bindings: Mapping[str, str],
    corpus_token_count: int,
) -> tuple[PilotAction | None, bool]:
    history = _mapping(value, "INVALID_HISTORY_STATE")
    if set(history) - {"checkpoints", "evaluations"}:
        raise ContractViolation("UNKNOWN_HISTORY_STATE_FIELD")
    checkpoints = _mapping(history.get("checkpoints", {}), "INVALID_HISTORY_CHECKPOINTS")
    evaluations = _mapping(history.get("evaluations", {}), "INVALID_HISTORY_EVALUATIONS")
    if set(checkpoints) - set(H_PHASES) or set(evaluations) - {"H_A", "H_B"}:
        raise ContractViolation("UNKNOWN_HISTORY_PHASE_EVIDENCE")
    if "H_BASE" not in checkpoints:
        if checkpoints or evaluations:
            raise ContractViolation("HISTORY_BRANCH_BEFORE_BASE")
        return _action("TRANSITION_HISTORY_BASE", root_id, "H_BASE", "T3"), False
    base = _checkpoint_marker(
        checkpoints["H_BASE"],
        root_id=root_id,
        phase="H_BASE",
        parent=t3,
        bindings=bindings,
        corpus_token_count=corpus_token_count,
    )
    for branch_phase in ("H_A", "H_B"):
        if branch_phase not in checkpoints:
            if branch_phase == "H_A" and ("H_B" in checkpoints or "H_B" in evaluations):
                raise ContractViolation("HISTORY_B_BEFORE_A")
            return _action("TRAIN_HISTORY_BRANCH", root_id, branch_phase, "H_BASE"), False
        checkpoint = _checkpoint_marker(
            checkpoints[branch_phase],
            root_id=root_id,
            phase=branch_phase,
            parent=base,
            bindings=bindings,
            corpus_token_count=corpus_token_count,
        )
        if branch_phase not in evaluations:
            if branch_phase == "H_A" and "H_B" in checkpoints:
                raise ContractViolation("HISTORY_B_BEFORE_A_EVALUATION")
            return _action("EVALUATE_HISTORY_BRANCH", root_id, branch_phase, branch_phase), False
        _evaluation_marker(
            evaluations[branch_phase],
            checkpoint=checkpoint,
            expected_steps=H_DIAGNOSTIC_STEPS,
            bindings=bindings,
        )
    return None, True


def derive_pilot_decision(state: Mapping[str, Any], *, requested_main: bool = False) -> PilotDecision:
    """Validate evidence ordering and return only currently authorized actions."""
    bindings = _state_bindings(state)
    if requested_main:
        return PilotDecision("MAIN_NOT_AUTHORIZED", False, (), ("MAIN_ENABLED_FALSE",))
    base_passed, base_failures = _validate_base_gates(state, bindings)
    roots = _mapping(state.get("roots", {}), "INVALID_ROOTS_STATE")
    normalized_roots: dict[int, Any] = {}
    for key, value in roots.items():
        try:
            root_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ContractViolation("INVALID_ROOT_ID") from exc
        if root_id not in PLANNED_ROOTS or root_id in normalized_roots:
            raise ContractViolation("INVALID_ROOT_ID")
        normalized_roots[root_id] = value
    history = _mapping(state.get("history", {}), "INVALID_HISTORY_STATE")
    if not base_passed:
        if normalized_roots or history:
            raise ContractViolation("TRAINING_EVIDENCE_PRECEDES_BASE_GATES")
        return PilotDecision("BLOCKED_PREREQUISITES", False, (), base_failures)

    first_action, first_qualified, first_reasons, first_checkpoints = _root_progress(
        4101,
        normalized_roots.get(4101, {}),
        bindings=bindings,
        corpus_token_count=int(state["corpus_token_count"]),
    )
    if not first_qualified:
        if any(root in normalized_roots for root in PLANNED_ROOTS[1:]) or history:
            raise ContractViolation("ROOT_EXPANDED_BEFORE_4101_GATES")
        if first_action is not None:
            return PilotDecision("AUTHORIZED", True, (first_action,), ())
        return PilotDecision("BLOCKED_ROOT_4101", False, (), first_reasons)

    actions: list[PilotAction] = []
    failures: list[str] = []
    qualified: dict[int, Mapping[str, Mapping[str, Any]]] = {4101: first_checkpoints}
    observed_initializations: dict[int, Mapping[str, Mapping[str, Any]]] = {
        4101: first_checkpoints
    }
    for root_id in PLANNED_ROOTS[1:]:
        action, passed, reasons, checkpoints = _root_progress(
            root_id,
            normalized_roots.get(root_id, {}),
            bindings=bindings,
            corpus_token_count=int(state["corpus_token_count"]),
        )
        if action is not None:
            actions.append(action)
        if passed:
            qualified[root_id] = checkpoints
        if "INIT" in checkpoints:
            observed_initializations[root_id] = checkpoints
        failures.extend(reasons)
    # A later root whose INIT collides with an already observed root must not be
    # allowed to spend corpus/lexical GPU time and fail only at the H boundary.
    _validate_distinct_root_initializations(observed_initializations)
    if actions:
        if history:
            raise ContractViolation("HISTORY_BEFORE_ALL_ROOT_GATES")
        return PilotDecision("AUTHORIZED", True, tuple(actions), tuple(failures))
    if failures or set(qualified) != set(PLANNED_ROOTS):
        if history:
            raise ContractViolation("HISTORY_BEFORE_ALL_ROOT_GATES")
        return PilotDecision("BLOCKED_REMAINING_ROOTS", False, (), tuple(failures))

    _validate_distinct_root_initializations(qualified, require_all_roots=True)

    normalized_history: dict[int, Any] = {}
    for key, value in history.items():
        try:
            root_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ContractViolation("INVALID_HISTORY_ROOT") from exc
        if root_id not in PLANNED_ROOTS or root_id in normalized_history:
            raise ContractViolation("INVALID_HISTORY_ROOT")
        normalized_history[root_id] = value
    history_actions: list[PilotAction] = []
    completed = True
    for root_id in PLANNED_ROOTS:
        action, done = _history_progress(
            root_id,
            normalized_history.get(root_id, {}),
            t3=qualified[root_id]["T3"],
            bindings=bindings,
            corpus_token_count=int(state["corpus_token_count"]),
        )
        completed &= done
        if action is not None:
            history_actions.append(action)
    if history_actions:
        return PilotDecision("AUTHORIZED", True, tuple(history_actions), ())
    if completed:
        return PilotDecision("PILOT_COMPLETE_CANDIDATE", False, (), ())
    raise ContractViolation("INVALID_PILOT_STATE")


def _validate_production_freeze(
    experiment_freeze_path: Path,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    from .prepare_data import verify_experiment_freeze

    freeze = verify_experiment_freeze(Path(experiment_freeze_path), production=True)
    if (
        freeze.get("verified") is not True
        or freeze.get("status") != "PASS"
        or freeze.get("freeze_gate_status") != "PASS"
        or freeze.get("data_kind") != "REAL"
    ):
        raise ContractViolation("EXPERIMENT_FREEZE_GATE_NOT_PASS")
    artifacts = _mapping(freeze.get("artifacts"), "EXPERIMENT_FREEZE_ARTIFACTS_MISSING")
    expected = {
        "freeze_sha256": freeze.get("freeze_sha256"),
        "config_sha256": _mapping(artifacts.get("pilot_config"), "PILOT_CONFIG_REF_MISSING").get("sha256"),
        "tokenizer_sha256": freeze.get("tokenizer_file_sha256"),
        "evaluation_plan_sha256": _mapping(artifacts.get("evaluation_plan"), "EVALUATION_PLAN_REF_MISSING").get("sha256"),
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise ContractViolation("PILOT_STATE_FREEZE_BINDING_MISMATCH")
    if state.get("corpus_token_count") != freeze.get("corpus_token_count"):
        raise ContractViolation("PILOT_STATE_FREEZE_BINDING_MISMATCH")
    return freeze


def _validate_budget(ledger: Any, *, planned_reservation_seconds: float) -> dict[str, Any]:
    if not math.isfinite(planned_reservation_seconds) or planned_reservation_seconds <= 0:
        raise ContractViolation("INVALID_PLANNED_GPU_RESERVATION")
    try:
        snapshot = ledger.snapshot()
    except AttributeError as exc:
        raise ContractViolation("GPU_BUDGET_LEDGER_REQUIRED") from exc
    policy = load_json(PROJECT_ROOT / "implementation/config/campaign_policy.json")["budget"]
    if (
        snapshot.get("campaign_cap_gpu_hours") != policy["campaign_gpu_hours_cap"]
        or snapshot.get("prior_gpu_hours_user_reported") != policy["prior_gpu_hours_user_reported"]
    ):
        raise ContractViolation("GPU_BUDGET_POLICY_MISMATCH")
    used_hours = snapshot.get("campaign_gpu_hours_charged_or_reserved")
    remaining_hours = snapshot.get("campaign_gpu_hours_remaining")
    if (
        not isinstance(used_hours, (int, float))
        or not isinstance(remaining_hours, (int, float))
        or not math.isfinite(float(used_hours))
        or not math.isfinite(float(remaining_hours))
        or used_hours < policy["prior_gpu_hours_user_reported"]
        or used_hours > policy["campaign_gpu_hours_cap"]
        or abs(
            float(remaining_hours)
            - max(0.0, float(policy["campaign_gpu_hours_cap"]) - float(used_hours))
        )
        > 1e-9
        or remaining_hours * 3600.0 + 1e-9 < planned_reservation_seconds
    ):
        raise ContractViolation("BLOCKED_CAMPAIGN_GPU_BUDGET")
    root_usage = _mapping(
        snapshot.get("per_root_gpu_seconds_charged_or_reserved", {}),
        "INVALID_ROOT_GPU_BUDGET",
    )
    root_cap = float(policy["per_root_gpu_hours_cap"]) * 3600.0
    for key, value in root_usage.items():
        if str(key) not in {str(root) for root in PLANNED_ROOTS}:
            raise ContractViolation("INVALID_ROOT_GPU_BUDGET")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0 or value > root_cap:
            raise ContractViolation("BLOCKED_ROOT_GPU_BUDGET")
    return dict(snapshot)


def _validate_action_root_budgets(
    snapshot: Mapping[str, Any],
    actions: Sequence[PilotAction],
    *,
    planned_reservation_seconds: float,
) -> None:
    policy = load_json(PROJECT_ROOT / "implementation/config/campaign_policy.json")["budget"]
    root_cap = float(policy["per_root_gpu_hours_cap"]) * 3600.0
    usage = _mapping(
        snapshot.get("per_root_gpu_seconds_charged_or_reserved", {}),
        "INVALID_ROOT_GPU_BUDGET",
    )
    for action in actions:
        if not action.requires_gpu_session or action.root_id is None:
            continue
        current = float(usage.get(str(action.root_id), 0.0))
        if current + planned_reservation_seconds > root_cap + 1e-9:
            raise ContractViolation("BLOCKED_ROOT_GPU_BUDGET")


def run_production_pilot(
    *,
    state: Mapping[str, Any],
    experiment_freeze_path: Path,
    ledger: Any,
    code_sha256: str,
    requested_main: bool = False,
    planned_reservation_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Perform fail-closed production preflight; never start partial training.

    The returned ``NOT_IMPLEMENTED`` is intentional until every interface in
    ``missing_interfaces`` exists.  In particular this function never imports
    :mod:`implementation.src.train`, so it cannot accidentally create a model,
    initialize CUDA, reserve GPU time, or execute an optimizer step.
    """
    require_sha256(code_sha256, "INVALID_CODE_SHA256")
    if state.get("code_sha256") != code_sha256:
        raise ContractViolation("PILOT_CODE_BINDING_MISMATCH")
    if requested_main:
        return {
            "schema_version": PILOT_PREFLIGHT_SCHEMA,
            "status": "MAIN_NOT_AUTHORIZED",
            "authorized": False,
            "scientific_training_started": False,
            "gpu_session_started": False,
            "reasons": ["MAIN_ENABLED_FALSE"],
        }
    pilot = load_pilot_config()
    if pilot["policy"]["main_enabled"] is not False:
        raise ContractViolation("MAIN_NOT_AUTHORIZED")
    freeze = _validate_production_freeze(
        Path(experiment_freeze_path), state=state
    )
    budget = _validate_budget(
        ledger, planned_reservation_seconds=planned_reservation_seconds
    )
    decision = derive_pilot_decision(state)
    _validate_action_root_budgets(
        budget,
        decision.actions,
        planned_reservation_seconds=planned_reservation_seconds,
    )
    return {
        "schema_version": PILOT_PREFLIGHT_SCHEMA,
        "status": "NOT_IMPLEMENTED",
        "authorized": False,
        "scientific_training_started": False,
        "gpu_session_started": False,
        "main_enabled": False,
        "verified_freeze_sha256": freeze["freeze_sha256"],
        "budget": budget,
        "next_authorization": decision.as_dict(),
        "required_learning_context": "implementation.src.train.gpu2_training_session",
        "missing_interfaces": list(MISSING_PRODUCTION_INTERFACES),
        "next_action": "Implement and CPU-test every listed orchestration interface before enabling a production pilot command.",
    }


__all__ = [
    "BASE_GATES",
    "H_DIAGNOSTIC_STEPS",
    "LEXICAL_DIAGNOSTIC_STEPS",
    "MISSING_PRODUCTION_INTERFACES",
    "PILOT_STATE_SCHEMA",
    "PLANNED_ROOTS",
    "PilotAction",
    "PilotDecision",
    "derive_pilot_decision",
    "run_production_pilot",
]
