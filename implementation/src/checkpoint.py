"""Durable, full-state checkpoints with strict resume verification."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .artifacts import canonical_json_bytes, publish_bytes_once, publish_json_once, read_verified_json, sha256_file
from .contracts import ContractViolation, PROJECT_ROOT, require_sha256


CHECKPOINT_SCHEMA_VERSION = "v4-full-state-1"
REQUIRED_PROGRESS = {
    "global_step",
    "phase_step",
    "loader_state",
    "accumulation_step",
    "examples_seen",
    "model_tokens_seen",
    "plan_sha256",
}
REQUIRED_LINEAGE = {
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
    "evaluation_plan_sha256",
    "data_kind",
}
REQUIRED_PAYLOAD = {
    "schema_version",
    "model",
    "optimizer",
    "optimizer_class",
    "scheduler",
    "scaler",
    "rng",
    "progress",
    "lineage",
    "parameter_group_signature",
    "deterministic_settings",
    "semantic_fingerprints",
}


@dataclass(frozen=True)
class CheckpointRef:
    path: str
    state_sha256: str
    manifest_sha256: str
    model_fingerprint: str
    optimizer_fingerprint: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResumeContract:
    root_id: int
    phase: str
    stage: str
    branch: str | None
    freeze_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    code_sha256: str
    evaluation_plan_sha256: str
    plan_sha256: str
    data_kind: str
    require_cuda_rng: bool
    initial_model_fingerprint: str | None = None
    phase_parent_sha256: str | None = None
    resume_parent_sha256: str | None = None


def semantic_fingerprint(value: Any) -> str:
    """Hash nested state by type, shape, dtype, and exact tensor bytes."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"torch:")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy:")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.reshape(-1).view(np.uint8).tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, tuple):
            digest.update(b"tuple[")
            for nested in item:
                visit(nested)
            digest.update(b"]")
        elif isinstance(item, list):
            digest.update(b"list[")
            for nested in item:
                visit(nested)
            digest.update(b"]")
        elif is_dataclass(item):
            digest.update(type(item).__qualname__.encode("utf-8"))
            visit(asdict(item))
        else:
            digest.update(type(item).__qualname__.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def _model_fingerprint(model: Any) -> str:
    # Scoring and checkpoints intentionally share one exact algorithm.
    from .score import semantic_model_fingerprint

    return semantic_model_fingerprint(model)


def _model_state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    """The state-dict form of score.semantic_model_fingerprint."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(dtype=torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _optimizer_class_name(optimizer: torch.optim.Optimizer) -> str:
    kind = type(optimizer)
    return f"{kind.__module__}.{kind.__qualname__}"


def _validate_optimizer_parameter_groups(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    signature: Mapping[str, Any],
) -> None:
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    expected_groups: list[tuple[str, ...]] = []
    try:
        for names in signature.values():
            normalized = tuple(str(name) for name in names)
            if not normalized or len(set(normalized)) != len(normalized):
                raise ContractViolation("INVALID_PARAMETER_GROUP_SIGNATURE")
            expected_groups.append(tuple(sorted(normalized)))
    except (TypeError, AttributeError) as exc:
        raise ContractViolation("INVALID_PARAMETER_GROUP_SIGNATURE") from exc
    actual_groups: list[tuple[str, ...]] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group.get("params", ()):  # state IDs never substitute here
            identity = id(parameter)
            if identity in seen or identity not in name_by_id:
                raise ContractViolation("OPTIMIZER_PARAMETER_GROUP_MISMATCH")
            seen.add(identity)
            names.append(name_by_id[identity])
        if not names:
            raise ContractViolation("OPTIMIZER_PARAMETER_GROUP_MISMATCH")
        actual_groups.append(tuple(sorted(names)))
    if sorted(actual_groups) != sorted(expected_groups) or seen != set(name_by_id):
        raise ContractViolation("OPTIMIZER_PARAMETER_GROUP_MISMATCH")


def _require_literal_gpu2_mask() -> None:
    # Lazy import keeps checkpoint's CPU path independent of the CUDA guard.
    from .gpu_guard import require_literal_gpu2_mask

    require_literal_gpu2_mask()


def _validate_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    if not REQUIRED_LINEAGE.issubset(lineage):
        raise ContractViolation("INCOMPLETE_CHECKPOINT_LINEAGE")
    value = copy.deepcopy(dict(lineage))
    if int(value["root_id"]) not in {4101, 4102, 4103, 4104}:
        raise ContractViolation("INVALID_ROOT_ID")
    if value["phase"] not in {"INIT", "CORPUS", "T0", "T1", "T2", "T3", "H_BASE", "H_A", "H_B", "GPU_REPLAY"}:
        raise ContractViolation("INVALID_CHECKPOINT_PHASE")
    if value["branch"] not in {None, "A", "B"}:
        raise ContractViolation("INVALID_CHECKPOINT_BRANCH")
    for key in (
        "initial_model_fingerprint",
        "freeze_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "code_sha256",
        "evaluation_plan_sha256",
    ):
        require_sha256(str(value[key]), "INVALID_LINEAGE_HASH")
    for key in ("phase_parent_sha256", "resume_parent_sha256"):
        if value[key] is not None:
            require_sha256(str(value[key]), "INVALID_PARENT_HASH")
    if value["data_kind"] not in {"REAL", "SYNTHETIC_TEST_FIXTURE"}:
        raise ContractViolation("INVALID_DATA_KIND")
    if value["data_kind"] == "REAL" and value.get("freeze_gate_status") != "PASS":
        raise ContractViolation("REAL_DATA_FREEZE_GATE_NOT_PASS")
    if value["data_kind"] == "REAL":
        binding = value.get("gpu_binding")
        try:
            gpu_policy = json.loads(
                (PROJECT_ROOT / "implementation/config/campaign_policy.json").read_text(encoding="utf-8")
            )["gpu"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ContractViolation("BLOCKED_GPU_POLICY") from exc
        if (
            not isinstance(binding, Mapping)
            or binding.get("physical_index") != 2
            or binding.get("logical_index") != 0
            or binding.get("uuid") != gpu_policy.get("expected_uuid")
        ):
            raise ContractViolation("REAL_CHECKPOINT_NOT_BOUND_TO_GPU2")
        if not isinstance(value.get("budget_lease_id"), str) or not value["budget_lease_id"]:
            raise ContractViolation("REAL_CHECKPOINT_MISSING_BUDGET_LEASE")
        runtime = value.get("runtime")
        if not isinstance(runtime, Mapping) or not all(
            runtime.get(key) for key in ("python", "torch", "transformers", "torch_cuda", "precision")
        ):
            raise ContractViolation("REAL_CHECKPOINT_MISSING_RUNTIME")
    expected_stage_branch = {
        "INIT": ("INIT", None),
        "CORPUS": ("CORPUS", None),
        "T0": ("T0", None),
        "T1": ("T1", None),
        "T2": ("T2", None),
        "T3": ("T3", None),
        "H_BASE": ("H", None),
        "H_A": ("H", "A"),
        "H_B": ("H", "B"),
        "GPU_REPLAY": ("GPU_REPLAY", None),
    }[value["phase"]]
    if (value["stage"], value["branch"]) != expected_stage_branch:
        raise ContractViolation("PHASE_STAGE_BRANCH_MISMATCH")
    if value["phase"] in {"INIT", "GPU_REPLAY"}:
        if value["phase_parent_sha256"] is not None or value["resume_parent_sha256"] is not None:
            raise ContractViolation("INIT_MUST_NOT_HAVE_PARENT")
    elif value["phase_parent_sha256"] is None:
        raise ContractViolation("NONINIT_REQUIRES_PHASE_PARENT")
    return value


def _validate_progress(progress: Mapping[str, Any]) -> dict[str, Any]:
    if not REQUIRED_PROGRESS.issubset(progress):
        raise ContractViolation("INCOMPLETE_CHECKPOINT_PROGRESS")
    value = copy.deepcopy(dict(progress))
    for key in ("global_step", "phase_step", "accumulation_step", "examples_seen", "model_tokens_seen"):
        if not isinstance(value[key], int) or value[key] < 0:
            raise ContractViolation("INVALID_CHECKPOINT_PROGRESS")
    if value["accumulation_step"] != 0:
        raise ContractViolation("SAVE_ONLY_AT_ACCUMULATION_BOUNDARY")
    if not isinstance(value["loader_state"], Mapping):
        raise ContractViolation("INVALID_LOADER_STATE")
    require_sha256(str(value["plan_sha256"]), "INVALID_PLAN_HASH")
    return value


def capture_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any,
    scaler: Any | None,
    generators: Mapping[str, torch.Generator],
    progress: Mapping[str, Any],
    lineage: Mapping[str, Any],
    parameter_group_signature: Mapping[str, Any],
    deterministic_settings: Mapping[str, Any],
    require_cuda_rng: bool,
) -> dict[str, Any]:
    if optimizer is None:
        raise ContractViolation("OPTIMIZER_STATE_REQUIRED")
    if scheduler is None or not hasattr(scheduler, "state_dict"):
        raise ContractViolation("SCHEDULER_STATE_REQUIRED")
    if "loader" not in generators or any(not hasattr(generator, "get_state") for generator in generators.values()):
        raise ContractViolation("LOADER_GENERATOR_STATE_REQUIRED")
    checked_progress = _validate_progress(progress)
    checked_lineage = _validate_lineage(lineage)
    _validate_optimizer_parameter_groups(model, optimizer, parameter_group_signature)
    model_fingerprint = _model_fingerprint(model)
    if checked_lineage.get("model_fingerprint") not in {None, model_fingerprint}:
        raise ContractViolation("LINEAGE_MODEL_FINGERPRINT_MISMATCH")
    checked_lineage["model_fingerprint"] = model_fingerprint
    if checked_lineage["data_kind"] == "REAL":
        required_determinism = {
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
        }
        if any(deterministic_settings.get(key) != value for key, value in required_determinism.items()):
            raise ContractViolation("REAL_CHECKPOINT_NONDETERMINISTIC_SETTINGS")
    cuda_states: list[torch.Tensor] = []
    if require_cuda_rng:
        _require_literal_gpu2_mask()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ContractViolation("EXACTLY_ONE_CUDA_RNG_REQUIRED")
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
        if len(cuda_states) != 1:
            raise ContractViolation("EXACTLY_ONE_CUDA_RNG_REQUIRED")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "optimizer_class": _optimizer_class_name(optimizer),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": copy.deepcopy(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": cuda_states,
            "generators": {
                name: generator.get_state().clone()
                for name, generator in sorted(generators.items())
            },
        },
        "progress": checked_progress,
        "lineage": checked_lineage,
        "parameter_group_signature": copy.deepcopy(dict(parameter_group_signature)),
        "deterministic_settings": copy.deepcopy(dict(deterministic_settings)),
    }
    payload["semantic_fingerprints"] = {
        "model": model_fingerprint,
        "optimizer": semantic_fingerprint(payload["optimizer"]),
        "scheduler": semantic_fingerprint(payload["scheduler"]),
        "scaler": semantic_fingerprint(payload["scaler"]),
        "rng": semantic_fingerprint(payload["rng"]),
        "progress": semantic_fingerprint(payload["progress"]),
    }
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(directory: Path, payload: Mapping[str, Any]) -> CheckpointRef:
    """Publish state, manifest, then a final commit marker; never overwrite."""
    destination = Path(directory)
    if destination.is_symlink() or destination.exists():
        raise ContractViolation("CHECKPOINT_EXISTS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ContractViolation("CHECKPOINT_EXISTS") from exc
    state_path = destination / "state.pt"
    try:
        with state_path.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(state_path, 0o444)
        state_hash = sha256_file(state_path)
        state_bytes = state_path.stat().st_size
        fingerprints = payload.get("semantic_fingerprints")
        if not isinstance(fingerprints, Mapping):
            raise ContractViolation("MISSING_SEMANTIC_FINGERPRINTS")
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_file": "state.pt",
            "state_sha256": state_hash,
            "state_bytes": state_bytes,
            "model_fingerprint": fingerprints["model"],
            "optimizer_fingerprint": fingerprints["optimizer"],
            "optimizer_class": payload["optimizer_class"],
            "scheduler_fingerprint": fingerprints["scheduler"],
            "scaler_fingerprint": fingerprints["scaler"],
            "rng_fingerprint": fingerprints["rng"],
            "progress_fingerprint": fingerprints["progress"],
            "lineage": copy.deepcopy(payload["lineage"]),
            "progress": copy.deepcopy(payload["progress"]),
        }
        manifest_ref = publish_json_once(destination / "manifest.json", manifest)
        commit_content = (str(manifest_ref["sha256"]) + "\n").encode("ascii")
        publish_bytes_once(destination / "COMMITTED", commit_content, mode=0o444)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        return CheckpointRef(
            path=str(destination),
            state_sha256=state_hash,
            manifest_sha256=str(manifest_ref["sha256"]),
            model_fingerprint=str(fingerprints["model"]),
            optimizer_fingerprint=str(fingerprints["optimizer"]),
            bytes=state_bytes,
        )
    except Exception:
        # An incomplete directory is intentionally retained and lacks COMMITTED.
        # It can neither masquerade as a checkpoint nor overwrite a later run.
        raise


def load_checkpoint_payload(
    directory: Path,
    *,
    expected_state_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = Path(directory)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ContractViolation("CHECKPOINT_NOT_COMMITTED")
    commit_path = candidate / "COMMITTED"
    manifest_path = candidate / "manifest.json"
    state_path = candidate / "state.pt"
    if any(path.is_symlink() or not path.is_file() for path in (commit_path, manifest_path, state_path)):
        raise ContractViolation("CHECKPOINT_NOT_COMMITTED")
    try:
        committed_manifest_hash = commit_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractViolation("CHECKPOINT_NOT_COMMITTED") from exc
    require_sha256(committed_manifest_hash, "INVALID_CHECKPOINT_COMMIT")
    if expected_manifest_sha256 is not None and committed_manifest_hash != expected_manifest_sha256:
        raise ContractViolation("CHECKPOINT_MANIFEST_HASH_MISMATCH")
    manifest = read_verified_json(manifest_path, expected_sha256=committed_manifest_hash)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ContractViolation("CHECKPOINT_SCHEMA_MISMATCH")
    actual_state_hash = sha256_file(state_path)
    required_state_hash = expected_state_sha256 or manifest.get("state_sha256")
    if actual_state_hash != required_state_hash or actual_state_hash != manifest.get("state_sha256"):
        raise ContractViolation("CHECKPOINT_STATE_HASH_MISMATCH")
    try:
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ContractViolation("CHECKPOINT_STATE_UNREADABLE") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ContractViolation("CHECKPOINT_SCHEMA_MISMATCH")
    if not REQUIRED_PAYLOAD.issubset(payload):
        raise ContractViolation("INCOMPLETE_CHECKPOINT_STATE")
    fingerprints = payload.get("semantic_fingerprints", {})
    if _model_state_fingerprint(payload.get("model", {})) != manifest.get("model_fingerprint"):
        raise ContractViolation("CHECKPOINT_MODEL_HASH_MISMATCH")
    if fingerprints.get("model") != manifest.get("model_fingerprint"):
        raise ContractViolation("CHECKPOINT_MODEL_HASH_MISMATCH")
    if payload.get("optimizer_class") != manifest.get("optimizer_class"):
        raise ContractViolation("CHECKPOINT_OPTIMIZER_CLASS_MISMATCH")
    checks = {
        "optimizer": "optimizer_fingerprint",
        "scheduler": "scheduler_fingerprint",
        "scaler": "scaler_fingerprint",
        "rng": "rng_fingerprint",
        "progress": "progress_fingerprint",
    }
    for payload_key, manifest_key in checks.items():
        if semantic_fingerprint(payload[payload_key]) != manifest[manifest_key]:
            raise ContractViolation("CHECKPOINT_SEMANTIC_HASH_MISMATCH")
        if fingerprints.get(payload_key) != manifest[manifest_key]:
            raise ContractViolation("CHECKPOINT_SEMANTIC_HASH_MISMATCH")
    if payload.get("lineage") != manifest.get("lineage") or payload.get("progress") != manifest.get("progress"):
        raise ContractViolation("CHECKPOINT_MANIFEST_PAYLOAD_MISMATCH")
    return payload, manifest


def _validate_resume_contract(payload: Mapping[str, Any], expected: ResumeContract) -> None:
    lineage = _validate_lineage(payload["lineage"])
    progress = _validate_progress(payload["progress"])
    comparisons = {
        "root_id": expected.root_id,
        "phase": expected.phase,
        "stage": expected.stage,
        "branch": expected.branch,
        "freeze_sha256": expected.freeze_sha256,
        "config_sha256": expected.config_sha256,
        "tokenizer_sha256": expected.tokenizer_sha256,
        "code_sha256": expected.code_sha256,
        "evaluation_plan_sha256": expected.evaluation_plan_sha256,
        "data_kind": expected.data_kind,
    }
    if any(lineage.get(key) != value for key, value in comparisons.items()):
        raise ContractViolation("CHECKPOINT_RESUME_CONTRACT_MISMATCH")
    if progress["plan_sha256"] != expected.plan_sha256:
        raise ContractViolation("CHECKPOINT_LOADER_PLAN_MISMATCH")
    if lineage.get("phase_parent_sha256") != expected.phase_parent_sha256:
        raise ContractViolation("CHECKPOINT_PARENT_OR_INIT_MISMATCH")
    if lineage.get("resume_parent_sha256") != expected.resume_parent_sha256:
        raise ContractViolation("CHECKPOINT_PARENT_OR_INIT_MISMATCH")
    optional_comparisons = {
        "initial_model_fingerprint": expected.initial_model_fingerprint,
    }
    if any(
        value is not None and lineage.get(key) != value
        for key, value in optional_comparisons.items()
    ):
        raise ContractViolation("CHECKPOINT_PARENT_OR_INIT_MISMATCH")
    cuda_states = payload.get("rng", {}).get("torch_cuda")
    if expected.require_cuda_rng and (not isinstance(cuda_states, list) or len(cuda_states) != 1):
        raise ContractViolation("EXACTLY_ONE_CUDA_RNG_REQUIRED")
    if not expected.require_cuda_rng and cuda_states:
        raise ContractViolation("CPU_FIXTURE_CAPTURED_CUDA_STATE")


def restore_training_state(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any,
    scaler: Any | None,
    generators: Mapping[str, torch.Generator],
    expected: ResumeContract,
    parameter_group_signature: Mapping[str, Any],
    expected_deterministic_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not REQUIRED_PAYLOAD.issubset(payload):
        raise ContractViolation("INCOMPLETE_CHECKPOINT_STATE")
    if optimizer is None:
        raise ContractViolation("OPTIMIZER_STATE_REQUIRED")
    if scheduler is None or not hasattr(scheduler, "load_state_dict"):
        raise ContractViolation("SCHEDULER_STATE_REQUIRED")
    _validate_resume_contract(payload, expected)
    if _optimizer_class_name(optimizer) != payload.get("optimizer_class"):
        raise ContractViolation("OPTIMIZER_CLASS_MISMATCH")
    _validate_optimizer_parameter_groups(model, optimizer, parameter_group_signature)
    if dict(payload["parameter_group_signature"]) != dict(parameter_group_signature):
        raise ContractViolation("OPTIMIZER_PARAMETER_GROUP_MISMATCH")
    if (
        expected_deterministic_settings is not None
        and dict(payload["deterministic_settings"]) != dict(expected_deterministic_settings)
    ):
        raise ContractViolation("DETERMINISTIC_SETTINGS_MISMATCH")
    if set(generators) != set(payload["rng"]["generators"]):
        raise ContractViolation("GENERATOR_SET_MISMATCH")
    if (scaler is None) != (payload["scaler"] is None):
        raise ContractViolation("SCALER_MISMATCH")

    model.load_state_dict(copy.deepcopy(payload["model"]), strict=True)
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    scheduler.load_state_dict(copy.deepcopy(payload["scheduler"]))
    if scaler is not None:
        scaler.load_state_dict(copy.deepcopy(payload["scaler"]))
    if _model_fingerprint(model) != payload["semantic_fingerprints"]["model"]:
        raise ContractViolation("RESTORED_MODEL_FINGERPRINT_MISMATCH")
    if semantic_fingerprint(optimizer.state_dict()) != payload["semantic_fingerprints"]["optimizer"]:
        raise ContractViolation("RESTORED_OPTIMIZER_FINGERPRINT_MISMATCH")
    if semantic_fingerprint(scheduler.state_dict()) != payload["semantic_fingerprints"]["scheduler"]:
        raise ContractViolation("RESTORED_SCHEDULER_FINGERPRINT_MISMATCH")
    if scaler is not None and semantic_fingerprint(scaler.state_dict()) != payload["semantic_fingerprints"]["scaler"]:
        raise ContractViolation("RESTORED_SCALER_FINGERPRINT_MISMATCH")

    # RNG is restored last: constructing/loading objects is allowed to consume
    # randomness, but the next training record must see the captured state.
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(copy.deepcopy(rng["numpy"]))
    torch.set_rng_state(rng["torch_cpu"].clone())
    cuda_states = rng["torch_cuda"]
    if cuda_states:
        _require_literal_gpu2_mask()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ContractViolation("CUDA_STATE_WITHOUT_EXACT_GPU2_VISIBILITY")
        torch.cuda.set_rng_state_all([state.clone() for state in cuda_states])
    for name, generator in generators.items():
        generator.set_state(rng["generators"][name].clone())
    return copy.deepcopy(payload["progress"])


def checkpoint_state_fingerprint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    progress: Mapping[str, Any],
    *,
    generators: Mapping[str, torch.Generator] | None = None,
    include_cuda_rng: bool = False,
) -> dict[str, str]:
    result = {
        "model": _model_fingerprint(model),
        "optimizer": semantic_fingerprint(optimizer.state_dict()),
        "scheduler": semantic_fingerprint(scheduler.state_dict()),
        "progress": semantic_fingerprint(dict(progress)),
    }
    if generators is not None:
        if include_cuda_rng:
            _require_literal_gpu2_mask()
        rng = {
            "python": random.getstate(),
            "numpy": copy.deepcopy(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if include_cuda_rng
                else []
            ),
            "generators": {
                name: generator.get_state().clone()
                for name, generator in sorted(generators.items())
            },
        }
        result["rng"] = semantic_fingerprint(rng)
    return result
