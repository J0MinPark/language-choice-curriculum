"""Fixed-endpoint training engine shared by CPU fixtures and GPU-2 runs.

The public production entry path acquires a budget lease and the project GPU
lock before proving physical GPU 2.  CPU execution is exposed only through the
explicit synthetic-fixture helper used by integration tests.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import random
import signal
import hashlib
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from .budget import (
    BudgetLease,
    BudgetStop,
    ExclusiveGpu2Lock,
    GpuBudgetLedger,
    require_disk_reservation,
)
from .checkpoint import (
    ResumeContract,
    capture_training_state,
    checkpoint_state_fingerprint,
    load_checkpoint_payload,
    restore_training_state,
    save_checkpoint,
    semantic_fingerprint,
)
from .contracts import ContractViolation, PROJECT_ROOT, canonical_json_bytes, load_json, require_sha256
from .gpu_guard import (
    GpuBinding,
    assert_model_on_bound_device,
    require_literal_gpu2_mask,
    require_physical_gpu2,
)
from .model import (
    ConstantSchedule,
    build_adamw,
    build_random_model,
    configure_determinism,
    corpus_causal_loss,
    lexical_response_loss,
    seed_all,
)
from .records import EncodedRecord


PHASE_PARENT = {
    "CORPUS": "INIT",
    "T0": "CORPUS",
    "T1": "T0",
    "T2": "T1",
    "T3": "T2",
    "H_BASE": "T3",
    "H_A": "H_BASE",
    "H_B": "H_BASE",
}
BASE_TRAINING_GATES = (
    "DATA_QA",
    "EXPERIMENT_FREEZE",
    "CPU_INTEGRATION",
    "GPU_REPLAY",
)


@dataclass(frozen=True)
class TrainBatch:
    record_ids: tuple[str, ...]
    content_sha256s: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor | None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ContractViolation("INVALID_TRAIN_BATCH")
        if len(self.record_ids) != self.input_ids.shape[0] or len(self.content_sha256s) != len(self.record_ids):
            raise ContractViolation("INVALID_TRAIN_BATCH")
        if self.response_mask is not None and self.response_mask.shape != self.input_ids.shape:
            raise ContractViolation("INVALID_TRAIN_BATCH")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ContractViolation("DUPLICATE_RECORD_IN_BATCH")


@dataclass(frozen=True)
class StepTrace:
    global_step: int
    phase_step: int
    record_ids: tuple[str, ...]
    content_sha256s: tuple[str, ...]
    loss_hex: str
    learning_rates: tuple[float, ...]


@dataclass(frozen=True)
class GpuTrainingSession:
    binding: GpuBinding
    lease: BudgetLease
    ledger: GpuBudgetLedger
    experiment_freeze: Mapping[str, Any]


class BoundaryStopController:
    """Signal handler that requests a stop only after the current update."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None
        self._previous: dict[int, Any] = {}

    def _handle(self, signal_number: int, frame: Any) -> None:
        del frame
        self.requested = True
        self.signal_number = signal_number

    def __enter__(self) -> "BoundaryStopController":
        for value in (signal.SIGTERM, signal.SIGINT):
            self._previous[value] = signal.getsignal(value)
            signal.signal(value, self._handle)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for value, previous in self._previous.items():
            signal.signal(value, previous)


class StatefulBatchStream:
    """A pre-materialized, worker-free plan whose cursor is checkpointable."""

    def __init__(
        self,
        batches: Sequence[TrainBatch],
        *,
        plan_sha256: str,
        generator: torch.Generator,
        cursor: int = 0,
    ) -> None:
        if len(plan_sha256) != 64 or not batches or cursor < 0 or cursor > len(batches):
            raise ContractViolation("INVALID_BATCH_STREAM")
        self.batches = tuple(batches)
        self.plan_sha256 = plan_sha256
        self.generator = generator
        self.cursor = cursor

    def __len__(self) -> int:
        return len(self.batches)

    def next_batch(self) -> TrainBatch:
        if self.cursor >= len(self.batches):
            raise ContractViolation("BATCH_STREAM_EXHAUSTED")
        batch = self.batches[self.cursor]
        self.cursor += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        next_ids = () if self.cursor >= len(self.batches) else self.batches[self.cursor].record_ids
        return {
            "kind": "PREMATERIALIZED_NO_WORKERS",
            "plan_sha256": self.plan_sha256,
            "cursor": self.cursor,
            "total_batches": len(self.batches),
            "next_record_ids": list(next_ids),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("kind") != "PREMATERIALIZED_NO_WORKERS":
            raise ContractViolation("LOADER_KIND_MISMATCH")
        if state.get("plan_sha256") != self.plan_sha256 or int(state.get("total_batches", -1)) != len(self.batches):
            raise ContractViolation("LOADER_PLAN_MISMATCH")
        cursor = int(state.get("cursor", -1))
        if cursor < 0 or cursor > len(self.batches):
            raise ContractViolation("INVALID_LOADER_CURSOR")
        expected_next = () if cursor == len(self.batches) else self.batches[cursor].record_ids
        if tuple(state.get("next_record_ids", ())) != expected_next:
            raise ContractViolation("LOADER_NEXT_RECORD_MISMATCH")
        self.cursor = cursor


def collate_encoded_records(
    records: Sequence[EncodedRecord],
    *,
    pad_token_id: int,
) -> TrainBatch:
    if not records:
        raise ContractViolation("EMPTY_TRAIN_BATCH")
    maximum = max(len(record.input_ids) for record in records)
    if maximum < 2:
        raise ContractViolation("TRAIN_SEQUENCE_TOO_SHORT")
    input_ids = torch.full((len(records), maximum), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros_like(input_ids, dtype=torch.bool)
    response = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, record in enumerate(records):
        length = len(record.input_ids)
        input_ids[index, :length] = torch.tensor(record.input_ids, dtype=torch.long)
        attention[index, :length] = True
        response[index, :length] = torch.tensor(record.response_mask, dtype=torch.bool)
    return TrainBatch(
        record_ids=tuple(record.record_id for record in records),
        content_sha256s=tuple(record.content_sha256 for record in records),
        input_ids=input_ids,
        attention_mask=attention,
        response_mask=response,
    )


def corpus_batch(
    input_ids: torch.Tensor,
    *,
    record_ids: Sequence[str],
    content_sha256s: Sequence[str],
    attention_mask: torch.Tensor | None = None,
) -> TrainBatch:
    attention = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    return TrainBatch(
        tuple(record_ids),
        tuple(content_sha256s),
        input_ids.long(),
        attention,
        None,
    )


def validate_phase_transition(
    *,
    child_phase: str,
    child_root_id: int,
    parent_manifest: Mapping[str, Any] | None,
    resume: bool,
) -> None:
    if child_phase == "INIT":
        if parent_manifest is not None or resume:
            raise ContractViolation("INIT_MUST_NOT_HAVE_PARENT")
        return
    if child_phase not in PHASE_PARENT or parent_manifest is None:
        raise ContractViolation("MISSING_OR_INVALID_PHASE_PARENT")
    lineage = parent_manifest.get("lineage")
    if not isinstance(lineage, Mapping) or int(lineage.get("root_id", -1)) != child_root_id:
        raise ContractViolation("CROSS_ROOT_LINEAGE")
    expected_phase = child_phase if resume else PHASE_PARENT[child_phase]
    if lineage.get("phase") != expected_phase:
        raise ContractViolation("INVALID_PHASE_TRANSITION")
    if resume:
        expected_branch = {"H_A": "A", "H_B": "B"}.get(child_phase)
        if lineage.get("branch") != expected_branch:
            raise ContractViolation("INVALID_RESUME_BRANCH")
    elif child_phase in {"H_A", "H_B"} and lineage.get("branch") is not None:
        raise ContractViolation("INVALID_HISTORY_FORK_PARENT")


def require_campaign_training_gates(
    *,
    root_id: int,
    phase: str,
    gates: Mapping[str, str],
    main_enabled: bool,
) -> None:
    """Enforce first-root and all-root expansion gates without averaging."""
    if main_enabled:
        raise ContractViolation("MAIN_NOT_AUTHORIZED")
    if root_id not in {4101, 4102, 4103, 4104} or phase not in set(PHASE_PARENT) | {"INIT"}:
        raise ContractViolation("INVALID_TRAINING_ACTION")
    required = list(BASE_TRAINING_GATES)
    if root_id != 4101:
        required.extend(("ROOT_4101_T3_READINESS", "ROOT_4101_T3_MEASUREMENT"))
    if phase in {"H_BASE", "H_A", "H_B"}:
        for planned_root in (4101, 4102, 4103, 4104):
            required.extend(
                (
                    f"ROOT_{planned_root}_T3_READINESS",
                    f"ROOT_{planned_root}_T3_MEASUREMENT",
                )
            )
    failed = [name for name in required if gates.get(name) != "PASS"]
    if failed:
        raise ContractViolation("BLOCKED_TRAINING_GATES:" + ",".join(sorted(set(failed))))


def audit_initialization_fingerprints(
    fingerprints: Mapping[int, str],
    *,
    require_all_roots: bool,
) -> dict[str, Any]:
    planned = {4101, 4102, 4103, 4104}
    roots = set(fingerprints)
    if not roots or not roots.issubset(planned) or (require_all_roots and roots != planned):
        raise ContractViolation("INVALID_INITIALIZATION_ROOT_SET")
    values = list(fingerprints.values())
    if any(len(value) != 64 for value in values) or len(set(values)) != len(values):
        raise ContractViolation("ROOT_INITIALIZATION_NOT_DISTINCT")
    return {
        "status": "PASS",
        "roots": sorted(roots),
        "distinct_initialization_fingerprints": len(set(values)),
    }
def _capture_rng(
    generators: Mapping[str, torch.Generator],
    *,
    include_cuda: bool,
) -> dict[str, Any]:
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if include_cuda else [],
        "generators": {name: generator.get_state().clone() for name, generator in generators.items()},
    }


def _restore_rng(state: Mapping[str, Any], generators: Mapping[str, torch.Generator]) -> None:
    random.setstate(state["python"])
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch"].clone())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.clone() for value in state["cuda"]])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name].clone())


def evaluate_without_rng_consumption(
    callback: Callable[[], Any],
    *,
    generators: Mapping[str, torch.Generator],
    include_cuda: bool = False,
) -> Any:
    before = _capture_rng(generators, include_cuda=include_cuda)
    before_hash = semantic_fingerprint(before)
    try:
        result = callback()
    except Exception:
        _restore_rng(before, generators)
        raise
    after = _capture_rng(generators, include_cuda=include_cuda)
    if semantic_fingerprint(after) != before_hash:
        _restore_rng(before, generators)
        raise ContractViolation("EVALUATION_CONSUMED_TRAINING_RNG")
    return result


def _model_device(model: torch.nn.Module) -> torch.device:
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise ContractViolation("MODEL_ON_MULTIPLE_DEVICES")
    return next(iter(devices))


def _run_optimizer_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ConstantSchedule,
    stream: StatefulBatchStream,
    progress: dict[str, Any],
    *,
    steps: int,
    loss_kind: str,
    grad_clip_norm: float,
    autocast_bfloat16: bool,
    evaluation_interval: int | None = None,
    evaluation_callback: Callable[[int], Any] | None = None,
    budget_check: Callable[[], Any] | None = None,
    stop_controller: BoundaryStopController | None = None,
    boundary_checkpoint: Callable[[str], Any] | None = None,
) -> tuple[list[StepTrace], str]:
    """Run exact optimizer updates; this internal engine never selects a device."""
    if steps < 0 or not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0:
        raise ContractViolation("INVALID_TRAINING_LOOP_ARGUMENT")
    if loss_kind not in {"corpus", "lexical"}:
        raise ContractViolation("UNKNOWN_LOSS_KIND")
    if int(progress.get("accumulation_step", -1)) != 0:
        raise ContractViolation("TRAINING_NOT_AT_ACCUMULATION_BOUNDARY")
    if progress.get("plan_sha256") != stream.plan_sha256:
        raise ContractViolation("TRAINING_PLAN_MISMATCH")
    if progress.get("loader_state") != stream.state_dict():
        raise ContractViolation("TRAINING_LOADER_STATE_MISMATCH")
    device = _model_device(model)
    if autocast_bfloat16 and device.type != "cuda":
        raise ContractViolation("BF16_PRODUCTION_REQUIRES_CUDA")
    traces: list[StepTrace] = []
    model.train()

    def check_budget() -> None:
        if budget_check is None:
            return
        try:
            budget_check()
        except BudgetStop as exc:
            if boundary_checkpoint is not None:
                boundary_checkpoint(exc.code)
            raise

    for _ in range(steps):
        check_budget()
        batch = stream.next_batch()
        input_ids = batch.input_ids.to(device=device, non_blocking=False)
        attention = batch.attention_mask.to(device=device, non_blocking=False)
        response = batch.response_mask.to(device=device, non_blocking=False) if batch.response_mask is not None else None
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast_bfloat16
            else contextlib.nullcontext()
        )
        with context:
            output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
            if loss_kind == "lexical":
                if response is None:
                    raise ContractViolation("LEXICAL_BATCH_MISSING_RESPONSE_MASK")
                loss = lexical_response_loss(output.logits, input_ids, response, attention)
            else:
                loss = corpus_causal_loss(output.logits, input_ids, attention)
        if not bool(torch.isfinite(loss).item()):
            raise ContractViolation("NONFINITE_TRAINING_LOSS")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if not bool(torch.isfinite(norm).item()):
            raise ContractViolation("NONFINITE_GRADIENT_NORM")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        progress["global_step"] = int(progress["global_step"]) + 1
        progress["phase_step"] = int(progress["phase_step"]) + 1
        progress["examples_seen"] = int(progress["examples_seen"]) + len(batch.record_ids)
        progress["model_tokens_seen"] = int(progress["model_tokens_seen"]) + int(attention.sum().item())
        progress["loader_state"] = stream.state_dict()
        traces.append(
            StepTrace(
                global_step=progress["global_step"],
                phase_step=progress["phase_step"],
                record_ids=batch.record_ids,
                content_sha256s=batch.content_sha256s,
                loss_hex=float(loss.detach().cpu()).hex(),
                learning_rates=tuple(float(group["lr"]) for group in optimizer.param_groups),
            )
        )
        check_budget()
        if evaluation_interval and evaluation_callback and progress["phase_step"] % evaluation_interval == 0:
            evaluate_without_rng_consumption(
                lambda: evaluation_callback(progress["phase_step"]),
                generators={"loader": stream.generator},
                include_cuda=device.type == "cuda",
            )
        if stop_controller is not None and stop_controller.requested:
            if boundary_checkpoint is None:
                raise ContractViolation("INTERRUPT_WITHOUT_CHECKPOINT_CALLBACK")
            boundary_checkpoint("INTERRUPTED_AT_BOUNDARY")
            return traces, "INTERRUPTED_AT_BOUNDARY"
    return traces, "FIXED_ENDPOINT_REACHED"


def run_cpu_fixture_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ConstantSchedule,
    stream: StatefulBatchStream,
    progress: dict[str, Any],
    **kwargs: Any,
) -> tuple[list[StepTrace], str]:
    if _model_device(model).type != "cpu":
        raise ContractViolation("CPU_FIXTURE_MODEL_NOT_CPU")
    if kwargs.pop("data_kind", None) != "SYNTHETIC_TEST_FIXTURE":
        raise ContractViolation("CPU_PATH_REQUIRES_SYNTHETIC_FIXTURE")
    return _run_optimizer_steps(
        model,
        optimizer,
        scheduler,
        stream,
        progress,
        autocast_bfloat16=False,
        **kwargs,
    )


@contextlib.contextmanager
def gpu2_training_session(
    *,
    ledger: GpuBudgetLedger,
    lock_path: Path,
    run_id: str,
    root_id: int | None,
    phase: str,
    reserved_seconds: float,
    checkpoint_grace_seconds: float,
    experiment_freeze_path: Path,
) -> Iterator[GpuTrainingSession]:
    """The only supported production CUDA session; no fallback exists."""
    # Check the literal mask at the public boundary, before data verification,
    # budget reservation, or any CUDA call.  The CLI performs the same check
    # before importing this Torch-bearing module, while this protects direct
    # library callers as well.
    require_literal_gpu2_mask()
    # This happens before budget reservation or CUDA initialization.  Loose
    # concept files and synthetic freezes can never reach production training.
    from .prepare_data import verify_experiment_freeze

    verified_freeze = verify_experiment_freeze(
        Path(experiment_freeze_path),
        production=True,
    )
    if (
        verified_freeze.get("status") != "PASS"
        or verified_freeze.get("freeze_gate_status") != "PASS"
        or verified_freeze.get("data_kind") != "REAL"
        or verified_freeze.get("verified") is not True
    ):
        raise ContractViolation("EXPERIMENT_FREEZE_GATE_NOT_PASS")
    if (
        verified_freeze.get("schema_version") in {"semantic-experiment-freeze-v4.1.2", "pilot-execution-freeze-v1"}
        and phase not in verified_freeze.get("allowed_execution_phases", [])
    ):
        raise ContractViolation("SEMANTIC_EXPERIMENT_PHASE_NOT_AUTHORIZED")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {None, ":4096:8"}:
        raise ContractViolation("BLOCKED_CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with ExclusiveGpu2Lock(lock_path):
        lease = ledger.reserve(
            run_id=run_id,
            root_id=root_id,
            phase=phase,
            reserved_seconds=reserved_seconds,
        )
        status = "FAILED_BEFORE_TRAINING"
        try:
            binding = require_physical_gpu2()
            deterministic_settings = configure_determinism()
            if deterministic_settings.get("cublas_workspace_config") != ":4096:8":
                raise ContractViolation("BLOCKED_CUBLAS_WORKSPACE_CONFIG")
            session = GpuTrainingSession(
                binding=binding,
                lease=lease,
                ledger=ledger,
                experiment_freeze=verified_freeze,
            )
            ledger.require_time_remaining(
                lease,
                checkpoint_grace_seconds=checkpoint_grace_seconds,
            )
            status = "FAILED_DURING_SESSION"
            yield session
            status = "FINISHED"
        finally:
            ledger.finish(lease, status)


def run_gpu2_stage_steps(
    session: GpuTrainingSession,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ConstantSchedule,
    stream: StatefulBatchStream,
    progress: dict[str, Any],
    *,
    checkpoint_grace_seconds: float,
    **kwargs: Any,
) -> tuple[list[StepTrace], str]:
    require_literal_gpu2_mask()
    assert_model_on_bound_device(model, session.binding)
    if (
        session.experiment_freeze.get("status") != "PASS"
        or session.experiment_freeze.get("freeze_gate_status") != "PASS"
        or session.experiment_freeze.get("data_kind") != "REAL"
        or session.experiment_freeze.get("verified") is not True
    ):
        raise ContractViolation("EXPERIMENT_FREEZE_GATE_NOT_PASS")
    return _run_optimizer_steps(
        model,
        optimizer,
        scheduler,
        stream,
        progress,
        autocast_bfloat16=True,
        budget_check=lambda: session.ledger.require_time_remaining(
            session.lease,
            checkpoint_grace_seconds=checkpoint_grace_seconds,
        ),
        **kwargs,
    )


def transition_to_history_base(
    *,
    parent_manifest: Mapping[str, Any],
    root_id: int,
    optimizer: torch.optim.Optimizer,
    scheduler: ConstantSchedule,
    history_learning_rate: float,
) -> None:
    validate_phase_transition(
        child_phase="H_BASE",
        child_root_id=root_id,
        parent_manifest=parent_manifest,
        resume=False,
    )
    scheduler.transition(phase="H_BASE", learning_rate=history_learning_rate)
    if any(float(group["lr"]) != float(history_learning_rate) for group in optimizer.param_groups):
        raise ContractViolation("HISTORY_LR_TRANSITION_FAILED")


def _corpus_replay_batches(
    tokens: np.ndarray,
    *,
    batch_examples: int,
    context_length: int,
    batch_count: int,
    token_artifact_sha256: str,
) -> tuple[list[TrainBatch], str]:
    required = batch_examples * context_length * batch_count
    if tokens.ndim != 1 or len(tokens) < required:
        raise ContractViolation("CORPUS_TOO_SMALL_FOR_GPU_REPLAY")
    batches: list[TrainBatch] = []
    plan_rows: list[dict[str, Any]] = []
    offset = 0
    for batch_index in range(batch_count):
        array = np.asarray(tokens[offset : offset + batch_examples * context_length], dtype=np.int64)
        matrix = torch.from_numpy(array.copy()).reshape(batch_examples, context_length)
        record_ids: list[str] = []
        content_hashes: list[str] = []
        for example_index in range(batch_examples):
            start = offset + example_index * context_length
            stop = start + context_length
            content_hash = hashlib.sha256(
                np.ascontiguousarray(array[example_index * context_length : (example_index + 1) * context_length])
                .view(np.uint8)
                .tobytes()
            ).hexdigest()
            record_id = f"GPU_REPLAY|{batch_index}|{example_index}|{start}|{stop}"
            record_ids.append(record_id)
            content_hashes.append(content_hash)
            plan_rows.append(
                {
                    "record_id": record_id,
                    "content_sha256": content_hash,
                    "token_start": start,
                    "token_stop": stop,
                }
            )
        batches.append(
            corpus_batch(
                matrix,
                record_ids=record_ids,
                content_sha256s=content_hashes,
            )
        )
        offset += batch_examples * context_length
    plan_payload = {
        "kind": "GPU2_PRODUCTION_SHAPED_CORPUS_REPLAY",
        "token_artifact_sha256": token_artifact_sha256,
        "batch_examples": batch_examples,
        "context_length": context_length,
        "batch_count": batch_count,
        "rows": plan_rows,
    }
    return batches, hashlib.sha256(canonical_json_bytes(plan_payload)).hexdigest()


def _replay_progress(stream: StatefulBatchStream) -> dict[str, Any]:
    return {
        "global_step": 0,
        "phase_step": 0,
        "loader_state": stream.state_dict(),
        "accumulation_step": 0,
        "examples_seen": 0,
        "model_tokens_seen": 0,
        "plan_sha256": stream.plan_sha256,
    }


def run_gpu2_replay(
    *,
    experiment_freeze_path: Path,
    checkpoint_directory: Path,
    ledger: GpuBudgetLedger,
    lock_path: Path,
    run_id: str,
    code_sha256: str,
    cpu_check_sha256: str,
    reserved_seconds: float = 1800.0,
    checkpoint_grace_seconds: float = 120.0,
) -> dict[str, Any]:
    """Production-shaped, exact-model GPU replay gate; never a study result.

    It runs two warm-up updates, saves once, then compares the uninterrupted
    next ten updates with two independent restores of that checkpoint.  Every
    optimizer step is charged and can only execute inside ``gpu2_training_session``.
    """
    require_literal_gpu2_mask()
    require_sha256(code_sha256, "INVALID_CODE_SHA256")
    require_sha256(cpu_check_sha256, "INVALID_CPU_CHECK_HASH")
    policy = load_json(PROJECT_ROOT / "implementation/config/campaign_policy.json")
    emergency = int(policy["budget"]["minimum_emergency_free_bytes"])
    # Importing these helpers does not load or mutate data.  They verify the
    # complete REAL freeze before any output directory or GPU lease is made.
    from .build_tokenizer import load_corpus_token_memmap
    from .prepare_data import verify_experiment_freeze

    preverified_freeze = verify_experiment_freeze(
        Path(experiment_freeze_path),
        production=True,
    )
    if (
        preverified_freeze.get("status") != "PASS"
        or preverified_freeze.get("freeze_gate_status") != "PASS"
        or preverified_freeze.get("data_kind") != "REAL"
        or preverified_freeze.get("verified") is not True
    ):
        raise ContractViolation("EXPERIMENT_FREEZE_GATE_NOT_PASS")
    corpus_manifest_path = Path(preverified_freeze["resolved_artifacts"]["corpus"])
    from .build_tokenizer import load_evaluation_plan

    evaluation_ref = preverified_freeze["artifacts"]["evaluation_plan"]
    evaluation_plan = load_evaluation_plan(
        Path(preverified_freeze["resolved_artifacts"]["evaluation_plan"]),
        expected_sha256=evaluation_ref["sha256"],
        expected_bytes=evaluation_ref["bytes"],
    )
    tokens, corpus_manifest = load_corpus_token_memmap(corpus_manifest_path, production=True)

    checkpoint_directory = Path(checkpoint_directory)
    if checkpoint_directory.is_symlink() or checkpoint_directory.exists():
        raise ContractViolation("CHECKPOINT_EXISTS")
    checkpoint_directory.parent.mkdir(parents=True, exist_ok=True)
    disk = require_disk_reservation(
        checkpoint_directory.parent,
        planned_bytes=2 * 1024**3,
        emergency_free_bytes=emergency,
    )
    session_value: GpuTrainingSession | None = None
    checkpoint_ref = None
    comparison: dict[str, Any] | None = None
    with gpu2_training_session(
        ledger=ledger,
        lock_path=lock_path,
        run_id=run_id,
        root_id=None,
        phase="GPU_REPLAY",
        reserved_seconds=reserved_seconds,
        checkpoint_grace_seconds=checkpoint_grace_seconds,
        experiment_freeze_path=experiment_freeze_path,
    ) as session:
        session_value = session
        freeze = session.experiment_freeze
        if freeze["freeze_sha256"] != preverified_freeze["freeze_sha256"]:
            raise ContractViolation("EXPERIMENT_FREEZE_CHANGED_BEFORE_GPU")
        training_policy = evaluation_plan["training"]
        model_policy = evaluation_plan["model"]
        if (
            training_policy["microbatch_examples"] != 32
            or training_policy["gradient_accumulation_steps"] != 1
            or model_policy["precision_candidate"] != "BF16_AUTOCAST_WITH_FP32_PARAMETERS_AND_ADAMW"
        ):
            raise ContractViolation("GPU_REPLAY_POLICY_MISMATCH")
        batches, plan_sha256 = _corpus_replay_batches(
            tokens,
            batch_examples=32,
            context_length=int(model_policy["context_length"]),
            batch_count=12,
            token_artifact_sha256=str(corpus_manifest["token_stream_sha256"]),
        )
        deterministic_settings = configure_determinism()
        replay_seed = 904101

        def fresh_objects():
            seed_all(replay_seed, include_cuda=True)
            model = build_random_model(seed=replay_seed).to("cuda:0")
            optimizer, signature = build_adamw(model, training_policy)
            scheduler = ConstantSchedule(
                optimizer,
                float(training_policy["learning_rate"]),
                "GPU_REPLAY",
            )
            generator = torch.Generator(device="cpu").manual_seed(replay_seed + 1)
            stream = StatefulBatchStream(
                batches,
                plan_sha256=plan_sha256,
                generator=generator,
            )
            return model, optimizer, scheduler, generator, stream, signature

        model, optimizer, scheduler, generator, stream, signature = fresh_objects()
        initial_model_fingerprint = __import__(
            "implementation.src.score", fromlist=["semantic_model_fingerprint"]
        ).semantic_model_fingerprint(model)
        progress = _replay_progress(stream)
        run_gpu2_stage_steps(
            session,
            model,
            optimizer,
            scheduler,
            stream,
            progress,
            checkpoint_grace_seconds=checkpoint_grace_seconds,
            steps=2,
            loss_kind="corpus",
            grad_clip_norm=float(training_policy["grad_clip_norm"]),
        )
        artifacts = freeze["artifacts"]
        lineage = {
            "root_id": 4101,
            "phase": "GPU_REPLAY",
            "stage": "GPU_REPLAY",
            "branch": None,
            "phase_parent_sha256": None,
            "resume_parent_sha256": None,
            "initial_model_fingerprint": initial_model_fingerprint,
            "freeze_sha256": freeze["freeze_sha256"],
            "config_sha256": artifacts["pilot_config"]["sha256"],
            "tokenizer_sha256": freeze["tokenizer_file_sha256"],
            "code_sha256": code_sha256,
            "cpu_check_sha256": cpu_check_sha256,
            "evaluation_plan_sha256": artifacts["evaluation_plan"]["sha256"],
            "data_kind": "REAL",
            "freeze_gate_status": "PASS",
            "gpu_binding": session.binding.as_dict(),
            "budget_lease_id": session.lease.lease_id,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "torch_cuda": torch.version.cuda,
                "precision": "BF16_AUTOCAST_WITH_FP32_PARAMETERS_AND_ADAMW",
            },
            "scientific_result": False,
        }
        payload = capture_training_state(
            model,
            optimizer,
            scheduler=scheduler,
            scaler=None,
            generators={"loader": generator},
            progress=progress,
            lineage=lineage,
            parameter_group_signature=signature,
            deterministic_settings=deterministic_settings,
            require_cuda_rng=True,
        )
        checkpoint_ref = save_checkpoint(Path(checkpoint_directory), payload)
        loaded_payload, _ = load_checkpoint_payload(
            Path(checkpoint_directory),
            expected_state_sha256=checkpoint_ref.state_sha256,
            expected_manifest_sha256=checkpoint_ref.manifest_sha256,
        )
        if loaded_payload.get("lineage", {}).get("cpu_check_sha256") != cpu_check_sha256:
            raise ContractViolation("CHECKPOINT_CPU_EVIDENCE_MISMATCH")
        del payload

        continuous_trace, _ = run_gpu2_stage_steps(
            session,
            model,
            optimizer,
            scheduler,
            stream,
            progress,
            checkpoint_grace_seconds=checkpoint_grace_seconds,
            steps=10,
            loss_kind="corpus",
            grad_clip_norm=float(training_policy["grad_clip_norm"]),
        )
        continuous_state = checkpoint_state_fingerprint(
            model,
            optimizer,
            scheduler,
            progress,
            generators={"loader": generator},
            include_cuda_rng=True,
        )
        del model, optimizer, scheduler, generator, stream
        torch.cuda.empty_cache()

        resume_contract = ResumeContract(
            root_id=4101,
            phase="GPU_REPLAY",
            stage="GPU_REPLAY",
            branch=None,
            freeze_sha256=freeze["freeze_sha256"],
            config_sha256=artifacts["pilot_config"]["sha256"],
            tokenizer_sha256=freeze["tokenizer_file_sha256"],
            code_sha256=code_sha256,
            evaluation_plan_sha256=artifacts["evaluation_plan"]["sha256"],
            plan_sha256=plan_sha256,
            data_kind="REAL",
            require_cuda_rng=True,
            initial_model_fingerprint=initial_model_fingerprint,
            phase_parent_sha256=None,
            resume_parent_sha256=None,
        )

        restored_runs = []
        for _ in range(2):
            restored_model, restored_optimizer, restored_scheduler, restored_generator, restored_stream, restored_signature = fresh_objects()
            restored_progress = restore_training_state(
                loaded_payload,
                restored_model,
                restored_optimizer,
                scheduler=restored_scheduler,
                scaler=None,
                generators={"loader": restored_generator},
                expected=resume_contract,
                parameter_group_signature=restored_signature,
                expected_deterministic_settings=deterministic_settings,
            )
            restored_stream.load_state_dict(restored_progress["loader_state"])
            trace, _ = run_gpu2_stage_steps(
                session,
                restored_model,
                restored_optimizer,
                restored_scheduler,
                restored_stream,
                restored_progress,
                checkpoint_grace_seconds=checkpoint_grace_seconds,
                steps=10,
                loss_kind="corpus",
                grad_clip_norm=float(training_policy["grad_clip_norm"]),
            )
            state = checkpoint_state_fingerprint(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                restored_progress,
                generators={"loader": restored_generator},
                include_cuda_rng=True,
            )
            restored_runs.append((trace, state))
            del restored_model, restored_optimizer, restored_scheduler, restored_generator, restored_stream
            torch.cuda.empty_cache()

        traces_equal = continuous_trace == restored_runs[0][0] == restored_runs[1][0]
        states_equal = continuous_state == restored_runs[0][1] == restored_runs[1][1]
        if not traces_equal or not states_equal:
            raise ContractViolation("BLOCKED_GPU_REPLAY_MISMATCH")
        trace_rows = [asdict(trace) for trace in continuous_trace]
        resume_trace_rows = [
            [asdict(trace) for trace in restored_trace]
            for restored_trace, _restored_state in restored_runs
        ]
        resume_state_fingerprints = [
            restored_state for _restored_trace, restored_state in restored_runs
        ]
        comparison = {
            "continuous_vs_resume_1": True,
            "resume_1_vs_resume_2": True,
            "steps_compared": 10,
            "traces": trace_rows,
            "resume_traces": resume_trace_rows,
            "loss_trace_sha256": hashlib.sha256(
                canonical_json_bytes([row["loss_hex"] for row in trace_rows])
            ).hexdigest(),
            "record_trace_sha256": hashlib.sha256(
                canonical_json_bytes(
                    [
                        {
                            "record_ids": row["record_ids"],
                            "content_sha256s": row["content_sha256s"],
                        }
                        for row in trace_rows
                    ]
                )
            ).hexdigest(),
            "final_state_fingerprints": continuous_state,
            "resume_final_state_fingerprints": resume_state_fingerprints,
        }

    if session_value is None or checkpoint_ref is None or comparison is None:
        raise ContractViolation("GPU_REPLAY_DID_NOT_COMPLETE")
    return {
        "schema_version": "gpu2-replay-v1",
        "status": "PASS",
        "scientific_result": False,
        "scope": "exact v4 model and real frozen corpus; deterministic integration gate only",
        "freeze_sha256": preverified_freeze["freeze_sha256"],
        "code_sha256": code_sha256,
        "cpu_check_sha256": cpu_check_sha256,
        "gpu_binding": session_value.binding.as_dict(),
        "checkpoint": checkpoint_ref.as_dict(),
        "comparison": comparison,
        "optimizer_steps_charged": 32,
        "gpu_seconds_charged": ledger.charged_seconds(session_value.lease),
        "budget": ledger.snapshot(),
        "disk_reservation": disk,
    }
