"""Fail-closed binding to the one physical GPU authorized by the v4 pilot.

This module deliberately does not import :mod:`torch` at import time.  CUDA's
visibility mask must be checked before Torch is allowed to initialize CUDA.
There is no device-selection or CPU-fallback interface here.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import ContractViolation, PROJECT_ROOT, load_json


POLICY_PATH = PROJECT_ROOT / "implementation" / "config" / "campaign_policy.json"
CommandOutput = Callable[[list[str]], str]


@dataclass(frozen=True)
class GpuBinding:
    physical_index: int
    logical_index: int
    uuid: str
    pci_bus_id: str
    name: str
    total_memory_mib: int
    torch_cuda_version: str
    torch_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _command_output(argv: list[str]) -> str:
    try:
        result = subprocess.run(
            argv,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise ContractViolation("BLOCKED_NVIDIA_SMI") from exc
    return result.stdout


def _policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    try:
        value = load_json(path)
        gpu = value["gpu"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ContractViolation("BLOCKED_GPU_POLICY") from exc
    if gpu.get("allow_fallback") is not False or gpu.get("allow_multi_gpu") is not False:
        raise ContractViolation("BLOCKED_GPU_POLICY")
    return gpu


def require_literal_gpu2_mask(
    environ: Mapping[str, str] | None = None,
    *,
    policy_path: Path = POLICY_PATH,
) -> None:
    """Reject anything except the literal pre-import mask ``2``."""
    values = os.environ if environ is None else environ
    expected = str(_policy(policy_path)["required_cuda_visible_devices"])
    if expected != "2":
        raise ContractViolation("BLOCKED_GPU_POLICY")
    if values.get("CUDA_VISIBLE_DEVICES") != expected:
        raise ContractViolation("BLOCKED_GPU2_MASK")
    if values.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ContractViolation("BLOCKED_GPU_DEVICE_ORDER")


def _physical_gpu2(
    command_output: CommandOutput,
    required_index: int,
) -> dict[str, Any]:
    raw = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    matches: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",", 4)]
        if len(fields) != 5:
            raise ContractViolation("BLOCKED_GPU_INVENTORY")
        try:
            index = int(fields[0])
            total_memory_mib = int(fields[4])
        except ValueError as exc:
            raise ContractViolation("BLOCKED_GPU_INVENTORY") from exc
        if index == required_index:
            matches.append(
                {
                    "physical_index": index,
                    "uuid": fields[1],
                    "pci_bus_id": fields[2],
                    "name": fields[3],
                    "total_memory_mib": total_memory_mib,
                }
            )
    if len(matches) != 1:
        raise ContractViolation("BLOCKED_GPU2_UNAVAILABLE")
    return matches[0]


def _compute_processes(command_output: CommandOutput) -> dict[str, set[int]]:
    raw = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    result: dict[str, set[int]] = {}
    for line in raw.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        fields = [part.strip() for part in line.split(",", 1)]
        if len(fields) != 2:
            raise ContractViolation("BLOCKED_GPU_PROCESS_INVENTORY")
        try:
            pid = int(fields[1])
        except ValueError as exc:
            raise ContractViolation("BLOCKED_GPU_PROCESS_INVENTORY") from exc
        result.setdefault(fields[0], set()).add(pid)
    return result


def _cuda_visible_memory_mib(
    physical: Mapping[str, Any], command_output: CommandOutput,
) -> int:
    """Subtract documented driver/firmware reservation, never a free-memory value.

    Query the already verified UUID and reject missing/unsupported accounting.
    NVIDIA reports FB total including reserved memory; CUDA excludes it.
    https://docs.nvidia.com/deploy/nvml-api/structnvmlMemory__v2__t.html
    """
    raw = command_output([
        "nvidia-smi", "--id=" + physical["uuid"],
        "--query-gpu=uuid,memory.total,memory.reserved",
        "--format=csv,noheader,nounits",
    ])
    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ContractViolation("BLOCKED_GPU2_MEMORY_ACCOUNTING")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 3 or fields[0] != physical["uuid"]:
        raise ContractViolation("BLOCKED_GPU2_MEMORY_ACCOUNTING")
    try:
        total, reserved = int(fields[1]), int(fields[2])
    except ValueError as exc:
        raise ContractViolation("BLOCKED_GPU2_MEMORY_ACCOUNTING") from exc
    if total != physical["total_memory_mib"] or not 0 <= reserved < total:
        raise ContractViolation("BLOCKED_GPU2_MEMORY_ACCOUNTING")
    return total - reserved


def inspect_gpu2_without_cuda(
    *,
    environ: Mapping[str, str] | None = None,
    policy_path: Path = POLICY_PATH,
    command_output: CommandOutput = _command_output,
    require_idle: bool = True,
) -> dict[str, Any]:
    """Read-only inventory check performed before Torch touches CUDA."""
    require_literal_gpu2_mask(environ, policy_path=policy_path)
    gpu_policy = _policy(policy_path)
    required_index = int(gpu_policy["required_physical_index"])
    if required_index != 2:
        raise ContractViolation("BLOCKED_GPU_POLICY")
    physical = _physical_gpu2(command_output, required_index)
    if physical["uuid"] != gpu_policy.get("expected_uuid"):
        raise ContractViolation("BLOCKED_GPU2_IDENTITY")
    processes = _compute_processes(command_output)
    foreign = sorted(processes.get(physical["uuid"], set()) - {os.getpid()})
    if require_idle and foreign:
        raise ContractViolation("BLOCKED_GPU2_BUSY")
    return {**physical, "foreign_compute_pids": foreign}


def require_physical_gpu2(
    *,
    environ: Mapping[str, str] | None = None,
    policy_path: Path = POLICY_PATH,
    command_output: CommandOutput = _command_output,
    torch_module: Any | None = None,
    require_idle: bool = True,
    identity_attempts: int = 5,
) -> GpuBinding:
    """Initialize CUDA only after proving the configured physical device.

    Process-to-UUID confirmation is mandatory.  Merely seeing logical
    ``cuda:0`` after masking is insufficient evidence of physical identity.
    """
    physical = inspect_gpu2_without_cuda(
        environ=environ,
        policy_path=policy_path,
        command_output=command_output,
        require_idle=require_idle,
    )
    torch = torch_module
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ContractViolation("BLOCKED_CUDA_TORCH_MISSING") from exc
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not cuda_version:
        raise ContractViolation("BLOCKED_CUDA_TORCH_BUILD")
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
    except Exception as exc:
        raise ContractViolation("BLOCKED_GPU2_UNAVAILABLE") from exc
    if not available or count != 1:
        raise ContractViolation("BLOCKED_GPU2_UNAVAILABLE")
    try:
        torch.cuda.set_device(0)
        # Force creation of this process's CUDA context before querying NVML via
        # nvidia-smi.  This is an identity sentinel, not a training operation.
        sentinel = torch.empty(1, device="cuda:0")
        torch.cuda.synchronize(0)
        del sentinel
        properties = torch.cuda.get_device_properties(0)
    except Exception as exc:
        raise ContractViolation("BLOCKED_GPU2_UNAVAILABLE") from exc

    this_pid = os.getpid()
    confirmed = False
    attempts = max(1, int(identity_attempts))
    for attempt in range(attempts):
        processes = _compute_processes(command_output)
        process_uuids = {
            gpu_uuid for gpu_uuid, pids in processes.items() if this_pid in pids
        }
        foreign_on_gpu2 = processes.get(physical["uuid"], set()) - {this_pid}
        if foreign_on_gpu2:
            raise ContractViolation("BLOCKED_GPU2_BUSY")
        if this_pid in processes.get(physical["uuid"], set()):
            if process_uuids != {physical["uuid"]}:
                raise ContractViolation("BLOCKED_GPU2_PROCESS_IDENTITY")
            confirmed = True
            break
        if attempt + 1 < attempts:
            time.sleep(0.1)
    if not confirmed:
        raise ContractViolation("BLOCKED_GPU2_PROCESS_IDENTITY")
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if bf16_supported is None or not bool(bf16_supported()):
        raise ContractViolation("BLOCKED_GPU2_BF16_UNAVAILABLE")

    property_name = str(getattr(properties, "name", ""))
    property_memory = int(getattr(properties, "total_memory", 0)) // (1024 * 1024)
    expected_memory = _cuda_visible_memory_mib(physical, command_output)
    if property_name != physical["name"] or property_memory <= 0 or abs(property_memory - expected_memory) > 1:
        raise ContractViolation("BLOCKED_GPU2_PROPERTY_MISMATCH")
    return GpuBinding(
        physical_index=2,
        logical_index=0,
        uuid=physical["uuid"],
        pci_bus_id=physical["pci_bus_id"],
        name=physical["name"],
        total_memory_mib=physical["total_memory_mib"],
        torch_cuda_version=str(cuda_version),
        torch_version=str(getattr(torch, "__version__", "UNKNOWN")),
    )


def assert_model_on_bound_device(model: Any, binding: GpuBinding) -> None:
    require_literal_gpu2_mask()
    gpu_policy = _policy()
    if (
        binding.logical_index != 0
        or binding.physical_index != 2
        or binding.uuid != gpu_policy.get("expected_uuid")
    ):
        raise ContractViolation("BLOCKED_GPU2_IDENTITY")
    devices = {str(parameter.device) for parameter in model.parameters()}
    if devices != {"cuda:0"}:
        raise ContractViolation("MODEL_NOT_EXCLUSIVELY_ON_GPU2")
