"""Append-only GPU-hour accounting and single-process GPU-2 leases."""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .contracts import ContractViolation, canonical_json_bytes


@dataclass(frozen=True)
class BudgetCaps:
    campaign_gpu_hours_cap: float
    prior_gpu_hours: float
    per_root_gpu_hours_cap: float

    def __post_init__(self) -> None:
        values = (
            self.campaign_gpu_hours_cap,
            self.prior_gpu_hours,
            self.per_root_gpu_hours_cap,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ContractViolation("INVALID_BUDGET_CAP")
        if self.prior_gpu_hours > self.campaign_gpu_hours_cap:
            raise ContractViolation("CAMPAIGN_BUDGET_ALREADY_EXCEEDED")


@dataclass(frozen=True)
class BudgetLease:
    lease_id: str
    run_id: str
    root_id: int | None
    phase: str
    reserved_seconds: float
    started_monotonic: float

    @property
    def deadline_monotonic(self) -> float:
        return self.started_monotonic + self.reserved_seconds


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class BudgetStop(ContractViolation):
    pass


class GpuBudgetLedger:
    """A conservative ledger: unfinished leases consume their full reservation."""

    def __init__(
        self,
        path: Path,
        caps: BudgetCaps,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.caps = caps
        self.monotonic = monotonic

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise ContractViolation("INVALID_GPU_LEDGER")
        events: list[dict[str, Any]] = []
        previous = "0" * 64
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ContractViolation("INVALID_GPU_LEDGER") from exc
        for line in lines:
            if not line.strip():
                raise ContractViolation("INVALID_GPU_LEDGER")
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ContractViolation("INVALID_GPU_LEDGER") from exc
            if not isinstance(event, dict):
                raise ContractViolation("INVALID_GPU_LEDGER")
            recorded_hash = event.pop("event_sha256", None)
            if event.get("previous_event_sha256") != previous:
                raise ContractViolation("GPU_LEDGER_CHAIN_BROKEN")
            calculated = hashlib.sha256(canonical_json_bytes(event)).hexdigest()
            if recorded_hash != calculated:
                raise ContractViolation("GPU_LEDGER_EVENT_HASH_MISMATCH")
            event["event_sha256"] = recorded_hash
            events.append(event)
            previous = recorded_hash
        return events

    def _append_unlocked(self, payload: Mapping[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
        event = dict(payload)
        event["schema_version"] = 1
        event["recorded_utc"] = _utc_now()
        event["previous_event_sha256"] = events[-1]["event_sha256"] if events else "0" * 64
        event_hash = hashlib.sha256(canonical_json_bytes(event)).hexdigest()
        event["event_sha256"] = event_hash
        encoded = json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            written = 0
            while written < len(encoded):
                count = os.write(descriptor, encoded[written:])
                if count <= 0:
                    raise ContractViolation("GPU_LEDGER_WRITE_FAILED")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return event

    @staticmethod
    def _account(events: list[dict[str, Any]]) -> tuple[float, dict[int, float], set[str]]:
        reservations: dict[str, dict[str, Any]] = {}
        finishes: dict[str, dict[str, Any]] = {}
        run_ids: set[str] = set()
        for event in events:
            kind = event.get("event")
            lease_id = event.get("lease_id")
            if kind == "RESERVE":
                if not isinstance(lease_id, str) or lease_id in reservations:
                    raise ContractViolation("INVALID_GPU_LEDGER_SEQUENCE")
                if event.get("run_id") in run_ids:
                    raise ContractViolation("DUPLICATE_GPU_RUN_ID")
                reservations[lease_id] = event
                run_ids.add(str(event.get("run_id")))
            elif kind == "FINISH":
                if lease_id not in reservations or lease_id in finishes:
                    raise ContractViolation("INVALID_GPU_LEDGER_SEQUENCE")
                finishes[lease_id] = event
            elif kind == "HEARTBEAT":
                if lease_id not in reservations or lease_id in finishes:
                    raise ContractViolation("INVALID_GPU_LEDGER_SEQUENCE")
            else:
                raise ContractViolation("INVALID_GPU_LEDGER_EVENT")
        campaign_seconds = 0.0
        root_seconds: dict[int, float] = {}
        for lease_id, reserve in reservations.items():
            if lease_id in finishes:
                seconds = float(finishes[lease_id]["charged_seconds"])
            else:
                # A hard-killed process cannot make its GPU use disappear.
                seconds = float(reserve["reserved_seconds"])
            if not math.isfinite(seconds) or seconds < 0:
                raise ContractViolation("INVALID_GPU_LEDGER_SECONDS")
            campaign_seconds += seconds
            root = reserve.get("root_id")
            if root is not None:
                root = int(root)
                root_seconds[root] = root_seconds.get(root, 0.0) + seconds
        return campaign_seconds, root_seconds, run_ids

    @staticmethod
    def _require_active_lease(events: list[dict[str, Any]], lease: BudgetLease) -> None:
        reservations = [
            event for event in events
            if event.get("event") == "RESERVE" and event.get("lease_id") == lease.lease_id
        ]
        finishes = [
            event for event in events
            if event.get("event") == "FINISH" and event.get("lease_id") == lease.lease_id
        ]
        if len(reservations) != 1 or finishes:
            raise ContractViolation("GPU_LEASE_NOT_ACTIVE")
        reservation = reservations[0]
        if (
            reservation.get("run_id") != lease.run_id
            or reservation.get("root_id") != lease.root_id
            or reservation.get("phase") != lease.phase
            or float(reservation.get("reserved_seconds", -1)) != lease.reserved_seconds
        ):
            raise ContractViolation("GPU_LEASE_IDENTITY_MISMATCH")

    def snapshot(self) -> dict[str, Any]:
        with self._lock():
            events = self._read_unlocked()
        campaign_seconds, root_seconds, _ = self._account(events)
        prior_seconds = self.caps.prior_gpu_hours * 3600.0
        return {
            "campaign_cap_gpu_hours": self.caps.campaign_gpu_hours_cap,
            "prior_gpu_hours_user_reported": self.caps.prior_gpu_hours,
            "new_gpu_seconds_charged_or_reserved": campaign_seconds,
            "campaign_gpu_hours_charged_or_reserved": (prior_seconds + campaign_seconds) / 3600.0,
            "campaign_gpu_hours_remaining": max(
                0.0,
                self.caps.campaign_gpu_hours_cap - (prior_seconds + campaign_seconds) / 3600.0,
            ),
            "per_root_gpu_seconds_charged_or_reserved": {
                str(root): seconds for root, seconds in sorted(root_seconds.items())
            },
            "ledger_events": len(events),
            "ledger_tip_sha256": events[-1]["event_sha256"] if events else "0" * 64,
        }

    def charged_seconds(self, lease: BudgetLease) -> float:
        with self._lock():
            events = self._read_unlocked()
        matches = [
            event for event in events
            if event.get("event") == "FINISH" and event.get("lease_id") == lease.lease_id
        ]
        if len(matches) != 1:
            raise ContractViolation("GPU_LEASE_NOT_FINISHED")
        value = float(matches[0]["charged_seconds"])
        if not math.isfinite(value) or value < 0:
            raise ContractViolation("INVALID_GPU_LEDGER_SECONDS")
        return value

    def reserve(
        self,
        *,
        run_id: str,
        root_id: int | None,
        phase: str,
        reserved_seconds: float,
    ) -> BudgetLease:
        if not run_id or not phase or any(ch.isspace() for ch in run_id):
            raise ContractViolation("INVALID_BUDGET_LEASE")
        if root_id is not None and root_id not in {4101, 4102, 4103, 4104}:
            raise ContractViolation("INVALID_BUDGET_ROOT")
        if not math.isfinite(reserved_seconds) or reserved_seconds <= 0:
            raise ContractViolation("INVALID_BUDGET_LEASE")
        started = self.monotonic()
        lease = BudgetLease(
            lease_id=uuid.uuid4().hex,
            run_id=run_id,
            root_id=root_id,
            phase=phase,
            reserved_seconds=float(reserved_seconds),
            started_monotonic=started,
        )
        with self._lock():
            events = self._read_unlocked()
            new_seconds, roots, run_ids = self._account(events)
            if run_id in run_ids:
                raise ContractViolation("DUPLICATE_GPU_RUN_ID")
            campaign_projected = self.caps.prior_gpu_hours * 3600.0 + new_seconds + reserved_seconds
            if campaign_projected > self.caps.campaign_gpu_hours_cap * 3600.0 + 1e-9:
                raise ContractViolation("BLOCKED_CAMPAIGN_GPU_BUDGET")
            if root_id is not None:
                root_projected = roots.get(int(root_id), 0.0) + reserved_seconds
                if root_projected > self.caps.per_root_gpu_hours_cap * 3600.0 + 1e-9:
                    raise ContractViolation("BLOCKED_ROOT_GPU_BUDGET")
            self._append_unlocked(
                {
                    "event": "RESERVE",
                    "lease_id": lease.lease_id,
                    "run_id": run_id,
                    "root_id": root_id,
                    "phase": phase,
                    "reserved_seconds": reserved_seconds,
                    "pid": os.getpid(),
                    "physical_gpu_index": 2,
                },
                events,
            )
        return lease

    def heartbeat(self, lease: BudgetLease) -> dict[str, Any]:
        elapsed = max(0.0, self.monotonic() - lease.started_monotonic)
        with self._lock():
            events = self._read_unlocked()
            self._require_active_lease(events, lease)
            return self._append_unlocked(
                {
                    "event": "HEARTBEAT",
                    "lease_id": lease.lease_id,
                    "elapsed_seconds": elapsed,
                },
                events,
            )

    def finish(self, lease: BudgetLease, status: str) -> dict[str, Any]:
        elapsed = max(0.0, self.monotonic() - lease.started_monotonic)
        # Going over a reservation is itself a contract failure, but elapsed GPU
        # time is still charged in full and is never clipped away.
        with self._lock():
            events = self._read_unlocked()
            self._require_active_lease(events, lease)
            event = self._append_unlocked(
                {
                    "event": "FINISH",
                    "lease_id": lease.lease_id,
                    "charged_seconds": elapsed,
                    "status": status,
                },
                events,
            )
        if elapsed > lease.reserved_seconds + 1e-9:
            raise ContractViolation("GPU_BUDGET_RESERVATION_OVERRUN")
        return event

    def require_time_remaining(
        self,
        lease: BudgetLease,
        *,
        checkpoint_grace_seconds: float = 0.0,
    ) -> float:
        if checkpoint_grace_seconds < 0 or not math.isfinite(checkpoint_grace_seconds):
            raise ContractViolation("INVALID_CHECKPOINT_GRACE")
        remaining = lease.deadline_monotonic - self.monotonic()
        if remaining <= checkpoint_grace_seconds:
            raise BudgetStop("BLOCKED_GPU_BUDGET_DEADLINE")
        return remaining


class ExclusiveGpu2Lock:
    """Project-local nonblocking lock; never kills or migrates another job."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._descriptor: int | None = None

    def __enter__(self) -> "ExclusiveGpu2Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise ContractViolation("BLOCKED_GPU2_PROJECT_LOCK") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        self._descriptor = descriptor
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


def require_disk_reservation(
    path: Path,
    *,
    planned_bytes: int,
    emergency_free_bytes: int,
    available_bytes: int | None = None,
) -> dict[str, int]:
    """Fail before writes if planned artifacts would consume safety headroom."""
    if planned_bytes < 0 or emergency_free_bytes < 0:
        raise ContractViolation("INVALID_DISK_RESERVATION")
    if available_bytes is None:
        try:
            stats = os.statvfs(path)
        except OSError as exc:
            raise ContractViolation("BLOCKED_DISK_INVENTORY") from exc
        available_bytes = stats.f_bavail * stats.f_frsize
    required = planned_bytes + emergency_free_bytes
    if available_bytes < required:
        raise ContractViolation("BLOCKED_DISK_BUDGET")
    return {
        "available_bytes": int(available_bytes),
        "planned_bytes": int(planned_bytes),
        "emergency_free_bytes": int(emergency_free_bytes),
        "projected_free_bytes": int(available_bytes - planned_bytes),
    }
