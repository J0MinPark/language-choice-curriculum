"""Shared, dependency-light contracts for the production implementation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


VERSION = "4.0.0"
IMPLEMENTATION_REVISION = "4.0.0-r1"
LANGUAGES = ("ko", "en", "zh", "fr")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT / "work"


class ContractViolation(ValueError):
    """A stable, safe-to-log contract failure."""

    def __init__(self, code: str):
        if not code or any(ch.isspace() for ch in code):
            raise ValueError("contract codes must be nonempty and whitespace-free")
        self.code = code
        super().__init__(code)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pilot_config() -> dict[str, Any]:
    config = load_json(PROJECT_ROOT / "spec" / "pilot.json")
    if config.get("version") != VERSION:
        raise ContractViolation("BLOCKED_SPEC_VERSION")
    if tuple(config.get("languages", ())) != LANGUAGES:
        raise ContractViolation("BLOCKED_LANGUAGE_ORDER")
    if config.get("policy", {}).get("main_enabled") is not False:
        raise ContractViolation("MAIN_NOT_AUTHORIZED")
    return config


def require_relative_to(path: Path, root: Path, code: str = "PATH_OUTSIDE_SCOPE") -> Path:
    resolved = path.resolve(strict=False)
    root_resolved = root.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ContractViolation(code) from exc
    return resolved


def require_sha256(value: str, code: str = "INVALID_SHA256") -> str:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ContractViolation(code)
    return value


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        require_sha256(self.sha256)
        if self.bytes < 0:
            raise ContractViolation("INVALID_ARTIFACT_SIZE")

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    reasons: tuple[str, ...]
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "BLOCKED", "NOT_RUN", "NOT_IMPLEMENTED"}:
            raise ContractViolation("INVALID_GATE_STATUS")
        if self.status == "PASS" and self.reasons:
            raise ContractViolation("PASS_WITH_REASONS")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
        }
