"""Hashing and fail-closed, write-once artifact publication.

Run JSON is authoritative and must never be replaced.  Publication therefore
writes and fsyncs a private temporary inode, then atomically links that inode at
the final name.  ``os.replace`` is intentionally reserved for reproducible,
derived files such as Markdown.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ContractViolation as ContractError,
    canonical_json_bytes as _canonical_json_bytes,
    sha256_bytes,
)


JSONValue = Any


def read_regular_file_bytes(path: str | Path) -> bytes:
    """Read one regular-file inode without following a final symlink.

    Validation and reading share the same descriptor, so replacing the path
    between a metadata check and a later ``read_bytes`` call cannot launder
    different content into a verified artifact.
    """
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ContractError("ARTIFACT_NOT_REGULAR_FILE") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("ARTIFACT_NOT_REGULAR_FILE")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_regular_file_bytes_exact(
    path: str | Path,
    *,
    expected_bytes: int,
) -> bytes:
    """Read exactly ``expected_bytes`` from one regular-file descriptor.

    The descriptor's size is checked before any payload bytes are allocated.
    Reads are then capped at the frozen size and one trailing byte is probed,
    so growth or truncation after ``fstat`` also fails closed.
    """
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise ContractError("INVALID_ARTIFACT_SIZE")
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ContractError("ARTIFACT_NOT_REGULAR_FILE") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("ARTIFACT_NOT_REGULAR_FILE")
        if metadata.st_size != expected_bytes:
            raise ContractError("ARTIFACT_SIZE_MISMATCH")
        remaining = expected_bytes
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ContractError("ARTIFACT_SIZE_MISMATCH")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ContractError("ARTIFACT_SIZE_MISMATCH")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ContractError("ARTIFACT_NOT_REGULAR_FILE") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ContractError("ARTIFACT_NOT_REGULAR_FILE")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def canonical_json_bytes(payload: JSONValue) -> bytes:
    """Return deterministic UTF-8 JSON and reject NaN/Infinity."""
    try:
        return _canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_JSON_PAYLOAD") from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes_once(
    path: str | Path,
    content: bytes,
    *,
    mode: int = 0o444,
) -> dict[str, JSONValue]:
    """Atomically publish complete bytes without ever replacing ``path``.

    A filesystem that cannot atomically create a hard link fails closed.  A
    stale private temporary file can be removed safely; the requested final
    name is either absent or points at fully written, fsynced content.
    """
    destination = Path(path)
    if not destination.name or destination.name in {".", ".."}:
        raise ContractError("INVALID_ARTIFACT_PATH")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        raise ContractError("ARTIFACT_EXISTS")

    temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise ContractError("ARTIFACT_EXISTS") from exc
        except OSError as exc:
            if exc.errno in {
                errno.EPERM,
                errno.EOPNOTSUPP,
                getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
                errno.EXDEV,
            }:
                raise ContractError("ATOMIC_NOREPLACE_UNSUPPORTED") from exc
            raise
        published = True
        _fsync_directory(destination.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if published:
            _fsync_directory(destination.parent)

    return {
        "path": str(destination),
        "sha256": sha256_bytes(content),
        "bytes": len(content),
        "write_semantics": "ATOMIC_WRITE_ONCE",
    }


def publish_verified_file_once(
    source: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    mode: int = 0o444,
) -> dict[str, JSONValue]:
    """Stream one verified regular-file inode into a write-once artifact.

    The source descriptor is opened once with ``O_NOFOLLOW`` and the digest is
    computed over the exact bytes copied.  This avoids both path-swap races and
    loading an evidence payload of attacker-controlled size into memory.
    """
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise ContractError("INVALID_ARTIFACT_SIZE")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ContractError("INVALID_SHA256")
    source_path = Path(source)
    target = Path(destination)
    if not target.name or target.name in {".", ".."}:
        raise ContractError("INVALID_ARTIFACT_PATH")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        raise ContractError("ARTIFACT_EXISTS")

    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    temporary = target.parent / (
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    published = False
    digest = hashlib.sha256()
    copied = 0
    try:
        try:
            source_descriptor = os.open(
                source_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
        except OSError as exc:
            raise ContractError("ARTIFACT_NOT_REGULAR_FILE") from exc
        metadata = os.fstat(source_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("ARTIFACT_NOT_REGULAR_FILE")
        if metadata.st_size != expected_bytes:
            raise ContractError("ARTIFACT_SIZE_MISMATCH")
        destination_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > expected_bytes:
                raise ContractError("ARTIFACT_SIZE_MISMATCH")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise ContractError("ARTIFACT_WRITE_FAILED")
                view = view[written:]
        if copied != expected_bytes:
            raise ContractError("ARTIFACT_SIZE_MISMATCH")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ContractError("ARTIFACT_HASH_MISMATCH")
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = None
        os.chmod(temporary, mode)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise ContractError("ARTIFACT_EXISTS") from exc
        except OSError as exc:
            if exc.errno in {
                errno.EPERM,
                errno.EOPNOTSUPP,
                getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
                errno.EXDEV,
            }:
                raise ContractError("ATOMIC_NOREPLACE_UNSUPPORTED") from exc
            raise
        published = True
        _fsync_directory(target.parent)
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if published:
            _fsync_directory(target.parent)
    return {
        "path": str(target),
        "sha256": expected_sha256,
        "bytes": expected_bytes,
        "write_semantics": "ATOMIC_WRITE_ONCE_VERIFIED_COPY",
    }


def publish_json_once(
    path: str | Path,
    payload: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    if not isinstance(payload, Mapping):
        raise ContractError("JSON_ROOT_MUST_BE_OBJECT")
    return publish_bytes_once(path, canonical_json_bytes(dict(payload)))


def _reject_nonstandard_number(token: str) -> None:
    raise ContractError("NONFINITE_JSON_NUMBER:" + token)


def read_verified_json(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, JSONValue]:
    parsed, _actual = read_verified_json_with_sha256(
        path, expected_sha256=expected_sha256
    )
    return parsed


def read_verified_json_with_sha256(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, JSONValue], str]:
    """Parse and hash the exact same bytes from a single file descriptor."""
    raw = read_regular_file_bytes(path)
    actual = sha256_bytes(raw)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ContractError("ARTIFACT_HASH_MISMATCH")
    try:
        parsed = json.loads(raw, parse_constant=_reject_nonstandard_number)
    except ContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("INVALID_JSON_ARTIFACT") from exc
    if not isinstance(parsed, dict):
        raise ContractError("JSON_ROOT_MUST_BE_OBJECT")
    return parsed, actual


def replace_derived_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> dict[str, JSONValue]:
    """Atomically regenerate a non-authoritative text projection."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    data = text.encode(encoding)
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {
        "path": str(destination),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "write_semantics": "REGENERABLE_DERIVED_PROJECTION",
    }
