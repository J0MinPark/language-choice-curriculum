"""Git-bound implementation integrity and reproducible CPU-check evidence."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import (
    publish_bytes_once,
    publish_json_once,
    read_regular_file_bytes,
    read_verified_json,
    read_verified_json_with_sha256,
    sha256_file,
)
from .contracts import (
    IMPLEMENTATION_REVISION,
    PROJECT_ROOT,
    VERSION,
    WORK_ROOT,
    ContractViolation,
    canonical_json_bytes,
    require_relative_to,
    require_sha256,
    sha256_bytes,
)


IMPLEMENTATION_MANIFEST_SCHEMA = "implementation-integrity-v1"
CPU_CHECK_SCHEMA = "cpu-checks-v3"
GitRunner = Callable[..., subprocess.CompletedProcess[bytes]]


# This program is passed literally to an isolated Python interpreter.  Keeping
# the guard in the child makes it effective before unittest discovery imports
# any project test module.  CPython audit hooks cannot police a native syscall
# or a separately executed program, so the evidence below states that limit
# explicitly instead of claiming OS-level network isolation.
_OFFLINE_UNITTEST_PROGRAM = r"""
import os
import socket
import sys
import unittest

root = os.path.realpath(sys.argv[1])
start_name = sys.argv[2]
if start_name not in {"tests", "implementation/tests"}:
    raise SystemExit(97)
start = os.path.realpath(os.path.join(root, start_name))
if os.path.commonpath((root, start)) != root:
    raise SystemExit(98)

blocked_events = {
    "socket.bind",
    "socket.connect",
    "socket.connect_ex",
    "socket.getaddrinfo",
    "socket.gethostbyaddr",
    "socket.gethostbyname",
    "socket.gethostname",
    "socket.getnameinfo",
    "socket.getservbyname",
    "socket.getservbyport",
}

def deny_network(event, args):
    if event == "socket.__new__":
        family = args[1]
        if family in {socket.AF_INET, socket.AF_INET6}:
            raise PermissionError("PYTHON_INET_NETWORK_DISABLED")
    elif event in blocked_events:
        raise PermissionError("PYTHON_INET_NETWORK_DISABLED")

sys.addaudithook(deny_network)
os.chdir(root)
sys.path.insert(0, root)
suite = unittest.defaultTestLoader.discover(start_dir=start, top_level_dir=start)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
""".strip()


def _cpu_subprocess_environment() -> dict[str, str]:
    """Return the complete environment supplied to CPU-check children.

    Values are fixed here rather than copied from ``os.environ``.  In
    particular, credentials, proxy configuration, PYTHONPATH, and loader
    injection variables are not forwarded.
    """

    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PATH": os.defpath,
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _cpu_test_commands(root: Path) -> tuple[tuple[str, list[str]], ...]:
    prefix = [
        sys.executable,
        "-I",
        "-B",
        "-X",
        "utf8",
        "-c",
        _OFFLINE_UNITTEST_PROGRAM,
        str(root),
    ]
    return (
        ("reference_cpu_tests", [*prefix, "tests"]),
        ("implementation_cpu_tests", [*prefix, "implementation/tests"]),
    )


def _cpu_execution_controls() -> dict[str, Any]:
    return {
        "subprocess_environment_policy": "FIXED_ALLOWLIST_V1",
        "subprocess_environment": _cpu_subprocess_environment(),
        "python_flags": ["-I", "-B", "-X", "utf8"],
        "python_network_guard": "CPYTHON_AUDIT_INET_DENY_V1",
        "guard_installed": "AFTER_INTERPRETER_STARTUP_BEFORE_TEST_DISCOVERY",
    }


def _cpu_evidence_limitations() -> dict[str, Any]:
    return {
        "network_nonuse_proven": False,
        "os_network_namespace_isolated": False,
        "external_subprocess_network_blocked": False,
        "native_syscall_network_blocked": False,
        "interpreter_startup_network_blocked": False,
        "gpu_execution_monitoring_performed": False,
        "gpu_nonuse_proven": False,
        "cuda_control_scope": "CUDA_VISIBLE_DEVICES_MASK_ONLY",
    }


def _logical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float = 300.0,
    runner: GitRunner = subprocess.run,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = runner(
            list(argv),
            cwd=cwd,
            env=None if env is None else dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractViolation("SUBPROCESS_UNAVAILABLE") from exc
    if not isinstance(result.returncode, int):
        raise ContractViolation("INVALID_SUBPROCESS_RESULT")
    return result


def _git_bytes(
    root: Path,
    args: Sequence[str],
    *,
    runner: GitRunner = subprocess.run,
) -> bytes:
    result = _run(["git", *args], cwd=root, timeout=30.0, runner=runner)
    if result.returncode != 0:
        raise ContractViolation("GIT_INTEGRITY_COMMAND_FAILED")
    return bytes(result.stdout)


def build_implementation_manifest(
    *,
    project_root: Path = PROJECT_ROOT,
    runner: GitRunner = subprocess.run,
) -> dict[str, Any]:
    """Describe the clean tracked ``implementation/`` tree at exact HEAD."""
    root = Path(project_root).resolve(strict=True)
    reported_root = Path(
        _git_bytes(root, ["rev-parse", "--show-toplevel"], runner=runner)
        .decode("utf-8")
        .strip()
    ).resolve(strict=True)
    if reported_root != root:
        raise ContractViolation("IMPLEMENTATION_REPOSITORY_ROOT_MISMATCH")
    dirty = _git_bytes(
        root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "implementation"],
        runner=runner,
    )
    if dirty:
        raise ContractViolation("IMPLEMENTATION_WORKTREE_NOT_CLEAN")
    commit = _git_bytes(root, ["rev-parse", "HEAD"], runner=runner).decode().strip()
    tree = _git_bytes(root, ["rev-parse", "HEAD^{tree}"], runner=runner).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit) or not re.fullmatch(
        r"[0-9a-f]{40,64}", tree
    ):
        raise ContractViolation("INVALID_GIT_OBJECT_ID")
    raw = _git_bytes(
        root, ["ls-tree", "-r", "-z", "HEAD", "--", "implementation"], runner=runner
    )
    rows: list[dict[str, Any]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            header, encoded_path = entry.split(b"\t", 1)
            mode, kind, object_id = header.decode("ascii").split(" ")
            relative = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ContractViolation("INVALID_GIT_TREE_ENTRY") from exc
        if kind != "blob" or mode == "120000":
            raise ContractViolation("IMPLEMENTATION_TREE_CONTAINS_UNSAFE_ENTRY")
        path = require_relative_to(root / relative, root / "implementation")
        if path.is_symlink() or not path.is_file():
            raise ContractViolation("IMPLEMENTATION_FILE_NOT_REGULAR")
        try:
            working_bytes = path.read_bytes()
        except OSError as exc:
            raise ContractViolation("IMPLEMENTATION_FILE_NOT_REGULAR") from exc
        # `git status` intentionally honors index hints such as
        # assume-unchanged.  Compare with the committed blob itself so those
        # hints cannot turn modified runtime code into "exact HEAD" evidence.
        committed_bytes = _git_bytes(
            root, ["cat-file", "blob", object_id], runner=runner
        )
        if working_bytes != committed_bytes:
            raise ContractViolation("IMPLEMENTATION_WORKTREE_NOT_CLEAN")
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "git_mode": mode,
                "git_blob_oid": object_id,
                "bytes": len(working_bytes),
                "sha256": sha256_bytes(working_bytes),
            }
        )
    rows.sort(key=lambda row: row["path"])
    if not rows:
        raise ContractViolation("IMPLEMENTATION_TREE_EMPTY")
    core = {
        "schema_version": IMPLEMENTATION_MANIFEST_SCHEMA,
        "spec_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "PASS",
        "scope": "all files tracked at HEAD below implementation/; original MANIFEST.json is delivery-only",
        "git_commit": commit,
        "git_tree": tree,
        "file_count": len(rows),
        "files": rows,
        "content_set_sha256": _logical_hash(rows),
    }
    return {**core, "implementation_manifest_sha256": _logical_hash(core)}


def publish_implementation_manifest(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    scope_root: Path = WORK_ROOT,
    runner: GitRunner = subprocess.run,
) -> dict[str, Any]:
    destination = require_relative_to(Path(path), scope_root, "INTEGRITY_OUTPUT_OUTSIDE_SCOPE")
    manifest = build_implementation_manifest(project_root=project_root, runner=runner)
    artifact = publish_json_once(destination, manifest)
    return {**manifest, "manifest_artifact": artifact}


def verify_implementation_manifest(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    scope_root: Path | None = None,
    runner: GitRunner = subprocess.run,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    scope = Path(scope_root) if scope_root is not None else root / "work"
    manifest_path = require_relative_to(
        Path(path), scope, "INTEGRITY_MANIFEST_OUTSIDE_SCOPE"
    )
    manifest, manifest_file_sha256 = read_verified_json_with_sha256(manifest_path)
    claimed = manifest.get("implementation_manifest_sha256")
    require_sha256(str(claimed or ""), "INVALID_IMPLEMENTATION_MANIFEST_HASH")
    core = {key: value for key, value in manifest.items() if key != "implementation_manifest_sha256"}
    if claimed != _logical_hash(core):
        raise ContractViolation("IMPLEMENTATION_MANIFEST_HASH_MISMATCH")
    current = build_implementation_manifest(project_root=root, runner=runner)
    if current != manifest:
        raise ContractViolation("IMPLEMENTATION_MANIFEST_NOT_CURRENT_HEAD")
    return {
        **manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": manifest_file_sha256,
    }


def verify_delivery_manifest(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify the immutable delivery while explicitly limiting its claim."""
    root = Path(project_root).resolve(strict=True)
    path = root / "MANIFEST.json"
    manifest = read_verified_json(path)
    members = manifest.get("sha256")
    if (
        manifest.get("version") != VERSION
        or not isinstance(members, Mapping)
        or not members
    ):
        raise ContractViolation("INVALID_DELIVERY_MANIFEST")
    failures: list[str] = []
    for relative, expected in sorted(members.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            failures.append(str(relative))
            continue
        require_sha256(expected, "INVALID_DELIVERY_FILE_HASH")
        candidate = require_relative_to(root / relative, root, "DELIVERY_PATH_OUTSIDE_ROOT")
        if candidate.is_symlink() or not candidate.is_file() or sha256_file(candidate) != expected:
            failures.append(relative)
    if failures:
        raise ContractViolation("DELIVERY_INTEGRITY_FAILED")
    return {
        "status": "PASS",
        "scope": "only files enumerated by the original delivery MANIFEST.json; excludes implementation/",
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "checked_files": len(members),
        "failed_files": 0,
    }


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in (
        "numpy",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "tokenizers",
        "torch",
        "transformers",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "NOT_INSTALLED"
    return result


def _cpu_runtime_snapshot() -> dict[str, Any]:
    """Return the runtime identity to which CPU-check evidence is bound."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "dependencies": _dependency_versions(),
    }


def _test_count(stdout: bytes, stderr: bytes) -> int | None:
    text = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    matches = re.findall(r"Ran ([0-9]+) tests?", text)
    return int(matches[-1]) if matches else None


def run_cpu_check_evidence(
    implementation_manifest_path: Path,
    *,
    output_dir: Path,
    project_root: Path = PROJECT_ROOT,
    scope_root: Path = WORK_ROOT,
    runner: GitRunner = subprocess.run,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Run fixed offline CPU suites and publish their complete, code-bound logs."""
    root = Path(project_root).resolve(strict=True)
    output = require_relative_to(Path(output_dir), scope_root, "CPU_CHECK_OUTPUT_OUTSIDE_SCOPE")
    if output.exists() or output.is_symlink():
        raise ContractViolation("CPU_CHECK_OUTPUT_EXISTS")
    implementation = verify_implementation_manifest(
        implementation_manifest_path,
        project_root=root,
        scope_root=scope_root,
        runner=runner,
    )
    delivery = verify_delivery_manifest(project_root=root)
    environment = _cpu_subprocess_environment()
    commands = _cpu_test_commands(root)
    completed_rows: list[tuple[str, list[str], subprocess.CompletedProcess[bytes]]] = []
    for name, argv in commands:
        completed_rows.append(
            (name, argv, _run(argv, cwd=root, env=environment, timeout=300.0, runner=runner))
        )
    output.mkdir(parents=True, exist_ok=False)
    checks: list[dict[str, Any]] = []
    for name, argv, completed in completed_rows:
        stdout_ref = publish_bytes_once(output / f"{name}.stdout.log", bytes(completed.stdout))
        stderr_ref = publish_bytes_once(output / f"{name}.stderr.log", bytes(completed.stderr))
        checks.append(
            {
                "name": name,
                "status": "PASS" if completed.returncode == 0 else "BLOCKED",
                "return_code": completed.returncode,
                "test_count": _test_count(bytes(completed.stdout), bytes(completed.stderr)),
                "command": argv,
                "stdout": stdout_ref,
                "stderr": stderr_ref,
            }
        )
    timestamp = (now or (lambda: datetime.now(timezone.utc)))()
    if timestamp.tzinfo is None:
        raise ContractViolation("CPU_CHECK_TIMEZONE_REQUIRED")
    created_at = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    passed = all(row["status"] == "PASS" for row in checks)
    core = {
        "schema_version": CPU_CHECK_SCHEMA,
        "spec_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "PASS" if passed else "BLOCKED_CPU_CHECKS",
        "scientific_result": False,
        "created_at_utc": created_at,
        "runtime": _cpu_runtime_snapshot(),
        "execution_controls": _cpu_execution_controls(),
        "limitations": _cpu_evidence_limitations(),
        "implementation": {
            "manifest_path": str(Path(implementation_manifest_path).resolve()),
            "manifest_file_sha256": implementation["manifest_file_sha256"],
            "implementation_manifest_sha256": implementation[
                "implementation_manifest_sha256"
            ],
            "git_commit": implementation["git_commit"],
            "git_tree": implementation["git_tree"],
            "file_count": implementation["file_count"],
        },
        "delivery": delivery,
        "checks": checks,
    }
    evidence = {**core, "cpu_check_sha256": _logical_hash(core)}
    manifest_ref = publish_json_once(output / "cpu_checks.json", evidence)
    return {**evidence, "manifest_artifact": manifest_ref}


def verify_cpu_check_evidence(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    scope_root: Path | None = None,
    runner: GitRunner = subprocess.run,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    scope = Path(scope_root) if scope_root is not None else root / "work"
    source = require_relative_to(Path(path), scope, "CPU_CHECK_OUTSIDE_SCOPE")
    evidence, evidence_file_sha256 = read_verified_json_with_sha256(source)
    if set(evidence) != {
        "schema_version",
        "spec_version",
        "implementation_revision",
        "status",
        "scientific_result",
        "created_at_utc",
        "runtime",
        "execution_controls",
        "limitations",
        "implementation",
        "delivery",
        "checks",
        "cpu_check_sha256",
    }:
        raise ContractViolation("CPU_CHECK_FIELD_SET_MISMATCH")
    claimed = evidence.get("cpu_check_sha256")
    require_sha256(str(claimed or ""), "INVALID_CPU_CHECK_HASH")
    core = {key: value for key, value in evidence.items() if key != "cpu_check_sha256"}
    if claimed != _logical_hash(core):
        raise ContractViolation("CPU_CHECK_HASH_MISMATCH")
    if (
        evidence.get("schema_version") != CPU_CHECK_SCHEMA
        or evidence.get("spec_version") != VERSION
        or evidence.get("implementation_revision") != IMPLEMENTATION_REVISION
        or evidence.get("status") != "PASS"
        or evidence.get("scientific_result") is not False
    ):
        raise ContractViolation("CPU_CHECK_NOT_PASS")
    if (
        evidence.get("execution_controls") != _cpu_execution_controls()
        or evidence.get("limitations") != _cpu_evidence_limitations()
    ):
        raise ContractViolation("CPU_CHECK_CONTROL_CLAIM_MISMATCH")
    if evidence.get("runtime") != _cpu_runtime_snapshot():
        raise ContractViolation("CPU_CHECK_RUNTIME_MISMATCH")
    try:
        created = datetime.fromisoformat(
            str(evidence.get("created_at_utc", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ContractViolation("INVALID_CPU_CHECK_TIMESTAMP") from exc
    if created.tzinfo is None or created.utcoffset().total_seconds() != 0:
        raise ContractViolation("INVALID_CPU_CHECK_TIMESTAMP")
    implementation = evidence.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ContractViolation("CPU_CHECK_IMPLEMENTATION_REF_MISSING")
    manifest_path = require_relative_to(
        Path(str(implementation.get("manifest_path", ""))),
        scope,
        "INTEGRITY_MANIFEST_OUTSIDE_SCOPE",
    )
    verified = verify_implementation_manifest(
        manifest_path, project_root=root, scope_root=scope, runner=runner
    )
    if (
        verified.get("manifest_file_sha256")
        != implementation.get("manifest_file_sha256")
        or verified.get("implementation_manifest_sha256")
        != implementation.get("implementation_manifest_sha256")
    ):
        raise ContractViolation("CPU_CHECK_IMPLEMENTATION_REF_MISMATCH")
    current_delivery = verify_delivery_manifest(project_root=root)
    if evidence.get("delivery") != current_delivery:
        raise ContractViolation("CPU_CHECK_DELIVERY_REF_MISMATCH")
    checks = evidence.get("checks")
    expected_commands = dict(_cpu_test_commands(root))
    if (
        not isinstance(checks, list)
        or len(checks) != 2
        or {row.get("name") for row in checks if isinstance(row, Mapping)}
        != set(expected_commands)
    ):
        raise ContractViolation("CPU_CHECK_SET_MISMATCH")
    for check in checks:
        if (
            not isinstance(check, Mapping)
            or check.get("status") != "PASS"
            or check.get("return_code") != 0
            or check.get("command") != expected_commands.get(check.get("name"))
            or not isinstance(check.get("test_count"), int)
            or isinstance(check.get("test_count"), bool)
            or check["test_count"] < 1
        ):
            raise ContractViolation("CPU_CHECK_NOT_PASS")
        for stream in ("stdout", "stderr"):
            ref = check.get(stream)
            if not isinstance(ref, Mapping):
                raise ContractViolation("CPU_CHECK_LOG_REF_MISSING")
            log_path = require_relative_to(
                Path(str(ref.get("path", ""))),
                source.parent,
                "CPU_CHECK_LOG_OUTSIDE_EVIDENCE_DIRECTORY",
            )
            try:
                log_bytes = read_regular_file_bytes(log_path)
            except ContractViolation as exc:
                raise ContractViolation("CPU_CHECK_LOG_REF_MISMATCH") from exc
            if len(log_bytes) != ref.get("bytes") or sha256_bytes(log_bytes) != ref.get(
                "sha256"
            ):
                raise ContractViolation("CPU_CHECK_LOG_REF_MISMATCH")
    return {
        **evidence,
        "manifest_path": str(source.resolve()),
        "manifest_sha256": evidence_file_sha256,
    }


__all__ = [
    "build_implementation_manifest",
    "publish_implementation_manifest",
    "run_cpu_check_evidence",
    "verify_cpu_check_evidence",
    "verify_delivery_manifest",
    "verify_implementation_manifest",
]
