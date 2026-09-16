from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from implementation.src.artifacts import publish_json_once
from implementation.src.contracts import ContractViolation, canonical_json_bytes
from implementation.src.integrity import (
    build_implementation_manifest,
    publish_implementation_manifest,
    run_cpu_check_evidence,
    verify_cpu_check_evidence,
    verify_implementation_manifest,
)


def run_git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def make_repo(root: Path) -> None:
    run_git(root, "init")
    run_git(root, "config", "user.email", "test@example.invalid")
    run_git(root, "config", "user.name", "Integrity Test")
    (root / "implementation").mkdir()
    (root / "implementation/a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "implementation/b.txt").write_text("stable\n", encoding="utf-8")
    (root / "MANIFEST.json").write_text(
        json.dumps({"version": "4.0.0", "sha256": {}}) + "\n", encoding="utf-8"
    )
    run_git(root, "add", "implementation", "MANIFEST.json")
    run_git(root, "commit", "-m", "fixture")


class ImplementationIntegrityTests(unittest.TestCase):
    def test_manifest_binds_every_tracked_file_and_current_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            make_repo(root)
            work = root / "work"
            work.mkdir()
            result = publish_implementation_manifest(
                work / "implementation_manifest.json",
                project_root=root,
                scope_root=work,
            )
            self.assertEqual(result["file_count"], 2)
            self.assertEqual(
                [row["path"] for row in result["files"]],
                ["implementation/a.py", "implementation/b.txt"],
            )
            verified = verify_implementation_manifest(
                work / "implementation_manifest.json", project_root=root
            )
            self.assertEqual(verified["git_commit"], result["git_commit"])

    def test_dirty_or_untracked_implementation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            make_repo(root)
            (root / "implementation/a.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ContractViolation, "IMPLEMENTATION_WORKTREE_NOT_CLEAN"
            ):
                build_implementation_manifest(project_root=root)
            run_git(root, "checkout", "--", "implementation/a.py")
            (root / "implementation/untracked.py").write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ContractViolation, "IMPLEMENTATION_WORKTREE_NOT_CLEAN"
            ):
                build_implementation_manifest(project_root=root)
            (root / "implementation/untracked.py").unlink()
            run_git(root, "update-index", "--assume-unchanged", "implementation/a.py")
            (root / "implementation/a.py").write_text("VALUE = 99\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ContractViolation, "IMPLEMENTATION_WORKTREE_NOT_CLEAN"
            ):
                build_implementation_manifest(project_root=root)

    def test_manifest_tamper_and_old_head_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            make_repo(root)
            work = root / "work"
            work.mkdir()
            path = work / "implementation_manifest.json"
            manifest = build_implementation_manifest(project_root=root)
            publish_json_once(path, manifest)
            (root / "implementation/a.py").write_text("VALUE = 3\n", encoding="utf-8")
            run_git(root, "add", "implementation/a.py")
            run_git(root, "commit", "-m", "change")
            with self.assertRaisesRegex(
                ContractViolation, "IMPLEMENTATION_MANIFEST_NOT_CURRENT_HEAD"
            ):
                verify_implementation_manifest(path, project_root=root)

    def test_cpu_evidence_contains_logs_runtime_counts_and_code_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            make_repo(root)
            (root / "tests").mkdir()
            reference_test = root / "tests/test_reference.py"
            reference_test.write_text(
                "import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)\n",
                encoding="utf-8",
            )
            (root / "implementation/tests").mkdir()
            implementation_test = root / "implementation/tests/test_impl.py"
            implementation_test.write_text(
                "import os\n"
                "import socket\n"
                "import unittest\n"
                "class T(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertEqual(2 + 2, 4)\n"
                "        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')\n"
                "        for name in ('PYTHONPATH', 'HTTP_PROXY', 'HTTPS_PROXY', "
                "'ALL_PROXY', 'NO_PROXY', 'KRDICT_API_KEY', 'AWS_SECRET_ACCESS_KEY', "
                "'CPU_CHECK_SECRET_SENTINEL'):\n"
                "            self.assertNotIn(name, os.environ)\n"
                "        with self.assertRaisesRegex(PermissionError, "
                "'PYTHON_INET_NETWORK_DISABLED'):\n"
                "            socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n",
                encoding="utf-8",
            )
            delivered = root / "delivered.txt"
            delivered.write_text("delivered\n", encoding="utf-8")
            digest = hashlib.sha256(delivered.read_bytes()).hexdigest()
            (root / "MANIFEST.json").write_text(
                json.dumps({"version": "4.0.0", "sha256": {"delivered.txt": digest}})
                + "\n",
                encoding="utf-8",
            )
            run_git(root, "add", "implementation/tests/test_impl.py")
            run_git(root, "commit", "-m", "add implementation test")
            work = root / "work"
            work.mkdir()
            manifest_path = work / "implementation_manifest.json"
            publish_implementation_manifest(
                manifest_path, project_root=root, scope_root=work
            )
            hostile_parent_environment = {
                "PYTHONPATH": "/tmp/should-not-be-inherited",
                "HTTP_PROXY": "http://proxy.invalid",
                "HTTPS_PROXY": "http://proxy.invalid",
                "ALL_PROXY": "socks5://proxy.invalid",
                "NO_PROXY": "internal.invalid",
                "KRDICT_API_KEY": "not-a-real-key",
                "AWS_SECRET_ACCESS_KEY": "not-a-real-secret",
                "CPU_CHECK_SECRET_SENTINEL": "must-not-cross-process-boundary",
            }
            with mock.patch.dict(os.environ, hostile_parent_environment, clear=False):
                result = run_cpu_check_evidence(
                    manifest_path,
                    output_dir=work / "cpu",
                    project_root=root,
                    scope_root=work,
                )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["schema_version"], "cpu-checks-v3")
            self.assertNotIn("offline_environment", result)
            self.assertNotIn("side_effects", result)
            self.assertEqual([row["test_count"] for row in result["checks"]], [1, 1])
            self.assertIn("python", result["runtime"])
            self.assertEqual(result["implementation"]["file_count"], 3)
            controls = result["execution_controls"]
            self.assertEqual(controls["subprocess_environment_policy"], "FIXED_ALLOWLIST_V1")
            self.assertEqual(
                set(controls["subprocess_environment"]),
                {
                    "CUDA_DEVICE_ORDER",
                    "CUDA_VISIBLE_DEVICES",
                    "HF_DATASETS_OFFLINE",
                    "HF_HUB_OFFLINE",
                    "PATH",
                    "TOKENIZERS_PARALLELISM",
                    "TRANSFORMERS_OFFLINE",
                },
            )
            self.assertEqual(
                controls["guard_installed"],
                "AFTER_INTERPRETER_STARTUP_BEFORE_TEST_DISCOVERY",
            )
            self.assertFalse(result["limitations"]["os_network_namespace_isolated"])
            self.assertFalse(result["limitations"]["network_nonuse_proven"])
            self.assertFalse(result["limitations"]["gpu_nonuse_proven"])
            verified = verify_cpu_check_evidence(
                work / "cpu/cpu_checks.json", project_root=root
            )
            self.assertEqual(verified["cpu_check_sha256"], result["cpu_check_sha256"])

            original = json.loads(
                (work / "cpu/cpu_checks.json").read_text(encoding="utf-8")
            )
            for field in ("python", "python_implementation", "platform", "executable"):
                with self.subTest(stale_runtime_field=field):
                    stale = json.loads(json.dumps(original))
                    stale["runtime"][field] = f"STALE_{field}"
                    stale_core = {
                        key: value
                        for key, value in stale.items()
                        if key != "cpu_check_sha256"
                    }
                    stale["cpu_check_sha256"] = hashlib.sha256(
                        canonical_json_bytes(stale_core)
                    ).hexdigest()
                    stale_path = work / f"cpu/stale_{field}.json"
                    publish_json_once(stale_path, stale)
                    with self.assertRaisesRegex(
                        ContractViolation, "CPU_CHECK_RUNTIME_MISMATCH"
                    ):
                        verify_cpu_check_evidence(stale_path, project_root=root)

            stale_dependencies = json.loads(json.dumps(original))
            current_numpy = stale_dependencies["runtime"]["dependencies"]["numpy"]
            stale_dependencies["runtime"]["dependencies"]["numpy"] = (
                "0.0.0-STALE" if current_numpy != "0.0.0-STALE" else "0.0.1-STALE"
            )
            stale_dependencies_core = {
                key: value
                for key, value in stale_dependencies.items()
                if key != "cpu_check_sha256"
            }
            stale_dependencies["cpu_check_sha256"] = hashlib.sha256(
                canonical_json_bytes(stale_dependencies_core)
            ).hexdigest()
            stale_dependencies_path = work / "cpu/stale_dependencies.json"
            publish_json_once(stale_dependencies_path, stale_dependencies)
            with self.assertRaisesRegex(
                ContractViolation, "CPU_CHECK_RUNTIME_MISMATCH"
            ):
                verify_cpu_check_evidence(
                    stale_dependencies_path, project_root=root
                )

            forged = json.loads((work / "cpu/cpu_checks.json").read_text(encoding="utf-8"))
            forged["limitations"]["gpu_nonuse_proven"] = True
            forged_core = {
                key: value for key, value in forged.items() if key != "cpu_check_sha256"
            }
            forged["cpu_check_sha256"] = hashlib.sha256(
                canonical_json_bytes(forged_core)
            ).hexdigest()
            publish_json_once(work / "cpu/forged_claims.json", forged)
            with self.assertRaisesRegex(
                ContractViolation, "CPU_CHECK_CONTROL_CLAIM_MISMATCH"
            ):
                verify_cpu_check_evidence(
                    work / "cpu/forged_claims.json", project_root=root
                )

            log_path = Path(result["checks"][0]["stderr"]["path"])
            log_path.chmod(0o644)
            log_path.write_bytes(log_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(
                ContractViolation, "CPU_CHECK_LOG_REF_MISMATCH"
            ):
                verify_cpu_check_evidence(
                    work / "cpu/cpu_checks.json", project_root=root
                )


if __name__ == "__main__":
    unittest.main()
