from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from implementation.src.artifacts import (
    publish_verified_file_once,
    read_regular_file_bytes_exact,
)
from implementation.src.contracts import ContractViolation


class VerifiedFilePublicationTests(unittest.TestCase):
    def test_exact_reader_rejects_size_before_reading_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.bin"
            path.write_bytes(b"oversized")
            with mock.patch(
                "implementation.src.artifacts.os.read", wraps=os.read
            ) as read:
                with self.assertRaisesRegex(
                    ContractViolation, "ARTIFACT_SIZE_MISMATCH"
                ):
                    read_regular_file_bytes_exact(path, expected_bytes=1)
            read.assert_not_called()

    def test_exact_reader_rejects_growth_after_fstat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "growing.bin"
            path.write_bytes(b"abc")
            real_read = os.read
            calls = 0
            requested_sizes: list[int] = []

            def grow_after_exact_read(descriptor: int, size: int) -> bytes:
                nonlocal calls
                calls += 1
                requested_sizes.append(size)
                if calls == 2:
                    return b"x"
                return real_read(descriptor, size)

            with mock.patch(
                "implementation.src.artifacts.os.read",
                side_effect=grow_after_exact_read,
            ):
                with self.assertRaisesRegex(
                    ContractViolation, "ARTIFACT_SIZE_MISMATCH"
                ):
                    read_regular_file_bytes_exact(path, expected_bytes=3)
            self.assertEqual(requested_sizes, [3, 1])

    def test_exact_reader_rejects_early_eof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated-during-read.bin"
            path.write_bytes(b"abc")
            requested_sizes: list[int] = []

            def truncate_during_read(_descriptor: int, size: int) -> bytes:
                requested_sizes.append(size)
                return b"ab" if len(requested_sizes) == 1 else b""

            with mock.patch(
                "implementation.src.artifacts.os.read",
                side_effect=truncate_during_read,
            ):
                with self.assertRaisesRegex(
                    ContractViolation, "ARTIFACT_SIZE_MISMATCH"
                ):
                    read_regular_file_bytes_exact(path, expected_bytes=3)
            self.assertEqual(requested_sizes, [3, 1])

    def test_exact_reader_rejects_invalid_expected_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(b"abc")
            with mock.patch(
                "implementation.src.artifacts.os.open", wraps=os.open
            ) as opened:
                for invalid in (-1, True, 1.5, "3", None):
                    with self.subTest(invalid=invalid):
                        with self.assertRaisesRegex(
                            ContractViolation, "INVALID_ARTIFACT_SIZE"
                        ):
                            read_regular_file_bytes_exact(
                                path, expected_bytes=invalid  # type: ignore[arg-type]
                            )
            opened.assert_not_called()

    def test_verified_copy_is_exact_read_only_and_write_once(self) -> None:
        payload = (b"verified-evidence-payload\n" * 50_000) + b"final"
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "published" / "evidence.bin"
            source.write_bytes(payload)

            artifact = publish_verified_file_once(
                source,
                destination,
                expected_sha256=expected_sha256,
                expected_bytes=len(payload),
            )

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(artifact["sha256"], expected_sha256)
            self.assertEqual(artifact["bytes"], len(payload))
            self.assertEqual(
                artifact["write_semantics"], "ATOMIC_WRITE_ONCE_VERIFIED_COPY"
            )
            self.assertEqual(destination.stat().st_mode & 0o777, 0o444)
            with self.assertRaisesRegex(ContractViolation, "ARTIFACT_EXISTS"):
                publish_verified_file_once(
                    source,
                    destination,
                    expected_sha256=expected_sha256,
                    expected_bytes=len(payload),
                )

    def test_size_mismatch_never_publishes_destination(self) -> None:
        payload = b"evidence"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(payload)

            with self.assertRaisesRegex(
                ContractViolation, "ARTIFACT_SIZE_MISMATCH"
            ):
                publish_verified_file_once(
                    source,
                    destination,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_bytes=len(payload) + 1,
                )

            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".destination.bin.*.tmp")), [])

    def test_hash_mismatch_never_publishes_destination(self) -> None:
        payload = b"evidence"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(payload)

            with self.assertRaisesRegex(
                ContractViolation, "ARTIFACT_HASH_MISMATCH"
            ):
                publish_verified_file_once(
                    source,
                    destination,
                    expected_sha256="0" * 64,
                    expected_bytes=len(payload),
                )

            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".destination.bin.*.tmp")), [])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_symlink_source_is_rejected_without_publication(self) -> None:
        payload = b"evidence"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual.bin"
            source = root / "source.bin"
            destination = root / "destination.bin"
            actual.write_bytes(payload)
            source.symlink_to(actual)

            with self.assertRaisesRegex(
                ContractViolation, "ARTIFACT_NOT_REGULAR_FILE"
            ):
                publish_verified_file_once(
                    source,
                    destination,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_bytes=len(payload),
                )

            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
