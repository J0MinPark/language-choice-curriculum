"""Strict, Git-reviewed trust anchors for production artifacts.

Artifact-local SHA-256 fields detect accidental changes, but an attacker can
rewrite an artifact and recompute unkeyed hashes.  The independent authority
for this registry is therefore the reviewed Git commit containing it, together
with the verified implementation-integrity manifest for that commit.  This
module does not claim a human or keyed signature.

Only exact registry entries are accepted: kind and artifact ID select one
entry, after which both the manifest SHA-256 and the complete binding object
must match.  New production artifacts require a reviewed registry change.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .artifacts import read_regular_file_bytes
from .contracts import (
    IMPLEMENTATION_REVISION,
    PROJECT_ROOT,
    ContractViolation,
    canonical_json_bytes,
    require_sha256,
)


TRUSTED_ARTIFACT_REGISTRY_SCHEMA = "trusted-artifact-anchors-v1"
TRUSTED_ARTIFACT_REGISTRY_REVISION = "1"
TRUSTED_ARTIFACT_REGISTRY_PATH = (
    PROJECT_ROOT / "implementation" / "config" / "trusted_artifact_anchors.json"
)
TRUST_AUTHORITY = (
    "REVIEWED_GIT_COMMIT_AND_VERIFIED_IMPLEMENTATION_INTEGRITY_MANIFEST"
)
UNKEYED_HASH_ROLE = "INTEGRITY_BINDING_ONLY_NOT_AUTHENTICATION"
SUPPORTED_ARTIFACT_KINDS = frozenset(
    {
        "KRDICT_SOURCE_COLLECTION",
        "ANNOTATION_FREEZE",
        "EXPERIMENT_FREEZE",
    }
)

_REGISTRY_FIELDS = frozenset(
    {
        "schema_version",
        "registry_revision",
        "implementation_revision",
        "trust_model",
        "anchors",
    }
)
_TRUST_MODEL = {
    "authority": TRUST_AUTHORITY,
    "human_signature_claimed": False,
    "keyed_signature_claimed": False,
    "unkeyed_hash_role": UNKEYED_HASH_ROLE,
}
_TRUST_MODEL_FIELDS = frozenset(_TRUST_MODEL)
_ANCHOR_FIELDS = frozenset(
    {"artifact_kind", "artifact_id", "manifest_sha256", "bindings"}
)
_ARTIFACT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_BINDING_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")
_MAX_REGISTRY_BYTES = 1024 * 1024


def _reject_nonstandard_number(token: str) -> None:
    raise ContractViolation("INVALID_TRUST_REGISTRY_JSON")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation("DUPLICATE_TRUST_REGISTRY_KEY")
        result[key] = value
    return result


def _parse_registry(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_REGISTRY_BYTES:
        raise ContractViolation("INVALID_TRUST_REGISTRY_SIZE")
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_number,
        )
    except ContractViolation:
        raise
    except (UnicodeError, ValueError) as exc:
        raise ContractViolation("INVALID_TRUST_REGISTRY_JSON") from exc
    if not isinstance(parsed, dict):
        raise ContractViolation("INVALID_TRUST_REGISTRY")
    return parsed


def _validate_json_value(value: Any) -> None:
    """Require an ordinary finite JSON value with string object keys."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        # Artifact identities should not depend on lossy JSON numeric parsing.
        raise ContractViolation("INVALID_TRUST_ANCHOR_BINDINGS")
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not _BINDING_KEY_RE.fullmatch(key):
                raise ContractViolation("INVALID_TRUST_ANCHOR_BINDINGS")
            _validate_json_value(item)
        return
    raise ContractViolation("INVALID_TRUST_ANCHOR_BINDINGS")


def _validated_bindings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ContractViolation("INVALID_TRUST_ANCHOR_BINDINGS")
    _validate_json_value(value)
    # Exercise the canonical encoder here as an additional fail-closed check.
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ContractViolation("INVALID_TRUST_ANCHOR_BINDINGS") from exc
    return value


def _validate_anchor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ANCHOR_FIELDS:
        raise ContractViolation("INVALID_TRUST_ANCHOR")
    artifact_kind = value.get("artifact_kind")
    if (
        not isinstance(artifact_kind, str)
        or artifact_kind not in SUPPORTED_ARTIFACT_KINDS
    ):
        raise ContractViolation("INVALID_TRUST_ANCHOR_KIND")
    artifact_id = value.get("artifact_id")
    if not isinstance(artifact_id, str) or not _ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise ContractViolation("INVALID_TRUST_ANCHOR_ID")
    manifest_sha256 = value.get("manifest_sha256")
    if not isinstance(manifest_sha256, str):
        raise ContractViolation("INVALID_TRUST_ANCHOR_MANIFEST_SHA256")
    require_sha256(manifest_sha256, "INVALID_TRUST_ANCHOR_MANIFEST_SHA256")
    _validated_bindings(value.get("bindings"))
    return value


def validate_trusted_artifact_registry(value: Any) -> dict[str, Any]:
    """Validate and return a defensive copy of one complete registry."""
    if not isinstance(value, dict) or set(value) != _REGISTRY_FIELDS:
        raise ContractViolation("INVALID_TRUST_REGISTRY")
    if value.get("schema_version") != TRUSTED_ARTIFACT_REGISTRY_SCHEMA:
        raise ContractViolation("TRUST_REGISTRY_SCHEMA_MISMATCH")
    if value.get("registry_revision") != TRUSTED_ARTIFACT_REGISTRY_REVISION:
        raise ContractViolation("TRUST_REGISTRY_REVISION_MISMATCH")
    if value.get("implementation_revision") != IMPLEMENTATION_REVISION:
        raise ContractViolation("TRUST_REGISTRY_IMPLEMENTATION_REVISION_MISMATCH")
    trust_model = value.get("trust_model")
    if (
        not isinstance(trust_model, dict)
        or set(trust_model) != _TRUST_MODEL_FIELDS
        or trust_model.get("authority") != TRUST_AUTHORITY
        or trust_model.get("human_signature_claimed") is not False
        or trust_model.get("keyed_signature_claimed") is not False
        or trust_model.get("unkeyed_hash_role") != UNKEYED_HASH_ROLE
    ):
        raise ContractViolation("TRUST_REGISTRY_TRUST_MODEL_MISMATCH")
    anchors = value.get("anchors")
    if not isinstance(anchors, list):
        raise ContractViolation("INVALID_TRUST_REGISTRY_ANCHORS")
    seen: set[tuple[str, str]] = set()
    previous: tuple[str, str] | None = None
    for raw_anchor in anchors:
        anchor = _validate_anchor(raw_anchor)
        identity = (anchor["artifact_kind"], anchor["artifact_id"])
        if identity in seen:
            raise ContractViolation("DUPLICATE_TRUST_ANCHOR")
        if previous is not None and identity <= previous:
            raise ContractViolation("NONCANONICAL_TRUST_ANCHOR_ORDER")
        seen.add(identity)
        previous = identity
    return copy.deepcopy(value)


def load_trusted_artifact_registry(
    registry_path: Path | str = TRUSTED_ARTIFACT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Read a regular registry file and validate its fixed schema."""
    try:
        raw = read_regular_file_bytes(registry_path)
    except ContractViolation as exc:
        raise ContractViolation("TRUST_REGISTRY_NOT_REGULAR_FILE") from exc
    return validate_trusted_artifact_registry(_parse_registry(raw))


def require_trusted_artifact_anchor(
    *,
    artifact_kind: str,
    artifact_id: str,
    manifest_sha256: str,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an exact artifact anchor from the reviewed registry.

    ``artifact_kind`` and ``artifact_id`` select an entry.  They are never
    inferred from a digest or prefix.  The selected entry then has to match the
    full manifest digest and binding object, including key set and JSON types.
    """
    if (
        not isinstance(artifact_kind, str)
        or artifact_kind not in SUPPORTED_ARTIFACT_KINDS
    ):
        raise ContractViolation("INVALID_TRUST_ANCHOR_KIND")
    if not isinstance(artifact_id, str) or not _ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise ContractViolation("INVALID_TRUST_ANCHOR_ID")
    if not isinstance(manifest_sha256, str):
        raise ContractViolation("INVALID_TRUST_ANCHOR_MANIFEST_SHA256")
    require_sha256(manifest_sha256, "INVALID_TRUST_ANCHOR_MANIFEST_SHA256")
    if not isinstance(bindings, Mapping):
        raise ContractViolation("INVALID_TRUST_ANCHOR_BINDINGS")
    supplied_bindings = dict(bindings)
    _validated_bindings(supplied_bindings)

    # Authorization always uses the code-owned path.  Accepting a runtime path
    # here would let a caller nominate its own trust authority.
    registry = load_trusted_artifact_registry(TRUSTED_ARTIFACT_REGISTRY_PATH)
    selected = next(
        (
            anchor
            for anchor in registry["anchors"]
            if anchor["artifact_kind"] == artifact_kind
            and anchor["artifact_id"] == artifact_id
        ),
        None,
    )
    if selected is None:
        raise ContractViolation("TRUSTED_ARTIFACT_ANCHOR_MISSING")
    if selected["manifest_sha256"] != manifest_sha256:
        raise ContractViolation("TRUSTED_ARTIFACT_MANIFEST_SHA256_MISMATCH")
    if canonical_json_bytes(selected["bindings"]) != canonical_json_bytes(
        supplied_bindings
    ):
        raise ContractViolation("TRUSTED_ARTIFACT_BINDINGS_MISMATCH")
    return copy.deepcopy(selected)


__all__ = [
    "SUPPORTED_ARTIFACT_KINDS",
    "TRUSTED_ARTIFACT_REGISTRY_PATH",
    "TRUSTED_ARTIFACT_REGISTRY_REVISION",
    "TRUSTED_ARTIFACT_REGISTRY_SCHEMA",
    "load_trusted_artifact_registry",
    "require_trusted_artifact_anchor",
    "validate_trusted_artifact_registry",
]
