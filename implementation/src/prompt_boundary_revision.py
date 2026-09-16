"""Exact v4.1.2 pre-training correction of 36 trailing-space templates."""
import copy
import json
from pathlib import Path

from .artifacts import read_regular_file_bytes_exact
from .contracts import PROJECT_ROOT, ContractViolation, sha256_bytes

PLAN_PATH = PROJECT_ROOT / "implementation/config/evaluation_plan_v4_1_2.json"
AMENDMENT_PATH = PROJECT_ROOT / "implementation/protocol/AMENDMENT_V4_1_2_PROMPT_BOUNDARY.json"
AMENDMENT_SHA256 = "313cf33ad929384fd16ed0a8af225155f9064b3daad7e6e004e74d6e3eba4f3d"
AMENDMENT_BYTES = 1583


def _read(ref):
    declared = PROJECT_ROOT / ref["path"]
    path = declared.resolve(strict=True)
    if path != declared.absolute() or not path.is_relative_to(PROJECT_ROOT):
        raise ContractViolation("PROMPT_REVISION_PATH_MISMATCH")
    raw = read_regular_file_bytes_exact(path, expected_bytes=ref["bytes"])
    if sha256_bytes(raw) != ref["sha256"]:
        raise ContractViolation("PROMPT_REVISION_HASH_MISMATCH")
    return json.loads(raw)


def audit_prompt_revision():
    amendment_ref = {"path": str(AMENDMENT_PATH.relative_to(PROJECT_ROOT)), "sha256": AMENDMENT_SHA256, "bytes": AMENDMENT_BYTES}
    amendment = _read(amendment_ref)
    parent = _read(amendment["parent_evaluation_plan"])
    _read(amendment["parent_protocol_amendment"])
    revised = _read(amendment["revised_evaluation_plan"])
    expected = copy.deepcopy(parent)
    changes = []
    for wrapper, languages in expected["prompt_plan"]["templates"].items():
        for language, modes in languages.items():
            for mode, template in modes.items():
                stripped = template.rstrip(" ")
                if stripped != template:
                    changes.append((wrapper, language, mode))
                modes[mode] = stripped
    if revised != expected or len(changes) != 36:
        raise ContractViolation("PROMPT_REVISION_SCOPE_MISMATCH")
    return {
        "status": "PASS_PROMPT_BOUNDARY_REVISION",
        "amendment_artifact": {**amendment_ref, "path": str(AMENDMENT_PATH)},
        "parent_evaluation_plan": amendment["parent_evaluation_plan"],
        "evaluation_plan": {**amendment["revised_evaluation_plan"], "path": str(PLAN_PATH)},
        "changed_template_count": len(changes),
        "scientific_outcome_invariance_claimed": False,
    }


def resolve_boundary_plan(plan_path, tokenizer_plan_ref):
    """Allow the exact new lexical plan with the unchanged tokenizer policy."""
    declared = Path(plan_path).absolute()
    if declared != declared.resolve() or declared != PLAN_PATH:
        raise ContractViolation("PROMPT_REVISION_PATH_MISMATCH")
    result = audit_prompt_revision()
    parent = result["parent_evaluation_plan"]
    if (Path(tokenizer_plan_ref["path"]) != PROJECT_ROOT / parent["path"]
        or tokenizer_plan_ref["sha256"] != parent["sha256"]
        or tokenizer_plan_ref["bytes"] != parent["bytes"]):
        raise ContractViolation("PROMPT_REVISION_TOKENIZER_PARENT_MISMATCH")
    return result["evaluation_plan"]
