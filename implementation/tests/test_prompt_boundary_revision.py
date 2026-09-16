import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from implementation.src import prompt_boundary_revision as revision
from implementation.src.contracts import ContractViolation, PROJECT_ROOT


class PromptBoundaryRevisionTests(unittest.TestCase):
    def test_only_36_trailing_spaces_change(self):
        audit = revision.audit_prompt_revision()
        self.assertEqual(audit["status"], "PASS_PROMPT_BOUNDARY_REVISION")
        self.assertEqual(audit["changed_template_count"], 36)
        parent = json.loads((PROJECT_ROOT / audit["parent_evaluation_plan"]["path"]).read_bytes())
        new = json.loads(Path(audit["evaluation_plan"]["path"]).read_bytes())
        for key in parent:
            if key != "prompt_plan":
                self.assertEqual(parent[key], new[key])
        for wrapper, languages in parent["prompt_plan"]["templates"].items():
            for language, modes in languages.items():
                for mode, text in modes.items():
                    actual = new["prompt_plan"]["templates"][wrapper][language][mode]
                    self.assertEqual(actual, text.rstrip(" "))
                    if language == "zh":
                        self.assertEqual(actual, text)

    def test_wrong_tokenizer_parent_or_unregistered_plan_rejected(self):
        audit = revision.audit_prompt_revision()
        parent = audit["parent_evaluation_plan"]
        ref = {**parent, "path": str(PROJECT_ROOT / parent["path"])}
        self.assertEqual(revision.resolve_boundary_plan(revision.PLAN_PATH, ref), audit["evaluation_plan"])
        ref["sha256"] = "a" * 64
        with self.assertRaisesRegex(ContractViolation, "PROMPT_REVISION_TOKENIZER_PARENT_MISMATCH"):
            revision.resolve_boundary_plan(revision.PLAN_PATH, ref)
        with self.assertRaisesRegex(ContractViolation, "PROMPT_REVISION_PATH_MISMATCH"):
            revision.resolve_boundary_plan(PROJECT_ROOT / "other.json", ref)

    def test_same_size_plan_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = [revision.AMENDMENT_PATH, revision.PLAN_PATH,
                     PROJECT_ROOT / "implementation/config/evaluation_plan.json",
                     PROJECT_ROOT / "implementation/protocol/AMENDMENT_V4_1_1_EGG.json"]
            for source in paths:
                target = root / source.relative_to(PROJECT_ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            new_plan = root / revision.PLAN_PATH.relative_to(PROJECT_ROOT)
            raw = new_plan.read_bytes().replace(b'"context_length": 256', b'"context_length": 512')
            new_plan.write_bytes(raw)
            with mock.patch.object(revision, "PROJECT_ROOT", root), \
                 mock.patch.object(revision, "PLAN_PATH", new_plan), \
                 mock.patch.object(revision, "AMENDMENT_PATH", root / revision.AMENDMENT_PATH.relative_to(PROJECT_ROOT)):
                with self.assertRaisesRegex(ContractViolation, "PROMPT_REVISION_HASH_MISMATCH"):
                    revision.audit_prompt_revision()


if __name__ == "__main__":
    unittest.main()
