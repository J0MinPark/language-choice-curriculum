from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest import mock

from implementation.src.contracts import ContractViolation
from implementation.src.pilot import (
    BASE_GATES,
    H_DIAGNOSTIC_STEPS,
    LEXICAL_DIAGNOSTIC_STEPS,
    MISSING_PRODUCTION_INTERFACES,
    derive_pilot_decision,
    run_production_pilot,
)


GPU2_UUID = "GPU-de18b86c-419a-795a-0667-c7ae03bf800f"


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def base_state() -> dict:
    state = {
        "schema_version": "pilot-state-v1",
        "project_id": "lexical-freshstart-4lang-ety-v4",
        "main_enabled": False,
        "freeze_sha256": digest("freeze"),
        "config_sha256": digest("config"),
        "tokenizer_sha256": digest("tokenizer"),
        "code_sha256": digest("code"),
        "evaluation_plan_sha256": digest("evaluation"),
        "corpus_token_count": 2_000_000,
        "base_gates": {},
        "roots": {},
        "history": {},
    }
    for name in BASE_GATES:
        state["base_gates"][name] = {
            "status": "PASS",
            "evidence_sha256": digest("gate:" + name),
        }
    state["base_gates"]["freeze"]["freeze_sha256"] = state["freeze_sha256"]
    state["base_gates"]["gpu_replay"].update(
        {
            "freeze_sha256": state["freeze_sha256"],
            "code_sha256": state["code_sha256"],
            "scientific_result": False,
        }
    )
    return state


def checkpoint(
    state: dict,
    root_id: int,
    phase: str,
    *,
    parent: dict | None,
    initial: str,
) -> dict:
    if phase == "INIT":
        global_step, phase_step, tokens = 0, 0, 0
    elif phase == "CORPUS":
        global_step, phase_step, tokens = 100, 100, 2_000_000
    elif phase in {"T0", "T1", "T2", "T3"}:
        global_step = parent["global_step"] + 3000
        phase_step, tokens = 3000, parent["model_tokens_seen"] + 1000
    elif phase == "H_BASE":
        global_step = parent["global_step"]
        phase_step, tokens = 0, parent["model_tokens_seen"]
    else:
        global_step = parent["global_step"] + 300
        phase_step, tokens = 300, parent["model_tokens_seen"] + 100
    result = {
        "status": "PASS",
        "root_id": root_id,
        "phase": phase,
        "stage": "H" if phase.startswith("H_") else phase,
        "branch": {"H_A": "A", "H_B": "B"}.get(phase),
        "data_kind": "REAL",
        "freeze_gate_status": "PASS",
        "checkpoint_sha256": digest(f"checkpoint:{root_id}:{phase}"),
        "manifest_sha256": digest(f"manifest:{root_id}:{phase}"),
        "model_fingerprint": initial if phase == "INIT" else digest(f"model:{root_id}:{phase}"),
        "initial_model_fingerprint": initial,
        "plan_sha256": digest(f"plan:{root_id}:{phase}"),
        "gpu_binding": {
            "physical_index": 2,
            "logical_index": 0,
            "uuid": GPU2_UUID,
        },
        "budget_lease_id": f"lease-{root_id}-{phase}",
        "global_step": global_step,
        "phase_step": phase_step,
        "model_tokens_seen": tokens,
        "accumulation_step": 0,
        "phase_parent_sha256": None if parent is None else parent["checkpoint_sha256"],
        "resume_parent_sha256": None,
    }
    for name in (
        "freeze_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "code_sha256",
        "evaluation_plan_sha256",
    ):
        result[name] = state[name]
    if phase == "H_BASE":
        result["learning_rate"] = 0.0001
    return result


def evaluation(state: dict, checkpoint_value: dict, *, history: bool = False) -> dict:
    result = {
        "status": "PASS",
        "root_id": checkpoint_value["root_id"],
        "phase": checkpoint_value["phase"],
        "stage": checkpoint_value["stage"],
        "branch": checkpoint_value["branch"],
        "checkpoint_sha256": checkpoint_value["checkpoint_sha256"],
        "model_fingerprint": checkpoint_value["model_fingerprint"],
        "diagnostic_steps": list(
            H_DIAGNOSTIC_STEPS if history else LEXICAL_DIAGNOSTIC_STEPS
        ),
        "rng_unchanged": True,
        "fixed_endpoint_used": True,
        "data_kind": "REAL",
        "score_artifact_sha256": digest(
            "score:" + str(checkpoint_value["checkpoint_sha256"])
        ),
    }
    for name in (
        "freeze_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "code_sha256",
        "evaluation_plan_sha256",
    ):
        result[name] = state[name]
    return result


def complete_root(state: dict, root_id: int, *, readiness: str = "PASS", measurement: str = "PASS") -> dict:
    checkpoints = {}
    evaluations = {}
    parent = None
    initial = digest(f"initial:{root_id}")
    for phase in ("INIT", "CORPUS", "T0", "T1", "T2", "T3"):
        parent = checkpoint(state, root_id, phase, parent=parent, initial=initial)
        checkpoints[phase] = parent
        if phase.startswith("T"):
            evaluations[phase] = evaluation(state, parent)
    root = {"checkpoints": checkpoints, "evaluations": evaluations}
    root["readiness"] = {
        "status": readiness,
        "root_id": root_id,
        "checkpoint_sha256": checkpoints["T3"]["checkpoint_sha256"],
        "score_artifact_sha256": evaluations["T3"]["score_artifact_sha256"],
        "gate_artifact_sha256": digest(f"readiness:{root_id}"),
    }
    if readiness == "PASS":
        root["measurement"] = {
            "status": measurement,
            "root_id": root_id,
            "checkpoint_sha256": checkpoints["T3"]["checkpoint_sha256"],
            "score_artifact_sha256": evaluations["T3"]["score_artifact_sha256"],
            "gate_artifact_sha256": digest(f"measurement:{root_id}"),
        }
    return root


def complete_history(state: dict, root_id: int) -> dict:
    t3 = state["roots"][str(root_id)]["checkpoints"]["T3"]
    initial = t3["initial_model_fingerprint"]
    base = checkpoint(state, root_id, "H_BASE", parent=t3, initial=initial)
    a = checkpoint(state, root_id, "H_A", parent=base, initial=initial)
    b = checkpoint(state, root_id, "H_B", parent=base, initial=initial)
    return {
        "checkpoints": {"H_BASE": base, "H_A": a, "H_B": b},
        "evaluations": {
            "H_A": evaluation(state, a, history=True),
            "H_B": evaluation(state, b, history=True),
        },
    }


class PilotOrderingTests(unittest.TestCase):
    def test_base_gates_then_root_4101_only(self):
        state = base_state()
        state["base_gates"].pop("gpu_replay")
        decision = derive_pilot_decision(state)
        self.assertEqual(decision.status, "BLOCKED_PREREQUISITES")
        self.assertFalse(decision.authorized)

        state = base_state()
        decision = derive_pilot_decision(state)
        self.assertEqual([(a.root_id, a.phase) for a in decision.actions], [(4101, "INIT")])
        self.assertFalse(decision.actions[0].performs_learning)
        self.assertTrue(decision.actions[0].requires_gpu_session)

    def test_expansion_before_4101_gates_is_rejected(self):
        state = base_state()
        state["roots"]["4102"] = {"checkpoints": {}, "evaluations": {}}
        with self.assertRaisesRegex(ContractViolation, "ROOT_EXPANDED_BEFORE_4101_GATES"):
            derive_pilot_decision(state)

    def test_exact_stage_checkpoint_then_evaluation_order(self):
        state = base_state()
        initial = digest("initial:4101")
        init = checkpoint(state, 4101, "INIT", parent=None, initial=initial)
        corpus = checkpoint(state, 4101, "CORPUS", parent=init, initial=initial)
        t0 = checkpoint(state, 4101, "T0", parent=corpus, initial=initial)
        state["roots"]["4101"] = {
            "checkpoints": {"INIT": init, "CORPUS": corpus, "T0": t0},
            "evaluations": {},
        }
        decision = derive_pilot_decision(state)
        self.assertEqual((decision.actions[0].kind, decision.actions[0].phase), ("EVALUATE_STAGE", "T0"))
        state["roots"]["4101"]["checkpoints"]["T1"] = checkpoint(
            state, 4101, "T1", parent=t0, initial=initial
        )
        with self.assertRaisesRegex(ContractViolation, "NEXT_STAGE_BEFORE_EVALUATION"):
            derive_pilot_decision(state)

    def test_only_root4101_pass_unlocks_remaining_roots(self):
        state = base_state()
        state["roots"]["4101"] = complete_root(state, 4101)
        decision = derive_pilot_decision(state)
        self.assertEqual(
            {(action.root_id, action.phase) for action in decision.actions},
            {(4102, "INIT"), (4103, "INIT"), (4104, "INIT")},
        )

    def test_history_waits_for_every_root_gate(self):
        state = base_state()
        for root_id in (4101, 4102, 4103):
            state["roots"][str(root_id)] = complete_root(state, root_id)
        state["roots"]["4104"] = complete_root(
            state, 4104, measurement="BLOCKED_MEASUREMENT"
        )
        decision = derive_pilot_decision(state)
        self.assertEqual(decision.status, "BLOCKED_REMAINING_ROOTS")
        self.assertFalse(decision.authorized)
        state["history"]["4101"] = {}
        with self.assertRaisesRegex(ContractViolation, "HISTORY_BEFORE_ALL_ROOT_GATES"):
            derive_pilot_decision(state)

    def test_all_roots_unlock_H_and_branches_share_H_base(self):
        state = base_state()
        for root_id in (4101, 4102, 4103, 4104):
            state["roots"][str(root_id)] = complete_root(state, root_id)
        decision = derive_pilot_decision(state)
        self.assertEqual(
            {(action.root_id, action.phase) for action in decision.actions},
            {(4101, "H_BASE"), (4102, "H_BASE"), (4103, "H_BASE"), (4104, "H_BASE")},
        )
        for root_id in (4101, 4102, 4103, 4104):
            state["history"][str(root_id)] = complete_history(state, root_id)
        complete = derive_pilot_decision(state)
        self.assertEqual(complete.status, "PILOT_COMPLETE_CANDIDATE")
        self.assertFalse(complete.authorized)

    def test_cross_root_initializations_are_distinct_before_history(self):
        state = base_state()
        for root_id in (4101, 4102, 4103, 4104):
            state["roots"][str(root_id)] = complete_root(state, root_id)

        distinct = derive_pilot_decision(state)
        self.assertEqual(distinct.status, "AUTHORIZED")
        self.assertEqual({action.phase for action in distinct.actions}, {"H_BASE"})

        shared_initial = state["roots"]["4101"]["checkpoints"]["INIT"][
            "initial_model_fingerprint"
        ]
        duplicate_root = state["roots"]["4102"]
        for marker in duplicate_root["checkpoints"].values():
            marker["initial_model_fingerprint"] = shared_initial
        duplicate_root["checkpoints"]["INIT"]["model_fingerprint"] = shared_initial

        with self.assertRaisesRegex(
            ContractViolation, "ROOT_INITIALIZATION_NOT_DISTINCT"
        ):
            derive_pilot_decision(state)

    def test_duplicate_later_root_init_blocks_its_next_gpu_action(self):
        state = base_state()
        state["roots"]["4101"] = complete_root(state, 4101)
        root_4102 = complete_root(state, 4102)
        root_4102 = {
            "checkpoints": {"INIT": root_4102["checkpoints"]["INIT"]},
            "evaluations": {},
        }
        shared_initial = state["roots"]["4101"]["checkpoints"]["INIT"][
            "initial_model_fingerprint"
        ]
        root_4102["checkpoints"]["INIT"][
            "initial_model_fingerprint"
        ] = shared_initial
        root_4102["checkpoints"]["INIT"]["model_fingerprint"] = shared_initial
        state["roots"]["4102"] = root_4102

        with self.assertRaisesRegex(
            ContractViolation, "ROOT_INITIALIZATION_NOT_DISTINCT"
        ):
            derive_pilot_decision(state)

    def test_gate_computation_actions_do_not_require_gpu_session(self):
        state = base_state()
        root = complete_root(state, 4101)
        readiness = root.pop("readiness")
        root.pop("measurement")
        state["roots"]["4101"] = root

        readiness_decision = derive_pilot_decision(state)
        self.assertEqual(readiness_decision.actions[0].kind, "COMPUTE_READINESS")
        self.assertFalse(readiness_decision.actions[0].requires_gpu_session)

        root["readiness"] = readiness
        measurement_decision = derive_pilot_decision(state)
        self.assertEqual(measurement_decision.actions[0].kind, "COMPUTE_MEASUREMENT")
        self.assertFalse(measurement_decision.actions[0].requires_gpu_session)

    def test_main_is_never_authorized(self):
        decision = derive_pilot_decision(base_state(), requested_main=True)
        self.assertEqual(decision.status, "MAIN_NOT_AUTHORIZED")
        self.assertFalse(decision.authorized)
        state = base_state()
        state["main_enabled"] = True
        with self.assertRaisesRegex(ContractViolation, "MAIN_NOT_AUTHORIZED"):
            derive_pilot_decision(state)


class ProductionPreflightTests(unittest.TestCase):
    def test_production_preflight_is_explicit_not_implemented_and_starts_no_gpu(self):
        state = base_state()

        class Ledger:
            def snapshot(self):
                return {
                    "campaign_cap_gpu_hours": 24.0,
                    "prior_gpu_hours_user_reported": 0.953,
                    "campaign_gpu_hours_charged_or_reserved": 1.0,
                    "campaign_gpu_hours_remaining": 23.0,
                    "per_root_gpu_seconds_charged_or_reserved": {},
                }

        freeze = {
            "verified": True,
            "status": "PASS",
            "freeze_gate_status": "PASS",
            "data_kind": "REAL",
            "freeze_sha256": state["freeze_sha256"],
            "tokenizer_file_sha256": state["tokenizer_sha256"],
            "corpus_token_count": state["corpus_token_count"],
            "artifacts": {
                "pilot_config": {"sha256": state["config_sha256"]},
                "evaluation_plan": {"sha256": state["evaluation_plan_sha256"]},
            },
        }
        with mock.patch(
            "implementation.src.prepare_data.verify_experiment_freeze",
            return_value=freeze,
        ) as verify:
            result = run_production_pilot(
                state=state,
                experiment_freeze_path=Path("unused-in-test.json"),
                ledger=Ledger(),
                code_sha256=state["code_sha256"],
            )
        verify.assert_called_once_with(Path("unused-in-test.json"), production=True)
        self.assertEqual(result["status"], "NOT_IMPLEMENTED")
        self.assertFalse(result["scientific_training_started"])
        self.assertFalse(result["gpu_session_started"])
        self.assertEqual(tuple(result["missing_interfaces"]), MISSING_PRODUCTION_INTERFACES)
        self.assertEqual(result["next_authorization"]["actions"][0]["root_id"], 4101)

    def test_live_budget_mismatch_fails_closed(self):
        state = base_state()

        class BadLedger:
            def snapshot(self):
                return {
                    "campaign_cap_gpu_hours": 25.0,
                    "prior_gpu_hours_user_reported": 0.953,
                }

        freeze = {
            "verified": True,
            "status": "PASS",
            "freeze_gate_status": "PASS",
            "data_kind": "REAL",
            "freeze_sha256": state["freeze_sha256"],
            "tokenizer_file_sha256": state["tokenizer_sha256"],
            "corpus_token_count": state["corpus_token_count"],
            "artifacts": {
                "pilot_config": {"sha256": state["config_sha256"]},
                "evaluation_plan": {"sha256": state["evaluation_plan_sha256"]},
            },
        }
        with mock.patch(
            "implementation.src.prepare_data.verify_experiment_freeze",
            return_value=freeze,
        ):
            with self.assertRaisesRegex(ContractViolation, "GPU_BUDGET_POLICY_MISMATCH"):
                run_production_pilot(
                    state=state,
                    experiment_freeze_path=Path("unused-in-test.json"),
                    ledger=BadLedger(),
                    code_sha256=state["code_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
