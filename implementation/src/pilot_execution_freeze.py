"""Additive executable pilot contract derived from the approved replay freeze.

No old freeze is relabelled. All scientific inputs and budgets are inherited;
only the already specified orchestration and source-record projection are added.
"""
from pathlib import Path

from .artifacts import publish_json_once, read_verified_json, sha256_file
from .contracts import ContractViolation, WORK_ROOT
from .semantic_experiment_freeze import verify_semantic_experiment_freeze, _hash

SCHEMA = "pilot-execution-freeze-v1"
PARENT = WORK_ROOT / "freeze/semantic_experiment_v4_1_2_jm02/experiment_freeze.json"
PARENT_SHA = "c3c98b8ac8dcb727ca8e3551dfdbae6e827bcac690514aed0e17466d6dd03b1a"
PHASES = ("INIT", "CORPUS", "T0", "T1", "T2", "T3", "H_BASE", "H_A", "H_B")
RETRAIN_REVISION = "zh-fr-balanced-5400-v1"
PROBE_REVISION = "accuracy-controlled-360-v1"
PROBE_NEXT_REVISION = "accuracy-refinement-360-v2"
PROBE_CONTINUE_REVISION = "accuracy-selected-360-v3"
TRAJECTORY_REVISION = "choice-trajectory-independent-seeds-v1"
REVERSE_REVISION = "choice-trajectory-paired-reverse-4101-v1"
REVERSE_SOURCE = {
    "path": str(WORK_ROOT / "runs/pilot_jm02_4e30ec2/checkpoints/4101_T0_003000_28f4069ccae8"),
    "state_sha256": "2b68d278f49a6811c72ad0f9664d11c00986cfb0db10e84b526c72977bb36e3b",
    "manifest_sha256": "1a3eb0ea3cd112ffffdb4706f78719275d864949796398115a0bc7fbf818c0d4",
    "freeze_sha256": "c26156ca836960913bfd1e1ae7b0c228099eda2aed94ee511fed818db99f7167",
    "code_sha256": "7700760479ec4a5271d7744d8289f421bf2b9e13156d300057b9ebefd2252f2c",
}
RETRAIN_SOURCE = {
    "path": str(WORK_ROOT / "runs/pilot_jm02_4e30ec2/checkpoints/4101_T1_003000_c81a25e0031f"),
    "state_sha256": "90f82104d268a3a52178bcc70acf981a453fe9294cf58dbb78bb3db4f80e7285",
    "manifest_sha256": "cda145d81da5cbb5bcadb424d7bf3c727c5558d3f716bad1180c4e6a340c4d52",
    "freeze_sha256": "c26156ca836960913bfd1e1ae7b0c228099eda2aed94ee511fed818db99f7167",
    "code_sha256": "7700760479ec4a5271d7744d8289f421bf2b9e13156d300057b9ebefd2252f2c",
}


def history_pairs(ids):
    ids = sorted(ids)
    if len(ids) != 60 or len(set(ids)) != 60:
        raise ContractViolation("PILOT_REQUIRES_60_DISTINCT_CONCEPTS")
    rows = []
    for i in range(30):
        core = {"pair_id": f"B36-pair-{i:02d}", "pair_index": i,
                "concept_ids": ids[2*i:2*i+2]}
        rows.append({**core, "pair_sha256": _hash(core)})
    core = {"status": "PASS", "history_family": "B36", "concept_order": ids,
            "pairing_rule": "lexicographically sort frozen concept_id and pair adjacent positions (0,1),(2,3),...",
            "pairs": rows}
    return {**core, "history_pair_set_sha256": _hash(core)}


def build_execution_freeze(*, revision=None):
    if revision == RETRAIN_REVISION:
        raise ContractViolation("DEFERRED_5400_PROPOSAL_NOT_EXECUTABLE")
    if revision not in (None, RETRAIN_REVISION, PROBE_REVISION, PROBE_NEXT_REVISION, PROBE_CONTINUE_REVISION, TRAJECTORY_REVISION, REVERSE_REVISION):
        raise ContractViolation("UNKNOWN_EXPLORATORY_REVISION")
    if sha256_file(PARENT) != PARENT_SHA:
        raise ContractViolation("PILOT_PARENT_FREEZE_MISMATCH")
    parent = verify_semantic_experiment_freeze(PARENT)
    annotations = read_verified_json(Path(parent["resolved_artifacts"]["annotation"]))
    sources = {row["candidate_id"]: sorted({ref["raw_sha256"]
        for term in row["terms"].values() for ref in term["source_refs"]})
        for row in annotations["concepts"]}
    concepts = [{**row, "source_record_hashes": sources[row["concept_id"]]}
                for row in parent["concepts"]]
    core = {"schema_version": SCHEMA, "protocol_version": "4.1.2",
        "parent_freeze": {"path": str(PARENT), "sha256": PARENT_SHA},
        "status": "PASS", "freeze_gate_status": "PASS", "data_kind": "REAL",
        "main_enabled": False, "scientific_training_authorized": True,
        "allowed_execution_phases": ["GPU_REPLAY", *PHASES],
        "artifacts": parent["artifacts"],
        "tokenizer_file_sha256": parent["tokenizer_file_sha256"],
        "evaluation_record_hash_schema": "score-record-v1",
        "concepts": concepts, "history_pairs": history_pairs(sources),
        "execution": {"roots": [4101, 4102, 4103, 4104],
            "corpus_tokens": 2000000, "lexical_updates": 3000,
            "batch_examples": 32, "diagnostic_interval": 200,
            "fixed_endpoint_only": True, "automatic_retry_on_error": False,
            "require_all_roots_before_history": True,
            "etymology_required": False, "optional_etymology": "NOT_RUN",
            "checkpoint_policy": "WRITE_ONCE_EVERY_200_AND_PHASE_ENDPOINT_AND_BOUNDARY_STOP",
            "retention_policy": "KEEP_ALL_PHASE_ENDPOINTS_AND_LATEST_TWO_INTERMEDIATE_STATES; KEEP_ALL_MANIFESTS_SCORES_TRACES",
            "disk_policy": "STOP_BEFORE_CHECKPOINT_IF_5_GIB_EMERGENCY_RESERVE_WOULD_BE_USED"}}
    if revision is not None and revision not in (TRAJECTORY_REVISION, REVERSE_REVISION):
        # Post-outcome exploration, never a replacement for the failed pilot.
        core.update({"protocol_version": "4.1.3-exploratory", "exploratory_revision": revision,
            "result_informed_revision": True, "allowed_execution_phases": ["GPU_REPLAY", *PHASES[:6]],
            "revision_authorization": "USER_REQUEST_IMPLEMENT_ZH_FR_ACCURACY_IMPROVEMENT_AND_RETRAIN",
            "interpretation": "single-root exploratory recipe; accuracy improvement not guaranteed; no H or main study"})
        core["execution"].update({"roots": [4101],
            "start_phase": "T2", "import_full_state": RETRAIN_SOURCE,
            "lexical_updates_by_stage": {"T0": 3000, "T1": 3000, "T2": 5400, "T3": 5400},
            "concept_factor_balance_stages": ["T2", "T3"],
            "learning_rate_reduction": {"stages": ["T2", "T3"], "after_updates": 3000, "learning_rate": 0.0001},
            "history_enabled": False})
    if revision in (PROBE_REVISION, PROBE_NEXT_REVISION, PROBE_CONTINUE_REVISION):
        core["allowed_execution_phases"] = ["GPU_REPLAY", "T3"]
        core["interpretation"] = "paired exploratory diagnostic; no full retraining, H, other roots, or main study"
        core["execution"] = {"runner":"accuracy_probe", "roots":[4101],"history_enabled":False,
            "source_summary_path":str(WORK_ROOT/"runs/pilot_jm02_4e30ec2/run_summary_000263.json"),
            "source_summary_sha256":"a8d55cd39abbe55b3124b74ff72921c15d3700aa305393848948598ff13d3ac4",
            "additional_updates":360,"batch_examples":32,
            "arms":{"constant":{"learning_rate":0.0003,"prompt_mix":False},
                    "low_lr":{"learning_rate":0.0001,"prompt_mix":False},
                    "prompt_mix":{"learning_rate":0.0001,"prompt_mix":True}},
            "selection_rule":"both ZH and FR improve >=2/120 over baseline AND constant; no KO/EN dev cell loses >1/60 versus either; all final gates remain required",
            "evaluation":"original KO/RD/dev only; train wrappers diagnostic; test not used for selection",
            "checkpoint_steps":[180,360],"automatic_expansion":False}
        if revision in (PROBE_NEXT_REVISION, PROBE_CONTINUE_REVISION):
            core["execution"].update({
                "source_summary_path":str(WORK_ROOT/"runs/accuracy_probe_47dc5c6/run_summary.json"),
                "source_summary_sha256":"014596eb3cc54d35b203761a9eea2432b4cfa560b6556a8c3f2cba7fc594ba65",
                "source_arm":"prompt_mix", "selection_version":2,
                "arms":{
                    "constant":{"learning_rate":0.0001,"prompt_mix":True},
                    "lower_lr":{"learning_rate":0.00003,"prompt_mix":True},
                    "balanced_cells":{"learning_rate":0.0001,"prompt_mix":True,"concept_factor_balance":True},
                    "full_prompt_mix":{"learning_rate":0.0001,"prompt_mix":True,"all_ko_prompts":True}},
                "selection_rule":"all readiness cells pass; ZH improves >=2/120 over source and >=1/120 over control; FR no aggregate loss vs source and <=1/120 loss vs control; KO/EN per cell <=1/60 loss vs either; measurement gates unchanged",
                "prune_verified_midpoints":True,
                "retention":"keep every final model and all scores/traces/manifests; delete midpoint state only after endpoint readback and scored evaluation"})
        if revision == PROBE_CONTINUE_REVISION:
            core["interpretation"] = "single selected candidate continuation; descriptive source comparison only; no H, other roots, or main study"
            core["execution"].update({
                "source_summary_path":str(WORK_ROOT/"runs/accuracy_refinement_fdfa878/run_summary.json"),
                "source_summary_sha256":"e9687662872c8abdec310e573ebb4de2bb517de066b831329ac228a7aedcf425",
                "source_arm":"lower_lr", "single_candidate_continuation":True,
                "arms":{"lower_lr":{"learning_rate":0.00003,"prompt_mix":True}},
                "selection_rule":"no concurrent control or further candidate selection; report source deltas and unchanged readiness/measurement gates; stop after 360 additional updates"})
    if revision == TRAJECTORY_REVISION:
        core.update(protocol_version="4.1.4-exploratory", exploratory_revision=revision,
            result_informed_revision=True, allowed_execution_phases=["GPU_REPLAY", *PHASES[:6]],
            revision_authorization="USER_REQUEST_20260916_MEASURE_ONE_FULL_SEED_THEN_ADD_2_TO_3_SEEDS_WITHIN_EXISTING_BUDGET",
            interpretation="Independent-init replication of observed S trajectories; readiness and measurement are reported outcomes, not expansion gates; no H or main study")
        core["execution"].update(roots=[4102,4103,4104], start_phase="INIT", history_enabled=False,
            require_all_roots_before_history=False, outcome_gates_are_diagnostic=True,
            benchmark_first_root=4102, language_order=["ko","en","zh","fr"],
            reference_run={"path":str(WORK_ROOT/"runs/pilot_jm02_4e30ec2/run_summary_000263.json"),
                "sha256":"a8d55cd39abbe55b3124b74ff72921c15d3700aa305393848948598ff13d3ac4"},
            minimum_remaining_hours_before_later_root=2.5,
            minimum_disk_bytes_before_root=5*1024**3+4*1024**3)
    if revision == REVERSE_REVISION:
        core.update(protocol_version="4.1.5-exploratory", exploratory_revision=revision,
            result_informed_revision=True, allowed_execution_phases=["GPU_REPLAY", "T1", "T2", "T3"],
            revision_authorization="USER_REQUEST_20260916_ONE_REVERSE_ORDER_RUN_IN_REMAINING_BUDGET",
            interpretation="One paired schedule contrast from the same T0 state; not another independent seed, not exposure-matched H, not reverse-seed variance")
        core["execution"].update(roots=[4101], start_phase="T1", import_root_id=4101, import_phase="T0",
            import_full_state=REVERSE_SOURCE, language_order=["ko","fr","zh","en"], history_enabled=False,
            require_all_roots_before_history=False, outcome_gates_are_diagnostic=True,
            pairing="preserve T0 model, optimizer, scheduler, RNG; same seed/stage role-index schedule; EN/FR roles exchanged",
            root_selection_rule="smallest existing seed ID; post-result exploratory choice, not a new preregistered experiment",
            forward_reference={"path":str(WORK_ROOT/"runs/pilot_jm02_4e30ec2/run_summary_000263.json"),
                "sha256":"a8d55cd39abbe55b3124b74ff72921c15d3700aa305393848948598ff13d3ac4"},
            compared_factors=["introduction_order","language-specific_total_exposure","recency"],
            additional_training_updates=9000, automatic_expansion=False)
    return {**core, "freeze_id": "pilot-execution-" + _hash(core)[:20]}


def publish_execution_freeze(path, *, revision=None):
    return publish_json_once(Path(path), build_execution_freeze(revision=revision))


def verify_execution_freeze(path, *, production=True):
    if production is not True:
        raise ContractViolation("EXECUTION_FREEZE_REQUIRES_REAL_DATA")
    value = read_verified_json(Path(path))
    # Verify the fixed parent before rebuilding; caller-supplied phase/budget
    # lists cannot authorize execution, even with a recomputed JSON digest.
    if value.get("parent_freeze") != {"path": str(PARENT), "sha256": PARENT_SHA}:
        raise ContractViolation("PILOT_PARENT_FREEZE_MISMATCH")
    revision = value.get("exploratory_revision")
    expected = build_execution_freeze() if revision is None else build_execution_freeze(revision=revision)
    if value != expected:
        raise ContractViolation("PILOT_EXECUTION_FREEZE_MISMATCH")
    return {**value, "verified": True, "freeze_sha256": sha256_file(Path(path)),
        "manifest_path": str(Path(path).resolve()),
        "resolved_artifacts": {k: r["path"] for k, r in value["artifacts"].items()}}
