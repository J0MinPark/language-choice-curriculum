"""Read-only, budgeted train/dev prompt diagnosis of the pinned failed T3.

Training-wrapper scores are diagnostics, never primary gate evidence.
"""
import argparse
from pathlib import Path

from .src.artifacts import publish_json_once, read_verified_json, sha256_file
from .src.contracts import ContractViolation, WORK_ROOT, PROJECT_ROOT, LANGUAGES
from .src.gpu_guard import require_literal_gpu2_mask

SOURCE = WORK_ROOT / "runs/pilot_jm02_4e30ec2/run_summary_000263.json"
SOURCE_SHA = "a8d55cd39abbe55b3124b74ff72921c15d3700aa305393848948598ff13d3ac4"
FREEZE = WORK_ROOT / "freeze/pilot_execution_v4_1_2_jm02_r1/experiment_freeze.json"


def diagnostic_records(concepts, plan):
    prompt = plan["prompt_plan"]
    rows=[]
    for wrapper in ("train1","train2","dev1","dev2"):
        for concept in concepts:
            for language in LANGUAGES:
                prefix=prompt["templates"][wrapper]["ko"]["REQUESTED"].format(
                    gloss=concept["glosses"]["ko"],
                    target_language_name=prompt["target_language_names"]["ko"][language])
                rows.append({"concept_id":concept["concept_id"],"wrapper":wrapper,
                    "requested_language":language,"prefix":prefix,"answers":concept["answers"]})
    return rows


def summarize(rows):
    cells={}
    for row in rows:
        key=row["wrapper"]+":"+row["requested_language"]
        cell=cells.setdefault(key,{"correct":0,"total":0,"replacement_character_outputs":0})
        cell["correct"]+=int(row["generation"]["request_compatible"])
        cell["total"]+=1
        cell["replacement_character_outputs"]+=int("\ufffd" in row["generation"]["first_line"])
    for cell in cells.values(): cell["accuracy"]=cell["correct"]/cell["total"]
    return cells


def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); args=p.parse_args()
    require_literal_gpu2_mask()
    output=Path(args.out).absolute()
    if output!=output.resolve() or not output.is_relative_to(WORK_ROOT) or output.exists():
        raise ContractViolation("INVALID_DIAGNOSTIC_OUTPUT")
    from .src.budget import BudgetCaps, GpuBudgetLedger
    from .src.checkpoint import load_checkpoint_payload
    from .src.build_tokenizer import load_verified_tokenizer
    from .src.model import build_random_model, configure_determinism
    from .src.train import gpu2_training_session, BoundaryStopController
    from .src.score import greedy_first_line, semantic_model_fingerprint, preserved_evaluation_state
    source=read_verified_json(SOURCE,expected_sha256=SOURCE_SHA)
    endpoint=[e["data"] for e in source["events"] if e["kind"]=="PHASE_COMPLETE" and e["data"]["phase"]=="T3"][-1]
    ref=endpoint["checkpoint"]
    payload,_=load_checkpoint_payload(Path(ref["path"]),expected_state_sha256=ref["state_sha256"],
        expected_manifest_sha256=ref["manifest_sha256"])
    policy=read_verified_json(PROJECT_ROOT/"implementation/config/campaign_policy.json")["budget"]
    ledger=GpuBudgetLedger(WORK_ROOT/"ledger/gpu_budget.json",BudgetCaps(policy["campaign_gpu_hours_cap"],
        policy["prior_gpu_hours_user_reported"],policy["per_root_gpu_hours_cap"]))
    with gpu2_training_session(ledger=ledger,lock_path=WORK_ROOT/"locks/gpu2.lock",run_id=output.stem,
            root_id=4101,phase="T3",reserved_seconds=900,checkpoint_grace_seconds=120,
            experiment_freeze_path=FREEZE) as session, BoundaryStopController() as stop:
        freeze=session.experiment_freeze
        if payload["lineage"]["freeze_sha256"]!=freeze["freeze_sha256"]:
            raise ContractViolation("DIAGNOSTIC_SOURCE_FREEZE_MISMATCH")
        plan=read_verified_json(Path(freeze["resolved_artifacts"]["evaluation_plan"]))
        tokenizer,_=load_verified_tokenizer(Path(freeze["resolved_artifacts"]["tokenizer"]))
        configure_determinism(); model=build_random_model(seed=4101).to("cuda:0")
        model.load_state_dict(payload["model"],strict=True)
        if semantic_model_fingerprint(model)!=ref["model_fingerprint"]:
            raise ContractViolation("DIAGNOSTIC_MODEL_MISMATCH")
        rows=[]
        with preserved_evaluation_state(model):
            for row in diagnostic_records(freeze["concepts"],plan):
                session.ledger.require_time_remaining(session.lease,checkpoint_grace_seconds=120)
                if stop.requested: raise ContractViolation("INTERRUPTED_AT_BOUNDARY")
                generated=greedy_first_line(model,tokenizer,row["prefix"],row["answers"],row["requested_language"],
                    context_length=256,max_new_tokens=64)
                rows.append({**row,"generation":generated})
                if len(rows)%120==0: print(f"DIAGNOSTIC {len(rows)}/960",flush=True)
        if semantic_model_fingerprint(model)!=ref["model_fingerprint"]:
            raise ContractViolation("DIAGNOSTIC_CHANGED_WEIGHTS")
        result={"status":"COMPLETE","scope":"READ_ONLY_TRAIN_VS_DEV_REQUESTED_DIAGNOSTIC_NOT_GATE",
            "source_summary_sha256":SOURCE_SHA,"checkpoint":ref,"freeze_sha256":freeze["freeze_sha256"],
            "diagnostic_code_sha256":sha256_file(Path(__file__)),"gpu_binding":session.binding.as_dict(),
            "updates":0,"rows":rows,"cells":summarize(rows)}
    result["budget"]=ledger.snapshot()
    print(publish_json_once(output,result),flush=True)
    print(result["cells"],flush=True)


if __name__=="__main__": main()
