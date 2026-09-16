"""Dedicated production pilot entry point; no device/seed/length overrides.

python -m implementation.pilot_cli --help
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .src.artifacts import publish_json_once
from .src.contracts import ContractViolation, PROJECT_ROOT, WORK_ROOT
from .src.gpu_guard import require_literal_gpu2_mask


def parser():
    p = argparse.ArgumentParser(description="Fixed GPU2 pilot. Client disconnection is safe; server shutdown is not.")
    p.add_argument("--freeze",required=True)
    p.add_argument("--implementation-manifest",required=True)
    p.add_argument("--cpu-checks",required=True)
    p.add_argument("--replay",required=True)
    p.add_argument("--run-directory",required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--launch",action="store_true",help="Detach into a new server session, logging to run-directory.")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--pause-after",help="Operational checkpoint pause ROOT:PHASE:STEP; never changes an endpoint.")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        require_literal_gpu2_mask()  # before importing Torch
        directory = Path(args.run_directory).absolute()
        if directory != directory.resolve() or not directory.is_relative_to(WORK_ROOT):
            raise ContractViolation("PILOT_DIRECTORY_OUTSIDE_WORK")
        pause = None
        if args.pause_after:
            r,p,s = args.pause_after.split(":")
            pause = (int(r),p,int(s))
            if pause[0] not in (4101,4102,4103,4104) or p not in ("CORPUS","T0","T1","T2","T3","H_A","H_B") or pause[2] < 1 or pause[2]%200:
                raise ContractViolation("INVALID_OPERATIONAL_PAUSE")
        if args.launch:
            if args.preflight_only: raise ContractViolation("LAUNCH_PREFLIGHT_CONFLICT")
            directory.mkdir(parents=True,exist_ok=True)
            tag = __import__("uuid").uuid4().hex[:12]
            log_path = directory/f"server_{tag}.log"
            child_args = [a for a in (list(argv) if argv is not None else sys.argv[1:]) if a != "--launch"]
            command = [sys.executable,"-m","implementation.pilot_cli",*child_args]
            # nohup sets SIG_IGN across exec; setsid disconnects from SSH's session.
            with log_path.open("xb") as log:
                child = subprocess.Popen(["nohup",*command],cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,close_fds=True)
            ref = publish_json_once(directory/f"launch_{tag}.json",{
                "pid":child.pid,"command":command,"log":str(log_path),
                "server":__import__("socket").gethostname(),"gpu":2,
                "status":"STARTING_NOT_YET_VALIDATED","automatic_restart":False,
                "server_shutdown_survival":False,"client_disconnect_survival":True})
            print(json.dumps(ref)); return 0
        from .src.pilot_runtime import PilotRuntime
        runtime = PilotRuntime(freeze_path=args.freeze,implementation_manifest=args.implementation_manifest,
            cpu_checks=args.cpu_checks,replay=args.replay,run_directory=directory,resume=args.resume,pause_after=pause)
        if args.preflight_only:
            print(json.dumps({"status":"PASS","scope":"runtime input gates only","training_started":False})); return 0
        result = runtime.run()
        return 0 if result["status"] in ("PILOT_COMPLETE","EXPLORATORY_PREPARATION_COMPLETE","TRAJECTORY_REPLICATION_COMPLETE","REVERSE_ORDER_COMPLETE","PAUSED_AFTER_REQUESTED_BOUNDARY") else 20
    except (ContractViolation,OSError,ValueError) as exc:
        print(json.dumps({"status":"BLOCKED","error":str(exc)}),flush=True)
        return 20


if __name__ == "__main__":
    raise SystemExit(main())
