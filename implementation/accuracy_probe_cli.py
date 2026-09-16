"""Exact bounded three-arm probe, not the full pilot launcher."""
import argparse
from pathlib import Path
from .src.gpu_guard import require_literal_gpu2_mask


def main():
    p=argparse.ArgumentParser()
    for name in ("freeze","implementation-manifest","cpu-checks","replay","run-directory"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args(); require_literal_gpu2_mask()
    from .src.accuracy_probe import run_probe
    result=run_probe(freeze_path=args.freeze,manifest_path=args.implementation_manifest,
        cpu_path=args.cpu_checks,replay_path=args.replay,directory=args.run_directory)
    return 0 if result["status"]=="EXPLORATORY_COMPARISON_COMPLETE" else 20


if __name__=="__main__": raise SystemExit(main())
