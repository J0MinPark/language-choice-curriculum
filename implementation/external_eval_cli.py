"""Physical GPU2 only, read-only external matched-definition diagnostic."""
import argparse
from pathlib import Path
from .src.gpu_guard import require_literal_gpu2_mask


def main():
    p=argparse.ArgumentParser()
    for key in ('contract','manifest','cpu','directory'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();require_literal_gpu2_mask()
    from .src.external_eval import run
    result=run(args.contract,args.manifest,args.cpu,args.directory)
    return 0 if result['status']=='COMPLETE' else 20


if __name__=='__main__':raise SystemExit(main())
