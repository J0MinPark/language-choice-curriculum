#!/usr/bin/env python3
"""Verify delivery, run CPU reference tests, then check credentials. No GPU jobs."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
for cmd in ([sys.executable,'verify_package.py'],[sys.executable,'-m','unittest','discover','-s','tests','-v']):
    result=subprocess.run(cmd,cwd=ROOT,check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
raise SystemExit(subprocess.run([sys.executable,'-m','freshstart','preflight',*sys.argv[1:]],cwd=ROOT,check=False).returncode)
