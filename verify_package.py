#!/usr/bin/env python3
"""Delivery integrity only. Not authentication or a proof of scientific correctness."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
try:
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['version']!='4.0.0':raise ValueError('VERSION')
    errors=[]
    for name, expected in manifest['sha256'].items():
        path=ROOT/name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            errors.append(name+':UNSAFE_PATH');continue
        if not path.is_file():errors.append(name+':MISSING');continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:errors.append(name+':HASH_MISMATCH')
    print(json.dumps({'status':'PASS' if not errors else 'BLOCKED_SPEC_VERSION',
                      'checked_files':len(manifest['sha256']),'errors':errors},ensure_ascii=False))
    sys.exit(0 if not errors else 20)
except (OSError,ValueError,KeyError):
    print('{"status":"BLOCKED_SPEC_VERSION","reason":"missing_or_invalid_manifest"}')
    sys.exit(20)
