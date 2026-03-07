#!/usr/bin/env python
"""Run all v2 experiments sequentially."""
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    'v2_multimodel.py',
    'v2_robustness_deployment.py',
    'v2_detection_strength.py',
    'v2_content_science.py',
    'v2_serious_dataset.py',
]

exp_dir = Path(__file__).resolve().parent
t0 = time.time()

for script in SCRIPTS:
    path = exp_dir / script
    print(f"\n{'='*70}", flush=True)
    print(f"RUNNING: {script}", flush=True)
    print(f"{'='*70}", flush=True)
    result = subprocess.run(
        [sys.executable, str(path)],
        env={**__import__('os').environ, 'PYTHONUNBUFFERED': '1'},
    )
    if result.returncode != 0:
        print(f"WARNING: {script} exited with code {result.returncode}", flush=True)
    print(f"Finished {script} ({time.time()-t0:.0f}s elapsed)", flush=True)

print(f"\n{'='*70}", flush=True)
print(f"ALL V2 EXPERIMENTS COMPLETE in {time.time()-t0:.0f}s", flush=True)
print(f"{'='*70}", flush=True)
