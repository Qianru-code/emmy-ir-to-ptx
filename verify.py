#!/usr/bin/env python3
"""Run the official GPU suite and save complete, reproducible results."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
report_path = root / "verification.json"
previous = json.loads(report_path.read_text())["results"] if "--retry-failed" in sys.argv and report_path.exists() else []
results = []
for problem in sorted((root/'problems').iterdir()):
    if not (problem/'meta.json').exists():
        continue
    prior = next((r for r in previous if r['problem'] == problem.name), None)
    if prior and prior['passed']:
        results.append(prior)
        continue
    result = subprocess.run([sys.executable,str(root/'submit.py'),str(problem),'--compiler',str(root/'compile.py')],capture_output=True,text=True)
    print(problem.name, result.stdout, result.stderr, flush=True)
    results.append({'problem':problem.name,'passed':result.returncode==0,'log':result.stdout+result.stderr})
report = {'verified_at_utc':datetime.now(timezone.utc).isoformat(),'runner':'http://runner.riftstack.ai:48701','results':results}
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
passed=sum(r['passed'] for r in results)
print(f'{passed}/{len(results)} passed')
sys.exit(0 if passed==len(results) else 1)
