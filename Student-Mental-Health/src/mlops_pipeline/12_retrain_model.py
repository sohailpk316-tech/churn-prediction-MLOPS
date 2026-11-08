"""Lab 12 optional retraining entry point. Runs the project's validated training pipeline."""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
  trigger=ROOT/'reports'/'lab12_monitor_trigger.json'
  if not trigger.exists(): raise SystemExit('Run Lab 12 drift and trigger reports first.')
  import json
  if not json.loads(trigger.read_text()).get('retraining_recommended'):
    raise SystemExit('Retraining not recommended by current monitoring report; no model changed.')
  print('[INFO] Retraining recommended. Starting the existing model-training workflow...')
  subprocess.run([sys.executable,str(ROOT/'src'/'train.py')],cwd=ROOT,check=True)
