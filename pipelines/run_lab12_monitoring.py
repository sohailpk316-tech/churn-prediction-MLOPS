from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=[
  '12_data_drift.py','12_prediction_drift.py','12_performance_monitoring.py',
  '12_api_performance_monitoring.py','12_performance_trend.py','12_drift_analysis.py'
]
if __name__=='__main__':
  for name in SCRIPTS:
    path=ROOT/'src'/'mlops_pipeline'/name
    if path.exists():
      print(f'\n===== Running {name} =====',flush=True)
      subprocess.run([sys.executable,str(path)],cwd=ROOT,check=True)
