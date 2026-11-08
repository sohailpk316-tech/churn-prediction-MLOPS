from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=['12_create_drift_dataset.py','12_data_drift.py','12_performance_monitoring.py','12_monitor_trigger.py']
if __name__=='__main__':
  for name in SCRIPTS:
    print(f'\n===== Lab 12: {name} =====',flush=True)
    subprocess.run([sys.executable,str(ROOT/'src/mlops_pipeline'/name)],cwd=ROOT,check=True)
  print('API log analysis is optional: run 12_api_performance_monitoring.py after testing the API.')
