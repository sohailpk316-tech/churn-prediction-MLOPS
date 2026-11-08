"""Lab 12: combine drift and model-performance reports into a retraining recommendation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
def read(path):
  if not path.exists(): raise SystemExit(f'Missing prerequisite report: {path.name}')
  return json.loads(path.read_text(encoding='utf-8'))
def main():
  drift=read(ROOT/'reports'/'lab12_data_drift_report.json'); perf=read(ROOT/'reports'/'lab12_model_performance.json')
  trigger=drift.get('overall_status')=='DRIFT_DETECTED' or perf.get('f1',1)<0.60
  result={'lab':12,'retraining_recommended':bool(trigger),'reasons':[],
    'note':'Recommendation only. Review data quality and obtain appropriate approval before retraining or using any model.'}
  if drift.get('overall_status')=='DRIFT_DETECTED': result['reasons'].append('Data drift threshold reached')
  if perf.get('f1',1)<0.60: result['reasons'].append('Held-out F1 below coursework threshold 0.60')
  if not result['reasons']: result['reasons'].append('No trigger threshold reached')
  out=ROOT/'reports'/'lab12_monitor_trigger.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
