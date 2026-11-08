"""Lab 12: summarize API request status and latency without storing student inputs."""
from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[2]; LOG=ROOT/'logs'/'inference'/'student_prediction_log.jsonl'
def main():
  if not LOG.exists(): raise SystemExit('No API log yet. Start the API and send a /predict request first.')
  rows=[json.loads(x) for x in LOG.read_text(encoding='utf-8').splitlines() if x.strip()]
  if not rows: raise SystemExit('API log is empty')
  df=pd.DataFrame(rows); lat=pd.to_numeric(df.get('latency_ms'),errors='coerce').dropna()
  report={'total_requests':len(df),'successful_requests':int((df.status=='success').sum()),
    'failed_requests':int((df.status=='failure').sum()),'success_rate_pct':round(float((df.status=='success').mean()*100),2),
    'average_latency_ms':float(lat.mean()) if len(lat) else None,'p95_latency_ms':float(lat.quantile(.95)) if len(lat) else None,
    'privacy':'Request payloads and student answers are not written to this log.'}
  out=ROOT/'reports'/'lab12_api_performance.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
