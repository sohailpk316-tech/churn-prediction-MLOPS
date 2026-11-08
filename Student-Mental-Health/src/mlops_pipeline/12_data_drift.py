"""Lab 12: population-stability-index comparison of baseline vs incoming survey data."""
from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data'/'raw'/'Student_Mental_health.csv'
CURRENT=ROOT/'data'/'monitoring'/'student_survey_drifted.csv'
OUT=ROOT/'reports'/'lab12_data_drift_report.json'
def psi(expected,actual,bins=10):
  e=pd.to_numeric(expected,errors='coerce').dropna(); a=pd.to_numeric(actual,errors='coerce').dropna()
  if e.empty or a.empty: return None
  cuts=np.unique(np.quantile(e,np.linspace(0,1,bins+1)))
  if len(cuts)<3: cuts=np.array([-np.inf,0,np.inf])
  else: cuts[0]=-np.inf; cuts[-1]=np.inf
  ep=np.histogram(e,bins=cuts)[0]/len(e); ap=np.histogram(a,bins=cuts)[0]/len(a)
  ep=np.clip(ep,1e-6,None); ap=np.clip(ap,1e-6,None)
  return float(np.sum((ap-ep)*np.log(ap/ep)))
def main():
  if not CURRENT.exists(): raise SystemExit('Run: python src/mlops_pipeline/12_create_drift_dataset.py first')
  b=pd.read_csv(BASE); c=pd.read_csv(CURRENT); results={}
  for col in b.columns:
    if col in ['Timestamp','Do you have Depression?'] or col not in c: continue
    if pd.api.types.is_numeric_dtype(b[col]): val=psi(b[col],c[col])
    else:
      cats=set(b[col].astype(str))|set(c[col].astype(str)); eb=b[col].astype(str).value_counts(normalize=True); ac=c[col].astype(str).value_counts(normalize=True)
      e=np.array([eb.get(x,0) for x in cats]); a=np.array([ac.get(x,0) for x in cats]); e=np.clip(e,1e-6,None); a=np.clip(a,1e-6,None)
      val=float(np.sum((a-e)*np.log(a/e)))
    results[col]={'psi':val,'status':'DRIFT' if val is not None and val>=0.20 else 'STABLE'}
  report={'lab':12,'project':'Student Mental Health','baseline_rows':len(b),'current_rows':len(c),'threshold':0.20,'features':results,
    'overall_status':'DRIFT_DETECTED' if any(v['status']=='DRIFT' for v in results.values()) else 'NO_SIGNIFICANT_DRIFT',
    'note':'Synthetic drift data is for coursework demonstration only; PSI is a monitoring signal, not a clinical conclusion.'}
  OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
