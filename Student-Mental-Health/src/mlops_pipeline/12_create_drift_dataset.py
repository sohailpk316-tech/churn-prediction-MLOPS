"""Lab 12: create a reproducible synthetic survey batch for drift exercises."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data'/'raw'/'Student_Mental_health.csv'
OUT=ROOT/'data'/'monitoring'/'student_survey_drifted.csv'
def main():
  df=pd.read_csv(RAW)
  if df.empty: raise ValueError('Source survey is empty')
  drift=df.sample(n=len(df),replace=True,random_state=42).reset_index(drop=True)
  # Synthetic distribution shift only; original source data is never modified.
  if 'Age' in drift:
    drift['Age']=(pd.to_numeric(drift['Age'],errors='coerce')+3).clip(15,60)
  if 'Do you have Anxiety?' in drift:
    mask=drift.index % 5 == 0
    drift.loc[mask,'Do you have Anxiety?']=drift.loc[mask,'Do you have Anxiety?'].map({'Yes':'No','No':'Yes'}).fillna('No')
  OUT.parent.mkdir(parents=True,exist_ok=True); drift.to_csv(OUT,index=False)
  print(f'[OK] Synthetic drift batch saved: {OUT.relative_to(ROOT)} ({len(drift)} rows)')
if __name__=='__main__': main()
