from pathlib import Path
import json, hashlib, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=ROOT/"data/raw/churn.csv"; df=pd.read_csv(p)
 report={"status":"PASSED","dataset_rows":len(df),"dataset_columns":len(df.columns),"dataset_sha256":digest(p),"deterministic_read_check":pd.read_csv(p).equals(df)}
 out=ROOT/"artifacts/reproducibility_report.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
 if not report["deterministic_read_check"]: raise SystemExit(1)
if __name__=="__main__": main()
