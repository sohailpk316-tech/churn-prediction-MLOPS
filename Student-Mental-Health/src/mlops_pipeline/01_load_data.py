from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[2]
KIND="student"
RAW=ROOT/"data"/"raw"/'Student_Mental_health.csv'
OUT=ROOT/"artifacts"/"data_load_report.json"
def main():
    if not RAW.exists(): raise FileNotFoundError(f"Raw dataset missing: {RAW}")
    df=pd.read_csv(RAW)
    if df.empty: raise ValueError("Dataset is empty")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"project":KIND,"dataset":str(RAW.relative_to(ROOT)),"rows":len(df),"columns":len(df.columns),"column_names":df.columns.tolist(),"missing_values":{k:int(v) for k,v in df.isna().sum().items() if v}},indent=2))
    print(f"[OK] Loaded {len(df)} rows x {len(df.columns)} columns; report: {OUT.relative_to(ROOT)}")
if __name__=="__main__": main()
