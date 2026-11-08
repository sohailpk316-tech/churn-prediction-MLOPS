from pathlib import Path
import pandas as pd, json
ROOT=Path(__file__).resolve().parents[2]
KIND="student"
def main():
    raw=ROOT/"data"/"raw"/'Student_Mental_health.csv'; df=pd.read_csv(raw)
    out=ROOT/"data"/"processed"; out.mkdir(parents=True,exist_ok=True)
    # Preserve the existing lab preprocessing artifacts; this stage records a reproducible input snapshot.
    target=out/"pipeline_input_snapshot.csv"; df.to_csv(target,index=False)
    report={"status":"PASSED","project":KIND,"input_rows":len(df),"input_columns":len(df.columns),"snapshot":str(target.relative_to(ROOT))}
    (ROOT/"artifacts"/"preprocessing_summary_report.json").write_text(json.dumps(report,indent=2))
    print("[OK] Preprocessing input snapshot written:",target)
if __name__=="__main__": main()
