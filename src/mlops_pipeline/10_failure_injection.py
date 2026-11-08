from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def main():
    raw=ROOT/"data"/"raw"/'churn.csv'; df=pd.read_csv(raw)
    cases=[]
    # Test 1: missing required field must be detected.
    if len(df.columns)>1:
        damaged=df.drop(columns=[df.columns[0]])
        cases.append({"test":"missing_column","detected":len(damaged.columns)!=len(df.columns)})
    # Test 2: empty dataset must be detected.
    empty=df.iloc[0:0]
    cases.append({"test":"empty_dataset","detected":empty.empty})
    # Test 3: duplicate rows are measurable.
    dup=pd.concat([df.head(1),df.head(1)],ignore_index=True)
    cases.append({"test":"duplicate_rows","detected":int(dup.duplicated().sum())>0})
    failed=[c["test"] for c in cases if not c["detected"]]
    result={"status":"PASSED" if not failed else "FAILED","tests":cases,"failed_tests":failed}
    (ROOT/"reports").mkdir(exist_ok=True); (ROOT/"reports"/"failure_report.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if failed: raise SystemExit(1)
if __name__=="__main__": main()
