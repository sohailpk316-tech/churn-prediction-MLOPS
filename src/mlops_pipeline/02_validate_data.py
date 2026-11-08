from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/"data"/"raw"/'churn.csv'
EXPECTED=['customerID','gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn']
def main():
    df=pd.read_csv(RAW)
    missing=[c for c in EXPECTED if c not in df.columns]
    errors=[]
    if missing: errors.append("Missing required columns: "+", ".join(missing))
    if df.empty: errors.append("Dataset is empty")
    if len(df.columns)!=len(set(df.columns)): errors.append("Duplicate column names")
    report={"status":"FAILED" if errors else "PASSED","rows":len(df),"columns":len(df.columns),"missing_required_columns":missing,"duplicate_rows":int(df.duplicated().sum()),"errors":errors}
    out=ROOT/"artifacts"/"baseline_validation.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{"status":report["status"],"error":e} for e in (errors or [""])]).to_csv(out,index=False)
    (ROOT/"artifacts"/"data_validation_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if errors: raise SystemExit(1)
if __name__=="__main__": main()
