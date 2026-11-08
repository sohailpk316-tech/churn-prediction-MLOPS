from pathlib import Path
import pandas as pd, json
ROOT=Path(__file__).resolve().parents[2]
def main():
    # Baseline inference handoff: verifies incoming data and emits a transparent placeholder,
    # rather than loading an incompatible model artifact from another project.
    source=ROOT/"data"/"raw"/'new_churn.csv'
    if not source.exists(): source=ROOT/"data"/"raw"/'churn.csv'
    df=pd.read_csv(source)
    result=pd.DataFrame({"row_number":range(1,len(df)+1),"inference_status":["INPUT_VALIDATED_NO_MODEL_INFERENCE"]*len(df)})
    out=ROOT/"outputs"/"inference_predictions.csv"; out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False)
    (ROOT/"reports"/"inference_report.json").write_text(json.dumps({"status":"INPUT_VALIDATED","input_file":str(source.relative_to(ROOT)),"rows":len(df),"note":"No prediction generated because this lab handoff does not assume a compatible trained pipeline."},indent=2))
    print(f"[OK] Validated {len(df)} inference rows; output: {out}")
if __name__=="__main__": main()
