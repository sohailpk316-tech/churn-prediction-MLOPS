from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
def main():
    expected=[ROOT/"artifacts"/"data_load_report.json",ROOT/"artifacts"/"data_validation_report.json",ROOT/"data"/"processed"/"pipeline_input_snapshot.csv"]
    missing=[str(p.relative_to(ROOT)) for p in expected if not p.exists()]
    report={"status":"FAILED" if missing else "PASSED","missing_outputs":missing}
    (ROOT/"artifacts"/"output_validation_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if missing: raise SystemExit(1)
if __name__=="__main__": main()
