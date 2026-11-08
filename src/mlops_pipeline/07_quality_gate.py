from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
def main():
    reports=[ROOT/"artifacts"/"data_load_report.json",ROOT/"artifacts"/"data_validation_report.json",ROOT/"artifacts"/"preprocessing_summary_report.json"]
    problems=[]
    for p in reports:
        if not p.exists(): problems.append(f"Missing report: {p.name}")
        else:
            try:
                data=json.loads(p.read_text())
                if data.get("status")=="FAILED": problems.append(f"Failed report: {p.name}")
            except Exception as e: problems.append(f"Unreadable report {p.name}: {e}")
    result={"status":"FAILED" if problems else "PASSED","checks":len(reports),"errors":problems}
    (ROOT/"reports").mkdir(exist_ok=True); (ROOT/"reports"/"quality_gate_report.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if problems: raise SystemExit(1)
if __name__=="__main__": main()
