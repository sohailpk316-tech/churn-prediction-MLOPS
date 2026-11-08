from pathlib import Path
import json, shutil
ROOT=Path(__file__).resolve().parents[2]
def main():
    # Safe local lifecycle placeholder: records validation status without promoting a model blindly.
    report=ROOT/"reports"/"quality_gate_report.json"
    if not report.exists(): raise SystemExit("Run Lab 7 quality gate first")
    data=json.loads(report.read_text())
    result={"status":"READY" if data.get("status")=="PASSED" else "BLOCKED","action":"no registry promotion performed","reason":"Quality gate passed" if data.get("status")=="PASSED" else "Quality gate failed"}
    (ROOT/"reports"/"lifecycle_report.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result["status"]!="READY": raise SystemExit(1)
if __name__=="__main__": main()
