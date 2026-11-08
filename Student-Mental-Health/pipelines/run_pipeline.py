"""Run the whole project end to end:  python pipelines/run_pipeline.py

Stages run in order and the pipeline stops at the first failure.
Console output is also written to logs/pipeline_run.log.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = [
    "src/validate_data.py",     # 1. check the raw survey data
    "src/preprocess.py",        # 2. clean, engineer features, train/test split
    "src/eda.py",               # 3. exploratory plots
    "src/train.py",             # 4. compare, tune and select models (CV on train only)
    "src/evaluate.py",          # 5. final evaluation on the held-out test set
    "src/predict.py",           # 6. sample prediction on a new student
    "src/validate_outputs.py",  # 7. sanity-check all outputs
]


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    log_path = ROOT / "logs" / "pipeline_run.log"
    with open(log_path, "w", encoding="utf-8") as log:
        def emit(text):
            print(text, flush=True)
            log.write(text + "\n")

        emit(f"[INFO] Student Mental Health pipeline - {datetime.now():%Y-%m-%d %H:%M:%S}")
        for script in STAGES:
            emit(f"\n[INFO] ---> {script}")
            proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                                  capture_output=True, text=True)
            emit(proc.stdout.rstrip())
            if proc.returncode != 0:
                emit(proc.stderr.rstrip())
                emit(f"[ERROR] Pipeline halted: {script} failed (exit code {proc.returncode}).")
                sys.exit(1)
        emit("\n[SUCCESS] Pipeline completed. See outputs/ and artifacts/ for results.")


if __name__ == "__main__":
    main()
