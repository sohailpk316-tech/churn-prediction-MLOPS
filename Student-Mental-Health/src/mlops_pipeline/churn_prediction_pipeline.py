from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[2]
STEPS=["01_load_data.py","02_validate_data.py","03_preprocessing.py","04_validate_outputs.py","07_quality_gate.py","08_automate_lifecycle.py","09_inference.py","10_failure_injection.py"]
def main():
    for step in STEPS:
        path=ROOT/"src"/"mlops_pipeline"/step
        print(f"\n=== Running {step} ===",flush=True)
        subprocess.run([sys.executable,str(path)],cwd=ROOT,check=True)
    print("\n[PASS] Lab 1, 2, 7, 8, 9, 10 pipeline stages completed.")
if __name__=="__main__": main()
