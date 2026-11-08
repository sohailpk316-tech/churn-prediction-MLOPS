from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'src'/'mlops_pipeline'/'11_prepare_deployment.py'
if __name__=='__main__': subprocess.run([sys.executable,str(SCRIPT)],cwd=ROOT,check=True)
