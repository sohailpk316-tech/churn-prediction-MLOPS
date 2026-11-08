from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
    subprocess.run([sys.executable,str(ROOT/"src"/"mlops_pipeline"/"08_automate_lifecycle.py")],cwd=ROOT,check=True)
