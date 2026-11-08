from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def run(args): subprocess.run(args,cwd=ROOT,check=True)
if __name__=='__main__':
  run([sys.executable,str(ROOT/'src/mlops_pipeline/11_prepare_deployment.py')])
  print('Build image: docker build -t student-mental-health-api:lab11 .')
  print('Run container: docker run --rm -p 8000:8000 student-mental-health-api:lab11')
  print('Swagger UI: http://localhost:8000/docs')
