"""Lab 11: FastAPI service for the Student Mental Health ML pipeline.
This is an educational demo, not a clinical screening or diagnostic service.
"""
import json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import sys
from fastapi import FastAPI, HTTPException
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from predict import predict_student
from api.schemas import StudentProfile
app=FastAPI(title='Student Mental Health Prediction API',version='1.0.0',
  description='Educational model demonstration only; not for clinical decisions.')
LOG=ROOT/'logs'/'inference'/'student_prediction_log.jsonl'
@app.get('/health')
def health():
  return {'status':'healthy','service':'student-mental-health-api','model_exists':(ROOT/'models'/'best_model.pkl').exists()}
@app.post('/predict')
def predict(profile:StudentProfile):
  request_id=uuid.uuid4().hex[:12]; start=time.perf_counter()
  try:
    result=predict_student(profile.model_dump())
    event={'request_id':request_id,'timestamp':datetime.now(timezone.utc).isoformat(),
      'status':'success','predicted_class':result['predicted_class'],
      'probability_of_depression':result['probability_of_depression'],
      'latency_ms':round((time.perf_counter()-start)*1000,3)}
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
    return {'request_id':request_id,**result,
      'disclaimer':'Educational estimate only; not a diagnosis or clinical advice.'}
  except Exception as exc:
    event={'request_id':request_id,'timestamp':datetime.now(timezone.utc).isoformat(),
      'status':'failure','latency_ms':round((time.perf_counter()-start)*1000,3),'error_type':type(exc).__name__}
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
    raise HTTPException(status_code=400,detail='Prediction could not be generated; verify the student profile values.')
