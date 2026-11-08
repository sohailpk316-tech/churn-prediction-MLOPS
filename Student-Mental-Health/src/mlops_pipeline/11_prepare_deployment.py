"""Lab 11: verify and stage the project's trained sklearn pipeline for serving."""
from pathlib import Path
import json, joblib
ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/'models'/'best_model.pkl'
def main():
  if not MODEL.exists(): raise FileNotFoundError(f'Model missing: {MODEL}. Run Labs 1-10 first.')
  loaded=joblib.load(MODEL)
  if not hasattr(loaded,'predict') or not hasattr(loaded,'predict_proba'):
    raise TypeError('Expected a fitted sklearn classifier pipeline with predict and predict_proba.')
  metadata={'project':'Student Mental Health','lab':11,'model_artifact':'models/best_model.pkl',
    'model_type':type(loaded).__name__,'status':'READY',
    'purpose':'Educational prediction demonstration only; not a clinical diagnostic tool.'}
  out=ROOT/'deployment'; out.mkdir(exist_ok=True)
  (out/'model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
  # Keep a deployment copy; never alter the original trained model.
  joblib.dump(loaded,out/'best_model.pkl')
  print(json.dumps(metadata,indent=2))
if __name__=='__main__': main()
