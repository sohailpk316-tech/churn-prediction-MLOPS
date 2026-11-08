"""Lab 12: evaluate the saved model on the project's held-out test split."""
from pathlib import Path
import json, joblib, pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
def main():
  model=joblib.load(ROOT/'models'/'best_model.pkl'); test=pd.read_csv(ROOT/'data'/'processed'/'test.csv')
  import sys; sys.path.insert(0,str(ROOT/'src')); import config
  X=test[config.FEATURES]; y=test['target']; pred=model.predict(X); prob=model.predict_proba(X)[:,1]
  report={'lab':12,'rows_evaluated':len(test),'accuracy':float(accuracy_score(y,pred)),
    'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),
    'f1':float(f1_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,prob)),
    'note':'Uses the held-out test split for reporting only; do not tune model choices against this report.'}
  out=ROOT/'reports'/'lab12_model_performance.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
