# Predicting Depression Risk in University Students

**Project ID:** 241FA18461 | **Type:** Binary classification (scikit-learn)
**Dataset:** Student Mental Health survey - 101 responses, Google Forms export (`data/raw/Student_Mental_health.csv`)

## 1. Problem
Predict whether a student reports **Depression** (`Do you have Depression?` = Yes/No) from their
demographic and academic profile (gender, age, course, year of study, CGPA band, marital status)
and reported Anxiety, Panic attack and treatment-seeking status.
Useful as an early-screening signal for a university counselling cell.

The dataset is tiny (101 rows, 66 No / 35 Yes), so the project uses **stratified 5-fold
cross-validation**, simple interpretable models, and tunes only the two best candidates.

## 2. Project structure
```
241FA18461/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/Student_Mental_health.csv      # original survey data (never modified)
│   └── processed/                         # generated: clean data, train/test split, metadata
├── notebooks/
│   └── student_mental_health_project.ipynb  # original step-by-step notebook (fixed, still runs)
├── src/
│   ├── config.py            # paths, column names, encoding rules, seeds
│   ├── features.py          # cleaning + feature engineering + sklearn preprocessor (shared)
│   ├── validate_data.py     # 1. checks the raw data
│   ├── preprocess.py        # 2. cleans data, stratified 80/20 split, metadata
│   ├── eda.py               # 3. exploratory plots
│   ├── train.py             # 4. compares 7 models (CV), tunes top 2, selects final model
│   ├── evaluate.py          # 5. test-set metrics, plots, error analysis
│   ├── predict.py           # 6. predict for a new student
│   └── validate_outputs.py  # 7. sanity-checks everything produced
├── pipelines/
│   ├── run_pipeline.py      # runs steps 1-7 in order, logs to logs/pipeline_run.log
│   ├── run_lab3_baseline.py # Lab 3: dummy + default logistic-regression baselines
│   ├── run_lab4_tracking.py # Lab 4: MLflow experiment tracking (6 runs)
│   ├── run_lab5_pipeline.py # Lab 5: end-to-end tuned sklearn Pipeline, saved + reload-verified
│   ├── run_lab6_registry.py # Lab 6: MLflow model registry (register + alias + load)
│   └── _common.py           # helpers shared by the three lab scripts
├── models/                  # generated: best_model.pkl, all_models.pkl
├── outputs/                 # generated: metrics, result tables, error-analysis CSVs
├── artifacts/               # generated: plots and validation reports
└── logs/                    # generated: pipeline_run.log
```

## 3. How to run
```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows      |  source venv/bin/activate   (Mac/Linux)

# 2. install dependencies
pip install -r requirements.txt

# 3. run everything (from the project root)
python pipelines/run_pipeline.py
```
Each step can also be run on its own, e.g. `python src/train.py` (run them in the order above).
Scripts work from any working directory.

**Lab pipelines** (run in this order, from the project root; Windows PowerShell shown)
```bash
python .\pipelines\run_lab3_baseline.py   # baselines  -> outputs/lab3_baseline_metrics.json
python .\pipelines\run_lab4_tracking.py   # MLflow runs -> mlflow.db + mlartifacts/
python .\pipelines\run_lab5_pipeline.py   # pipeline  -> models/lab5_pipeline.pkl, outputs/lab5_pipeline_report.json
python .\pipelines\run_lab6_registry.py   # registry   -> outputs/lab6_registry_summary.json
```
Each script creates `data/processed/` automatically if it is missing. To browse the tracked
runs and the registered model: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
then open http://127.0.0.1:5000.

**Predict for a new student**
```bash
python src/predict.py                       # built-in sample student
python src/predict.py --json '{"gender":"Male","age":21,"course":"Law","year_of_study":"year 3","cgpa":"2.50 - 2.99","marital_status":"No","anxiety":"Yes","panic_attack":"No","sought_treatment":"No"}'
```
(On Windows PowerShell/CMD, wrap the JSON in double quotes and escape inner quotes, or edit `SAMPLE_STUDENT` in `src/predict.py`.)

**Notebook:** `cd notebooks && jupyter notebook student_mental_health_project.ipynb`

## 4. Method
| Stage | What is done |
|---|---|
| Cleaning | Strip whitespace; unify casing of `year_of_study` / `course`; merge same-meaning course spellings (`Laws`->`law`, `Engine`->`engineering`, ...); drop `timestamp` |
| Features | `year_of_study` -> 1-4, `cgpa` band -> ordinal 0-4, Yes/No & gender -> 0/1, `course` -> one-hot |
| Missing data | 1 missing `age` -> median imputation, **fit on training data only** |
| Split | 80/20 stratified (80 train / 21 test), `random_state=42` |
| Leakage control | Imputation, scaling and encoding sit **inside** each model's `Pipeline`, so they are re-fit per CV fold |
| Model comparison | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Naive Bayes - 5-fold stratified CV, scored by F1 |
| Tuning | `GridSearchCV` on the top 2 models only |
| Selection | Final model = best **cross-validated** F1 (test set is not used to choose) |

## 5. Results
Cross-validated F1 on the 80 training rows (`outputs/cv_model_comparison.csv`):
Logistic Regression 0.735, SVM 0.710, Random Forest 0.659, Gradient Boosting 0.642,
Decision Tree 0.641, KNN 0.593, Naive Bayes 0.565.

**Selected model: Logistic Regression (C=1)** - held-out test set (21 students):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.810 | 0.800 | 0.571 | 0.667 | 0.786 |

Confusion matrix: TN 13, FP 1, FN 3, TP 4. Full per-model test results are in `outputs/test_results.csv`.

Sample prediction (20-year-old female Engineering student, year 2, anxiety + panic attack, no
treatment): **Depression, probability 68.7%** (`outputs/sample_prediction.json`).

### Output files
- `outputs/` - `metrics.json`, `cv_model_comparison.csv`, `test_results.csv`, `classification_report.txt`, `false_negatives.csv`, `false_positives.csv`, `training_summary.json`, `sample_prediction.json`
- `artifacts/` - `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`, `cv_model_comparison.png`, `eda_*.png`, `data_validation_report.json`, `output_validation_report.json`

## 6. Limitations (please read)
- **Very small data.** The test set has 21 students (7 with depression), so one student changes recall by ~14 points. Several models tie or beat the final model on test F1 (e.g. SVM 0.727); the CV-based choice is more trustworthy than a single 21-row split, but all numbers are rough.
- Many courses have a single student, so course coefficients (and the large marital-status coefficient) reflect small-sample noise rather than real effects.
- Anxiety, panic attack and treatment-seeking are self-reported alongside depression, so this is a profile-matching exercise, **not** a clinical screening tool. Do not use it to make real decisions about students.


## Labs 1, 2, 7, 8, 9 and 10
This repository is the Student Mental Health project. Run `python src/mlops_pipeline/churn_prediction_pipeline.py` from the repository root. It validates the survey source data and exercises failure-handling checks.
