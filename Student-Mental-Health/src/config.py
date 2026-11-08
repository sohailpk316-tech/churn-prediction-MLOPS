"""Central configuration: paths, column names and constants shared by every script.

All paths are built from this file's location, so scripts work from any
working directory (e.g. `python src/train.py` or `python 241FA18461/src/train.py`).
"""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Student_Mental_health.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

CLEAN_DATA_PATH = PROCESSED_DIR / "student_mental_health_clean.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
METADATA_PATH = PROCESSED_DIR / "dataset_metadata.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

# ---- Reproducibility / split ----------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---- Column names ----------------------------------------------------------
# Original survey headers -> short snake_case names (same as the notebook)
RENAME_MAP = {
    "Timestamp": "timestamp",
    "Choose your gender": "gender",
    "Age": "age",
    "What is your course?": "course",
    "Your current year of Study": "year_of_study",
    "What is your CGPA?": "cgpa",
    "Marital status": "marital_status",
    "Do you have Depression?": "depression",
    "Do you have Anxiety?": "anxiety",
    "Do you have Panic attack?": "panic_attack",
    "Did you seek any specialist for a treatment?": "sought_treatment",
}
RAW_COLUMNS = list(RENAME_MAP.keys())

TARGET = "depression"                       # Yes -> 1, No -> 0
DROP_COLUMNS = ["timestamp"]                # form-submission time, not a student trait

NUMERIC_FEATURES = ["age", "year_of_study_num", "cgpa_ordinal"]
BINARY_FEATURES = ["gender_bin", "marital_status_bin", "anxiety_bin",
                   "panic_attack_bin", "sought_treatment_bin"]
CATEGORICAL_FEATURES = ["course"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# ---- Encoding rules --------------------------------------------------------
CGPA_ORDER = {
    "0 - 1.99": 0,
    "2.00 - 2.49": 1,
    "2.50 - 2.99": 2,
    "3.00 - 3.49": 3,
    "3.50 - 4.00": 4,
}
YES_NO = {"Yes": 1, "No": 0}
GENDER_MAP = {"Male": 1, "Female": 0}

# Same-meaning course spellings found in the data (after lower-casing/stripping)
COURSE_ALIASES = {
    "laws": "law",
    "engine": "engineering",
    "engin": "engineering",
    "pendidikan islam": "islamic education",   # Malay for "Islamic education"
}
