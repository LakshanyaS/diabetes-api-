from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR    = ROOT_DIR / "data"
MODEL_DIR   = ROOT_DIR / "model"
API_DIR     = ROOT_DIR / "api"

DATASET_PATH  = DATA_DIR  / "dataset.csv"
MODEL_PATH    = MODEL_DIR / "model.pkl"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"

METRICS_PATH           = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"
TRAINING_SUMMARY_PATH  = ARTIFACTS_DIR / "training_summary.json"

# Fallbacks
ROOT_DATASET_FALLBACK_PATH = ROOT_DIR / "diabetes.csv"
ROOT_MODEL_FALLBACK_PATH   = ROOT_DIR / "diabetes_model.pkl"
DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

RANDOM_STATE = 42

TARGET_COLUMN   = "Outcome"
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

RISK_LOW_MAX      = 35.0
RISK_MODERATE_MAX = 65.0

MODEL_SELECTION_METRIC  = "accuracy"
DIABETIC_PROB_THRESHOLD = 0.5

INPUT_RANGES = {
    "Pregnancies":             {"min": 0,   "max": 20,  "step": 1},
    "Glucose":                 {"min": 0,   "max": 300, "step": 1},
    "BloodPressure":           {"min": 0,   "max": 150, "step": 1},
    "SkinThickness":           {"min": 0,   "max": 100, "step": 1},
    "Insulin":                 {"min": 0,   "max": 900, "step": 1},
    "BMI":                     {"min": 0.0, "max": 70.0,"step": 0.1},
    "DiabetesPedigreeFunction":{"min": 0.0, "max": 2.5, "step": 0.001},
    "Age":                     {"min": 1,   "max": 120, "step": 1},
}

# Flask API
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False
