"""Train, validate and save the telecom churn model bundle."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Telecom Churn Model1.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "telecom_churn_model.joblib"

def load_dataset():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    target = data["Churn"].map({"Yes": 1, "No": 0})
    features = data.drop(columns=["customerID", "Churn"])
    return features, target

def make_pipeline(features):
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.columns.difference(numeric).tolist()
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocessing", preprocessing),
                     ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])

def choose_threshold(y_true, probabilities):
    candidates = []
    for threshold in np.arange(0.20, 0.61, 0.01):
        predictions = probabilities >= threshold
        recall = recall_score(y_true, predictions)
        if recall >= 0.70:
            candidates.append((f1_score(y_true, predictions), float(threshold)))
    return max(candidates)[1] if candidates else 0.50

def train_and_save():
    features, target = load_dataset()
    x_dev, x_test, y_dev, y_test = train_test_split(
        features, target, test_size=0.20, stratify=target, random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=RANDOM_STATE)
    threshold_model = make_pipeline(features).fit(x_train, y_train)
    threshold = choose_threshold(y_val, threshold_model.predict_proba(x_val)[:, 1])
    final_model = make_pipeline(features).fit(x_dev, y_dev)
    probability = final_model.predict_proba(x_test)[:, 1]
    prediction = probability >= threshold
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction),
        "recall": recall_score(y_test, prediction),
        "f1": f1_score(y_test, prediction),
        "roc_auc": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(y_test, probability),
    }
    bundle = {"model": final_model, "threshold": threshold, "metrics": metrics,
              "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
              "feature_columns": features.columns.tolist(), "data_rows": len(features),
              "random_state": RANDOM_STATE}
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH, compress=3)
    return bundle

if __name__ == "__main__":
    result = train_and_save()
    print(f"Saved: {ARTIFACT_PATH}")
    print(f"Threshold: {result['threshold']:.2f}")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")
