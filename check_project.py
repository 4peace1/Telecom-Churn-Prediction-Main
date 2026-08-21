"""Small, dependency-light validation suite for local use and GitHub Actions."""
from pathlib import Path
import json
import re
import subprocess
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md", "app.py", "train_model.py", "requirements.txt",
    "Telecom Churn Model1.csv", "telecom_churn_capstone.ipynb",
]
SENSITIVE_PATTERNS = {
    "ngrok token": re.compile(r"\b[0-9A-Za-z_]{45,}\b"),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
}

def fail(message):
    print(f"FAIL: {message}")
    raise SystemExit(1)

missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
if missing:
    fail(f"Missing required files: {missing}")

subprocess.run([sys.executable, "-m", "py_compile", "app.py", "train_model.py"], cwd=ROOT, check=True)

with (ROOT / "telecom_churn_capstone.ipynb").open(encoding="utf-8") as handle:
    notebook = json.load(handle)
if notebook.get("nbformat") != 4 or not notebook.get("cells"):
    fail("Notebook is not a valid non-empty nbformat 4 document")

data = pd.read_csv(ROOT / "Telecom Churn Model1.csv")
if data.shape != (7043, 21) or set(data["Churn"].dropna().unique()) != {"Yes", "No"}:
    fail(f"Unexpected dataset shape or target values: {data.shape}")

subprocess.run([sys.executable, "train_model.py"], cwd=ROOT, check=True)
bundle = joblib.load(ROOT / "artifacts" / "telecom_churn_model.joblib")
required_bundle_keys = {"model", "threshold", "metrics", "confusion_matrix", "feature_columns"}
if not required_bundle_keys.issubset(bundle):
    fail("Saved model bundle is incomplete")
if not 0 < bundle["threshold"] < 1:
    fail("Saved decision threshold is outside (0, 1)")

scan_files = [ROOT / "README.md", ROOT / "app.py", ROOT / "train_model.py", ROOT / "telecom_churn_capstone.ipynb"]
for path in scan_files:
    content = path.read_text(encoding="utf-8", errors="ignore")
    if "authtoken" in content.lower() or "pyngrok" in content.lower():
        fail(f"Tunnel credential/dependency reference found in {path.name}")
    if SENSITIVE_PATTERNS["private key"].search(content):
        fail(f"Private key material found in {path.name}")

print("PASS: repository structure, code, notebook, dataset, model and secret checks")
