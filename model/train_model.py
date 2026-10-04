"""Train the Random Forest. Run once:  python train_model.py
Uses the UCI Cleveland heart-disease data (downloaded automatically), or data/heart.csv
if you place one there (must use UCI encoding and have a 'target' column, 1 = disease)."""
import os, json, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
UCI = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")

def load_data():
    if os.path.exists("data/heart.csv"):
        return pd.read_csv("data/heart.csv")
    df = pd.read_csv(UCI, header=None, names=COLS + ["num"], na_values="?")
    df["target"] = (df["num"] > 0).astype(int)   # 0 = no disease, 1 = disease
    return df.drop(columns="num")

if __name__ == "__main__":
    df = load_data()
    X, y = df[COLS], df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                      min_samples_leaf=3, random_state=42)),
    ]).fit(Xtr, ytr)
    pred, proba = model.predict(Xte), model.predict_proba(Xte)[:, 1]
    print(classification_report(yte, pred))
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/rf_heart.joblib")
    json.dump({"accuracy": accuracy_score(yte, pred), "roc_auc": roc_auc_score(yte, proba)},
              open("model/metrics.json", "w"))
    print("Saved model/rf_heart.joblib")
