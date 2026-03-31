# train.py — MLflow experiment tracking for PortIQ GTM scoring model
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np

# ── Load your data ─────────────────────────────────────────────────────────────
# Replace this with your actual data loading logic from PortIQ
from src.gtm_opportunity_pipeline import load_data, engineer_features

df = load_data()
X, y = engineer_features(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── MLflow experiment ──────────────────────────────────────────────────────────
mlflow.set_experiment("portiq-gtm-scoring")

# Parameters to track
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42,
    "cv_folds": 5,
}

with mlflow.start_run(run_name="random_forest_baseline"):

    # Log parameters
    mlflow.log_params(params)

    # Train model
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"]
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=params["cv_folds"], scoring="roc_auc")
    mlflow.log_metric("cv_auc_mean", round(cv_scores.mean(), 4))
    mlflow.log_metric("cv_auc_std", round(cv_scores.std(), 4))

    # Final fit and test evaluation
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metric("test_auc", round(test_auc, 4))

    # Log model artifact
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Run logged to MLflow.")
