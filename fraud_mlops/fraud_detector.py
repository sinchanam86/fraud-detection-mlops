"""
fraud_detector.py
─────────────────
Layer 1: Fraud Detection System

Components:
  • FraudDetector  — trains a soft-voting ensemble (RF + GBT)
  • ThresholdTuner — finds optimal decision threshold for imbalanced data
  • evaluate()     — full evaluation suite (AUC, PR-AUC, F1, confusion matrix)

Design notes:
  • Class imbalance handled via class_weight='balanced' + SMOTE-lite oversampling
  • Threshold tuned to maximise F2 score (recall-weighted, fraud prefers fewer FN)
  • All models expose predict_proba for downstream calibration
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


# ── Threshold Tuning ──────────────────────────────────────────────────────────

class ThresholdTuner:
    """
    Finds the optimal probability threshold for a binary classifier.

    Strategy:
      - Sweep thresholds 0.01 → 0.99
      - Score each with F-beta (default beta=2: penalises FN more than FP)
      - Optionally constrain to minimum recall (e.g. "catch ≥ 80% of fraud")
    """

    def __init__(self, beta: float = 2.0, min_recall: float = 0.70):
        self.beta = beta
        self.min_recall = min_recall
        self.best_threshold: float = 0.5
        self.threshold_df: Optional[pd.DataFrame] = None

    def fit(self, y_true: np.ndarray, y_proba: np.ndarray) -> "ThresholdTuner":
        thresholds = np.linspace(0.01, 0.99, 200)
        records = []

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            fb = fbeta_score(y_true, y_pred, beta=self.beta, zero_division=0)
            records.append({"threshold": t, "precision": prec, "recall": rec, "fbeta": fb})

        self.threshold_df = pd.DataFrame(records)

        # Filter to minimum recall constraint, then pick best F-beta
        eligible = self.threshold_df[self.threshold_df["recall"] >= self.min_recall]
        if eligible.empty:
            eligible = self.threshold_df  # relax constraint if nothing qualifies

        best_row = eligible.loc[eligible["fbeta"].idxmax()]
        self.best_threshold = float(best_row["threshold"])
        return self

    def predict(self, y_proba: np.ndarray) -> np.ndarray:
        return (y_proba >= self.best_threshold).astype(int)


# ── Main Model ────────────────────────────────────────────────────────────────

class FraudDetector:
    """
    Soft-voting ensemble: Random Forest + Gradient Boosting.

    Pipeline per base estimator:
      StandardScaler → Classifier → Platt calibration (isotonic)

    Usage:
      detector = FraudDetector()
      detector.fit(X_train, y_train)
      proba = detector.predict_proba(X_test)
      preds = detector.predict(X_test)          # uses tuned threshold
      metrics = detector.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        n_estimators_rf: int = 200,
        n_estimators_gb: int = 150,
        rf_max_depth: int = 12,
        gb_max_depth: int = 5,
        gb_learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators_rf": n_estimators_rf,
            "n_estimators_gb": n_estimators_gb,
            "rf_max_depth": rf_max_depth,
            "gb_max_depth": gb_max_depth,
            "gb_learning_rate": gb_learning_rate,
            "random_state": random_state,
        }
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.tuner = ThresholdTuner(beta=2.0, min_recall=0.75)
        self.feature_names: Optional[list] = None
        self.feature_importances_: Optional[pd.Series] = None
        self.is_fitted = False

    def _build_pipeline(self) -> Pipeline:
        rf = RandomForestClassifier(
            n_estimators=self.params["n_estimators_rf"],
            max_depth=self.params["rf_max_depth"],
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=self.params["n_estimators_gb"],
            max_depth=self.params["gb_max_depth"],
            learning_rate=self.params["gb_learning_rate"],
            subsample=0.8,
            random_state=self.random_state,
        )

        # Calibrate GB (RF is already well-calibrated via averaging)
        gb_cal = CalibratedClassifierCV(gb, method="isotonic", cv=3)

        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb_cal)],
            voting="soft",
            weights=[0.45, 0.55],  # slightly favour GB for fraud
        )

        return Pipeline([
            ("scaler", StandardScaler()),
            ("ensemble", ensemble),
        ])

    @staticmethod
    def _oversample_minority(
        X: np.ndarray, y: np.ndarray, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple random oversampling of the minority class.
        (Drop-in replacement until imbalanced-learn is available.)
        """
        X_maj = X[y == 0]
        y_maj = y[y == 0]
        X_min = X[y == 1]
        y_min = y[y == 1]

        target_n = len(X_maj) // 3  # oversample to 1:3 ratio
        X_min_up, y_min_up = resample(X_min, y_min, replace=True, n_samples=target_n, random_state=seed)

        return np.vstack([X_maj, X_min_up]), np.concatenate([y_maj, y_min_up])

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_threshold: bool = True,
    ) -> "FraudDetector":
        self.feature_names = list(X_train.columns)

        X_np = X_train.values
        y_np = y_train.values

        # Oversample minority before fitting
        X_bal, y_bal = self._oversample_minority(X_np, y_np, seed=self.random_state)

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_bal, y_bal)

        # Feature importances from the RF component
        rf_model = self.pipeline.named_steps["ensemble"].estimators_[0]
        if hasattr(rf_model, "feature_importances_"):
            self.feature_importances_ = pd.Series(
                rf_model.feature_importances_,
                index=self.feature_names,
            ).sort_values(ascending=False)

        # Tune threshold on ORIGINAL (imbalanced) training set
        if tune_threshold:
            val_proba = cross_val_predict(
                self.pipeline,
                X_np,
                y_np,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                method="predict_proba",
            )[:, 1]
            self.tuner.fit(y_np, val_proba)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X.values)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.tuner.predict(proba)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        proba = self.predict_proba(X_test)
        y_pred = self.tuner.predict(proba)
        y_true = y_test.values

        auc_roc = roc_auc_score(y_true, proba)
        auc_pr  = average_precision_score(y_true, proba)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "auc_roc":    round(float(auc_roc), 4),
            "auc_pr":     round(float(auc_pr), 4),
            "precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1":         round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "f2":         round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
            "threshold":  round(self.tuner.best_threshold, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "fraud_caught_pct": round(float(tp / (tp + fn) * 100), 2),
            "false_alarm_rate": round(float(fp / (fp + tn) * 100), 2),
        }
        return metrics

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "FraudDetector":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Evaluation Report ─────────────────────────────────────────────────────────

def evaluate(
    detector: FraudDetector,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True,
) -> Dict:
    metrics = detector.evaluate(X_test, y_test)

    if verbose:
        print("\n" + "═" * 50)
        print("  FRAUD DETECTION — EVALUATION REPORT")
        print("═" * 50)
        print(f"  AUC-ROC        : {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR         : {metrics['auc_pr']:.4f}   (primary metric for imbalanced)")
        print(f"  Precision      : {metrics['precision']:.4f}")
        print(f"  Recall         : {metrics['recall']:.4f}")
        print(f"  F1             : {metrics['f1']:.4f}")
        print(f"  F2 (β=2)       : {metrics['f2']:.4f}   (recall-weighted)")
        print(f"  Threshold      : {metrics['threshold']:.4f}")
        print()
        print(f"  TP: {metrics['tp']:4d}  FP: {metrics['fp']:4d}")
        print(f"  FN: {metrics['fn']:4d}  TN: {metrics['tn']:4d}")
        print()
        print(f"  Fraud caught   : {metrics['fraud_caught_pct']:.1f}%")
        print(f"  False alarms   : {metrics['false_alarm_rate']:.1f}% of legit transactions")
        print("═" * 50)

        if detector.feature_importances_ is not None:
            print("\n  TOP-10 FEATURE IMPORTANCES")
            print("  " + "─" * 40)
            top10 = detector.feature_importances_.head(10)
            for feat, imp in top10.items():
                bar = "█" * int(imp * 200)
                print(f"  {feat:<35s} {imp:.4f}  {bar}")

    return metrics
