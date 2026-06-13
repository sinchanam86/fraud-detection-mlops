"""
main.py
───────
End-to-end demo:

  1. Generate synthetic bank-transfer dataset
  2. Train FraudDetector (Layer 1)
  3. Evaluate on held-out test set
  4. Register baseline model in ModelRegistry
  5. Simulate production batch with drift
  6. Run RetrainingPipeline (Layer 2)
  7. Print full experiment history

Run:
  python main.py
"""

import os
import sys
import time
from pathlib import Path

# Set working dir so relative paths work regardless of where script is called
os.chdir(Path(__file__).parent)

import pandas as pd
from sklearn.model_selection import train_test_split

from data_generator import generate_transfers, generate_drift_batch, FEATURE_COLS, TARGET_COL
from fraud_detector import FraudDetector, evaluate
from mlops import (
    DriftDetector,
    ExperimentTracker,
    ModelRegistry,
    RetrainingPipeline,
)


def main():
    print("=" * 55)
    print("  FRAUD DETECTION + MLOPS SYSTEM")
    print("  Bank Transfer Fraud Detection")
    print("=" * 55)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/6] Generating synthetic bank-transfer dataset...")
    df = generate_transfers(n_legit=8000, n_fraud=400, seed=42)
    print(f"      Total records  : {len(df):,}")
    print(f"      Fraud rate     : {df[TARGET_COL].mean()*100:.1f}%")
    print(f"      Features       : {len(FEATURE_COLS)}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    print(f"\n      Train : {len(X_train):,}  Val : {len(X_val):,}  Test : {len(X_test):,}")

    # ── 2. Train baseline model ───────────────────────────────────────────────
    print("\n[2/6] Training FraudDetector (RF + GBT ensemble)...")
    t0 = time.time()
    detector = FraudDetector(
        n_estimators_rf=200,
        n_estimators_gb=150,
        gb_learning_rate=0.05,
        random_state=42,
    )
    detector.fit(X_train, y_train)
    print(f"      Training time  : {time.time()-t0:.1f}s")

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("\n[3/6] Evaluating on held-out test set...")
    metrics = evaluate(detector, X_test, y_test, verbose=True)

    # ── 4. MLOps — initial setup ──────────────────────────────────────────────
    print("\n[4/6] Setting up MLOps layer...")

    tracker  = ExperimentTracker("fraud_detection", tracking_dir="logs")
    registry = ModelRegistry("registry")
    drift_detector = DriftDetector(
        psi_threshold=0.15,
        ks_pvalue_threshold=0.05,
        drift_feature_pct=0.20,
    )

    # Fit drift detector on training distribution
    drift_detector.fit(X_train, FEATURE_COLS)

    # Track baseline experiment
    with tracker.start_run(run_name="baseline_v1") as run:
        run.log_params(detector.params)
        run.log_metrics(metrics)
        run.log_dict(metrics, "baseline_metrics.json")

    # Register and promote baseline directly to production
    v1 = registry.register(detector, metrics, run_id="baseline", model_name="fraud_detector")
    registry.transition(v1, "production")

    print("\n      Registry state:")
    print(registry.list_versions().to_string(index=False))

    # ── 5. Simulate production drift ──────────────────────────────────────────
    print("\n[5/6] Simulating production batch with data drift...")
    drift_batch = generate_drift_batch(n=1200, drift_factor=0.45, seed=77)
    prod_X = drift_batch[FEATURE_COLS]
    print(f"      Production batch size : {len(prod_X):,}")
    print("      (Simulated fraud campaign: higher amounts, more velocity, more new accounts)")

    # Run quick drift check without retraining
    quick_report = drift_detector.detect(prod_X)
    drift_detector.print_report(quick_report)

    # ── 6. Retraining pipeline ────────────────────────────────────────────────
    print("\n[6/6] Running automated retraining pipeline...")

    pipeline = RetrainingPipeline(
        tracker=tracker,
        registry=registry,
        drift_detector=drift_detector,
        retrain_on_alert_only=True,
    )

    result = pipeline.run(
        production_batch_df=prod_X,
        X_historical=X_train,
        y_historical=y_train,
        feature_cols=FEATURE_COLS,
        X_val=X_val,
        y_val=y_val,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  EXPERIMENT HISTORY")
    print("=" * 55)
    tracker.print_summary()

    print("\n" + "=" * 55)
    print("  MODEL REGISTRY")
    print("=" * 55)
    print(registry.list_versions().to_string(index=False))

    print("\n" + "=" * 55)
    print("  PIPELINE RESULT")
    print("=" * 55)
    print(f"  Status     : {result['status']}")
    if result["status"] == "retrained":
        print(f"  Run        : {result['run_name']}")
        print(f"  Version    : v{result['version']}")
        print(f"  Promoted   : {'Yes' if result['promoted'] else 'No'}")
        m = result["metrics"]
        print(f"  AUC-PR     : {m.get('auc_pr', 'N/A')}")
        print(f"  Recall     : {m.get('recall', 'N/A')}")
        print(f"  F2         : {m.get('f2', 'N/A')}")

    print("\n  Artifacts saved to:")
    print("    logs/      — experiment runs (JSONL)")
    print("    registry/  — versioned model files + index.json")
    print("=" * 55)

    # ── Verify production model loads correctly ───────────────────────────────
    print("\n  Verifying production model...")
    prod_model = registry.load_production()
    sample = X_test.iloc[:5]
    proba = prod_model.predict_proba(sample)
    pred  = prod_model.predict(sample)
    print("  Sample predictions (first 5 test rows):")
    for i, (p, pr) in enumerate(zip(pred, proba)):
        label = "FRAUD" if p == 1 else "legit"
        print(f"    [{i}] {label}  (fraud prob: {pr:.3f})")

    print("\n  ✓ System end-to-end OK.\n")


if __name__ == "__main__":
    main()
