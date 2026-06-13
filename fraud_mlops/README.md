# Fraud Detection System + MLOps
## Bank Transfer Fraud Detection — Research Implementation

---

## Architecture

```
fraud_mlops/
├── data_generator.py   Layer 1 — Synthetic data + feature engineering
├── fraud_detector.py   Layer 1 — Ensemble model + threshold tuning
├── mlops.py            Layer 2 — Tracker / Registry / Drift / Pipeline
├── main.py             End-to-end demo runner
├── logs/               Experiment runs (JSONL)
└── registry/           Versioned model artifacts + index.json
```

### Layer 1: Fraud Detection

**Feature engineering** (18 features across 6 categories):
- Amount signals: `log_amount`, `amount_is_round`, `amount_vs_avg_ratio`
- Velocity: `n_transfers_sender_24h`, `n_transfers_sender_7d`, `velocity_amount_risk`
- Account age: `sender_account_age_days`, `receiver_account_age_days`, `combined_account_newness`
- Behavioural: `hour_of_day`, `time_since_last_login_hrs`, `device_fingerprint_mismatch`
- Cross-border: `is_cross_border`, `sender_is_domestic_only`, `cross_border_domestic_flag`
- Receiver: `receiver_is_new`

**Ensemble model**: Soft-voting (RF 45% + Gradient Boosting 55%)
- Random Forest: `n_estimators=200`, `class_weight='balanced'`
- Gradient Boosting: `n_estimators=150`, `lr=0.05`, calibrated via isotonic regression
- Class imbalance: random oversampling to 1:3 minority ratio before fit
- Threshold tuning: F2 score sweep (β=2 penalises false negatives), min recall ≥ 75%

**Metrics**: AUC-ROC, AUC-PR (primary for imbalanced), Precision, Recall, F1, F2

### Layer 2: MLOps

| Component | Class | Description |
|-----------|-------|-------------|
| Experiment tracking | `ExperimentTracker` | Logs params + metrics to JSONL. Drop-in MLflow API |
| Model versioning | `ModelRegistry` | Staged promotion: staging → production → archived |
| Drift detection | `DriftDetector` | PSI + KS-test per feature; alerts on ≥20% drifted |
| Retraining | `RetrainingPipeline` | Watches drift, retrains, registers, auto-promotes |

**Drift thresholds**:
- PSI < 0.10 → stable
- PSI 0.10–0.25 → moderate (monitor)
- PSI > 0.25 → significant (retrain candidate)
- KS test p < 0.05 → distributions differ

---

## Quickstart

```bash
cd fraud_mlops
python main.py
```

---

## Switching to Real MLflow

The `ExperimentTracker` mirrors the MLflow API exactly. To use real MLflow:

```python
# Current (local JSON backend):
from mlops import ExperimentTracker
tracker = ExperimentTracker("fraud_detection")

# Replace with real MLflow:
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")   # or databricks://...
mlflow.set_experiment("fraud_detection")

# In your run:
with mlflow.start_run(run_name="retrain_v2") as run:
    mlflow.log_params(detector.params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(detector, "model")
```

For the model registry, MLflow Model Registry replaces `ModelRegistry`:
```python
mlflow.register_model("runs:/<run_id>/model", "fraud_detector")
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage("fraud_detector", version=1, stage="Production")
```

---

## Extending the System

### Add a new feature
In `data_generator.py`, add a column to `generate_transfers()` and `generate_drift_batch()`,
then add the feature name to `FEATURE_COLS`.

### Swap in XGBoost
In `fraud_detector.py`, replace `GradientBoostingClassifier` with `xgboost.XGBClassifier`:
```python
from xgboost import XGBClassifier
gb = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    scale_pos_weight=20,  # handles imbalance natively
    use_label_encoder=False,
    eval_metric="aucpr",
)
```

### Scheduled retraining (production)
Wrap `RetrainingPipeline.run()` in a cron job or Airflow DAG:
```python
# airflow_dag.py
from airflow.decorators import task, dag
@dag(schedule="@daily")
def fraud_retrain():
    @task
    def run_pipeline():
        batch = fetch_todays_transfers()
        pipeline.run(production_batch_df=batch, ...)
```

### SHAP explainability
```python
import shap
explainer = shap.TreeExplainer(detector.pipeline.named_steps["ensemble"].estimators_[0])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, feature_names=FEATURE_COLS)
```

---

## Key Design Decisions

**Why F2 for threshold tuning?**
In fraud detection, a missed fraud (FN) is far more costly than a false alarm (FP).
F2 weights recall twice as heavily as precision. The `min_recall=0.75` constraint
ensures we never drop below 75% fraud-caught, regardless of precision trade-offs.

**Why PSI + KS together?**
PSI detects shifts in the full distribution shape (binned). KS detects any difference
in the CDF. They're complementary: PSI is more sensitive to tail shifts; KS has a
clean p-value for statistical rigour. Using both reduces false positives in drift alerts.

**Why random oversampling instead of SMOTE?**
SMOTE requires `imbalanced-learn`. Random oversampling is a conservative baseline
that is free of the synthetic-sample interpolation artefacts SMOTE can introduce in
high-cardinality tabular data. Replace with SMOTE once the dependency is available.
