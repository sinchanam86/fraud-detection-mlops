"""
mlops.py
────────
Layer 2: MLOps

Components:
  • ExperimentTracker  — MLflow-compatible tracker (local JSON backend)
                         Swap for real MLflow with one-line change (see bottom)
  • ModelRegistry      — versioned model store with stage promotion
  • DriftDetector      — PSI + KS-test per feature; alerts on threshold breach
  • RetrainingPipeline — watches drift signals, triggers retrain, registers result
"""

import hashlib
import json
import os
import pickle
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)): return int(o)
        if isinstance(o, np.floating): return float(o)
        return super().default(o)
import pandas as pd
from scipy import stats


# ── Experiment Tracker ────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    MLflow-compatible experiment tracker backed by local JSON.

    API mirrors mlflow:
        tracker = ExperimentTracker("fraud_detection")
        with tracker.start_run(run_name="baseline_v1") as run:
            run.log_params({"n_estimators": 200})
            run.log_metrics({"auc_roc": 0.97})
            run.log_artifact("model.pkl")

    To switch to real MLflow (when running locally):
        import mlflow
        mlflow.set_experiment("fraud_detection")
        with mlflow.start_run(run_name="baseline_v1") as run:
            mlflow.log_params(...)
            mlflow.log_metrics(...)
    """

    def __init__(self, experiment_name: str, tracking_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.tracking_dir / f"{experiment_name}.jsonl"

    def start_run(self, run_name: str = "") -> "Run":
        return Run(
            run_name=run_name or f"run_{int(time.time())}",
            tracker=self,
        )

    def load_runs(self) -> pd.DataFrame:
        if not self._log_file.exists():
            return pd.DataFrame()
        rows = []
        with open(self._log_file) as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    def best_run(self, metric: str = "auc_pr", higher_is_better: bool = True) -> Optional[Dict]:
        df = self.load_runs()
        if df.empty or metric not in df.columns:
            return None
        idx = df[metric].idxmax() if higher_is_better else df[metric].idxmin()
        return df.loc[idx].to_dict()

    def _save_run(self, record: Dict) -> None:
        with open(self._log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def print_summary(self) -> None:
        df = self.load_runs()
        if df.empty:
            print("No runs recorded yet.")
            return
        print(f"\n{'─'*60}")
        print(f"  Experiment: {self.experiment_name}  ({len(df)} runs)")
        print(f"{'─'*60}")
        cols = ["run_name", "auc_roc", "auc_pr", "recall", "precision", "f2", "threshold"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))


class Run:
    """Context-manager run handle — mirrors mlflow.ActiveRun."""

    def __init__(self, run_name: str, tracker: ExperimentTracker):
        self.run_name = run_name
        self.run_id = str(uuid.uuid4())[:8]
        self.tracker = tracker
        self._record: Dict = {
            "run_id": self.run_id,
            "run_name": run_name,
            "start_time": datetime.utcnow().isoformat(),
            "status": "RUNNING",
        }

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, *_):
        self._record["end_time"] = datetime.utcnow().isoformat()
        self._record["status"] = "FAILED" if exc_type else "FINISHED"
        self.tracker._save_run(self._record)

    def log_params(self, params: Dict) -> None:
        self._record.update(params)

    def log_metrics(self, metrics: Dict) -> None:
        self._record.update(metrics)

    def log_artifact(self, path: str) -> None:
        self._record.setdefault("artifacts", []).append(path)

    def log_dict(self, data: Dict, filename: str) -> None:
        out = self.tracker.tracking_dir / filename
        with open(out, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)
        self.log_artifact(str(out))


# ── Model Registry ────────────────────────────────────────────────────────────

STAGES = ("staging", "production", "archived")


class ModelRegistry:
    """
    Versioned model registry.

    Lifecycle: None → Staging → Production → Archived

    Usage:
        registry = ModelRegistry("registry")
        v = registry.register(detector, metrics, run_id="abc123")
        registry.transition(v, "production")
        model = registry.load_production()
    """

    def __init__(self, registry_dir: str = "registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "index.json"
        self._index: List[Dict] = self._load_index()

    def _load_index(self) -> List[Dict]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return []

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def register(
        self,
        model,
        metrics: Dict,
        run_id: str = "",
        model_name: str = "fraud_detector",
    ) -> int:
        version = len(self._index) + 1
        fname = f"{model_name}_v{version}.pkl"
        model_path = self.registry_dir / fname

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        record = {
            "version": version,
            "model_name": model_name,
            "run_id": run_id,
            "stage": "staging",
            "registered_at": datetime.utcnow().isoformat(),
            "model_path": str(model_path),
            "metrics": metrics,
        }
        self._index.append(record)
        self._save_index()

        print(f"  [Registry] Registered {model_name} v{version} → staging")
        return version

    def transition(self, version: int, stage: str) -> None:
        assert stage in STAGES, f"stage must be one of {STAGES}"
        for rec in self._index:
            if rec["version"] == version:
                old_stage = rec["stage"]
                rec["stage"] = stage
                rec["transitioned_at"] = datetime.utcnow().isoformat()
                self._save_index()
                print(f"  [Registry] v{version}: {old_stage} → {stage}")
                return
        raise ValueError(f"Version {version} not found in registry")

    def load_production(self, model_name: str = "fraud_detector"):
        prod = [r for r in self._index if r["stage"] == "production" and r["model_name"] == model_name]
        if not prod:
            raise RuntimeError(f"No production model for '{model_name}'")
        latest = max(prod, key=lambda r: r["version"])
        with open(latest["model_path"], "rb") as f:
            return pickle.load(f)

    def list_versions(self) -> pd.DataFrame:
        if not self._index:
            return pd.DataFrame()
        rows = []
        for r in self._index:
            row = {
                "version": r["version"],
                "stage": r["stage"],
                "registered_at": r["registered_at"][:19],
                **{f"metric_{k}": v for k, v in r.get("metrics", {}).items()
                   if k in ("auc_roc", "auc_pr", "recall", "f2")},
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def promote_best_staging(
        self,
        metric: str = "auc_pr",
        min_improvement: float = 0.005,
    ) -> Optional[int]:
        """
        Promote the best staging model to production if it beats the current
        production model by at least `min_improvement`.
        """
        staging = [r for r in self._index if r["stage"] == "staging"]
        if not staging:
            print("  [Registry] No staging models to promote.")
            return None

        best_staging = max(staging, key=lambda r: r["metrics"].get(metric, 0))
        prod = [r for r in self._index if r["stage"] == "production"]

        if prod:
            current_prod_score = max(p["metrics"].get(metric, 0) for p in prod)
            new_score = best_staging["metrics"].get(metric, 0)
            if new_score < current_prod_score + min_improvement:
                print(
                    f"  [Registry] Staging v{best_staging['version']} ({new_score:.4f}) "
                    f"does not beat production ({current_prod_score:.4f}) by >{min_improvement}. "
                    "No promotion."
                )
                return None
            # Archive old production models
            for p in prod:
                self.transition(p["version"], "archived")

        self.transition(best_staging["version"], "production")
        return best_staging["version"]


# ── Drift Detector ────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Detects data drift between a reference (training) distribution and
    an incoming production batch.

    Two complementary tests per feature:
      • PSI  (Population Stability Index)
            < 0.10  → no significant shift
            0.10–0.25 → moderate shift (monitor)
            > 0.25  → significant shift (retrain candidate)

      • KS test (Kolmogorov-Smirnov, two-sample)
            p-value < 0.05 → distributions differ significantly

    An overall drift alert fires when ≥ drift_threshold_pct% of features
    trigger either test.
    """

    PSI_BINS = 10
    PSI_EPSILON = 1e-6

    def __init__(
        self,
        psi_threshold: float = 0.15,
        ks_pvalue_threshold: float = 0.05,
        drift_feature_pct: float = 0.20,  # alert if ≥20% of features drift
    ):
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.drift_feature_pct = drift_feature_pct
        self._reference: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[List[str]] = None

    def fit(self, reference_df: pd.DataFrame, feature_cols: List[str]) -> "DriftDetector":
        """Store reference distribution from training data."""
        self._reference = reference_df[feature_cols].copy()
        self._feature_cols = feature_cols
        return self

    def detect(self, production_df: pd.DataFrame) -> Dict:
        """
        Compare production batch against reference.
        Returns a report dict and per-feature details.
        """
        assert self._reference is not None, "Call .fit() first"

        results = {}
        drifted_features = []

        for col in self._feature_cols:
            ref_vals = self._reference[col].dropna().values
            prod_vals = production_df[col].dropna().values

            psi = self._compute_psi(ref_vals, prod_vals)
            ks_stat, ks_pval = stats.ks_2samp(ref_vals, prod_vals)

            feature_drifted = (psi > self.psi_threshold) or (ks_pval < self.ks_pvalue_threshold)
            if feature_drifted:
                drifted_features.append(col)

            results[col] = {
                "psi": round(float(psi), 4),
                "ks_stat": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pval), 4),
                "drifted": feature_drifted,
                "severity": self._psi_severity(psi),
            }

        drift_pct = len(drifted_features) / len(self._feature_cols)
        overall_alert = drift_pct >= self.drift_feature_pct

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_features_checked": len(self._feature_cols),
            "n_features_drifted": len(drifted_features),
            "drift_percentage": round(drift_pct * 100, 1),
            "overall_drift_alert": overall_alert,
            "drifted_features": drifted_features,
            "feature_details": results,
        }

        return report

    def _compute_psi(self, ref: np.ndarray, prod: np.ndarray) -> float:
        breakpoints = np.linspace(ref.min(), ref.max(), self.PSI_BINS + 1)
        breakpoints[0] -= 1e-6
        breakpoints[-1] += 1e-6

        ref_counts, _ = np.histogram(ref, bins=breakpoints)
        prod_counts, _ = np.histogram(prod, bins=breakpoints)

        ref_pct = ref_counts / (len(ref) + self.PSI_EPSILON) + self.PSI_EPSILON
        prod_pct = prod_counts / (len(prod) + self.PSI_EPSILON) + self.PSI_EPSILON

        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        return float(psi)

    @staticmethod
    def _psi_severity(psi: float) -> str:
        if psi < 0.10:
            return "stable"
        elif psi < 0.25:
            return "moderate"
        else:
            return "significant"

    def print_report(self, report: Dict) -> None:
        print(f"\n{'═'*55}")
        print("  DATA DRIFT REPORT")
        print(f"{'═'*55}")
        print(f"  Timestamp        : {report['timestamp'][:19]}")
        print(f"  Features checked : {report['n_features_checked']}")
        print(f"  Features drifted : {report['n_features_drifted']} ({report['drift_percentage']:.1f}%)")
        alert_str = "🚨 ALERT — RETRAIN RECOMMENDED" if report["overall_drift_alert"] else "✓ Within tolerance"
        print(f"  Overall status   : {alert_str}")

        if report["drifted_features"]:
            print(f"\n  Drifted features:")
            for feat in report["drifted_features"]:
                d = report["feature_details"][feat]
                print(
                    f"    {feat:<38s}  PSI={d['psi']:.3f} [{d['severity']}]"
                    f"  KS p={d['ks_pvalue']:.3f}"
                )
        print("═" * 55)


# ── Retraining Pipeline ───────────────────────────────────────────────────────

class RetrainingPipeline:
    """
    Orchestrates the full MLOps retraining loop:

      1. Run drift detection on production batch
      2. If drift alert fires → trigger retraining
      3. Train new model on combined (old + new) data
      4. Track experiment with ExperimentTracker
      5. Register in ModelRegistry
      6. Promote to production if it beats current model

    Usage:
        pipeline = RetrainingPipeline(tracker, registry, drift_detector)
        pipeline.run(
            production_batch_df=new_data,
            X_historical=X_train,
            y_historical=y_train,
            feature_cols=FEATURE_COLS,
        )
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        registry: ModelRegistry,
        drift_detector: DriftDetector,
        min_new_samples: int = 200,
        retrain_on_alert_only: bool = True,
    ):
        self.tracker = tracker
        self.registry = registry
        self.drift_detector = drift_detector
        self.min_new_samples = min_new_samples
        self.retrain_on_alert_only = retrain_on_alert_only

    def run(
        self,
        production_batch_df: pd.DataFrame,
        X_historical: pd.DataFrame,
        y_historical: pd.Series,
        feature_cols: List[str],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        force_retrain: bool = False,
    ) -> Dict:
        """
        Full pipeline run. Returns a summary dict.

        Args:
            production_batch_df: new data arriving from production
            X_historical / y_historical: existing labeled training data
            feature_cols: list of feature column names
            X_val / y_val: optional held-out validation set for post-train eval
            force_retrain: bypass drift check and always retrain
        """
        print(f"\n{'━'*55}")
        print(f"  RETRAINING PIPELINE  [{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}]")
        print(f"{'━'*55}")

        # Step 1 — Drift detection
        drift_report = self.drift_detector.detect(production_batch_df)
        self.drift_detector.print_report(drift_report)

        should_retrain = force_retrain or drift_report["overall_drift_alert"]

        if self.retrain_on_alert_only and not should_retrain:
            print("\n  ✓ No drift detected. Skipping retrain.")
            return {"status": "skipped", "reason": "no_drift", "drift_report": drift_report}

        # Step 2 — Combine data (historical + any labeled new data)
        print(f"\n  [Pipeline] Triggering retrain (drift: {drift_report['drift_percentage']}%)")

        # In production you'd have newly labeled fraud data here.
        # For research we reuse historical for demonstration.
        X_combined = X_historical.copy()
        y_combined = y_historical.copy()
        print(f"  [Pipeline] Training on {len(X_combined):,} samples")

        # Step 3 — Train with experiment tracking
        from fraud_detector import FraudDetector, evaluate as fd_evaluate

        run_name = f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        with self.tracker.start_run(run_name=run_name) as run:
            detector = FraudDetector(random_state=int(time.time()) % 10000)
            run.log_params(detector.params)

            t0 = time.time()
            detector.fit(X_combined, y_combined)
            elapsed = round(time.time() - t0, 1)

            if X_val is not None and y_val is not None:
                metrics = fd_evaluate(detector, X_val, y_val, verbose=True)
            else:
                # Self-eval on training (optimistic — use only when no val set)
                metrics = fd_evaluate(detector, X_combined, y_combined, verbose=False)

            metrics["train_seconds"] = elapsed
            metrics["n_train_samples"] = len(X_combined)
            run.log_metrics(metrics)
            run.log_dict(drift_report, f"drift_{run_name}.json")

            # Step 4 — Register
            version = self.registry.register(detector, metrics, run_id=run.run_id)

        # Step 5 — Promote if better
        promoted_version = self.registry.promote_best_staging()

        print(f"\n  [Pipeline] Run complete in {elapsed}s")

        return {
            "status": "retrained",
            "run_name": run_name,
            "version": version,
            "promoted": promoted_version is not None,
            "metrics": metrics,
            "drift_report": drift_report,
        }
