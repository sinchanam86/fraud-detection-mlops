"""
data_generator.py
─────────────────
Synthetic bank-transfer dataset generator.

Simulates realistic fraud patterns:
  • Round-amount transfers (fraudsters prefer €500, €1000 etc.)
  • Velocity spikes (many small transfers in short windows)
  • New-account → high-value anomaly
  • Unusual hours (2–5 AM)
  • Cross-border transfers from domestic-only accounts
  • Mule accounts (high in-flow, immediate out-flow)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_transfers(
    n_legit: int = 8000,
    n_fraud: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Legitimate transfers ──────────────────────────────────────────────────
    legit_amounts = rng.lognormal(mean=5.5, sigma=1.2, size=n_legit)
    legit_amounts = np.clip(legit_amounts, 1, 50_000)

    hours_legit = rng.choice(
        np.arange(24),
        size=n_legit,
        p=_hour_probs_legit(),
    )

    legit = pd.DataFrame(
        {
            "amount": legit_amounts,
            "hour_of_day": hours_legit,
            "sender_account_age_days": rng.integers(180, 3650, n_legit),
            "receiver_account_age_days": rng.integers(30, 3650, n_legit),
            "n_transfers_sender_24h": rng.integers(0, 8, n_legit),
            "n_transfers_sender_7d": rng.integers(1, 30, n_legit),
            "sender_avg_amount_30d": rng.lognormal(5.2, 1.0, n_legit),
            "is_cross_border": rng.choice([0, 1], n_legit, p=[0.85, 0.15]),
            "sender_is_domestic_only": rng.choice([0, 1], n_legit, p=[0.3, 0.7]),
            "amount_is_round": _is_round(legit_amounts),
            "receiver_is_new": (rng.integers(30, 3650, n_legit) < 30).astype(int),
            "time_since_last_login_hrs": rng.exponential(12, n_legit),
            "device_fingerprint_mismatch": rng.choice([0, 1], n_legit, p=[0.97, 0.03]),
            "is_fraud": 0,
        }
    )

    # ── Fraudulent transfers ──────────────────────────────────────────────────
    fraud_amounts = _fraud_amounts(rng, n_fraud)
    hours_fraud = rng.choice(
        np.arange(24),
        size=n_fraud,
        p=_hour_probs_fraud(),
    )

    fraud = pd.DataFrame(
        {
            "amount": fraud_amounts,
            "hour_of_day": hours_fraud,
            "sender_account_age_days": rng.integers(1, 90, n_fraud),   # new accounts
            "receiver_account_age_days": rng.integers(1, 45, n_fraud),  # new receivers
            "n_transfers_sender_24h": rng.integers(5, 30, n_fraud),     # velocity
            "n_transfers_sender_7d": rng.integers(15, 80, n_fraud),
            "sender_avg_amount_30d": rng.lognormal(3.5, 0.8, n_fraud),
            "is_cross_border": rng.choice([0, 1], n_fraud, p=[0.3, 0.7]),
            "sender_is_domestic_only": rng.choice([0, 1], n_fraud, p=[0.5, 0.5]),
            "amount_is_round": _is_round(fraud_amounts, bias=0.6),
            "receiver_is_new": rng.choice([0, 1], n_fraud, p=[0.1, 0.9]),
            "time_since_last_login_hrs": rng.exponential(0.5, n_fraud),  # just logged in
            "device_fingerprint_mismatch": rng.choice([0, 1], n_fraud, p=[0.4, 0.6]),
            "is_fraud": 1,
        }
    )

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=seed)
    df = df.reset_index(drop=True)

    # Add engineered composite features
    df = _add_engineered_features(df)

    return df


def generate_drift_batch(
    n: int = 1000,
    drift_factor: float = 0.4,
    seed: int = 99,
) -> pd.DataFrame:
    """
    Generate a production batch with simulated data drift.

    drift_factor=0 → identical distribution to training
    drift_factor=1 → heavily drifted (e.g. after a new fraud campaign)
    """
    rng = np.random.default_rng(seed)

    # Shift amount distribution upward (larger transfers)
    amounts = rng.lognormal(5.5 + drift_factor * 1.5, 1.2, n)

    # Shift velocity higher
    velocity_bias = int(drift_factor * 10)

    df = pd.DataFrame(
        {
            "amount": amounts,
            "hour_of_day": rng.choice(np.arange(24), n, p=_hour_probs_legit()),
            "sender_account_age_days": rng.integers(1, 3650 - int(drift_factor * 3000), n),
            "receiver_account_age_days": rng.integers(1, 3650, n),
            "n_transfers_sender_24h": rng.integers(0 + velocity_bias, 15 + velocity_bias, n),
            "n_transfers_sender_7d": rng.integers(1, 50, n),
            "sender_avg_amount_30d": rng.lognormal(5.2 + drift_factor, 1.0, n),
            "is_cross_border": rng.choice([0, 1], n, p=[0.85 - drift_factor * 0.3, 0.15 + drift_factor * 0.3]),
            "sender_is_domestic_only": rng.choice([0, 1], n, p=[0.3, 0.7]),
            "amount_is_round": _is_round(amounts, bias=drift_factor * 0.4),
            "receiver_is_new": rng.choice([0, 1], n, p=[0.7 - drift_factor * 0.4, 0.3 + drift_factor * 0.4]),
            "time_since_last_login_hrs": rng.exponential(12 - drift_factor * 10, n),
            "device_fingerprint_mismatch": rng.choice([0, 1], n, p=[0.97 - drift_factor * 0.3, 0.03 + drift_factor * 0.3]),
            "is_fraud": 0,  # labels unknown in production
        }
    )

    return _add_engineered_features(df)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_round(amounts: np.ndarray, bias: float = 0.0) -> np.ndarray:
    """Mark amounts that are multiples of 50, 100, 500, or 1000."""
    base = ((amounts % 50 == 0) | (amounts % 100 == 0)).astype(float)
    noise = np.random.default_rng(0).random(len(amounts))
    return np.where(noise < bias, 1, base).astype(int)


def _fraud_amounts(rng, n: int) -> np.ndarray:
    """Mix of round-number amounts and just-below-threshold amounts."""
    round_amounts = rng.choice([500, 1000, 2000, 5000, 10000], size=n // 2)
    below_threshold = rng.uniform(990, 999, size=n - n // 2)  # just below £1000 AML threshold
    amounts = np.concatenate([round_amounts, below_threshold])
    rng.shuffle(amounts)
    return amounts


def _hour_probs_legit() -> list:
    """Peak banking hours 9–17, very low overnight."""
    probs = np.array(
        [0.3, 0.2, 0.1, 0.1, 0.1, 0.2,  # 0–5 AM
         0.5, 1.0, 2.0, 4.0, 5.0, 5.5,  # 6–11
         5.5, 5.5, 5.0, 5.0, 4.5, 4.0,  # 12–17
         3.5, 3.0, 2.5, 2.0, 1.5, 1.0]  # 18–23
    )
    return (probs / probs.sum()).tolist()


def _hour_probs_fraud() -> list:
    """Fraud peaks overnight 2–5 AM."""
    probs = np.array(
        [2.0, 3.0, 5.0, 6.0, 5.0, 3.0,  # 0–5 AM  ← fraud peak
         1.0, 0.5, 0.5, 0.5, 1.0, 1.5,  # 6–11
         1.5, 1.5, 1.0, 1.0, 1.0, 1.0,  # 12–17
         1.5, 2.0, 2.5, 2.5, 2.0, 2.0]  # 18–23
    )
    return (probs / probs.sum()).tolist()


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and ratio features."""
    df = df.copy()

    # Amount relative to sender's history
    df["amount_vs_avg_ratio"] = df["amount"] / (df["sender_avg_amount_30d"] + 1e-9)

    # Velocity × amount risk score
    df["velocity_amount_risk"] = df["n_transfers_sender_24h"] * np.log1p(df["amount"])

    # Account age risk (new sender + new receiver = high risk)
    df["combined_account_newness"] = (
        np.exp(-df["sender_account_age_days"] / 365)
        + np.exp(-df["receiver_account_age_days"] / 365)
    )

    # Cross-border with domestic-only sender
    df["cross_border_domestic_flag"] = (
        df["is_cross_border"] & df["sender_is_domestic_only"]
    ).astype(int)

    # Night-time with device mismatch
    df["night_device_mismatch"] = (
        (df["hour_of_day"].between(0, 5)) & (df["device_fingerprint_mismatch"] == 1)
    ).astype(int)

    # Log-scale amount (stabilizes distributions)
    df["log_amount"] = np.log1p(df["amount"])

    return df


FEATURE_COLS = [
    "log_amount",
    "hour_of_day",
    "sender_account_age_days",
    "receiver_account_age_days",
    "n_transfers_sender_24h",
    "n_transfers_sender_7d",
    "sender_avg_amount_30d",
    "is_cross_border",
    "sender_is_domestic_only",
    "amount_is_round",
    "receiver_is_new",
    "time_since_last_login_hrs",
    "device_fingerprint_mismatch",
    "amount_vs_avg_ratio",
    "velocity_amount_risk",
    "combined_account_newness",
    "cross_border_domestic_flag",
    "night_device_mismatch",
]

TARGET_COL = "is_fraud"
