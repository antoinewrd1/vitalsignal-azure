"""Batch inference -> gold.fct_outbreak_signal.

Deliberately loads the model from the MLflow *registry* by name, not from a
pickle on disk. That is the whole point of registering: the scoring job and the
training job are decoupled, and promoting a new model is a registry operation
rather than a code deploy.

The threshold is read from the training run's report rather than hardcoded,
so the alert budget the model was tuned for is the one it actually serves.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from vitalsignal.config import get_settings
from vitalsignal.ml.features import FEATURE_COLUMNS, FEATURE_TABLE, KEY_COLUMNS

SIGNAL_TABLE = "fct_outbreak_signal"


def load_model(stage_or_version: str = "latest"):
    import mlflow

    s = get_settings()
    uri = s.mlflow_tracking_uri
    if uri.startswith(("azureml:", "databricks")):
        mlflow.set_tracking_uri(uri)
    else:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(Path(uri).resolve().as_uri())

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{s.model_name}'")
    if not versions:
        raise RuntimeError(f"no registered versions of {s.model_name} -- run `make train`")
    latest = max(versions, key=lambda v: int(v.version))
    print(f"score: loading {s.model_name} v{latest.version}")
    return mlflow.sklearn.load_model(f"models:/{s.model_name}/{latest.version}"), latest.version


def run(window_days: int = 14) -> pd.DataFrame:
    s = get_settings()
    feats = pd.read_parquet(Path(s.uri("feature", FEATURE_TABLE)) / "features.parquet")

    report_path = Path("_out/train_report.json")
    threshold = json.loads(report_path.read_text())["threshold"] if report_path.exists() else 0.5

    dates = pd.to_datetime(feats["report_date"])
    recent = feats[dates > dates.max() - pd.Timedelta(days=window_days)].copy()

    model, version = load_model()
    recent["signal_score"] = model.predict_proba(recent[FEATURE_COLUMNS])[:, 1]
    recent["is_signal"] = (recent["signal_score"] >= threshold).astype(int)
    recent["model_version"] = version
    recent["threshold"] = threshold
    recent["scored_at"] = pd.Timestamp.now("UTC")

    out_cols = [
        *KEY_COLUMNS,
        "condition_name", "county", "case_count", "baseline_mean_28d", "baseline_z",
        "signal_score", "is_signal", "threshold", "model_version", "scored_at",
    ]
    signals = recent[out_cols].sort_values("signal_score", ascending=False)

    out_dir = Path(s.uri("gold", SIGNAL_TABLE))
    out_dir.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(out_dir / "signals.parquet", index=False)

    fired = int(signals["is_signal"].sum())
    print(
        f"score: {len(signals):,} rows scored over last {window_days}d, "
        f"{fired} above threshold {threshold:.4f} -> {out_dir}"
    )
    if fired:
        cols = ["report_date", "facility_id", "condition_name", "case_count",
                "baseline_mean_28d", "signal_score"]
        print(signals[signals.is_signal == 1][cols].head(10).to_string(index=False))
    return signals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=14)
    run(**vars(ap.parse_args()))
