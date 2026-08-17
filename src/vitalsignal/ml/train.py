"""Train the outbreak-signal classifier and log everything to MLflow.

Design decisions worth defending in an interview:

  * TEMPORAL SPLIT, not a random one. Rows are a time series per
    (facility, condition). A random split puts Tuesday in train and Wednesday
    in test, and the 7-day rolling features make that a leak. We hold out the
    last N days, which is also how the model would actually be used.

  * PR-AUC is the headline metric. Positives are ~0.5% of rows. ROC-AUC looks
    flattering at that base rate; average precision does not.

  * THE BASELINE IS A RULE, NOT A DUMMY. Public health surveillance already has
    an aberration rule (flag when z >= 3). A gradient-boosted model is only
    worth deploying if it beats that rule at the same alert volume. This script
    reports both, and prints the comparison whether or not the model wins.

  * THE THRESHOLD IS CHOSEN ON TRAIN, under an alert budget. Epidemiologists
    can investigate a fixed number of signals per week; picking the threshold
    on the test set would be a leak and would also produce an operationally
    meaningless number.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from vitalsignal.config import get_settings
from vitalsignal.ml.features import FEATURE_COLUMNS, FEATURE_TABLE, LABEL_COLUMN

warnings.filterwarnings("ignore", category=FutureWarning)

EXPERIMENT = "vitalsignal-outbreak-signal"


def _mlflow():
    """Configure MLflow for local file store or Azure ML, then return it."""
    import mlflow

    s = get_settings()
    uri = s.mlflow_tracking_uri
    if uri.startswith(("azureml:", "databricks")):
        mlflow.set_tracking_uri(uri)
    else:
        # Local file store is in maintenance mode in MLflow 3.x; we opt in
        # explicitly rather than standing up a tracking server for a demo.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(Path(uri).resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def load_features() -> pd.DataFrame:
    s = get_settings()
    path = Path(s.uri("feature", FEATURE_TABLE)) / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make features` first")
    return pd.read_parquet(path)


def temporal_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["report_date"])
    cutoff = dates.max() - pd.Timedelta(days=test_days)
    return df[dates <= cutoff].copy(), df[dates > cutoff].copy()


def threshold_for_alert_budget(
    scores: np.ndarray, n_days: int, n_series: int, alerts_per_week: float
) -> float:
    """Highest threshold that still yields roughly `alerts_per_week` signals."""
    weeks = max(n_days / 7.0, 1.0)
    budget = round(alerts_per_week * weeks)
    budget = min(max(budget, 1), len(scores) - 1)
    return float(np.sort(scores)[::-1][budget - 1])


def rule_baseline(df: pd.DataFrame, z_threshold: float) -> np.ndarray:
    """The status-quo aberration rule. NULL z (too little history) -> no alert."""
    return (df["baseline_z"].fillna(-np.inf) >= z_threshold).astype(int).to_numpy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_days: int) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "alerts": int(y_pred.sum()),
        "alerts_per_week": float(y_pred.sum() / max(n_days / 7.0, 1.0)),
    }


def run(test_days: int = 75, alerts_per_week: float = 8.0, seed: int = 42) -> dict:
    s = get_settings()
    mlflow = _mlflow()

    df = load_features()
    train_df, test_df = temporal_split(df, test_days)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN].to_numpy()
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[LABEL_COLUMN].to_numpy()

    n_series = df.groupby(["facility_id", "condition_code"]).ngroups
    train_days = train_df["report_date"].nunique()
    test_days_actual = test_df["report_date"].nunique()

    params = {
        "model": "HistGradientBoostingClassifier",
        "max_depth": 4,
        "learning_rate": 0.06,
        "max_iter": 300,
        "l2_regularization": 1.0,
        "min_samples_leaf": 15,
        "class_weight": "balanced",
        "random_state": seed,
        "test_days": test_days,
        "alerts_per_week_budget": alerts_per_week,
    }

    with mlflow.start_run(run_name=f"hgb-tw{test_days}") as run_ctx:
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.set_tags(
            {
                "layer": "ml",
                "label_source": "synthetic_generator",
                "split": "temporal",
                "data_version": str(len(df)),
            }
        )

        clf = HistGradientBoostingClassifier(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            l2_regularization=params["l2_regularization"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=seed,
        )
        # NaNs (a series with <28 days of history) are passed through on
        # purpose -- HistGradientBoosting learns a split direction for missing,
        # which is more honest than imputing a baseline we do not have.
        clf.fit(X_train, y_train)

        train_scores = clf.predict_proba(X_train)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]

        thr = threshold_for_alert_budget(
            train_scores, train_days, n_series, alerts_per_week
        )
        model_metrics = evaluate(y_test, (test_scores >= thr).astype(int), test_days_actual)
        model_metrics.update(
            {
                "pr_auc": float(average_precision_score(y_test, test_scores)),
                "roc_auc": float(roc_auc_score(y_test, test_scores)),
                "brier": float(brier_score_loss(y_test, test_scores)),
                "threshold": thr,
            }
        )

        rule_metrics = {
            f"rule_z{z}": evaluate(y_test, rule_baseline(test_df, z), test_days_actual)
            for z in (2.0, 3.0)
        }

        mlflow.log_metrics({f"test_{k}": v for k, v in model_metrics.items()})
        for name, m in rule_metrics.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})
        mlflow.log_metrics(
            {
                "n_train_rows": len(train_df),
                "n_test_rows": len(test_df),
                "n_test_positives": int(y_test.sum()),
                "base_rate_test": float(y_test.mean()),
            }
        )

        # Permutation importance on the HELD-OUT set. Impurity-based importance
        # would be computed on training data and would flatter high-cardinality
        # features; this measures what the model actually relies on to score
        # days it has never seen.
        perm = permutation_importance(
            clf, X_test, y_test, n_repeats=5, random_state=seed,
            scoring="average_precision",
        )
        importance = dict(
            sorted(
                zip(FEATURE_COLUMNS, perm.importances_mean.tolist(), strict=True),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
        mlflow.log_metrics({f"perm_imp__{k}": v for k, v in importance.items()})

        signature_example = X_train.head(3)
        info = mlflow.sklearn.log_model(
            clf,
            name="model",
            input_example=signature_example,
            registered_model_name=s.model_name,
        )

        out = Path("_out")
        out.mkdir(exist_ok=True)
        report = {
            "run_id": run_ctx.info.run_id,
            "model_uri": info.model_uri,
            "threshold": thr,
            "model": model_metrics,
            "rule_baselines": rule_metrics,
            "n_test_positives": int(y_test.sum()),
            "permutation_importance": importance,
            "caveat": (
                "Labels are injected by the synthetic generator in this repo. "
                "These numbers validate the pipeline, not real-world detection."
            ),
        }
        (out / "train_report.json").write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(out / "train_report.json"))

    print(
        f"\ntrain rows {len(train_df):,} ({train_days}d) | "
        f"test rows {len(test_df):,} ({test_days_actual}d, {int(y_test.sum())} positives)\n"
        f"  model   PR-AUC {model_metrics['pr_auc']:.3f}  ROC-AUC {model_metrics['roc_auc']:.3f}\n"
        f"  model @ budget  P {model_metrics['precision']:.3f}  R {model_metrics['recall']:.3f}  "
        f"({model_metrics['alerts_per_week']:.1f} alerts/wk)\n"
        f"  rule z>=2       P {rule_metrics['rule_z2.0']['precision']:.3f}  "
        f"R {rule_metrics['rule_z2.0']['recall']:.3f}  "
        f"({rule_metrics['rule_z2.0']['alerts_per_week']:.1f} alerts/wk)\n"
        f"  rule z>=3       P {rule_metrics['rule_z3.0']['precision']:.3f}  "
        f"R {rule_metrics['rule_z3.0']['recall']:.3f}  "
        f"({rule_metrics['rule_z3.0']['alerts_per_week']:.1f} alerts/wk)\n"
        f"  top features: "
        + ", ".join(f"{k} {v:.2f}" for k, v in list(importance.items())[:3])
        + "\n  report -> _out/train_report.json"
    )
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-days", type=int, default=75)
    ap.add_argument("--alerts-per-week", type=float, default=8.0)
    args = ap.parse_args()
    run(test_days=args.test_days, alerts_per_week=args.alerts_per_week)
