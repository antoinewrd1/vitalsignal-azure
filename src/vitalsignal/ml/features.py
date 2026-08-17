"""Gold -> ML feature table.

The ML layer reads the *warehouse*, not the lake. That is deliberate: the
z-score, the zero-fill and the 28-day baseline are already defined once in dbt
and tested there, so the model and the surveillance dashboard can never drift
apart on what "expected case count" means.

Leakage discipline in this file:
  * every rolling/lag feature is shifted so that row t only sees days < t,
    except `case_count` itself, which is legitimately known at scoring time
    (we score after the day closes).
  * the facility volume feature is an *expanding* mean of prior days, not a
    lifetime total, because a lifetime total computed over the full table
    leaks the future into the training rows.

The label is synthetic. It comes from the outbreak windows this repo's own
generator injected. Metrics computed against it measure whether the pipeline
works, not whether the model would find a real outbreak in Virginia.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vitalsignal.config import get_settings

FEATURE_TABLE = "outbreak_features"

FEATURE_COLUMNS = [
    "case_count",
    "cases_trailing_7d",
    "baseline_mean_28d",
    "baseline_sd_28d",
    "baseline_z",
    "ratio_to_baseline",
    "lag_1_cases",
    "lag_7_cases",
    "delta_1_cases",
    "prior_mean_cases",
    "day_of_week",
    "is_weekend",
    "is_enteric",
    "is_respiratory",
]
KEY_COLUMNS = ["report_date", "facility_id", "condition_code"]
LABEL_COLUMN = "is_outbreak_day"


def load_gold() -> pd.DataFrame:
    """Read fct_case_daily from whichever warehouse this environment has."""
    s = get_settings()
    if s.is_azure:
        # Databricks SQL Warehouse. Same SQL, same model, different driver.
        from databricks import sql as dbsql

        with dbsql.connect(
            server_hostname=__import__("os").environ["DATABRICKS_HOST"],
            http_path=__import__("os").environ["DATABRICKS_HTTP_PATH"],
            access_token=__import__("os").environ["DATABRICKS_TOKEN"],
        ) as conn:
            return pd.read_sql("select * from vitalsignal.gold.fct_case_daily", conn)

    import duckdb

    db = Path(s.local_root) / "gold" / "warehouse.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.sql("select * from main_marts.fct_case_daily").df()
    finally:
        con.close()


def load_labels() -> pd.DataFrame:
    """Explode injected outbreak windows into one row per affected day."""
    s = get_settings()
    truth = Path(s.local_root) / "landing" / "_truth" / "outbreaks.json"
    windows = json.loads(truth.read_text())

    rows = []
    for w in windows:
        for d in pd.date_range(w["start_date"], w["end_date"], freq="D"):
            rows.append(
                {
                    "report_date": d.date(),
                    "facility_id": w["facility_id"],
                    "condition_name": w["condition_display"],
                    LABEL_COLUMN: 1,
                }
            )
    return pd.DataFrame(rows)


def build_features(gold: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = gold.copy()
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.date
    df = df.sort_values(KEY_COLUMNS).reset_index(drop=True)

    grp = df.groupby(["facility_id", "condition_code"], sort=False)["case_count"]

    df["lag_1_cases"] = grp.shift(1)
    df["lag_7_cases"] = grp.shift(7)
    df["delta_1_cases"] = df["case_count"] - df["lag_1_cases"]
    # Expanding mean of *prior* days only: shift(1) before expanding().
    df["prior_mean_cases"] = grp.transform(
        lambda s: s.shift(1).expanding(min_periods=7).mean()
    )
    # +0.5 keeps the ratio finite when the baseline is a true zero.
    df["ratio_to_baseline"] = df["case_count"] / (df["baseline_mean_28d"].fillna(0) + 0.5)

    dt = pd.to_datetime(df["report_date"])
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_enteric"] = (df["condition_group"] == "enteric").astype(int)
    df["is_respiratory"] = (df["condition_group"] == "respiratory").astype(int)

    merged = df.merge(
        labels,
        how="left",
        left_on=["report_date", "facility_id", "condition_name"],
        right_on=["report_date", "facility_id", "condition_name"],
    )
    merged[LABEL_COLUMN] = merged[LABEL_COLUMN].fillna(0).astype(int)
    return merged


def run() -> pd.DataFrame:
    s = get_settings()
    gold = load_gold()
    labels = load_labels()
    feats = build_features(gold, labels)

    out_dir = Path(s.uri("feature", FEATURE_TABLE))
    out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_dir / "features.parquet", index=False)

    # The feature contract. train.py and the online endpoint both read this,
    # so a feature added here cannot silently fail to reach serving.
    (out_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLUMNS,
                "key_columns": KEY_COLUMNS,
                "label_column": LABEL_COLUMN,
                "source_table": "gold.fct_case_daily",
                "row_count": len(feats),
            },
            indent=2,
        )
    )

    pos = int(feats[LABEL_COLUMN].sum())
    print(
        f"features: {len(feats):,} rows x {len(FEATURE_COLUMNS)} features -> {out_dir}\n"
        f"  positives: {pos} ({pos / len(feats):.2%}) -- heavily imbalanced by design"
    )
    return feats


if __name__ == "__main__":
    run()
