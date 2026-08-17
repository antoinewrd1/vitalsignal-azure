"""features.py promises rolling features never see the current day. If someone
'optimizes' the shift away, PR-AUC jumps and these tests catch why."""

import pandas as pd

from vitalsignal.ml.features import build_features


def _gold(counts):
    days = pd.date_range("2026-01-01", periods=len(counts), freq="D")
    return pd.DataFrame(
        {
            "report_date": days,
            "facility_id": "FAC001",
            "condition_code": "X",
            "condition_name": "TestCond",
            "condition_group": "enteric",
            "county": "Fairfax",
            "case_count": counts,
            "distinct_patients": counts,
            "mean_age_years": 30.0,
            "retransmissions": 0,
            "cases_trailing_7d": pd.Series(counts).rolling(7, min_periods=1).sum(),
            "baseline_mean_28d": None,
            "baseline_sd_28d": None,
            "baseline_z": None,
            "baseline_days": 0,
            "epi_week": 1,
            "is_weekend": False,
        }
    )


def test_lag_and_prior_mean_exclude_current_day():
    counts = [1, 2, 3, 4, 5, 6, 7, 8, 100]  # spike on the last day
    feats = build_features(_gold(counts), labels=pd.DataFrame(
        columns=["report_date", "facility_id", "condition_name", "is_outbreak_day"]
    ))
    last = feats.iloc[-1]
    assert last["lag_1_cases"] == 8
    # prior mean over days 1..8 only; the 100 must not contaminate it
    assert last["prior_mean_cases"] == pd.Series(counts[:-1]).mean()


def test_first_row_lags_are_nan_not_zero():
    feats = build_features(_gold([5, 5, 5]), labels=pd.DataFrame(
        columns=["report_date", "facility_id", "condition_name", "is_outbreak_day"]
    ))
    assert pd.isna(feats.iloc[0]["lag_1_cases"])  # unknown, not "no cases"


def test_label_join_defaults_to_zero():
    labels = pd.DataFrame(
        [{"report_date": pd.Timestamp("2026-01-02").date(),
          "facility_id": "FAC001", "condition_name": "TestCond", "is_outbreak_day": 1}]
    )
    feats = build_features(_gold([1, 9, 1]), labels)
    assert feats["is_outbreak_day"].tolist() == [0, 1, 0]
