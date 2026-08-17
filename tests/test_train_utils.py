import numpy as np
import pandas as pd

from vitalsignal.ml.train import temporal_split, threshold_for_alert_budget


def test_temporal_split_is_a_clean_cut():
    df = pd.DataFrame({"report_date": pd.date_range("2026-01-01", periods=100)})
    train, test = temporal_split(df, test_days=30)
    assert train["report_date"].max() < test["report_date"].min()
    assert len(train) + len(test) == 100


def test_threshold_matches_alert_budget():
    rng = np.random.default_rng(0)
    scores = rng.random(700)  # ~100 days of 7 series
    thr = threshold_for_alert_budget(scores, n_days=70, n_series=7, alerts_per_week=5)
    assert (scores >= thr).sum() == 50  # 10 weeks x 5 alerts
