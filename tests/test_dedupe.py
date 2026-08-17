from datetime import datetime

from vitalsignal.transform.bronze_to_silver import dedupe


def test_dedupe_keeps_first_arrival_and_counts(spark):
    rows = [
        ("hashA", "2026-01-05T10:00:00", datetime(2026, 1, 5, 10)),
        ("hashA", "2026-01-06T02:00:00", datetime(2026, 1, 6, 2)),   # retransmit
        ("hashB", "2026-01-05T11:00:00", datetime(2026, 1, 5, 11)),
    ]
    df = spark.createDataFrame(rows, ["_payload_sha256", "received_at", "_ingest_ts"])
    out = {r._payload_sha256: r for r in dedupe(df).collect()}

    assert len(out) == 2
    assert out["hashA"].received_at == "2026-01-05T10:00:00"  # first arrival won
    assert out["hashA"].retransmission_count == 1
    assert out["hashB"].retransmission_count == 0
