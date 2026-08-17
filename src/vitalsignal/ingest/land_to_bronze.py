"""Landing -> Bronze.

Bronze rules I hold myself to here:
  1. Never modify the payload. Bronze is the replayable record of what arrived.
  2. Add lineage, not meaning: batch id, ingest timestamp, source file.
  3. Be idempotent. Re-running must not double-count. We keep a small control
     table of files already consumed and anti-join against it.

On Azure Databricks the same job would use Auto Loader
(`spark.readStream.format("cloudFiles")`) against the ADLS Gen2 landing
container, which gives you file notification-based discovery and RocksDB
checkpointing instead of the control table below. The control table is the
portable, zero-infrastructure equivalent -- see docs/WALKTHROUGH.md.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from vitalsignal.config import get_settings
from vitalsignal.spark import get_spark, read_table, write_table

# Explicit schema. Inferring JSON schema costs an extra full pass over the data
# and silently changes types when a new file has a different shape.
LANDING_SCHEMA = StructType(
    [
        StructField("message_id", StringType(), nullable=False),
        StructField("source_system", StringType(), nullable=True),
        StructField("received_at", StringType(), nullable=True),
        StructField("payload_hl7", StringType(), nullable=True),
    ]
)

BRONZE_TABLE = "ecr_messages"
INGEST_LOG = "_ingest_log"


def _already_ingested(spark, log_uri: str):
    """Files consumed by previous runs, or an empty frame on first run."""
    try:
        return read_table(spark, log_uri).select("source_file")
    except Exception:
        return spark.createDataFrame([], StructType([StructField("source_file", StringType())]))


def run() -> dict:
    s = get_settings()
    spark = get_spark("bronze_ingest")

    landing_root = s.uri("landing", "").rstrip("/")
    bronze_uri = s.uri("bronze", BRONZE_TABLE)
    log_uri = s.uri("bronze", INGEST_LOG)

    batch_id = str(uuid.uuid4())
    ingest_ts = datetime.now(timezone.utc)

    raw = (
        spark.read.schema(LANDING_SCHEMA)
        .option("basePath", landing_root)  # makes ingest_date a real column
        .json(f"{landing_root}/ingest_date=*/*.ndjson")
        .withColumn("source_file", F.col("_metadata.file_path"))
    )

    seen = _already_ingested(spark, log_uri)
    new = raw.join(seen, on="source_file", how="left_anti")

    n_new = new.count()
    if n_new == 0:
        print("bronze: nothing new to ingest")
        return {"rows": 0, "batch_id": batch_id}

    bronze = (
        new.withColumn("_batch_id", F.lit(batch_id))
        .withColumn("_ingest_ts", F.lit(ingest_ts))
        # payload hash lets silver detect true retransmissions vs. corrections
        .withColumn("_payload_sha256", F.sha2(F.col("payload_hl7"), 256))
        .select(
            "message_id",
            "source_system",
            "received_at",
            "payload_hl7",
            "_payload_sha256",
            "ingest_date",
            "source_file",
            "_batch_id",
            "_ingest_ts",
        )
    )

    write_table(bronze, bronze_uri, mode="append", partition_by=["ingest_date"])

    files = (
        new.select("source_file")
        .distinct()
        .withColumn("_batch_id", F.lit(batch_id))
        .withColumn("_ingest_ts", F.lit(ingest_ts))
    )
    write_table(files, log_uri, mode="append")

    n_files = files.count()
    print(f"bronze: +{n_new} rows from {n_files} files (batch {batch_id[:8]}) -> {bronze_uri}")
    return {"rows": n_new, "files": n_files, "batch_id": batch_id}


if __name__ == "__main__":
    run()
