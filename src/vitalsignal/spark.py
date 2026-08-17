"""SparkSession factory.

On Azure Databricks a session already exists and `getOrCreate()` returns it, so
this module is a no-op there. Locally it builds a single-machine session with
the Delta extensions registered only when we actually intend to write Delta.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from vitalsignal.config import get_settings


def get_spark(app_name: str = "vitalsignal") -> SparkSession:
    s = get_settings()
    builder = (
        SparkSession.builder.appName(app_name)
        # Small, deterministic shuffle for a laptop / single-node CI runner.
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        # Reject silently-wrong date parsing instead of returning NULL/garbage.
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
    )

    if not s.is_azure:
        builder = builder.master("local[*]").config("spark.driver.memory", "2g")

    if s.table_format == "delta":
        builder = (
            builder.config(
                "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
            ).config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def write_table(df, uri: str, mode: str = "overwrite", partition_by=None) -> None:
    """Format-agnostic writer so pipeline code never hardcodes parquet/delta."""
    s = get_settings()
    writer = df.write.mode(mode).format(s.table_format)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(uri)


def read_table(spark: SparkSession, uri: str):
    s = get_settings()
    return spark.read.format(s.table_format).load(uri)
