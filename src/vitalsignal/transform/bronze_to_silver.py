"""Bronze -> Silver.

This is where the message stops being a string and becomes a case.

Three jobs, in this order:
  1. PARSE     HL7 v2 segments -> typed columns, using Spark SQL expressions
               rather than Python UDFs (a UDF here would serialise every row to
               the Python worker and cost ~10x on a real cluster).
  2. QUARANTINE  Rows failing a data-quality contract are written to a separate
               table with the reason attached. They are not dropped and they do
               not poison the good rows.
  3. DEDUPE    A real eCR feed retransmits. We collapse on the SHA-256 of the
               full payload -- not on the message control ID alone, because a
               *correction* reuses neither, and a retransmission reuses both.

HL7 field numbering note: for MSH, MSH-n lands at index n of the pipe-split
array (MSH-1 *is* the separator); for every other segment MSH-style numbering
shifts by one, so PID-n lands at index n+1. That off-by-one is the single most
common source of silent HL7 parsing bugs, so it is asserted in tests.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from vitalsignal.config import get_settings
from vitalsignal.spark import get_spark, read_table, write_table

SILVER_TABLE = "ecr_cases"
QUARANTINE_TABLE = "ecr_rejects"


def seg(segments_col: str, prefix: str):
    """First segment whose type matches `prefix` (e.g. 'OBX'), else NULL."""
    return F.try_element_at(
        F.filter(F.col(segments_col), lambda s: s.startswith(F.lit(prefix + "|"))),
        F.lit(1),
    )


def field(segment, idx: int):
    """HL7 field at 1-based pipe index `idx` of a segment string."""
    return F.try_element_at(F.split(segment, r"\|"), F.lit(idx))


def component(fld, idx: int):
    """HL7 component at 1-based caret index `idx` of a field."""
    return F.try_element_at(F.split(fld, r"\^"), F.lit(idx))


def safe_ts(fld, fmt: str = "yyyyMMddHHmmss"):
    """Parse or NULL. `to_timestamp` under CORRECTED policy *throws* on bad
    input, which would kill the whole job because of one malformed message.
    `try_to_timestamp` returns NULL, and the DQ layer below turns that NULL
    into an explicit quarantine reason instead of a silent loss."""
    return F.try_to_timestamp(fld, F.lit(fmt))


def safe_date(fld, fmt: str = "yyyyMMdd"):
    return safe_ts(fld, fmt).cast("date")


def parse(bronze: DataFrame) -> DataFrame:
    df = bronze.withColumn("segments", F.split(F.col("payload_hl7"), "\r"))

    msh, pid, obr, obx, nte = (seg("segments", p) for p in ("MSH", "PID", "OBR", "OBX", "NTE"))

    return (
        df.withColumn("sending_application", field(msh, 3))
        .withColumn("facility_id", field(msh, 4))
        .withColumn("message_ts", safe_ts(field(msh, 7)))
        .withColumn("message_type", field(msh, 9))
        .withColumn("message_control_id", field(msh, 10))
        .withColumn("hl7_version", field(msh, 12))
        # ---- PID (index = HL7 position + 1) ------------------------------
        .withColumn("patient_key", component(field(pid, 4), 1))
        .withColumn("birth_date", safe_date(field(pid, 8)))
        .withColumn("sex", field(pid, 9))
        .withColumn("county", component(field(pid, 12), 3))
        .withColumn("state", component(field(pid, 12), 4))
        .withColumn("postal_code", component(field(pid, 12), 5))
        # ---- OBR / OBX ---------------------------------------------------
        .withColumn("specimen_collection_date", safe_date(field(obr, 8)))
        .withColumn("condition_code", component(field(obx, 4), 1))
        .withColumn("condition_display", component(field(obx, 4), 2))
        .withColumn("condition_system", component(field(obx, 4), 3))
        .withColumn("result_code", component(field(obx, 6), 1))
        .withColumn("result_display", component(field(obx, 6), 2))
        .withColumn("abnormal_flag", field(obx, 9))
        .withColumn("report_date", safe_date(field(obx, 15)))
        # ---- NTE ---------------------------------------------------------
        .withColumn("clinical_note", field(nte, 4))
        .drop("segments")
    )


def apply_dq(df: DataFrame) -> DataFrame:
    """Attach an array of failed-rule names. Empty array == clean row."""
    rules = {
        "missing_condition_code": F.col("condition_code").isNull(),
        "missing_report_date": F.col("report_date").isNull(),
        "unparseable_message_ts": F.col("message_ts").isNull(),
        "missing_patient_key": F.col("patient_key").isNull(),
        "report_date_in_future": F.col("report_date") > F.col("ingest_date").cast("date"),
        "negative_age": F.col("birth_date") > F.col("report_date"),
    }
    checks = [F.when(cond, F.lit(name)) for name, cond in rules.items()]
    return df.withColumn("dq_failures", F.array_compact(F.array(*checks)))


def dedupe(df: DataFrame) -> DataFrame:
    """Collapse retransmissions, keeping the first arrival and a count."""
    w = Window.partitionBy("_payload_sha256").orderBy(
        F.col("received_at").asc(), F.col("_ingest_ts").asc()
    )
    return (
        df.withColumn("_rn", F.row_number().over(w))
        .withColumn(
            "retransmission_count",
            F.count("*").over(Window.partitionBy("_payload_sha256")) - 1,
        )
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def run() -> dict:
    s = get_settings()
    spark = get_spark("silver_transform")

    bronze = read_table(spark, s.uri("bronze", "ecr_messages"))
    parsed = apply_dq(parse(bronze))

    clean = parsed.filter(F.size("dq_failures") == 0)
    rejects = parsed.filter(F.size("dq_failures") > 0)

    silver = (
        dedupe(clean)
        .withColumn(
            "age_years",
            F.floor(F.months_between(F.col("report_date"), F.col("birth_date")) / 12),
        )
        .withColumn(
            "age_band",
            F.when(F.col("age_years") < 5, "0-4")
            .when(F.col("age_years") < 18, "5-17")
            .when(F.col("age_years") < 50, "18-49")
            .when(F.col("age_years") < 65, "50-64")
            .otherwise("65+"),
        )
        .withColumn("is_positive", F.col("result_code") == F.lit("POS"))
        .withColumn("report_month", F.date_format("report_date", "yyyy-MM"))
        .select(
            "message_control_id", "message_id", "_payload_sha256", "facility_id",
            "sending_application", "message_ts", "message_type", "hl7_version",
            "patient_key", "birth_date", "age_years", "age_band", "sex",
            "county", "state", "postal_code", "specimen_collection_date",
            "condition_code", "condition_display", "condition_system",
            "result_code", "result_display", "abnormal_flag", "is_positive",
            "clinical_note", "report_date", "report_month",
            "retransmission_count", "ingest_date", "_ingest_ts", "_batch_id",
        )
    )

    write_table(silver, s.uri("silver", SILVER_TABLE), partition_by=["report_month"])
    write_table(
        rejects.select("message_id", "payload_hl7", "dq_failures", "ingest_date", "_batch_id"),
        s.uri("quarantine", QUARANTINE_TABLE),
    )

    n_bronze, n_silver = bronze.count(), silver.count()
    n_rejects = rejects.count()
    stats = {
        "bronze_rows": n_bronze,
        "silver_rows": n_silver,
        "quarantined": n_rejects,
        "deduped": n_bronze - n_rejects - n_silver,
    }
    print(
        f"silver: {n_bronze} bronze -> {n_silver} silver "
        f"({n_rejects} quarantined, {stats['deduped']} retransmissions collapsed)"
    )
    (
        rejects.select(F.explode("dq_failures").alias("dq_rule"))
        .groupBy("dq_rule")
        .count()
        .orderBy(F.desc("count"))
        .show(truncate=False)
    )
    return stats


if __name__ == "__main__":
    run()
