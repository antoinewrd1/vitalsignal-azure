"""The silver docstring claims MSH is offset differently from PID and that a
message failing DQ is quarantined, not dropped. These tests hold it to that."""


from vitalsignal.transform.bronze_to_silver import apply_dq, parse

GOOD = (
    "MSH|^~\\&|EHR_APP|FAC001|VDH_ECR|VA|20260105193000||ORU^R01|MSGCTRL1|P|2.5.1\r"
    "PID|1||PATKEY123^^^FAC001^MR||DOE^PATIENT||19900215|F|||100 MAIN ST^^Fairfax^VA^22030\r"
    "OBR|1||ORD000001||CULTURE^Laboratory culture^L|||20260101\r"
    "OBX|1|CWE|302231008^Salmonellosis^SCT||POS^Positive^HL70078|||A|||F|||20260105\r"
    "NTE|1||Symptom onset reported as 2026-01-02."
)


def _frame(spark, payload):
    return (
        spark.createDataFrame([(payload, "2026-01-05")], ["payload_hl7", "ingest_date"])
    )


def test_msh_and_pid_field_offsets(spark):
    row = parse(_frame(spark, GOOD)).collect()[0]
    # MSH-4 is sending facility; MSH-10 is control id (index == HL7 position).
    assert row.facility_id == "FAC001"
    assert row.message_control_id == "MSGCTRL1"
    # PID-3.1 patient id and PID-7 birth date land at index n+1 (the off-by-one).
    assert row.patient_key == "PATKEY123"
    assert str(row.birth_date) == "1990-02-15"
    assert row.sex == "F"
    assert row.county == "Fairfax"


def test_obx_condition_and_report_date(spark):
    row = parse(_frame(spark, GOOD)).collect()[0]
    assert row.condition_code == "302231008"
    assert row.condition_display == "Salmonellosis"
    assert str(row.report_date) == "2026-01-05"
    assert row.result_code == "POS"


def test_missing_obx_is_quarantined_not_dropped(spark):
    payload = "\r".join(s for s in GOOD.split("\r") if not s.startswith("OBX"))
    df = apply_dq(parse(_frame(spark, payload)))
    row = df.collect()[0]
    assert "missing_condition_code" in row.dq_failures
    assert "missing_report_date" in row.dq_failures


def test_garbage_timestamp_yields_null_not_crash(spark):
    payload = GOOD.replace("20260105193000", "00000000000000")
    df = apply_dq(parse(_frame(spark, payload)))
    row = df.collect()[0]
    assert row.message_ts is None
    assert "unparseable_message_ts" in row.dq_failures


def test_truncated_message_survives_parsing(spark):
    payload = GOOD.split("\r")[0]  # MSH only
    df = apply_dq(parse(_frame(spark, payload)))
    row = df.collect()[0]
    assert row.facility_id == "FAC001"
    assert len(row.dq_failures) >= 2  # no OBX, no PID -> multiple named reasons
