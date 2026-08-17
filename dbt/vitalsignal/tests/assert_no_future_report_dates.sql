-- A report date after the ingest date means either a clock problem at the
-- sending facility or a parsing bug on our side. Both are incidents.
select case_id, report_date, ingest_date
from {{ ref('stg_ecr_cases') }}
where report_date > ingest_date
