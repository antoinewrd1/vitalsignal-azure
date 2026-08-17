{{ config(materialized='table') }}

-- Statewide rollup: the grain a Power BI / Fabric report actually binds to.
-- Kept as its own model so the BI layer never aggregates a fact table live.

select
    report_date,
    condition_code,
    condition_name,
    condition_group,
    sum(case_count)                                       as case_count,
    sum(distinct_patients)                                as distinct_patients,
    count(distinct facility_id)                           as reporting_facilities,
    count(distinct case when case_count > 0 then facility_id end)
                                                          as facilities_with_cases,
    sum(cases_trailing_7d)                                as cases_trailing_7d,
    max(baseline_z)                                       as max_facility_z
from {{ ref('fct_case_daily') }}
group by 1, 2, 3, 4
