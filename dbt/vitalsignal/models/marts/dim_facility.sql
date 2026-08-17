{{ config(materialized='table') }}

-- Conformed facility dimension. Built from the fact stream because the eCR
-- feed is the only system of record we have; in production this would be
-- joined against the state facility registry and this model would become the
-- reconciliation point between the two.

with cases as (

    select * from {{ ref('stg_ecr_cases') }}

),

agg as (

    select
        facility_id,
        max(county)                        as county,
        max(state)                         as state,
        min(report_date)                   as first_reported_on,
        max(report_date)                   as last_reported_on,
        count(*)                           as lifetime_case_count,
        count(distinct condition_code)     as distinct_conditions_reported,
        count(distinct report_date)        as reporting_days
    from cases
    group by 1

)

select
    {{ surrogate_key(['facility_id']) }}   as facility_sk,
    agg.*,
    -- A facility that has not reported in 14 days is a reporting failure, not
    -- an absence of disease. Surveillance teams act on this column.
    {{ days_between('last_reported_on', "date '" ~ var('spine_end') ~ "'") }}
        as days_since_last_report,
    case
        when {{ days_between('last_reported_on', "date '" ~ var('spine_end') ~ "'") }} > 14
            then 'stale'
        else 'active'
    end                                     as reporting_status
from agg
