{{ config(materialized='table') }}

/*
  Grain: one row per (report_date, facility_id, condition_code).

  Two things make this model the backbone of the project:

  1. ZERO-FILL. We cross join a date spine against the facility and condition
     dimensions before joining counts. Without this, a facility that reports
     nothing on Tuesday simply has no Tuesday row, and every rolling average
     downstream is computed over the wrong denominator.

  2. LEAKAGE-SAFE BASELINE. The 28-day baseline window is
     `rows between 28 preceding and 1 preceding` -- it deliberately excludes
     the current day. If the current day were inside its own baseline, the
     z-score would be damped by the very spike it is supposed to detect, and
     the ML model downstream would be training on a feature that has already
     seen its own label.
*/

with spine as (

    {{ date_spine(var('spine_start'), var('spine_end')) }}

),

scaffold as (

    select
        spine.date_day        as report_date,
        f.facility_id,
        f.county,
        c.condition_code,
        c.condition_name,
        c.condition_group
    from spine
    cross join {{ ref('dim_facility') }} f
    cross join {{ ref('dim_condition') }} c

),

counts as (

    select
        report_date,
        facility_id,
        condition_code,
        count(*)                     as case_count,
        count(distinct patient_key)  as distinct_patients,
        avg(age_years)               as mean_age_years,
        sum(retransmission_count)    as retransmissions
    from {{ ref('stg_ecr_cases') }}
    group by 1, 2, 3

),

zero_filled as (

    select
        s.report_date,
        s.facility_id,
        s.county,
        s.condition_code,
        s.condition_name,
        s.condition_group,
        coalesce(c.case_count, 0)        as case_count,
        coalesce(c.distinct_patients, 0) as distinct_patients,
        c.mean_age_years,
        coalesce(c.retransmissions, 0)   as retransmissions
    from scaffold s
    left join counts c
        on  s.report_date    = c.report_date
        and s.facility_id    = c.facility_id
        and s.condition_code = c.condition_code

),

windowed as (

    select
        *,
        sum(case_count) over w7                as cases_trailing_7d,
        avg(case_count) over w28_lagged        as baseline_mean_28d,
        stddev_samp(case_count) over w28_lagged as baseline_sd_28d,
        count(*) over w28_lagged               as baseline_days
    from zero_filled
    window
        w7 as (
            partition by facility_id, condition_code
            order by report_date
            rows between 6 preceding and current row
        ),
        w28_lagged as (
            partition by facility_id, condition_code
            order by report_date
            rows between 28 preceding and 1 preceding
        )

)

select
    {{ surrogate_key(['report_date', 'facility_id', 'condition_code']) }} as case_daily_sk,
    report_date,
    facility_id,
    county,
    condition_code,
    condition_name,
    condition_group,
    case_count,
    distinct_patients,
    mean_age_years,
    retransmissions,
    cases_trailing_7d,
    baseline_mean_28d,
    baseline_sd_28d,
    baseline_days,
    -- Classic aberration-detection z-score. NULL (not 0) when we do not yet
    -- have enough history or the series is flat -- "unknown" must not be
    -- silently encoded as "normal".
    case
        when baseline_days >= {{ var('baseline_min_days') }}
             and baseline_sd_28d > 0
        then (case_count - baseline_mean_28d) / baseline_sd_28d
    end                                                      as baseline_z,
    extract(dow from report_date) in (0, 6)                  as is_weekend,
    extract(week from report_date)                           as epi_week
from windowed
