-- The zero-fill in fct_case_daily is a cross join; a mistake there inflates or
-- drops cases silently. This test pins the total: gold must equal silver.
with gold as (
    select sum(case_count) as n from {{ ref('fct_case_daily') }}
),
silver as (
    select count(*) as n
    from {{ ref('stg_ecr_cases') }}
    where report_date between date '{{ var("spine_start") }}'
                          and date '{{ var("spine_end") }}'
)
select gold.n as gold_n, silver.n as silver_n
from gold, silver
where gold.n <> silver.n
