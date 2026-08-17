{{ config(materialized='table') }}

with cases as (

    select * from {{ ref('stg_ecr_cases') }}

)

select
    {{ surrogate_key(['condition_code']) }}  as condition_sk,
    condition_code,
    max(condition_name)                      as condition_name,
    max(condition_system)                    as code_system,
    count(*)                                 as lifetime_case_count,
    -- Enteric conditions share an investigation playbook; the grouping lives
    -- here so that every downstream report uses the same definition.
    case
        when max(condition_name) in ('Salmonellosis', 'Shigellosis', 'Hepatitis A')
            then 'enteric'
        when max(condition_name) in ('Influenza', 'COVID-19', 'Pertussis')
            then 'respiratory'
        else 'other'
    end                                      as condition_group
from cases
group by condition_code
