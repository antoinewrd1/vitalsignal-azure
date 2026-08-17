{{ config(materialized='view') }}

-- Staging does exactly three things and nothing else:
--   rename to the warehouse's vocabulary, cast, and apply the one filter that
--   every downstream consumer would otherwise repeat (positives only).
-- No joins, no aggregation -- those belong in marts.

with source as (

    select * from {{ source('silver', 'ecr_cases') }}

),

renamed as (

    select
        message_control_id                          as case_id,
        facility_id,
        condition_code,
        condition_display                           as condition_name,
        condition_system,
        cast(report_date as date)                   as report_date,
        cast(specimen_collection_date as date)      as specimen_collection_date,
        patient_key,
        age_years,
        age_band,
        sex,
        county,
        state,
        postal_code,
        clinical_note,
        retransmission_count,
        cast(ingest_date as date)                   as ingest_date

    from source
    where is_positive          -- surveillance counts confirmed positives only

)

select * from renamed
