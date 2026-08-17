{#
  A tiny cross-adapter date spine.

  dbt_utils.date_spine would normally do this, but pulling a package adds a
  network dependency to CI for ~6 lines of SQL. `adapter.dispatch` is the
  idiomatic way to keep one model file working on both DuckDB and Databricks.
#}

{% macro date_spine(start_date, end_date) %}
  {{ return(adapter.dispatch('date_spine', 'vitalsignal')(start_date, end_date)) }}
{% endmacro %}

{% macro default__date_spine(start_date, end_date) %}
  select cast(unnest(generate_series(
           date '{{ start_date }}', date '{{ end_date }}', interval 1 day
         )) as date) as date_day
{% endmacro %}

{% macro databricks__date_spine(start_date, end_date) %}
  select explode(sequence(
           to_date('{{ start_date }}'), to_date('{{ end_date }}'), interval 1 day
         )) as date_day
{% endmacro %}
