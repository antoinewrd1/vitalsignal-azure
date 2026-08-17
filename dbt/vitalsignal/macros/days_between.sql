{# DuckDB and Databricks spell date arithmetic differently. One macro, one
   place to fix it, instead of an adapter check inside every model. #}

{% macro days_between(start_date, end_date) %}
  {{ return(adapter.dispatch('days_between', 'vitalsignal')(start_date, end_date)) }}
{% endmacro %}

{% macro default__days_between(start_date, end_date) %}
  date_diff('day', {{ start_date }}, {{ end_date }})
{% endmacro %}

{% macro databricks__days_between(start_date, end_date) %}
  datediff({{ end_date }}, {{ start_date }})
{% endmacro %}
