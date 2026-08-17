{% test dbt_utils_accepted_range(model, column_name, min_value=none, max_value=none) %}

select *
from {{ model }}
where 1 = 0
{% if min_value is not none %} or {{ column_name }} < {{ min_value }} {% endif %}
{% if max_value is not none %} or {{ column_name }} > {{ max_value }} {% endif %}

{% endtest %}
