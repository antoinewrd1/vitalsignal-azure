{# Deterministic surrogate key. NULLs are coalesced to a sentinel so that two
   rows differing only in a NULL do not collide into the same hash. #}
{% macro surrogate_key(columns) %}
  md5(
    {%- for c in columns %}
    coalesce(cast({{ c }} as varchar), '_null_')
    {%- if not loop.last %} || '||' || {% endif %}
    {%- endfor %}
  )
{% endmacro %}
