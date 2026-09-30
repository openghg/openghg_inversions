{{ "\n" }}
{% for section, categories in sections.items() %}
{% for category, fragments in categories.items() %}
### {{ definitions[category]["name"] }}

{% for text, issues in fragments.items() %}
- {{ text }}{% if issues %} ({{ issues|join(", ") }}){% endif %}{{ "\n" }}
{% endfor %}
{% endfor %}
{% endfor %}
