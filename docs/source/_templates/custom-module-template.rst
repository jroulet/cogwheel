{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module attributes

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: custom-class-template.rst
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{#
   Do not descend into the test suite. `api.rst` recurses over the bare
   `cogwheel` package, which otherwise generates an API page for every class
   in `cogwheel.tests` -- publishing the test suite as public API and, because
   test docstrings are prose (headings like "What Is And Is Not Covered"),
   making numpydoc parse them as malformed sections. That was the source of
   the great majority of the build's warnings.

   Expressed as a "do not descend" filter rather than a hand-maintained list
   of public modules, so a new top-level module or subpackage is still picked
   up automatically and nothing has to be remembered when one is added.
#}
{#- `modules` holds SHORT names ('tests', 'lensing', ...), not dotted paths;
    both forms are listed so this keeps working if that ever changes. -#}
{%- set skip_subpackages = ['tests', fullname ~ '.tests'] %}
{% block modules %}
{% set documented = modules | reject('in', skip_subpackages) | list %}
{% if documented %}
.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for item in documented %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
