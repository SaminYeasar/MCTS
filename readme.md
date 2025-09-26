
* name for method:
    * Reflect-Explore-Backtrack CoT (REB-CoT)
    * Self-Reflective Exploratory Traceback (SRET)

```
astro/                         # your package (already exists)
  |-__init__.py
  |-astro.py                      # where MCTSAstro/Node live (example name)
  |-# ...

pyproject.toml                 # new: installable package

tests/
  |-conftest.py                  # shared mocks/fixtures
  |-unit/
    test_select.py
    test_expand.py
    test_backprop.py
  |-integration/
    test_build_tree.py
```

* `pyproject.toml` need revision
* installation `pip install -e .`
* run unit test `pytest -m unit`