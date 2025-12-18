Neuroimaging tools developed by the `Neural Systems and Behaviour Lab <https://www.monash.edu/medicine/psych/alex-fornito-lab>`_.

Installation
------------
``nsbutils`` works with Python 3.9+, and can be pip-installed into your environment via ``pip install git+https://github.com/NSBLab/nsbutils.git``

This will clone ``main``, our most stable branch. To try out any newer features under development, clone from our ``dev`` branch instead via ``pip install git+https://github.com/NSBLab/nsbutils.git@dev``.

Alternatively, ``nsbutils`` can be added as a dependency to your ``pyproject.toml`` via `UV <https://docs.astral.sh/uv/>`_ by running ``uv add git+https://github.com/NSBLab/nsbutils.git``.

If you encounter any issues, try reproducing the exact environment used for development via UV:
::

  git clone https://github.com/NSBLab/nsbutils
  cd nsbutils
  uv venv --python 3.12.10
  uv sync --frozen

If issues persist, please consider opening an issue on the `GitHub repository <https://github.com/NSBLab/nsbutils/issues>`_.
Tests can be run with ``pytest``:

::

  cd nsbutils
  pip install pytest
  pytest tests

License information
-------------------
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (``cc-by-nc-sa``). See the `LICENSE <LICENCE-CC-BY-NC-SA-4.0.md>`_ file for details.
