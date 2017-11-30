.. _proj-org:

====================
Project Organization
====================

.. code::

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── environment.yml    <- The requirements file for reproducing the analysis environment, e.g.
    │                         the Anaconda environment used in this project.
    │
    ├── test_environment.py   <- Script that checks that you are running the correct python environment.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   └── data           <- Scripts to download or generate data
    │       ├── make_dataset.py
    │       │
    │       ├── One_halo_conformity <- Scripts to analyze 1-halo conformity
    │       │
    │       ├── Two_halo_conformity <- Scripts to analyze 2-halo conformity
    │       │
    │       └── utilities_python    <- Scripts to analyze 1-halo conformity
    │           └── pair_counter_rp <- Scripts used throughout both analyses.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

Project based on the `cookiecutter data science project template
<https://drivendata.github.io/cookiecutter-data-science/>`_
