SDSS Conformity
==============================

Repository for Analysis on Galactic Conformity in SDSS catalogues

**Author**: Victor Calderon ([victor.calderon@vanderbilt.edu](mailto:victor.calderon@vanderbilt.edu))

**Date**  : 2017-11-30

[![Documentation Status](https://readthedocs.org/projects/sdss-conformity-analysis/badge/?version=latest)](http://sdss-conformity-analysis.readthedocs.io/en/latest/?badge=latest)

---

### Analysis for the signature of galactic conformity in the Sloan Digital Sky Survey (SDSS) Data Release 7 (DR7).

The documentation details the codes used in Calderon et al. (2017) for the analysis of 1- and 2-halo galactic conformity using a set of different statistics on SDSS DR7 and synthetic catalogues.

### Documentation

The documentation of the project can be found at 
[https://sdss-conformity-analysis.readthedocs.io](https://sdss-conformity-analysis.readthedocs.io)

### Questions?
For any questions, please send an email to [victor.calderon@vanderbilt.edu](mailto:victor.calderon@vanderbilt.edu), or submit an _Issue_ on the repository.

Victor C.

---

### Note
The scripts have default values that were used in Calderon et al. (2017). If one wishes to perform the analyses using a different set of parameters, these can be changed in the files the end with *\*make.py* in the `src/data/One_halo_conformitiy` and `src/data/Two_halo_conformitiy`.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   └── data           <- Scripts to download or generate data
    │       ├── make_dataset.py
    │       │
    │       ├── One_halo_conformity <- Scripts to analyze 1-halo conformity
    │       │
    │       ├── Two_halo_conformity <- Scripts to analyze 2-halo conformity
    │       │
    │       └── utilities_python    <- Scripts to analyze 1-halo conformity
    │           └── pair_counter_rp <- Scripts used throughout both analyses.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>