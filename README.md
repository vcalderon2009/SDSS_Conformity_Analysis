[![Documentation Status](https://readthedocs.org/projects/galactic-conformity-in-sdss-dr7/badge/?version=latest)](https://galactic-conformity-in-sdss-dr7.readthedocs.io/en/latest/?badge=latest)
[![Documentation Status](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/1712.02797)
[![Documentation Status](https://img.shields.io/badge/Paper-MNRAS-orange.svg)](https://academic.oup.com/mnras/article/480/2/2031/5059600)

Small- and Large-Scale Galactic Conformity in SDSS DR7
========================================================

Repository for the analysis on **galactic conformity** in SDSS DR7

**Author**: Victor Calderon ([victor.calderon@vanderbilt.edu](mailto:victor.calderon@vanderbilt.edu))

**Date**  : 2018-08-17

---

### Analysis for the signature of galactic conformity in the Sloan Digital Sky Survey (SDSS) Data Release 7 (DR7).

The documentation details the codes used in [Calderon et al. (2018)](https://academic.oup.com/mnras/article/480/2/2031/5059600) for the analysis of **1- and 2-halo galactic conformity** using a set of different statistics on SDSS DR7 and synthetic catalogues.

### Documentation

The documentation of the project can be found at
[https://sdss-conformity-analysis.readthedocs.io](https://sdss-conformity-analysis.readthedocs.io)

### Questions?
For any questions, please send an email to [victor.calderon@vanderbilt.edu](mailto:victor.calderon@vanderbilt.edu), or submit an _Issue_ on the repository.

Victor C.

---

### Note
The scripts have default values that were used in [Calderon et al. (2018)](https://academic.oup.com/mnras/article/480/2/2031/5059600). If one wishes to perform the analyses using a different set of parameters, these can be changed in the **Makefile**.


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
    │   ├── __init__.py    <- Makes src a Python module. Install via `pip install -e .`
    │   │
    │   └── data           <- Scripts to download or generate data
    │       ├── One_halo_conformity <- Scripts to analyze 1-halo conformity
    │       │
    │       ├── Two_halo_conformity <- Scripts to analyze 2-halo conformity
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
