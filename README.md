SDSS Conformity
==============================

Repository for Analysis on Galactic Conformity in SDSS catalogues

**Author**: Victor Calderon ([victor.calderon@vanderbilt.edu](mailto:victor.calderon@vanderbilt.edu))

**Date**  : 2017-11-11

## Installing Environment & Dependencies

To use the scripts in this repository, you must have _Anaconda_ installed on the systems that will be running the scripts. This will simplify the process of installing all the dependencies.

For reference, see: [https://conda.io/docs/user-guide/tasks/manage-environments.html](https://conda.io/docs/user-guide/tasks/manage-environments.html)

The package counts with a __Makefile__ with useful functions. You must use this Makefile to ensure that you have all the necessary _dependencies_, as well as the correct _conda environment_. 

* Show all available functions in the _Makefile_

```
$:  make show-help
    
Available rules:

    1_halo_fracs_calc   1-halo Quenched Fractions - Calculations
    1_halo_mcf_calc     1-halo Marked Correlation Function - Calculations
    2_halo_fracs_calc   2-halo Quenched Fractions - Calculations
    2_halo_mcf_calc     2-halo Marked Correlation Function - Calculations
    clean               Delete all compiled Python files
    create_environment  Set up python interpreter environment
    download_dataset    Download required Dataset
    environment         Set up python interpreter environment - Using environment.yml
    lint                Lint using flake8
    plot_figures        Figures
    remove_calc_screens Remove Calc. screen session
    remove_environment  Delete python interpreter environment
    remove_plot_screens Remove Plot screen session
    test_environment    Test python environment is setup correctly
    update_environment  Update python interpreter environment
```

* __Create__ the environment from the `environment.yml` file:

```
    make environment
```

* __Activate__ the new environment __conformity__.

```
    source activate conformity
```

* To __update__ the `environment.yml` file (when the required packages have changed):

```
  make update_environment
```

* __Deactivate__ the new environment:

```
    source deactivate
```

### Auto-activate environment
To make it easier to activate the necessary environment, one can check out [*conda-auto-env*](https://github.com/chdoig/conda-auto-env), which activates the necessary environment automatically.

Usage
------
This project analyzes **1-halo** and **2-halo** conformity on SDSS DR7 data. This repository lets you analyze the SDSS samples used in Calderon et al. (2017).

### Download Dataset
In order to be able to run the scripts in this repository, one needs to first **download** the required datasets. One can do that by running the following command from the main directory and using the _Makefile_:

```
    make download_dataset
```
This command will download the required catalogues for the analysis to `data/external/`.

### 1-halo
There are **2** types of analysis for the 1-halo conformity. These are 1) 1-halo Quenched Fractions calculations, and 2) 1-halo Marked Correlation Function (MCF). One can run these two analyses by running the following commands from the Makefile:

```
    make 1_halo_fracs_calc
```

```
    make 1_halo_mcf_calc
```

These functions make use of most of a fraction of your CPU, so it is better to run them **one by one**. One can modify this fraction in the Makefile.

### 2-halo
There are **2** types of analysis for the 2-halo conformity. These are 1) 2-halo Central Quenched Fractions calculations, and 2) 2-halo Marked Correlation Function (MCF). One can run these two analyses by running the following commands from the Makefile:

```
    make 2_halo_fracs_calc
```

```
    make 2_halo_mcf_calc
```

These functions make use of most of a fraction of your CPU, so it is better to run them **one by one**. One can modify this fraction in the Makefile.


### Plots 
Once all of the analyses for 1-halo and 2-halo are done, i.e. after having run the **4 commands** above, one can plot all of the results 
by running the following command

```
    make plot_figures
```

This will produce the plots for `data` and `mocks` for all of the 4 different analyses for 1- and 2-halo conformity.
The figures will be saved at: `/reports/figures/SDSS/Paper_Figures/`.

### Note
The scripts have default values that were used in Calderon et al. (2017). If one wishes to perform the analyses using a different set of parameters, these can be changed in the files the end with *\*make.py* in the `src/data/One_halo_conformtiy` and `src/data/Two_halo_conformtiy`.


Project Organization
------------

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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
