.. _downloading-dataset:

-----------------------
Downloading Repository
-----------------------

The first thing that needs to be done is to download the
repository from Github:
`<https://github.com/vcalderon2009/SDSS_Conformity_Analysis>`_:

.. code::

    git clone https://github.com/vcalderon2009/SDSS_Conformity_Analysis.git

This will download all of the necessary scripts to run the analysis on the
SDSS DR7 catalogues.

.. _env-dependencies:

-------------------------------------
Installing Environment & Dependencies
-------------------------------------

To use the scripts in this repository, you **must have Anaconda installed**
on the system that will be running the scripts. This will simplify the
process of installing all the dependencies.

For reference, see:
`Anaconda - Managing environments
<https://conda.io/docs/user-guide/tasks/manage-environments.html>`_

The package counts with a **Makefile** with useful functions.
You must use this Makefile to ensure that you have all the necessary
*dependencies*, as well as the correct **conda environment**.

Once Anaconda has been installed, you can use the **Makefile**
to

- Install the Anaconda environment ``conformity``.
- Update the project environment ``conformity``.
- Install the ``src`` package via `pip`.

^^^^^^^^^^^^^^^^^^
Makefile functions
^^^^^^^^^^^^^^^^^^

* Show all available functions in the *Makefile*

.. code-block:: bash

    $: 	make show-help

      1_halo_fracs_calc   1-halo Quenched Fractions - Calculations
      1_halo_mcf_calc     1-halo Marked Correlation Function - Calculations
      2_halo_fracs_calc   2-halo Quenched Fractions - Calculations
      2_halo_mcf_calc     2-halo Marked Correlation Function - Calculations
      clean               Deletes all build, test, coverage, and Python artifacts
      clean-build         Remove build artifacts
      clean-pyc           Removes Python file artifacts
      clean-test          Remove test and coverage artifacts
      cosmo_utils_install Installing cosmo-utils
      cosmo_utils_remove  Removing cosmo-utils
      cosmo_utils_upgrade Upgrading cosmo-utils
      download_dataset    Download required Dataset
      environment         Set up python interpreter environment - Using environment.yml
      lint                Lint using flake8
      plot_figures        Figures
      remove_calc_screens Remove Calc. screen session
      remove_catalogues   Remove downloaded catalogues
      remove_environment  Delete python interpreter environment
      remove_plot_screens Remove Plot screen session
      src_env             Import local source directory package
      src_remove          Remove local source directory package
      src_update          Updated local source directory package
      test_environment    Test python environment is setup correctly
      update_environment  Update python interpreter environment

* **Create** the environment from the `environment.yml` file:

.. code-block:: bash

    $: 	make environment

* **Activate** the new environment **conformity**.

.. code-block:: bash

    $: 	source activate conformity

* To **update** the `environment.yml` file (when the required packages have changed):

.. code-block:: bash

    $: 	make update_environment

* **Deactivate** the new environment:

.. code-block:: bash

    $: 	source deactivate

^^^^^^^^^^^^^^^^^^^^^^^^^
Auto-activate environment
^^^^^^^^^^^^^^^^^^^^^^^^^

To make it easier to activate the necessary environment, one can check out
`*conda-auto-env* <https://github.com/chdoig/conda-auto-env>`_ which activates
the necessary environment automatically.

.. _download-dataset:

-----------------
Download Dataset
-----------------

In order to be able to run the scripts in this repository, one needs to first
**download** the required datasets. One can do that by running the following
command from the main directory and using the *Makefile*:

.. code::

    $: make download_dataset

This command will download the required catalogues for the analysis to
the ``data/external/`` directory.

Depending on the variables used for the analysis, one can download different
sets of catalogues, depending on what kind of catalogues they want to use it
for.

.. note::
    In order to make use of this commands, one will need
    `wget <https://www.gnu.org/software/wget/>`_. If `wget` is not
    available, one can download the files from
    `<http://lss.phy.vanderbilt.edu/groups/data_vc/DR7/sdss_catalogues/>`_
    and put them in ``/data/external/SDSS``.

.. _steps-commands:

-------------------
Steps and Commands
-------------------

By running the following commands, one is able to replicate the
results found in `Calderon et al. (2018)
<https://academic.oup.com/mnras/article/480/2/2031/5059600>`_.

.. code-block:: bash

    git clone https://github.com/vcalderon2009/SDSS_Conformity_Analysis.git
    cd SDSS_Conformity_Analysis/
    make environment
    source activate conformity
    python test_environment.py
    make download_dataset
    make 1_halo_fracs_calc
    make 1_halo_mcf_calc
    make 2_halo_fracs_calc
    make 2_halo_mcf_calc
    make plot_figures
    open /reports/figures/SDSS/Paper_Figures/

This is the sequence of commands used to create the results shown in
`Calderon et al. (2018)
<https://academic.oup.com/mnras/article/480/2/2031/5059600>`_.
The scripts already have default values. If one wishes to perform the
analysis using a different set of parameters, these can be changed in the
**Makefile**, or by simply calling the functions in the *Makefile* as:

.. code-block:: bash

  make SAMPLE="20" download_dataset

This command will download the datasets for the ``Mr20`` galaxy and
group galaxy catalogues.
