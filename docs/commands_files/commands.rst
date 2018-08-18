.. _commands:

========
Commands
========

This project analyzes **1-halo** and **2-halo** conformity on SDSS DR7 data.

After having downloaded the dataset by running the command
.. code::

    make download_dataset

you can start analyzing the dataset. This command will download the
required catalogues for the analysis to ``data/external/``.

.. _one-halo:

---------
1-halo
---------

There are **2** types of analysis for the 1-halo conformity.These are

* 1-halo Quenched Fractions calculations
* 1-halo Marked correlation function (MCF)

One can run these two analyses by running the following commands:

.. code::

  make 1_halo_fracs_calc
  make 1_halo_mcf_calc

.. _two-halo:

---------
2-halo
---------

There are **2** types of analysis for the 1-halo conformity.These are

* 2-halo Central Quenched Fractions calculations
* 2-halo Marked Correlation Function (MCF)

One can run these two analyses by running the following commands:

.. code::

    make 2_halo_fracs_calc
    make 2_halo_mcf_calc

.. note::
    These functions make use of a fraction of your CPU, so it is better
    to run them **one by one**. One can modify the allowed fraction of
    the CPU in the Makefile by setting the ``CPU_FRAC`` variable to be
    from 0 to 1.


.. _plotting:

-------------
Making Plots
-------------

Once **all of the analyses** for 1-halo and 2-halo are *done*, i.e.
after having run the **4 commands** above, one can plot all of the
results by running the following command:

.. code::

    make plot_figures

This will produce the plots for **data** and **mocks** for all of the
4 different analyses for 1- and 2-halo conformity.
The figures will be saved in:
``/reports/figures/SDSS/Paper_Figures/``

.. note::
    The scripts have default values that were used in
    `Calderon et al. (2018)
    <https://academic.oup.com/mnras/article/480/2/2031/5059600>`_.
    If one wishes to perform the analyses
    using a different set of parameters, these can be changed in **Makefiles**,
    or be given as input variables to the Makefile. Take into account that
    not all combinations of parameters are allowed for the analysis.
