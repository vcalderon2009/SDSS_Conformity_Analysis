#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 01/15/2018
# Last Modified: 01/15/2018
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Proposed analysis by the MNRAS referee.

To Do:
- Projected Corr. Functions for galaxy properties of SDSS
- Distributions of galaxy properties

"""
# Path to Custom Utilities folder
import os
import sys
import git
from path_variables import *

# Importing Modules
import src.data.utilities_python as cu
import numpy as num
import math
import os
import sys
import pandas as pd
import pickle
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rc('text', usetex=True)
import seaborn as sns
# sns.set()
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from datetime import datetime
import Corrfunc
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp

## Functions

#### --------- Argument Parsing --------- ####

class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        """
        Modifier for `argparse` help parameters, that sorts them alphabetically
        """
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _check_pos_val(val, val_min=0):
    """
    Checks if value is larger than `val_min`

    Parameters
    ----------
    val: int or float
        value to be evaluated by `val_min`

    val_min: float or int, optional (default = 0)
        minimum value that `val` can be

    Returns
    -------
    ival: float
        value if `val` is larger than `val_min`

    Raises
    -------
    ArgumentTypeError: Raised if `val` is NOT larger than `val_min`
    """
    ival = float(val)
    if ival <= val_min:
        msg  = '`{0}` is an invalid input!'.format(ival)
        msg += '`val` must be larger than `{0}`!!'.format(val_min)
        raise argparse.ArgumentTypeError(msg)

    return ival

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None

def get_parser():
    """
    Get parser object for `eco_mocks_create.py` script.

    Returns
    -------
    args: 
        input arguments to the script
    """
    ## Define parser object
    description_msg = 'Description of Script'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    ## SDSS Sample
    parser.add_argument('-sample',
                        dest='sample',
                        help='SDSS Luminosity sample to analyze',
                        type=int,
                        choices=[19,20,21],
                        default=19)
    ## SDSS Type
    parser.add_argument('-abopt',
                        dest='catl_type',
                        help='Type of Abund. Matching used in catalogue',
                        type=str,
                        choices=['mr'],
                        default='mr')
    ## Rpmin and Rpmax
    parser.add_argument('-rpmin',
                        dest='rpmin',
                        help='Minimum value for projected distance `rp`',
                        type=_check_pos_val,
                        default=0.01)
    parser.add_argument('-rpmax',
                        dest='rpmax',
                        help='Maximum value for projected distance `rp`',
                        type=_check_pos_val,
                        default=10.)
    ## Number of `rp` bins
    parser.add_argument('-nrp',
                        dest='nrpbins',
                        help='Number of bins for projected distance `rp`',
                        type=int,
                        default=10)
    ## Pimax
    parser.add_argument('-pimax',
                        dest='pimax',
                        help='Value for `pimax` for the proj. corr. function',
                        type=_check_pos_val,
                        default=20.)
    ## Random Seed
    parser.add_argument('-seed',
                        dest='seed',
                        help='Random seed to be used for the analysis',
                        type=int,
                        metavar='[0-4294967295]',
                        default=1)
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help='Delete pickle file containing pair counts',
                        type=_str2bool,
                        default=False)
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cu.Program_Msg(__file__))
    ## Parsing Objects
    args = parser.parse_args()

    return args

#### --------- Adding parameters --------- ####

def add_to_dict(param_dict):
    """
    Aggregates extra variables to dictionary

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with input parameters and values

    Returns
    ----------
    param_dict: python dictionary
        dictionary with old and new values added
    """
    ### Sample - Int
    sample_s = str(param_dict['sample'])
    ### Projected distance `rp` bins
    logrpmin    = num.log10(param_dict['rpmin'])
    logrpmax    = num.log10(param_dict['rpmax'])
    dlogrp      = (logrpmax - logrpmin)/float(param_dict['nrpbins'])
    rpbin_arr   = num.linspace(logrpmin, logrpmax, param_dict['nrpbins']+1)
    rpbins_cens = rpbin_arr[:-1]+0.5*(rpbin_arr[1:]-rpbin_arr[:-1])
    ### Survey Details
    sample_title = r'\boldmath$M_{r}< -%d$' %(param_dict['sample'])
    ### URL - Random catalogues
    url_rand = os.path.join('http://lss.phy.vanderbilt.edu/groups/data_vc/DR7',
                            'mr-vollim-randoms',
                            'random_Mr{0}.rdcz'.format(sample_s))
    # cu.url_checker(url_rand)
    ## Galaxy properties - Limits
    prop_lim = {'logssfr':  -11,
                'sersic' :  3.,
                'g_r'    :  0.75}
    prop_keys = num.sort(list(prop_lim.keys()))
    ###
    ### To dictionary
    param_dict['sample_s'    ] = sample_s
    param_dict['logrpmin'    ] = logrpmin
    param_dict['logrpmax'    ] = logrpmax
    param_dict['dlogrp'      ] = dlogrp
    param_dict['rpbin_arr'   ] = rpbin_arr
    param_dict['rpbins_cens' ] = rpbins_cens
    param_dict['sample_title'] = sample_title
    param_dict['url_rand'    ] = url_rand
    param_dict['prop_lim'    ] = prop_lim
    param_dict['prop_keys'   ] = prop_keys

    return param_dict

def param_vals_test(param_dict):
    """
    Checks if values are consistent with each other.

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with `project` variables

    Raises
    -----------
    ValueError: Error
        This function raises a `ValueError` error if one or more of the 
        required criteria are not met
    """
    ##
    ## This is where the tests for `param_dict` input parameters go.
    ##
    ## Testing if `wget` exists in the system
    if is_tool('wget'):
        pass
    else:
        msg = '{0} You need to have `wget` installed in your system to run '
        msg += 'this script. You can download the entire dataset at {1}.\n\t\t'
        msg += 'Exiting....'
        msg = msg.format(param_dict['Prog_msg'], param_dict['url_catl'])
        raise ValueError(msg)

def directory_skeleton(param_dict, proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    """
    ## Download Random catalogues
    # rand_path = os.path.join(param_dict['data'])
    ##
    ## Figure directory
    figdir = os.path.join(  proj_dict['plot_dir'],
                            'referee_report_figs_1.0')
    ##
    ## Randoms catalogue
    rand_dir = os.path.join(proj_dict['data_dir'],
                            'external',
                            'SDSS',
                            'randoms')
    ## Creating directories
    cu.Path_Folder(figdir  )
    cu.Path_Folder(rand_dir)
    ##
    ## Adding to dictionary
    proj_dict['figdir'  ] = figdir
    proj_dict['rand_dir'] = rand_dir

    return proj_dict

#### --------- Reading Catalogue --------- ####

def loading_catls(param_dict, proj_dict):
    """
    Plots the distribution of color, ssfr, and morphology for the 
    SDSS DR7 dataset

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    Returns
    ----------

    """
    ## Loading data from `data` and `mock` catalogues
    # Data
    data_pd = cu.read_hdf5_file_to_pandas_DF(
                cu.extract_catls(   'data',
                                    param_dict['catl_type'],
                                    param_dict['sample_s'])[0])
    data_cl_pd = cu.sdss_catl_clean(data_pd, 'data').copy()
    # Mocks
    mocks_pd = cu.read_hdf5_file_to_pandas_DF(
                cu.extract_catls(   'mocks',
                                    param_dict['catl_type'],
                                    param_dict['sample_s'])[0]).copy()
    ##
    ## Galaxy properties - Limits
    prop_lim  = param_dict['prop_lim' ]
    prop_keys = param_dict['prop_keys']
    ##
    ## Normalizing data
    ##
    ## Data
    for col_kk in prop_keys:
        if col_kk in data_cl_pd.columns.values:
            data_cl_pd.loc[:, col_kk+'_normed'] = data_cl_pd[col_kk]/prop_lim[col_kk]
    ##
    ## Mocks
    for col_kk in prop_keys:
        if col_kk in mocks_pd.columns.values:
            mocks_pd.loc[:, col_kk+'_normed'] = mocks_pd[col_kk]/prop_lim[col_kk]

    return data_cl_pd, mocks_pd

#### --------- Galaxy Properties - Distributions --------- ####

def galprop_distr_main(data_cl_pd, mocks_pd, param_dict, proj_dict,
    fig_fmt='pdf', figsize=(10,10)):
    """
    Plots the distribution of color, ssfr, and morphology for the 
    SDSS DR7 dataset - Main Function

    Parameters
    -----------
    data_cl_pd: pandas DataFrame
        DataFrame containig the on about the galaxy properties of the `data`
        sample

    mocks_pd: pandas DataFrame
        DataFrame containig the on about the galaxy properties of the `mocks`
        sample

    param_dict: python dictionary
        dictionary with `project` variables
    
    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    fig_fmt: string, optional (default = 'pdf')
        extension to use for the figure

    figsize: tuple, optional (default = (10,10))
        size of the figure, in inches.

    Returns
    -----------

    """
    Prog_msg       = param_dict['Prog_msg']
    ## Figure name
    fname = os.path.join(   proj_dict['figdir'],
                            'galprop_data_mocks_distr.{0}'.format(fig_fmt))
    ## Galaxy properties
    prop_keys      = param_dict['prop_keys']
    n_keys         = len(prop_keys)
    prop_keys_norm = [xx+'_normed' for xx in prop_keys]
    ## Colors
    act_color = 'blue'
    pas_color = 'red'
    ## Other properties
    size_label  = 20
    size_legend = 10
    size_text   = 14
    propbox     = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ##
    ## Separating between active and passive galaxies, for each 
    ## galaxy property
    #
    # Initializing figure
    ## Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['text.latex.unicode']=True
    matplotlib.rcParams['text.usetex']=True
    #
    plt.clf()
    plt.close()
    fig      = plt.figure(figsize=figsize)
    ax1      = fig.add_subplot(311, facecolor='white')
    ax2      = fig.add_subplot(312, facecolor='white')
    ax3      = fig.add_subplot(313, facecolor='white')
    axes_arr = [ax1, ax2, ax3]
    ## Galaxy Properties - Labels
    prop_labels = {'g_r':'Color', 'logssfr':'sSFR','sersic':'morphology'}
    ##
    ## Looping over galaxy properties
    for kk, prop in enumerate(prop_keys):
        # Normalized property
        prop_norm = prop_keys_norm[kk]
        ### Data
        # Active
        prop_act_data = data_cl_pd.loc[data_cl_pd[prop_norm] <= 1., prop_norm]
        # Passive
        prop_pas_data = data_cl_pd.loc[data_cl_pd[prop_norm] > 1., prop_norm]
        ##
        ## Plotting distributions
        axes_arr[kk] = galprop_distr_plot(  prop_act_data,
                                            axes_arr[kk],
                                            data_opt=True,
                                            act_pas_opt='act')
        axes_arr[kk] = galprop_distr_plot(  prop_pas_data,
                                            axes_arr[kk],
                                            data_opt=True,
                                            act_pas_opt='pas')
        ### Mocks
        if prop in mocks_pd.columns.values:
            # Active
            prop_act_mock = mocks_pd.loc[mocks_pd[prop_norm] <= 1., prop_norm]
            # Passive
            prop_pas_mock = mocks_pd.loc[mocks_pd[prop_norm] > 1., prop_norm]
            ##
            ## Plotting distributions
            axes_arr[kk] = galprop_distr_plot(  prop_act_mock,
                                                axes_arr[kk],
                                                data_opt=False,
                                                act_pas_opt='act')
            axes_arr[kk] = galprop_distr_plot(  prop_pas_mock,
                                                axes_arr[kk],
                                                data_opt=False,
                                                act_pas_opt='pas')
        ## Text
        axes_arr[kk].text(0.80, 0.95, prop_labels[prop],
                transform=axes_arr[kk].transAxes,
                verticalalignment='top',
                color='black',
                bbox=propbox,
                weight='bold', fontsize=size_text)
    ##
    ## Axes limits
    ax1.set_xlim(0.,1.5)
    ax2.set_xlim(0.8,1.2)
    ax3.set_xlim(0.0,2.)
    ##
    ## Saving figure
    if fig_fmt == 'pdf':
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
    plt.clf()
    plt.close()

def galprop_distr_plot(catl_pd, ax, data_opt=False, act_pas_opt='act'):
    """
    Plots the distributions of color, ssfr, and morphology for the 
    SDSS DR7 dataset.

    Parameters
    -----------
    catl_pd: pandas DataFrame
        DataFrame with the values used for the distributions

    ax: matplotlib.axes._subplots.AxesSubplot
        axis object for the current distribution

    data_opt: boolean, optional (default = False)
        option for determining if `catl_pd` is from real SDSS or from 
        mock catalogue.

    act_pas_opt: string, optional (default = 'act')
        option for determining is `catl_pd` is `active` or `passive` 
        population
        Options:
            - 'act' or 'pas'

    Returns
    -----------

    """
    # KDE plot - Color
    if data_opt:
        catl_line = '-'
        if act_pas_opt == 'act':
            col = 'blue'
        else:
            col = 'red'
    else:
        catl_line = '--'
        if act_pas_opt == 'act':
            col = 'orange'
        else:
            col = 'green'
    # Plotting
    mean_distr = catl_pd.values.mean()
    std_distr  = catl_pd.values.std()
    sns.kdeplot(catl_pd.values, shade=True, alpha=0.5, ax=ax, color=col)
    ax.axvline(mean_distr, color=col, linestyle=catl_line)
    ax.axvspan(mean_distr - std_distr, mean_distr + std_distr,
                alpha = 0.5, color=col)

    return ax

#### --------- Projected Correlation Function --------- ####

def projected_wp_main(data_cl_pd, mocks_pd, param_dict, proj_dict):
    """
    Computes the projected correlation functions for the different 
    datasets.
    And plots the results of wp(rp)

    Parameters
    -----------
    data_cl_pd: pandas DataFrame
        DataFrame containig the on about the galaxy properties of the `data`
        sample

    mocks_pd: pandas DataFrame
        DataFrame containig the on about the galaxy properties of the `mocks`
        sample

    param_dict: python dictionary
        dictionary with `project` variables
    
    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    Returns
    -----------
    act_pd_data: pandas DataFrame
        DataFrame with the data for `active` galaxies from `data`

    pas_pd_data: pandas DataFrame
        DataFrame with the data for `passive` galaxies from `data`

    act_pd_mock: pandas DataFrame
        DataFrame with the data for `active` galaxies from `mocks`

    pas_pd_mock: pandas DataFrame
        DataFrame with the data for `passive` galaxies from `mocks`
    """
    ##
    ## Catalogue of Randoms
    rand_file = os.path.join(   proj_dict['rand_dir'],
                                os.path.basename(param_dict['url_rand']))
    if os.path.exists(rand_file):
        rand_pd = pd.read_hdf(  rand_file)
    else:        
        rand_pd = pd.read_csv(  param_dict['url_rand'],
                                sep='\s+',
                                names=['ra','dec','cz'])
        rand_pd.to_hdf(rand_file, 'gal', compression='gzip', complevel=9)
    ##
    ## Removing files
    act_data_file = os.path.join(proj_dict['rand_dir'], 'wp_rp_act_data.hdf5')
    pas_data_file = os.path.join(proj_dict['rand_dir'], 'wp_rp_pas_data.hdf5')
    act_mock_file = os.path.join(proj_dict['rand_dir'], 'wp_rp_act_mock.hdf5')
    pas_mock_file = os.path.join(proj_dict['rand_dir'], 'wp_rp_pas_mock.hdf5')
    #
    # Check if file exists
    act_pas_list = [act_data_file, pas_data_file, act_mock_file, pas_mock_file]
    for file_ii in act_pas_list:
        if param_dict['remove_files']:
            if os.path.exists(file_ii):
                os.remove(file_ii)
    ##
    ## Only running analysis if files are not present
    if all([os.path.isfile(f) for f in act_pas_list]):
        ## Reading in files
        act_pd_data = pd.read_hdf(act_data_file, 'gal')
        pas_pd_data = pd.read_hdf(pas_data_file, 'gal')
        act_pd_mock = pd.read_hdf(act_mock_file, 'gal')
        pas_pd_mock = pd.read_hdf(pas_mock_file, 'gal')
    else:
        ## Normalized keys
        prop_keys     = param_dict['prop_keys']
        n_keys        = len(prop_keys)
        ##
        ## Defining dictionaries
        act_pd_data = pd.DataFrame({'rpbin':10**param_dict['rpbins_cens']})
        pas_pd_data = pd.DataFrame({'rpbin':10**param_dict['rpbins_cens']})
        act_pd_mock = pd.DataFrame({'rpbin':10**param_dict['rpbins_cens']})
        pas_pd_mock = pd.DataFrame({'rpbin':10**param_dict['rpbins_cens']})
        ## Looping over galaxy property
        # ProgressBar properties
        widgets   = [Bar('>'), 'wp(rp) Gals. Props ', ETA(), ' ', ReverseBar('<')]
        pbar_mock = ProgressBar( widgets=widgets, maxval= 10 * n_keys).start()
        for kk, prop in enumerate(prop_keys):
            print('{0} Galaxy Property: {1}'.format(param_dict['Prog_msg'],
                                                    prop))
            ## Data
            prop_normed = prop + '_normed'
            data_act    = data_cl_pd.loc[data_cl_pd[prop_normed]<= 1.]
            data_pas    = data_cl_pd.loc[data_cl_pd[prop_normed]>  1.]
            ##
            ## Computing wp(rp)
            # Active
            wp_act_data = projected_wp_calc(data_act  , rand_pd  ,
                                            param_dict, proj_dict,
                                            data_opt = True)
            # Passive
            wp_pas_data = projected_wp_calc(data_pas  , rand_pd  ,
                                            param_dict, proj_dict,
                                            data_opt = True)
            ##
            ## Saving to `active` and `passive` wp DataFrames
            act_pd_data.loc[:,prop+'_wp'] = wp_act_data
            pas_pd_data.loc[:,prop+'_wp'] = wp_pas_data
            ##
            ## Mocks
            if prop in mocks_pd.columns.values:
                # Active and Passive - Mock
                mocks_act = mocks_pd.loc[mocks_pd[prop_normed] <= 1.]
                mocks_pas = mocks_pd.loc[mocks_pd[prop_normed] >  1.]
                ##
                ## Computing wp(rp)
                # Active
                wp_act_mock = projected_wp_calc(mocks_act  , rand_pd  ,
                                                param_dict, proj_dict,
                                                data_opt = False)
                # Passive
                wp_pas_mock = projected_wp_calc(mocks_pas  , rand_pd  ,
                                                param_dict, proj_dict,
                                                data_opt = False)
                ##
                ## Saving to `active` and `passive` wp DataFrames
                act_pd_mock.loc[:,prop+'_wp'] = wp_act_mock
                pas_pd_mock.loc[:,prop+'_wp'] = wp_pas_mock
            pbar_mock.update(10*kk)
        pbar_mock.finish()
        ##
        ## Saving Data
        act_pd_data.to_hdf(act_data_file, 'gal')
        pas_pd_data.to_hdf(pas_data_file, 'gal')
        act_pd_mock.to_hdf(act_mock_file, 'gal')
        pas_pd_mock.to_hdf(pas_mock_file, 'gal')

    return act_pd_data, pas_pd_data, act_pd_mock, pas_pd_mock

def projected_wp_calc(catl_pd, rand_pd, param_dict, proj_dict, data_opt=False):
    """
    Separates between `active` and `passive` galaxies for a given galaxy 
    property

    Parameters
    ------------
    catl_pd: pandas DataFrame
        DataFrame containg the galaxy properties to analyze

    rand_pd: pandas DataFrame
        DataFrame containing positions of `randoms`

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    data_opt: boolean, optional (default = False)
        `True` if `catl_pd` is real SDSS data.
        `False` if `catl_pd` is a mock catalogue

    Returns
    ------------
    wp: numpy.ndarray, shape(N,)
        The projected correlation function, calculated using the chosen
        estimator, is returned. If *any* of the ``pi`` bins (in an ``rp``
        bin) contains 0 for the ``RR`` counts, then ``NAN`` is returned
        for that ``rp`` bin.

    """
    ## Corrfunc options
    cosmology = 1
    nthreads  = 2
    ## Number of elements in `catl_pd` and `rand_pd`
    N         = len(catl_pd)
    rand_N    = len(rand_pd)
    ## Weights
    if data_opt:
        weights1 = catl_pd['compl'].values**(-1)
    else:
        weights1 = num.ones(N)
    ## Auto pair counts in DD
    autocorr  = 1
    DD_counts = DDrppi_mocks(   autocorr,
                                cosmology,
                                nthreads,
                                param_dict['pimax'],
                                10**param_dict['rpbin_arr'],
                                catl_pd['ra'].values,
                                catl_pd['dec'].values,
                                catl_pd['cz'].values,
                                weights1 = weights1,
                                weights2 = weights1)
    ## Cross pair counts in DR
    autocorr  = 0
    DR_counts = DDrppi_mocks(   autocorr,
                                cosmology,
                                nthreads,
                                param_dict['pimax'],
                                10**param_dict['rpbin_arr'],
                                catl_pd['ra'].values,
                                catl_pd['dec'].values,
                                catl_pd['cz'].values,
                                RA2  =rand_pd['ra'].values,
                                DEC2 =rand_pd['dec'].values,
                                CZ2  =rand_pd['cz'].values,
                                weights1 = weights1,
                                weights2 = num.ones(rand_N))
    ## Auto apirs counts in RR
    autocorr  = 1
    RR_counts = DDrppi_mocks(   autocorr,
                                cosmology,
                                nthreads,
                                param_dict['pimax'],
                                10**param_dict['rpbin_arr'],
                                rand_pd['ra'].values,
                                rand_pd['dec'].values,
                                rand_pd['cz'].values)
    ## All the pair counts are done, get the angular correlation function
    wp = convert_rp_pi_counts_to_wp(    N,
                                        N,
                                        rand_N,
                                        rand_N,
                                        DD_counts,
                                        DR_counts,
                                        DR_counts,
                                        RR_counts,
                                        param_dict['nrpbins'],
                                        param_dict['pimax'])

    return wp
    
def projected_wp_plot(act_pd_data, pas_pd_data, act_pd_mock, pas_pd_mock, 
    param_dict, proj_dict, fig_fmt='pdf', figsize_2=(5.,5.)):
    """
    Plots the projected correlation function wp(rp)

    Parameters
    ------------
    act_pd_data: pandas DataFrame
        DataFrame with the data for `active` galaxies from `data`

    pas_pd_data: pandas DataFrame
        DataFrame with the data for `passive` galaxies from `data`

    act_pd_mock: pandas DataFrame
        DataFrame with the data for `active` galaxies from `mocks`

    pas_pd_mock: pandas DataFrame
        DataFrame with the data for `passive` galaxies from `mocks`

    param_dict: python dictionary
        dictionary with `project` variables
    
    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    """
    Prog_msg = param_dict['Prog_msg']
    ## Galaxy properties
    prop_keys = param_dict['prop_keys']
    ## Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['text.latex.unicode']=True
    matplotlib.rcParams['text.usetex']=True
    ##
    ## Labels
    xlabel       = r'\boldmath $r_{p}\ \left[h^{-1}\ \textrm{Mpc} \right]$'
    ylabel       = r'\boldmath $\xi(r_{p})$'
    ## Figure name
    fname = os.path.join(   proj_dict['figdir'],
                            'wprp_galprop_data_mocks.{0}'.format(fig_fmt))
    ##
    ## Figure details
    figsize     = figsize_2
    size_label  = 20
    size_legend = 10
    size_text   = 14
    #
    # Figure
    plt.clf()
    plt.close()
    fig     = plt.figure(figsize=figsize)
    ax_data = fig.add_subplot(111, facecolor='white')
    color_arr = ['blue','red','green','orange']
    ### Plot data
    # Color and linestyles
    lines_arr = ['-','--',':']
    propbox   = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ## Looping over galaxy properties
    for kk, prop in enumerate(prop_keys):
        # Active
        ax_data.plot(act_pd_data['rpbin'],
                act_pd_data[prop+'_wp'],
                color=color_arr[0],
                linestyle=lines_arr[kk],
                label=r'{0}'.format(prop.replace('_','-')))
        # Passive
        ax_data.plot(pas_pd_data['rpbin'],
                pas_pd_data[prop+'_wp'].values,
                color=color_arr[1],
                linestyle=lines_arr[kk])
        ##
        ## Mocks
        ## Active
        ax_data.plot(act_pd_mock['rpbin'],
                act_pd_mock[prop+'_wp'],
                color=color_arr[2],
                linestyle=lines_arr[kk],
                label=r'{0} - Mocks'.format(prop.replace('_','-')))
        ## Passive
        ax_data.plot(pas_pd_mock['rpbin'],
                pas_pd_mock[prop+'_wp'].values,
                color=color_arr[3],
                linestyle=lines_arr[kk])
    ##
    ## Legend
    ax_data.legend( loc='lower left', bbox_to_anchor=[0, 0],
                    ncol=2, title='Galaxy Properties',
                    prop={'size':size_legend})
    ax_data.get_legend().get_title().set_color("red")
    ax_data.set_xscale('log')
    ax_data.set_yscale('log')
    ax_data.set_xlabel(xlabel, fontsize=size_label)
    ax_data.set_ylabel(ylabel, fontsize=size_label)
    ax_data.text(0.80, 0.95, 'SDSS',
            transform=ax_data.transAxes,
            verticalalignment='top',
            color='black',
            bbox=propbox,
            weight='bold', fontsize=size_text)
    ##
    ## Saving figure
    if fig_fmt == 'pdf':
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
    plt.clf()
    plt.close()

#### --------- Main Function --------- ####

def main(args):
    """

    """
    ## Starting time
    start_time = datetime.now()
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Initializing random seed
    num.random.seed(param_dict['seed'])
    ## Checking for correct input
    param_vals_test(param_dict)
    ## ---- Adding to `param_dict` ----
    param_dict = add_to_dict(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ##
    ## Creating Folder Structure
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    # Distribution of galaxy properties
    data_cl_pd, mocks_pd = loading_catls(param_dict, proj_dict)
    ##
    ## Projected correlation function
    # Calculations
    (   act_pd_data,
        pas_pd_data,
        act_pd_mock,
        pas_pd_mock) = projected_wp_main(   data_cl_pd,
                                            mocks_pd  ,
                                            param_dict,
                                            proj_dict )
    # Plotting
    projected_wp_plot(  act_pd_data,
                        pas_pd_data,
                        act_pd_mock,
                        pas_pd_mock,
                        param_dict ,
                        proj_dict)
    ##
    ## Distributions of Galaxy Properties
    galprop_distr_main(data_cl_pd, mocks_pd, param_dict, proj_dict)


# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
