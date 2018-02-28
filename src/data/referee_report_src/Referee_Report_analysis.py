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
import matplotlib.gridspec as gridspec
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
from tqdm import tqdm
from multiprocessing import Pool, Process, cpu_count
from glob import glob
import copy

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
    ## CLF/CSMF method of assigning galaxy properties
    parser.add_argument('-clf_method',
                        dest='clf_method',
                        help="""
                        Method for assigning galaxy properties to mock 
                        galaxies. Options:
                        (1) = Independent assignment of (g-r), sersic, logssfr
                        (2) = (g-r) decides active/passive designation and 
                        draws values independently.
                        (3) (g-r) decides active/passive designation, and 
                        assigns other galaxy properties for that given 
                        galaxy.
                        """,
                        type=int,
                        choices=[1,2,3],
                        default=1)
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.5)
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

def array_insert(arr1, arr2, axis=1):
    """
    Joins the arrays into a signle multi-dimensional array

    Parameters
    ----------
    arr1: array_like
        first array to merge

    arr2: array_like
        second array to merge

    Return
    ---------
    arr3: array_like
        merged array from `arr1` and `arr2`
    """
    arr3 = num.insert(arr1, len(arr1.T), arr2, axis=axis)

    return arr3

def sigma_calcs(data_arr, type_sigma='std', perc_arr = [68., 95., 99.7],
    return_mean_std=False):
    """
    Calcualates the 1-, 2-, and 3-sigma ranges for `data_arr`

    Parameters
    -----------
    data_arr: numpy.ndarray, shape( param_dict['nrpbins'], param_dict['itern_tot'])
        array of values, from which to calculate percentiles or St. Dev.

    type_sigma: string, optional (default = 'std')
        option for calculating either `percentiles` or `standard deviations`
        Options:
            - 'perc': calculates percentiles
            - 'std' : uses standard deviations as 1-, 2-, and 3-sigmas

    perc_arr: array_like, optional (default = [68., 95., 99.7])
        array of percentiles to calculate

    return_mean_std: boolean, optional (default = False)
        option for returning mean and St. Dev. along with `sigma_dict`

    Return
    ----------
    sigma_dict: python dicitionary
        dictionary containg the 1-, 2-, and 3-sigma upper and lower 
        ranges for `data-arr`

    mark_mean: array_like
        array of the mean value of `data_arr`.
        Only returned if `return_mean_std == True`

    mark_std: array_like
        array of the St. Dev. value of `data_arr`.
        Only returned if `return_mean_std == True`
    """
    ## Creating dictionary for saving `sigma`s
    sigma_dict = {}
    for ii in range(len(perc_arr)):
        sigma_dict[ii] = []
    ## Using Percentiles to estimate errors
    if type_sigma=='perc':
        for ii, perc_ii in enumerate(perc_arr):
            mark_lower = num.nanpercentile(data_arr, 50.-(perc_ii/2.),axis=1)
            mark_upper = num.nanpercentile(data_arr, 50.+(perc_ii/2.),axis=1)
            # Saving to dictionary
            sigma_dict[ii] = num.column_stack((mark_lower, mark_upper)).T
    ## Using standard deviations to estimate errors
    if type_sigma=='std':
        mean_val = num.nanmean(data_arr, axis=1)
        std_val  = num.nanstd( data_arr, axis=1)
        for ii in range(len(perc_arr)):
            mark_lower = mean_val - ((ii+1) * std_val)
            mark_upper = mean_val + ((ii+1) * std_val)
            # Saving to dictionary
            sigma_dict[ii] = num.column_stack((mark_lower, mark_upper)).T
    ##
    ## Estimating mean and St. Dev. of `data_arr`
    mark_mean = num.nanmean(data_arr, axis=1)
    mark_std  = num.nanstd (data_arr, axis=1)

    if return_mean_std:
        return sigma_dict, mark_mean, mark_std
    else:
        return sigma_dict

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
    ##
    ## Checking `cpu_frac` range
    if (param_dict['cpu_frac'] > 0) and (param_dict['cpu_frac'] <= 1):
        pass
    else:
        msg = '{0} `cpu_frac` ({1}) must be between (0,1]'.format(
            param_dict['Prog_msg'],
            param_dict['cpu_frac'])
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
    data_cl_pd: pandas DataFrame
        data from the `data` catalogue

    mocks_pd_arr: list, shape (n_mocks,)
        list with pandas DataFrames for each mock catalogue being 
        analyzed

    """
    ##
    ## Galaxy properties - Limits
    prop_lim  = param_dict['prop_lim' ]
    prop_keys = param_dict['prop_keys']
    ## Loading data from `data` and `mock` catalogues
    # Data
    data_pd = cu.read_hdf5_file_to_pandas_DF(
                cu.extract_catls(   'data',
                                    param_dict['catl_type'],
                                    param_dict['sample_s'])[0])
    data_cl_pd = cu.sdss_catl_clean(data_pd, 'data').copy()
    ##
    ## Normalizing Data
    for col_kk in prop_keys:
        if col_kk in data_cl_pd.columns.values:
            data_cl_pd.loc[:, col_kk+'_normed'] = data_cl_pd[col_kk]/prop_lim[col_kk]
    ##
    ## Mocks
    mocks_arr = cu.Index(   cu.catl_sdss_dir( 'mocks',
                                param_dict['catl_type'],
                                param_dict['sample_s']), '.hdf5')
    n_mocks   = len(mocks_arr)
    # Saving mock data
    mocks_pd_arr = [[] for x in range(n_mocks)]
    for ii, mock_ii in enumerate(tqdm(mocks_arr)):
        ## Extracting data
        mock_ii_pd = cu.read_hdf5_file_to_pandas_DF(mock_ii)
        ## Normalizing data
        for col_kk in prop_keys:
            if col_kk in mock_ii_pd.columns.values:
                mock_ii_pd.loc[:, col_kk+'_normed'] = mock_ii_pd[col_kk]/prop_lim[col_kk]
        ## Saving to list
        mocks_pd_arr[ii] = mock_ii_pd

    return data_cl_pd, mocks_pd_arr

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
            col = 'cyan'
    # Plotting
    mean_distr = catl_pd.values.mean()
    std_distr  = catl_pd.values.std()
    sns.kdeplot(catl_pd.values, shade=True, alpha=0.5, ax=ax, color=col)
    ax.axvline(mean_distr, color=col, linestyle=catl_line)
    ax.axvspan(mean_distr - std_distr, mean_distr + std_distr,
                alpha = 0.5, color=col)

    return ax

#### --------- Projected Correlation Function --------- ####

def projected_wp_main(data_cl_pd, mocks_pd_arr, param_dict, proj_dict):
    """
    Computes the projected correlation functions for the different 
    datasets.
    And plots the results of wp(rp)

    Parameters
    -----------
    data_cl_pd: pandas DataFrame
        DataFrame containig the on about the galaxy properties of the `data`
        sample

    mocks_pd_arr: list, shape (n_mocks,)
        list with pandas DataFrames for each mock catalogue being 
        analyzed

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
    wp_rp_dir    = os.path.join(    proj_dict['rand_dir'], 'wp_rp')
    wp_data_dir  = os.path.join(    wp_rp_dir            , 'data')
    wp_mocks_dir = os.path.join(    wp_rp_dir            , 
                                    'mocks',
                                    'clf_method_{0}'.format(
                                        param_dict['clf_method']))
    cu.Path_Folder(wp_rp_dir   )
    cu.Path_Folder(wp_data_dir )
    cu.Path_Folder(wp_mocks_dir)
    ## Randoms file
    rand_file = os.path.join(   proj_dict['rand_dir'],
                                os.path.basename(param_dict['url_rand']))
    if os.path.exists(rand_file):
        rand_pd = pd.read_hdf(  rand_file)
    else:        
        rand_pd = pd.read_csv(  param_dict['url_rand'],
                                sep='\s+',
                                names=['ra','dec','cz'])
        rand_pd.to_hdf(rand_file, 'gal', compression='gzip', complevel=9)
    ###
    ### ------| DATA |------ ###
    ###
    ## Removing files
    # Data
    act_data_file = os.path.join(   wp_data_dir, 'wp_rp_act_data.hdf5')
    pas_data_file = os.path.join(   wp_data_dir, 'wp_rp_pas_data.hdf5')
    ## Checking if file exists
    act_pas_data_list = [act_data_file, pas_data_file]
    for file_ii in act_pas_data_list:
        if param_dict['remove_files']:
            if os.path.exists(file_ii):
                os.remove(file_ii)
    ##
    ## Reading Data if files present
    if all([os.path.isfile(f) for f in act_pas_data_list]):
        ## Reading in files
        act_pd_data = pd.read_hdf(act_data_file, 'gal')
        pas_pd_data = pd.read_hdf(pas_data_file, 'gal')
    else:
        ## Normalized keys
        prop_keys     = param_dict['prop_keys']
        n_keys        = len(prop_keys)
        ##
        ## Defining dictionaries
        act_pd_data = pd.DataFrame({'rpbin':10**param_dict['rpbins_cens']})
        pas_pd_data = pd.DataFrame({'rpbin':10**param_dict['rpbins_cens']})
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
                                            param_dict, data_opt = True)
            # Passive
            wp_pas_data = projected_wp_calc(data_pas  , rand_pd  ,
                                            param_dict, data_opt = True)
            ##
            ## Saving to `active` and `passive` wp DataFrames
            act_pd_data.loc[:,prop+'_wp'] = wp_act_data
            pas_pd_data.loc[:,prop+'_wp'] = wp_pas_data
        ##
        ## Saving Data
        act_pd_data.to_hdf(act_data_file, 'gal')
        pas_pd_data.to_hdf(pas_data_file, 'gal')
        cu.File_Exists(act_data_file)
        cu.File_Exists(pas_data_file)
    ###
    ### ------| Mock Catalogues |------ ###
    ###
    n_mocks = len(mocks_pd_arr)
    ## Number of CPUs to use
    cpu_number = int(cpu_count() * param_dict['cpu_frac'])
    ## Defining step-size for each CPU
    if cpu_number <= n_mocks:
        catl_step  = int(n_mocks / cpu_number)
    else:
        catl_step  = int(1)
    ## Array with designanted catalogue numbers for each CPU
    memb_arr     = num.arange(0, n_mocks+1, catl_step)
    memb_arr[-1] = n_mocks
    ## Tuples of the ID of each catalogue
    memb_tuples  = num.asarray([(memb_arr[xx], memb_arr[xx+1])
                            for xx in range(memb_arr.size-1)])
    ## Assigning `memb_tuples` to function `multiprocessing_catls`
    procs = []
    for ii in range(len(memb_tuples)):
        # Defining `proc` element
        proc = Process(target=projected_wp_multiprocessing, 
                        args=(memb_tuples[ii], mocks_pd_arr, rand_pd.copy(), 
                            wp_mocks_dir, param_dict, proj_dict))
        # Appending to main `procs` list
        procs.append(proc)
        proc.start()
    ##
    ## Joining `procs`
    for proc in procs:
        proc.join()

    return act_pd_data, pas_pd_data, wp_mocks_dir

def projected_wp_multiprocessing(memb_tuples_ii, mocks_pd_arr, rand_ii, 
    catl_out, param_dict, proj_dict):
    """
    Multiprocessing for wp(rp) for mock catalogues

    Parameters
    ------------
    memb_tuples_ii: tuple
        tuple with the indices of the catalogues to be analyzed

    mocks_pd_arr: list
        list of pandas DataFrames for each mock catalogue to be analyzed

    rand_ii: pandas DataFrame
        DataFrame with the 3-positions from the random galaxy catalogues

    catl_out: string
        path to the output folder for the wp-rp result

    param_dict: python dictionary
        dictionary with all of the script's variables

    proj_dict: python dictionary
        dictionary with all of the paths used throughout the script

    """
    ## Program Message
    Prog_msg = param_dict['Prog_msg']
    ## Reading in Catalogue IDs
    start_ii, end_ii = memb_tuples_ii
    print('{0}  start_ii: {1} | end_ii: {2}'.format(Prog_msg, start_ii, end_ii))
    ##
    ## Looping the desired catalogues
    for ii in range(start_ii,end_ii):
        print('{0} Analyzing `Mock {1}`\n'.format(Prog_msg, ii))
        mock_ii     = mocks_pd_arr[ii]
        act_file_ii = os.path.join( catl_out,
                                    'wp_rp_act_{0}_mock.hdf5'.format(ii))
        pas_file_ii = os.path.join( catl_out,
                                    'wp_rp_pas_{0}_mock.hdf5'.format(ii))
        ## Checking if files exist
        act_pas_list_ii = [act_file_ii, pas_file_ii]
        for file_ii in act_pas_list_ii:
            if param_dict['remove_files']:
                if os.path.exists(file_ii):
                    os.remove(file_ii)
        ##
        ##
        if all([os.path.isfile(f) for f in act_pas_list_ii]):
            ## Reading in files
            pass
        else:
            ## Normalized keys
            prop_keys     = param_dict['prop_keys']
            n_keys        = len(prop_keys)
            ##
            ## Defining dictionaries
            act_pd_ii = pd.DataFrame({'rpbin':10**param_dict['rpbins_cens']})
            pas_pd_ii = pd.DataFrame({'rpbin':10**param_dict['rpbins_cens']})
            ## Looping over galaxy property
            for kk, prop in enumerate(prop_keys):
                ## Data
                prop_normed = prop + '_normed'
                data_act_ii    = mock_ii.loc[mock_ii[prop_normed]<= 1.]
                data_pas_ii    = mock_ii.loc[mock_ii[prop_normed]>  1.]
                ##
                ## Computing wp(rp)
                # Active
                wp_act_data = projected_wp_calc(data_act_ii, rand_ii  ,
                                                param_dict , data_opt = False)
                # Passive
                wp_pas_data = projected_wp_calc(data_pas_ii, rand_ii  ,
                                                param_dict , data_opt = False)
                ##
                ## Saving to `active` and `passive` wp DataFrames
                act_pd_ii.loc[:,prop+'_wp'] = wp_act_data
                pas_pd_ii.loc[:,prop+'_wp'] = wp_pas_data
            ##
            ## Saving Data
            act_pd_ii.to_hdf(act_file_ii, 'gal')
            pas_pd_ii.to_hdf(pas_file_ii, 'gal')
            cu.File_Exists(act_file_ii)
            cu.File_Exists(pas_file_ii)

def projected_wp_mocks_range(wp_mocks_dir, param_dict, type_sigma='std'):
    """
    Returns the range of mocks at each rp-bin

    Parameters
    ------------
    wp_mocks_dir: string
        path to the mock results from wp-rp

    param_dict: python dictionary
        dictionary with all of the script's variables

    type_sigma: string, optional (default = 'std')
        option for calculating either `percentiles` or `standard deviations`
        Options:
            - 'perc': calculates percentiles
            - 'std' : uses standard deviations as 1-, 2-, and 3-sigmas

    Returns
    ------------
    wp_act_stats: python dictionary
        dictionary with `active` upper/lower limits for each galaxy property

    wp_pas_stats: python dictionary
        dictionary with `passive` upper/lower limits for each galaxy property

    """
    ## Constants
    prop_keys = param_dict['prop_keys']
    ## Arrays of `active` and `passive` wp-results
    mocks_act_arr = num.sort(glob(wp_mocks_dir+'/*act*.hdf5'))
    mocks_pas_arr = num.sort(glob(wp_mocks_dir+'/*pas*.hdf5'))
    # Variables
    n_catls = len(mocks_act_arr)
    # Initializing arrays
    zeros_arr = num.zeros((param_dict['nrpbins'],1))
    wp_act_results = dict(zip(prop_keys, [copy.deepcopy(zeros_arr) for xx in 
                            range(len(prop_keys))]))
    wp_pas_results = dict(zip(prop_keys, [copy.deepcopy(zeros_arr) for xx in 
                            range(len(prop_keys))]))
    ## Looping over galaxy properties
    for ii in range(n_catls):
        # Active
        catl_act_ii = cu.read_hdf5_file_to_pandas_DF(mocks_act_arr[ii])
        # Passive
        catl_pas_ii = cu.read_hdf5_file_to_pandas_DF(mocks_pas_arr[ii])
        # Looping over galaxy properties
        for prop_zz in prop_keys:
            # Active
            wp_act_results[prop_zz] = array_insert( wp_act_results[prop_zz],
                                                    catl_act_ii[prop_zz+'_wp'],
                                                    axis=1)
            # Passive
            wp_pas_results[prop_zz] = array_insert( wp_pas_results[prop_zz],
                                                    catl_pas_ii[prop_zz+'_wp'],
                                                    axis=1)
    ##
    ## Statistics for `active` and `passive`
    wp_act_stats = dict(zip(prop_keys, [{} for xx in range(len(prop_keys))]))
    wp_pas_stats = dict(zip(prop_keys, [{} for xx in range(len(prop_keys))]))
    # Looping over galaxy properties
    for prop_zz in prop_keys:
        # Deleting 1st row of zeros
        wp_act_ii = num.delete(wp_act_results[prop_zz], 0, axis=1)
        wp_pas_ii = num.delete(wp_pas_results[prop_zz], 0, axis=1)
        ##
        ## Statistics: Mean, and St. Dev.
        ## Errors, mean, and St. Dev.
        # Active
        (   sigma_act,
            mean_act ,
            std_act  ) = sigma_calcs(   wp_act_ii,
                                        type_sigma=type_sigma,
                                        return_mean_std=True)
        # Passive
        (   sigma_pas,
            mean_pas ,
            std_pas  ) = sigma_calcs(   wp_pas_ii,
                                        type_sigma=type_sigma,
                                        return_mean_std=True)
        ##
        ## Saving values
        # Active
        wp_act_stats[prop_zz]['mean' ] = mean_act
        wp_act_stats[prop_zz]['std'  ] = std_act
        wp_act_stats[prop_zz]['sigma'] = sigma_act
        # wp_act_stats[prop_zz]['wp_rp'] = wp_act_ii
        # Passive
        wp_pas_stats[prop_zz]['mean' ] = mean_pas
        wp_pas_stats[prop_zz]['std'  ] = std_pas
        wp_pas_stats[prop_zz]['sigma'] = sigma_pas
        # wp_pas_stats[prop_zz]['wp_rp'] = wp_pas_ii

    return wp_act_stats, wp_pas_stats

def projected_wp_calc(catl_pd, rand_pd, param_dict, data_opt=False):
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
    
def projected_wp_plot(act_pd_data, pas_pd_data, wp_act_stats, wp_pas_stats,
    param_dict, proj_dict, fig_fmt='pdf', figsize_2=(7.,10.), clf_method=1):
    """
    Plots the projected correlation function wp(rp)

    Parameters
    ------------
    act_pd_data: pandas DataFrame
        DataFrame with the data for `active` galaxies from `data`

    pas_pd_data: pandas DataFrame
        DataFrame with the data for `passive` galaxies from `data`

    wp_act_stats: python dictionary
        dictionary with `active` upper/lower limits for each galaxy property

    wp_pas_stats: python dictionary
        dictionary with `passive` upper/lower limits for each galaxy property

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
    ylabel       = r'\boldmath $w_{p}(r_{p})$'
    ylabel_res   = r'\boldmath $\Delta\ w_{p}(r_{p})$'
    ## Figure name
    fname = os.path.join(   proj_dict['figdir'],
                            'wprp_galprop_data_mocks_method_{0}.{1}'.format(
                                clf_method, fig_fmt))
    ##
    ## Figure details
    figsize     = figsize_2
    size_label  = 20
    size_legend = 10
    size_text   = 14
    color_arr   = ['blue','red','green','orange']
    alpha_arr   = [0.7, 0.5, 0.3]
    #
    # Figure
    plt.clf()
    plt.close()
    fig     = plt.figure(figsize=figsize)
    gs_prop = gridspec.GridSpec(1, 1, hspace=0.05, wspace=0.1)
    gs_prop_axes = gridspec.GridSpecFromSubplotSpec(2, 1, 
        gs_prop[0], height_ratios=[2,1], hspace=0 )
    ax_data  = plt.Subplot(fig, gs_prop_axes[0,:])
    ax_res   = plt.Subplot(fig, gs_prop_axes[1,:], sharex=ax_data)
    fig.add_subplot(ax_data)
    fig.add_subplot(ax_res)
    ax_data.set_facecolor('white')
    ax_res.set_facecolor('white')
    ## Hiding labels
    plt.setp(ax_data.get_xticklabels(), visible=False)
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
                label=r'{0} - Act'.format(prop.replace('_','-')))
        # Passive
        ax_data.plot(pas_pd_data['rpbin'],
                pas_pd_data[prop+'_wp'].values,
                color=color_arr[1],
                linestyle=lines_arr[kk],
                label=r'{0} - Pas'.format(prop.replace('_','-')))
        ##
        ## Mocks
        ## Active
        ax_data.plot(act_pd_data['rpbin'],
                wp_act_stats[prop]['mean'],
                color=color_arr[2],
                linestyle=lines_arr[kk],
                label=r'{0} (M) - Act'.format(prop.replace('_','-')))
        ## Passive
        ax_data.plot(act_pd_data['rpbin'],
                wp_pas_stats[prop]['mean'],
                color=color_arr[3],
                linestyle=lines_arr[kk],
                label=r'{0} (M) - Pas'.format(prop.replace('_','-')))
        ##
        ## Shaded contours for `mocks`
        # Active
        for zz in range(3):
            ## Active
            ax_data.fill_between(
                act_pd_data['rpbin'],
                wp_act_stats[prop]['sigma'][zz][0],
                wp_act_stats[prop]['sigma'][zz][1],
                facecolor=color_arr[2],
                alpha=alpha_arr[zz],
                zorder=zz+1)
            # Passive
            ax_data.fill_between(
                act_pd_data['rpbin'],
                wp_pas_stats[prop]['sigma'][zz][0],
                wp_pas_stats[prop]['sigma'][zz][1],
                facecolor=color_arr[3],
                alpha=alpha_arr[zz],
                zorder=zz+1)
        ##
        ## Residuals
        ## Active
        ax_res.plot(act_pd_data['rpbin'],
                100*(wp_act_stats[prop]['mean']-act_pd_data[prop+'_wp'])/\
                    act_pd_data[prop+'_wp'],
                    color=color_arr[0],
                    linestyle=lines_arr[kk],
                    label=r'{0} - Act'.format(prop.replace('_','-')))
        ## Passive
        ax_res.plot(act_pd_data['rpbin'],
                100*(wp_pas_stats[prop]['mean']-pas_pd_data[prop+'_wp'])/\
                    pas_pd_data[prop+'_wp'],
                    color=color_arr[1],
                    linestyle=lines_arr[kk],
                    label=r'{0} - Pas'.format(prop.replace('_','-')))
    ##
    ## Legend
    ax_data.legend( loc='lower left', bbox_to_anchor=[0, 0],
                    ncol=3, title='Galaxy Properties',
                    prop={'size':size_legend})
    ax_data.get_legend().get_title().set_color("red")
    ax_data.set_xscale('log')
    ax_data.set_yscale('log')
    ax_data.set_ylabel(ylabel, fontsize=size_label)
    ax_data.text(0.60, 0.95, 'SDSS - Method {0}'.format(clf_method),
            transform=ax_data.transAxes,
            verticalalignment='top',
            color='black',
            bbox=propbox,
            weight='bold', fontsize=size_text)
    ax_res.set_xlabel(xlabel, fontsize=size_label)
    ax_res.set_ylabel(ylabel_res, fontsize=size_label)
    ax_res.legend( loc='upper right', #bbox_to_anchor=[0.5, 0.5],
                    ncol=3,
                    prop={'size':size_legend})
    plt.subplots_adjust(hspace=0.)
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
    data_cl_pd, mocks_pd_arr = loading_catls(param_dict, proj_dict)
    ##
    ## Projected correlation function
    # Calculations
    (   act_pd_data ,
        pas_pd_data ,
        wp_mocks_dir) = projected_wp_main(  data_cl_pd,
                                            mocks_pd_arr,
                                            param_dict,
                                            proj_dict )
    ##
    ## Getting range for mocks
    (   wp_act_stats,
        wp_pas_stats) = projected_wp_mocks_range(wp_mocks_dir, param_dict)
    # Plotting
    projected_wp_plot(  act_pd_data ,
                        pas_pd_data ,
                        wp_act_stats,
                        wp_pas_stats,
                        param_dict  ,
                        proj_dict   ,
                        clf_method=param_dict['clf_method'])
    ##
    ## Distributions of Galaxy Properties
    galprop_distr_main(data_cl_pd, mocks_pd_arr[5], param_dict, proj_dict)
    ##
    ## End time for running the catalogues
    end_time   = datetime.now()
    total_time = end_time - start_time
    print('{0} Total Time taken (Create): {1}'.format(Prog_msg, total_time))


# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
