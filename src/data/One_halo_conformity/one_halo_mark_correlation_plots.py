#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 10/17/2017
# Last Modified: 04/06/2018
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Script that plots the 1-halo MCF results for `data` and `mocks`
"""
# Path to Custom Utilities folder
import os
import sys
import git
from path_variables import git_root_dir
sys.path.insert(0, os.path.realpath(git_root_dir(__file__)))

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
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import copy
import warnings
import subprocess

# Ignoring certain warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

## Functions

## --------- PARSING ARGUMENTS ------------##

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

def get_parser():
    """
    Get parser object for `eco_mocks_create.py` script.

    Returns
    -------
    args: 
        input arguments to the script
    """
    ## Define parser object
    description_msg = 'Script to evaluate 1-halo conformity on SDSS DR7'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    # 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    ## Number of HOD's to create. Dictates how many different types of 
    ##      mock catalogues to create
    parser.add_argument('-hod_model_n',
                        dest='hod_n',
                        help="Number of distinct HOD model to use. Default = 0",
                        type=int,
                        choices=range(0,1),
                        metavar='[0]',
                        default=0)
    ## Type of dark matter halo to use in the simulation
    parser.add_argument('-halotype',
                        dest='halotype',
                        help='Type of the DM halo.',
                        type=str,
                        choices=['so','fof'],
                        default='fof')
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
                        default=3)
    ## SDSS Sample
    parser.add_argument('-sample',
                        dest='sample',
                        help='SDSS Luminosity sample to analyze',
                        type=int,
                        choices=[19,20,21],
                        default=19)
    ## SDSS Kind
    parser.add_argument('-kind',
                        dest='catl_kind',
                        help='Type of data being analyzed.',
                        type=str,
                        choices=['data','mocks'],
                        default='data')
    ## SDSS Type
    parser.add_argument('-abopt',
                        dest='catl_type',
                        help='Type of Abund. Matching used in catalogue',
                        type=str,
                        choices=['mr'],
                        default='mr')
    ## Correlation Pair Type
    parser.add_argument('-pairtype',
                        dest='corr_pair_type',
                        help='Types of galaxy pairs to keep for the MFC and ',
                        type=str,
                        choices=['cen_sat'],
                        default='cen_sat')
    ## Shuffling Marks
    parser.add_argument('-shuffle',
                        dest='shuffle_marks',
                        help='Option for shuffling marks of Cens. and Sats.',
                        choices=['cen_sh', 'sat_sh', 'censat_sh'],
                        default='censat_sh')
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
    ## Total number of Iterations
    parser.add_argument('-itern',
                        dest='itern_tot',
                        help='Total number of iterations to perform on the `shuffled` scenario',
                        type=int,
                        choices=range(10,10000),
                        metavar='[10-10000]',
                        default=1000)
    ## Minimum Number of Galaxies in 1 group
    parser.add_argument('-nmin',
                        dest='ngals_min',
                        help='Minimum number of galaxies in a galaxy group',
                        type=int,
                        default=2)
    ## Bin in Group mass
    parser.add_argument('-mg',
                        dest='Mg_bin',
                        help='Bin width for the group masses',
                        type=_check_pos_val,
                        default=0.4)
    ## Pimax
    parser.add_argument('-pimax',
                        dest='pimax',
                        help='Value for `pimax` for the proj. corr. function',
                        type=_check_pos_val,
                        default=20.)
    ## Logarithm of the galaxy property
    parser.add_argument('-log',
                        dest='prop_log',
                        help='Use `log` or `non-log` units for `M*` and `sSFR`',
                        type=str,
                        choices=['log', 'nonlog'],
                        default='log')
    ## Mock Start
    parser.add_argument('-catl_start',
                        dest='catl_start',
                        help='Starting index of mock catalogues to use',
                        type=int,
                        choices=range(101),
                        metavar='[0-100]',
                        default=0)
    ## Mock Finish
    parser.add_argument('-catl_finish',
                        dest='catl_finish',
                        help='Finishing index of mock catalogues to use',
                        type=int,
                        choices=range(101),
                        metavar='[0-100]',
                        default=100)
    ## `Perfect Catalogue` Option
    parser.add_argument('-perf',
                        dest='perf_opt',
                        help='Option for using a `Perfect` catalogue',
                        type=_str2bool,
                        default=False)
    ## Type of correlation funciton to perform
    parser.add_argument('-corrtype',
                        dest='corr_type',
                        help='Type of correlation function to perform',
                        type=str,
                        choices=['galgal'],
                        default='galgal')
    ## Type of error or `sigma` to use
    parser.add_argument('-sigma',
                        dest='type_sigma',
                        help='Type of error or sigma to calculate for plotting',
                        type=str,
                        choices=['std','perc'],
                        default='std')
    ## Show Progbar
    parser.add_argument('-prog',
                        dest='prog_bar',
                        help='Option to print out progress bars for each for loop',
                        type=_str2bool,
                        default=True)
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cu.Program_Msg(__file__))
    ## Minimim mass bin to show
    parser.add_argument('-mg_min',
                        dest='mg_min',
                        help='Minimim mass bin to show in the results plot',
                        type=float,
                        default=12.41)
    ## Maximum mass bin to show
    parser.add_argument('-mg_max',
                        dest='mg_max',
                        help='Maximum mass bin to show in the results plot',
                        type=float,
                        default=14.)
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
                        default=3)
    ## Verbose
    parser.add_argument('-v','--verbose',
                        dest='verbose',
                        help='Option to print out project parameters',
                        type=_str2bool,
                        default=False)
    ## Maximum mass bin to show
    ## Parsing Objects
    args = parser.parse_args()

    return args

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
    ###
    ### Sample - Int
    sample_s = str(param_dict['sample'])
    ### Sample - Mr
    sample_Mr = 'Mr{0}'.format(param_dict['sample'])
    ###
    ### Perfect Catalogue
    if param_dict['perf_opt']:
        perf_str = 'haloperf'
    else:
        perf_str = ''
    ###
    ### Figure
    fig_idx = 21
    ###
    ### Projected distance `rp` bins
    logrpmin          = num.log10(param_dict['rpmin'])
    logrpmax          = num.log10(param_dict['rpmax'])
    dlogrp            = (logrpmax - logrpmin)/float(param_dict['nrpbins'])
    rpbin_arr         = num.linspace(logrpmin, logrpmax, param_dict['nrpbins']+1)
    rpbins_cens       = rpbin_arr[:-1]+0.5*(rpbin_arr[1:]-rpbin_arr[:-1])
    rpbins_cens_unlog = 10**rpbins_cens
    ###
    ### Survey Details
    sample_title = r'\boldmath$M_{r}< -%d$' %(param_dict['sample'])
    sample_name  = {'19':'Consuelo','20':'Esmeralda','21':'Carmen'}
    ###
    ### Project Details
    # String for Main directories
    param_str_arr = [   param_dict['rpmin']         , param_dict['rpmax']    ,
                        param_dict['nrpbins']       , param_dict['Mg_bin']   ,
                        param_dict['pimax' ]        , param_dict['itern_tot'],
                        param_dict['corr_pair_type'], param_dict['prop_log'] ,
                        param_dict['shuffle_marks'] , param_dict['ngals_min'],
                        perf_str ]
    param_str  = 'rpmin_{0}_rpmax_{1}_nrpbins_{2}_Mgbin_{3}_pimax_{4}_'
    param_str += 'itern_{5}_corrpair_type_{6}_proplog_{7}_shuffle_{8}_'
    param_str += 'nmin_{9}'
    if param_dict['perf_opt']:
        param_str += '_perf_opt_str_{10}/'
    else:
        param_str += '{10}/'
    param_str  = param_str.format(*param_str_arr)
    # String for Main Figures
    param_str_pic_arr = [param_dict['rpmin']  , param_dict['rpmax'] ,
                         param_dict['nrpbins'], param_dict['Mg_bin'],
                         param_dict['pimax']  , param_dict['ngals_min'],
                         perf_str ]
    param_str_pic  = 'rpmin_{0}_rpmax_{1}_nrpbins_{2}_Mgbin_{3}_pimax_{4}_'
    param_str_pic += 'nmin_{5}'
    if param_dict['perf_opt']:
        param_str_pic += '_perf_opt_str_{6}'
    else:
        param_str_pic += '{6}'
    param_str_pic = param_str_pic.format(*param_str_pic_arr)
    #
    # Figure Prefix
    fig_prefix = '{0}_{1}'.format(fig_idx, param_str_pic)
    ###
    ### Sigma Dictionary
    perc_arr = [68.,95., 99.7]
    sigma_dict = {}
    for ii, perc_ii in enumerate(perc_arr):
        low_sig        = 50.-(perc_ii/2.)
        high_sig       = 50.+(perc_ii/2.)
        sigma_dict[ii] = [low_sig, high_sig]
    ###
    ### To dictionary
    param_dict['sample_s'         ] = sample_s
    param_dict['sample_Mr'        ] = sample_Mr
    param_dict['perf_str'         ] = perf_str
    param_dict['fig_idx'          ] = fig_idx
    param_dict['logrpmin'         ] = logrpmin
    param_dict['logrpmax'         ] = logrpmax
    param_dict['dlogrp'           ] = dlogrp
    param_dict['rpbin_arr'        ] = rpbin_arr
    param_dict['rpbins_cens'      ] = rpbins_cens
    param_dict['rpbins_cens_unlog'] = rpbins_cens_unlog
    param_dict['sample_title'     ] = sample_title
    param_dict['param_str'        ] = param_str
    param_dict['param_str_pic'    ] = param_str_pic
    # Extras
    param_dict['sample_name'      ] = sample_name
    param_dict['sigma_dict'       ] = sigma_dict
    param_dict['perc_arr'         ] = perc_arr
    param_dict['fig_prefix'       ] = fig_prefix

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
    ## Check the `perf_opt` for when `catl_kind` is 'data'
    if (param_dict['catl_kind']=='data') and (param_dict['perf_opt']):
        msg = '{0} `catl_kind` ({1}) must smaller than `perf_opt` ({2})'\
            .format(
            param_dict['Prog_msg'],
            param_dict['catl_kind'],
            param_dict['perf_opt'])
        raise ValueError(msg)
    else:
        pass
    ##
    ## Checking that `nmin` is larger than 2
    if param_dict['ngals_min'] >= 2:
        pass
    else:
        msg = '{0} `ngals_min` ({1}) must be larger than `2`'.format(
            param_dict['Prog_msg'],
            param_dict['ngals_min'])
        raise ValueError(msg)
    ##
    ## Number of bins
    if (param_dict['nrpbins'] > 0):
        pass
    else:
        msg = '{0} `nrpbins` ({1}) must be larger than 0'.format(
            param_dict['Prog_msg'],
            param_dict['nrpbins'])
        raise ValueError(msg)
    ##
    ## Checking that `catl_start` < `catl_finish`
    if param_dict['catl_start'] < param_dict['catl_finish']:
        pass
    else:
        msg = '{0} `catl_start` ({1}) must smaller than `catl_finish` ({2})'\
            .format(
            param_dict['Prog_msg'],
            param_dict['catl_start'],
            param_dict['catl_finish'])
        raise ValueError(msg)
    ##
    ## Checking that `rpmin` < `rpmax`
    if param_dict['rpmin'] < param_dict['rpmax']:
        pass
    else:
        msg = '{0} `rpmin` ({1}) must smaller than `rpmax` ({2})'\
            .format(
            param_dict['Prog_msg'],
            param_dict['rpmin'],
            param_dict['rpmax'])
        raise ValueError(msg)
    ##
    ## Checking that `mg_min` < `mg_max`
    if param_dict['mg_min'] < param_dict['mg_max']:
        pass
    else:
        msg = '{0} `mg_min` ({1}) must smaller than `mg_max` ({2})'\
            .format(
            param_dict['Prog_msg'],
            param_dict['mg_min'],
            param_dict['mg_max'])
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
    ### MCF Folder prefix
    if param_dict['catl_kind'] == 'data':
        path_prefix = os.path.join( 'SDSS',
                                    param_dict['catl_kind'],
                                    param_dict['catl_type'],
                                    param_dict['sample_Mr'],
                                    'MCF_results')
    elif param_dict['catl_kind'] == 'mocks':
        path_prefix = os.path.join( 'SDSS',
                                    param_dict['catl_kind'],
                                    'halos_{0}'.format(param_dict['halotype']),
                                    'hod_model_{0}'.format(param_dict['hod_n']),
                                    'clf_method_{0}'.format(param_dict['clf_method']),
                                    param_dict['catl_type'],
                                    param_dict['sample_Mr'],
                                    'MCF_results')
    ### Pickle directory (result from 1-halo MCF analysis results)
    pickle_res = os.path.join(  proj_dict['data_dir'],
                                'processed',
                                path_prefix,
                                'catl_pickle_files',
                                param_dict['corr_type'],
                                param_dict['param_str'])
    ###
    ### MCF Folder prefix -- Data
    path_prefix_data = os.path.join('SDSS',
                                    'data',
                                    param_dict['catl_type'],
                                    param_dict['sample_Mr'],
                                    'MCF_results/')
    ### Pickle directory (results for `data`)
    pickle_data = os.path.join( proj_dict['data_dir'],
                                'processed',
                                path_prefix_data,
                                'catl_pickle_files',
                                param_dict['corr_type'],
                                param_dict['param_str'])
    ### Figure out directory
    if param_dict['catl_kind'] == 'data':
        figure_dir = os.path.join(  proj_dict['plot_dir'],
                                    'SDSS',
                                    param_dict['catl_kind'],
                                    param_dict['catl_type'],
                                    param_dict['sample_Mr'],
                                    'conformity_output',
                                    'MCF_figures',
                                    param_dict['corr_type'],
                                    param_dict['param_str'])
    elif param_dict['catl_kind'] == 'mocks':
        figure_dir = os.path.join(  proj_dict['plot_dir'],
                                    'SDSS',
                                    param_dict['catl_kind'],
                                    'halos_{0}'.format(param_dict['halotype']),
                                    'hod_model_{0}'.format(param_dict['hod_n']),
                                    'clf_method_{0}'.format(param_dict['clf_method']),
                                    param_dict['catl_type'],
                                    param_dict['sample_Mr'],
                                    'conformity_output',
                                    'MCF_figures',
                                    param_dict['corr_type'],
                                    param_dict['param_str'])
    ## Figure - Paper directory
    fig_paper_dir = os.path.join(   proj_dict['plot_dir'],
                                    'SDSS',
                                    'Paper_Figures/')


    ### MCF Folder prefix
    # path_prefix = 'SDSS/{0}/{1}/Mr{2}/MCF_results/'.format(
    #                     param_dict['catl_kind'],
    #                     param_dict['catl_type'],
    #                     param_dict['sample'   ])
    # ### ## Pickle directory (result from 1-halo MCF analysis results)
    # pickle_res = '{0}/processed/{1}/catl_pickle_files/{2}/{3}/'.format(
    #                 proj_dict['data_dir']  ,
    #                 path_prefix            ,
    #                 param_dict['corr_type'],
    #                 param_dict['param_str'])
    ###
    ### MCF Folder prefix -- Data
    # path_prefix_data = 'SDSS/{0}/{1}/Mr{2}/MCF_results/'.format(
    #                     'data',
    #                     param_dict['catl_type'],
    #                     param_dict['sample'   ])
    ### Pickle directory (results for `data`)
    # pickle_data = '{0}/processed/{1}/catl_pickle_files/{2}/{3}/'.format(
    #                 proj_dict['data_dir']  ,
    #                 path_prefix_data       ,
    #                 param_dict['corr_type'],
    #                 param_dict['param_str'])
    ###
    ### Figure out directory
    # figure_dir  = '{0}/SDSS/{1}/{2}/Mr{3}/conformity_output/'
    # figure_dir += 'MCF_figures/{4}/{5}'
    # figure_dir  = figure_dir.format(*[  proj_dict['plot_dir'],
    #                                     param_dict['catl_kind'],
    #                                     param_dict['catl_type'],
    #                                     param_dict['sample'],
    #                                     param_dict['corr_type'],
    #                                     param_dict['param_str'] ])
    ## Figure - Paper directory
    # fig_paper_dir = '{0}/SDSS/Paper_Figures/'.format(proj_dict['plot_dir'])
    ##
    ## Checking if `pickle_res` exists
    if os.path.exists(pickle_res):
        pass
    else:
        msg = '{0} `pickle_res` ({1}) does not exist! Exiting!!'.format(
            param_dict['Prog_msg'], pickle_res)
        raise ValueError(msg)
    ##
    ## Checking if `pickle_data` exists
    if os.path.exists(pickle_data):
        pass
    else:
        msg = '{0} `pickle_data` ({1}) does not exist! Exiting!!'.format(
            param_dict['Prog_msg'], pickle_data)
        raise ValueError(msg)
    ##
    ## Creating `figure_dir` folder
    cu.Path_Folder(figure_dir)
    cu.Path_Folder(fig_paper_dir)
    ##
    ## Adding to main dictionary `proj_dict`
    proj_dict['pickle_res'   ] = pickle_res
    proj_dict['pickle_data'  ] = pickle_data
    proj_dict['figure_dir'   ] = figure_dir
    proj_dict['fig_paper_dir'] = fig_paper_dir

    return proj_dict

## --------- DATA EXTRACTION ------------##

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

def mgroup_keys_lim(keys_arr, sep='_'):
    """
    Convers the list of string sto array of new strings with the 
    upper and lower limits of the masses.

    Parameters
    -----------
    keys_arr: numpy.ndarray
        array of keys for the different group mass bins

    sep: string
        character used for separating group bin edges

    Returns
    -----------
    gm_lims: array-like, shape (2,1)
        array of the minimum and maximum mass of the array
        Shape: [gm_min, gm_max]
    """
    ## New array for storing the `GM` keys
    keys_new = [[] for x in range(2*len(keys_arr))]
    ## Looping over all `GM` keys
    ii = 0
    for zz, mg_ii in enumerate(keys_arr):
        keys_new[ii],\
        keys_new[ii+1] = num.asarray(mg_ii.split(sep),dtype=float)
        ii += 2
    ## Obtaining minimum and maximum masses for the catalogue
    keys_new = num.unique(keys_new)[[0,-1]]

    return keys_new

def mgroup_bins_create(mgroup_lims, param_dict):
    """
    Creates the bins from all the catalogue(s)

    Parameters
    -----------
    mgroup_lims: array_like, shape (ncatls, 2)
        array of mass limits for the 1st and last mass bin

    param_dict: python dictionary
        dictionary with `project` variables

    Return
    -----------
    param_dict: python dictionary
        dictionary with 'updated' `project` variables
    """
    ## Transpoing `mgroup_lims` to obtain array of upper and lower limits
    mgroup_lims_vals = num.min(mgroup_lims.T, axis=1)
    ## Bin edges for all catalogues
    mgroup_bins = num.arange(   mgroup_lims_vals[0],
                                mgroup_lims_vals[1]+.5*param_dict['Mg_bin'],
                                param_dict['Mg_bin'])
    ##
    ## Bin edges keys in the form of the original keys
    mgroup_keys = ['{0:.2f}_{1:.2f}'.format(mgroup_bins[xx],mgroup_bins[xx+1])\
                    for xx in range(len(mgroup_bins)-1)]
    ##
    ## Labels for Plotting for each mass bin
    mgroup_labels = ['{0:.1f} - {1:.1f}'.format(
        *num.array(xx.split('_')).astype(float)) for xx in mgroup_keys]
    ##
    ## Indices for the DataFrame
    mgroup_idx    = [xx.replace('.','_') for xx in mgroup_keys]
    ##
    ## Creating dictionary that contains the label for each mass bin
    mgroup_dict = dict(zip(mgroup_keys, mgroup_labels))
    ##
    ## Adding to `param_dict`
    param_dict['mgroup_dict'] = mgroup_dict
    param_dict['mgroup_bins'] = mgroup_bins
    param_dict['mgroup_keys'] = mgroup_keys

    return param_dict

## --------- DATA ------------##

def data_shuffles_extraction(param_dict, proj_dict, pickle_ext='.p'):
    """
    Extracts the data from the `data` and `shuffles`.
    Functions used when `catl_kind=='data'`

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    pickle_ext: string, optional (default = '.p')
        extension of the pickle files to read

    Returns
    ---------
    prop_catl_dict: python dictionary
        dictionary with all the results from the 1-halo MCF for `data`
    """
    ## Reading in Catalogue
    catl_arr = cu.Index(proj_dict['pickle_res'], pickle_ext)
    ncatls   = len(catl_arr)
    ## Choosing catalogue
    if ncatls==1:
        catl_path = catl_arr[0]
    else:
        msg = '{0} `catl_arr` ({1}) has more than 1 catalogue in it! '
        msg += 'It can only contain 1 catalogue!! Exiting!'
        msg  = msg.format(  param_dict['Prog_msg'],
                            proj_dict['pickle_res'],
                            ncatls)
        raise ValueError(msg)
    ## Opening pickle file
    (   param_dict_kk,
        proj_dict_kk ,
        GM_prop_dict ,
        catl_name    ,
        GM_arr       ) = pickle.load(open(catl_path,'rb'))
    ##
    ## Reading array of masses
    mgroup_lims = [[[],[]] for x in range(ncatls)]
    for ii in range(ncatls): 
        mgroup_lims[ii] = mgroup_keys_lim(GM_prop_dict.keys())
    mgroup_lims = num.array(mgroup_lims)
    ##
    ## Creating mass bins
    param_dict = mgroup_bins_create(mgroup_lims, param_dict)
    ##
    ## Sving number of `mgroup` keys
    param_dict['n_mgroup' ] = len(param_dict['mgroup_keys'])
    ##
    ## Determining 'galaxy properties' list
    prop_keys = num.sort(
        list(GM_prop_dict[param_dict['mgroup_keys'][0]].keys()))
    n_prop    = len(prop_keys)
    ##
    ## Saving to `param_dict`
    param_dict['prop_keys'] = prop_keys
    param_dict['n_prop'   ] = n_prop
    ##
    ## Parsing the data into dictionaries
    prop_keys_tot = dict(zip(param_dict['prop_keys'],
                            [{} for xx in range(n_prop)]))
    mgroup_keys_dict = dict(zip(param_dict['mgroup_keys'],
                                [copy.deepcopy(prop_keys_tot) \
                                for xx in range(param_dict['n_mgroup' ])]))
    prop_catl_dict = copy.deepcopy(mgroup_keys_dict)
    ##
    ### Restructuring the data
    ## Looping over group mass bins
    for gm in param_dict['mgroup_keys']:
        ## Looping over `galaxy properties`
        for prop in param_dict['prop_keys']:
            ## Extracting the data from main dictionary
            mcf_dict_conf    ,\
            mcf_dict_conf_seg,\
            ngroups           = GM_prop_dict[gm][prop]
            ##
            ## Extracting MCFs
            # 'Conf Only'
            mcf_conf        = mcf_dict_conf['mcf'   ]
            mcf_conf_sh     = mcf_dict_conf['mcf_sh']
            # 'Conf + Seg'
            mcf_conf_seg    = mcf_dict_conf_seg['mcf'   ]
            mcf_conf_seg_sh = mcf_dict_conf_seg['mcf_sh']
            ##
            ## Errors, mean, and St. Dev.
            # 'Conf Only'
            (   sigma_conf_dict,
                conf_mean      ,
                conf_std       ) = sigma_calcs(mcf_conf_sh,
                                        type_sigma=param_dict['type_sigma'],
                                        return_mean_std=True)
            # 'Conf + Seg'
            (   sigma_conf_seg_dict,
                conf_seg_mean      ,
                conf_seg_std       ) = sigma_calcs(mcf_conf_seg_sh,
                                        type_sigma=param_dict['type_sigma'],
                                        return_mean_std=True)
            ##
            ## Fractional Difference
            # 'Conf. Only'
            mcf_conf_frac = (mcf_conf - conf_mean)/conf_std
            # 'Conf. + Seg.'
            mcf_conf_seg_frac = (mcf_conf_seg - conf_seg_mean)/conf_seg_std
            ## Saving data to restructured dictionary `prop_catl_dict`
            prop_catl_dict[gm][prop]['mcf_conf'    ] = mcf_conf
            prop_catl_dict[gm][prop]['mcf_conf_seg'] = mcf_conf_seg
            prop_catl_dict[gm][prop]['mcf_conf_sig'] = sigma_conf_dict
            prop_catl_dict[gm][prop]['conf_res'    ] = mcf_conf_frac
            prop_catl_dict[gm][prop]['conf_seg_res'] = mcf_conf_seg_frac

    return prop_catl_dict

## --------- Mocks ------------##

def mocks_data_extraction(param_dict, proj_dict, pickle_ext='.p'):
    """
    Extracts the data from the `mocks` and `data`.
    Functions used when `catl_kind=='mocks'`

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    pickle_ext: string, optional (default = '.p')
        extension of the pickle files to read

    Returns
    ---------
    prop_catl_dict_stats: python dictionary
        dictionary with all the results from the 1-halo MCF for `mocks` and 
        `data`
    """
    ## -------------- ##
    ## ---- DATA ---- ##
    ## -------------- ##
    catl_arr_data = cu.Index(proj_dict['pickle_data'], pickle_ext)
    ncatls_data   = len(catl_arr_data)
    ## Choosing catalogue
    if ncatls_data==1:
        catl_data_path = catl_arr_data[0]
    else:
        msg = '{0} `catl_arr_data` ({1}) has more than 1 catalogue in it! '
        msg += 'It can only contain 1 catalogue!! Exiting!'
        msg  = msg.format(  param_dict['Prog_msg'],
                            proj_dict['pickle_data'],
                            ncatls_data)
        raise ValueError(msg)
    ##
    ## Opening `data` pickle file
    (   param_dict_kk_data,
        proj_dict_kk_data ,
        GM_prop_dict_data ,
        catl_name_data    ,
        GM_arr_data       ) = pickle.load(open(catl_data_path,'rb'))
    ##
    ## Reading array of masses
    mgroup_lims_data = [[[],[]] for x in range(ncatls_data)]
    for ii in range(ncatls_data):
        mgroup_lims_data[ii] = mgroup_keys_lim(GM_prop_dict_data.keys())
    mgroup_lims_data = num.array(mgroup_lims_data)
    ##
    ## Creating mass bins
    param_dict_data = param_dict.copy()
    param_dict_data = mgroup_bins_create(mgroup_lims_data, param_dict_data)
    ##
    ## Saving number of `mgroup` keys
    param_dict_data['n_mgroup'] = len(param_dict_data['mgroup_keys'])
    ##
    ## Determining `gaalxy properties` list
    prop_keys_data = num.sort(
        list(GM_prop_dict_data[param_dict_data['mgroup_keys'][0]].keys()))
    n_prop_data = len(prop_keys_data)
    ##
    ## Saving to `param_dict_data`
    param_dict_data['prop_keys_data'] = prop_keys_data
    param_dict_data['n_prop_data'   ] = n_prop_data
    ##
    ## Parsing the data into dictionaries - `data`
    prop_keys_data_tot = dict(zip(param_dict_data['prop_keys_data'],
        [{} for xx in range(n_prop_data)]))
    mgroup_keys_data_dict = dict(zip(param_dict_data['mgroup_keys'],
                                [copy.deepcopy(prop_keys_data_tot) \
                                for xx in range(param_dict_data['n_mgroup'])]))
    prop_catl_data_dict = copy.deepcopy(mgroup_keys_data_dict)
    ##
    ## Restructuring the data for `data`
    ## Looping over group mass bins and galaxy properties
    for gm in param_dict_data['mgroup_keys']:
        ## Looping over `galaxy properties`
        for prop in param_dict_data['prop_keys_data']:
            ## Extracting the data from main dictionary
            mcf_dict_data_conf,\
            mcf_dict_data_conf_seg,\
            n_groups_data = GM_prop_dict_data[gm][prop]
            ## Saving data to restructured dictionary `prop_catl_data_dict`
            prop_catl_data_dict[gm][prop]['mcf_conf'    ] = mcf_dict_data_conf    ['mcf']
            prop_catl_data_dict[gm][prop]['mcf_conf_seg'] = mcf_dict_data_conf_seg['mcf']
    ## --------------- ##
    ## ---- MOCKS ---- ##
    ## --------------- ##
    catl_arr_mocks = cu.Index(proj_dict['pickle_res'], pickle_ext)
    catl_arr_mocks = catl_arr_mocks[param_dict['catl_start']:param_dict['catl_finish']]
    ncatls_mocks   = len(catl_arr_mocks)
    ##
    ## Reading in pickle files
    mgroup_lims_mocks = [[] for x in range(ncatls_mocks)]
    GM_prop_dict_arr  = [[] for x in range(ncatls_mocks)]
    ## Looping over all mock catalogues
    for ii, mock_ii in enumerate(catl_arr_mocks):
        ## Opening up pickle file
        (   param_dict_kk  ,
            proj_dict_kk   ,
            GM_prop_dict_kk,
            catl_name_kk   ,
            GM_arr_kk      ) = pickle.load(open(mock_ii,'rb'))
        ## Mass limits for `mock_ii`
        mgroup_lims_mocks[ii] = mgroup_keys_lim(GM_prop_dict_kk.keys())
        ## Saving MCF results from pickle file
        GM_prop_dict_arr[ii] = GM_prop_dict_kk
    ##
    ## Determining mass limits for each mock catalogue
    mgroup_lims_mocks = num.array(mgroup_lims_mocks)
    ##
    ## Creating mass bins
    param_dict = mgroup_bins_create(mgroup_lims_mocks, param_dict)
    ##
    ## Determining common `mass bin` keys and `prop` in common`
    mgroup_keys_intersect = num.intersect1d(param_dict     ['mgroup_keys'],
                                            param_dict_data['mgroup_keys'])
    param_dict['mgroup_keys'] = mgroup_keys_intersect
    ##
    ## Saving number of `mgroup` keys
    param_dict['n_mgroup'] = len(param_dict['mgroup_keys'])
    ##
    ## Determining `galaxy properties` list
    prop_keys = num.sort(
        list(GM_prop_dict_arr[0][param_dict['mgroup_keys'][0]].keys()))
    ##
    ## Common keys for the `galaxy property`
    prop_keys_intersect = num.intersect1d(prop_keys, prop_keys_data)
    n_prop              = len(prop_keys_intersect)
    ##
    ## Saving to `param_dict`
    param_dict['prop_keys'] = prop_keys
    param_dict['n_prop'   ] = n_prop
    ##
    ## Parsing the data into dictionaries
    zero_arr      = num.zeros((param_dict['nrpbins'],1))
    mcf_dicts     = {'mcf':zero_arr.copy(), 'mcf_seg':zero_arr.copy()}
    prop_keys_tot = dict(zip(param_dict['prop_keys'],
                        [copy.deepcopy(mcf_dicts) for xx in range(n_prop)]))
    mgroup_keys_dict = dict(zip(param_dict['mgroup_keys'],
                                [copy.deepcopy(prop_keys_tot) \
                                for xx in range(param_dict['n_mgroup' ])]))
    prop_catl_dict = copy.deepcopy(mgroup_keys_dict)
    ##
    ## Restructuring the data
    ## Looping over mock catalogues
    for ii, gm_prop_ii in enumerate(GM_prop_dict_arr):
        ## Looping over group mass bins
        for gm in param_dict['mgroup_keys']:
            ## Looping over `galaxy properties`
            for prop in param_dict['prop_keys']:
                ## Extracting the data from main dictionary
                (   mcf_dict_conf    ,
                    mcf_dict_conf_seg,
                    ngroups          ) = gm_prop_ii[gm][prop]
                ##
                ## MCF for 'Conf Only' and 'Conf + Seg'
                mcf_conf     = mcf_dict_conf    ['mcf']
                mcf_conf_seg = mcf_dict_conf_seg['mcf']
                ##
                ## Appending to `prop_catl_dict`
                # 'Conformity Only'
                prop_catl_dict[gm][prop]['mcf'] = array_insert(
                    prop_catl_dict[gm][prop]['mcf'],
                    mcf_conf, axis=1)
                # 'Conformity + Segregation'
                prop_catl_dict[gm][prop]['mcf_seg'] = array_insert(
                    prop_catl_dict[gm][prop]['mcf_seg'],
                    mcf_conf_seg, axis=1)
    ##
    ## Statistics of `prop_catl_dict`
    prop_keys_tot_stats = dict(zip(param_dict['prop_keys'],
                                [{} for xx in range(n_prop)]))
    mgroup_keys_dict_stats = dict(zip(param_dict['mgroup_keys'],
                                [copy.deepcopy(prop_keys_tot_stats) \
                                for xx in range(param_dict['n_mgroup' ])]))
    prop_catl_dict_stats = copy.deepcopy(mgroup_keys_dict_stats)
    ##
    ## Looping over group mass bins
    for gm in param_dict['mgroup_keys']:
        ## Looping over `galaxy properties`
        for prop in param_dict['prop_keys']:
            ## Deleting 1st row of zeros
            mcf_mocks     = num.delete(prop_catl_dict[gm][prop]['mcf'],
                                0, axis=1)
            mcf_seg_mocks = num.delete(prop_catl_dict[gm][prop]['mcf_seg'],
                                0,axis=1)
            ##
            ## Statistics: Sigmas, Means, and St. Dev.
            ## Errors, mean and St. Dev.
            # 'Conf Only'
            (   sigma_conf_dict,
                conf_mean      ,
                conf_std       ) = sigma_calcs(mcf_mocks,
                                        type_sigma=param_dict['type_sigma'],
                                        return_mean_std=True)
            # 'Conf + Seg'
            (   sigma_conf_seg_dict,
                conf_seg_mean      ,
                conf_seg_std       ) = sigma_calcs(mcf_seg_mocks,
                                        type_sigma=param_dict['type_sigma'],
                                        return_mean_std=True)
            ##
            ## Defining `mcf` from `data`
            mcf_conf_data     = prop_catl_data_dict[gm][prop]['mcf_conf']
            mcf_conf_seg_data = prop_catl_data_dict[gm][prop]['mcf_conf_seg']
            ##
            ## Fractional Differences
            # 'Conf Only'
            mcf_conf_frac     = (mcf_conf_data - conf_mean)/conf_std
            # 'Conf + Seg'
            mcf_conf_seg_frac = (mcf_conf_seg_data - conf_seg_mean)/conf_seg_std
            ## Saving to `prop_catl_dict_stats`
            prop_catl_dict_stats[gm][prop]['mcf_conf'    ] = mcf_conf_data
            prop_catl_dict_stats[gm][prop]['mcf_conf_seg'] = mcf_conf_seg_data
            prop_catl_dict_stats[gm][prop]['mcf_conf_sig'] = sigma_conf_dict
            prop_catl_dict_stats[gm][prop]['conf_res'    ] = mcf_conf_frac
            prop_catl_dict_stats[gm][prop]['conf_seg_res'] = mcf_conf_seg_frac

    return prop_catl_dict_stats

## --------- Plotting ------------##

def MCF_one_halo_plotting(prop_catl_dict, param_dict, proj_dict, fig_fmt='pdf',
    figsize_1=(4.,15.5), figsize_2=(10,15.5)):
    """
    Funtion to plot the MCF for `data` for the given group mass bins

    Parameters
    ----------
    prop_catl_dict: python dictionary
        dictionary with the necessary information to plot MCF for 
        `Conformity Only` and `Conformity + Segregation`

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    fig_fmt: string, optional (default = 'pdf')
        extension used to save the figure

    figsize: tuple, optional (10, 15,5)
        size of the output figure
    """
    Prog_msg = param_dict['Prog_msg']
    ## Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    ##
    ## Labels
    xlabel       = r'\boldmath $r_{p}\ \left[h^{-1}\ \textrm{Mpc} \right]$'
    ylabel       = r'\boldmath $\mathcal{M}(r_{p})$'
    sigma_ylabel = 'Res.'
    ## Text properties
    alpha_arr      = [0.7, 0.5, 0.3]
    color_sig      = 'red'
    color_prop     = 'black'
    color_prop_seg = 'dimgrey'
    ## Figure name
    fname_prefix = ('MCF_{0}_{1}_clf_{2}'.format(   param_dict['catl_kind'],
                                            param_dict['fig_prefix'],
                                            param_dict['clf_method'])
                                            ).replace('.', 'p')+'.'+fig_fmt
    fname = proj_dict['figure_dir']+fname_prefix
    ##
    ## Labels for Galaxy Properties
    # Choosing label for `ssfr`
    if param_dict['prop_log']=='log':
        ssfr_label = r'\boldmath $\log\ \textrm{ssfr}$'
    elif param_dict['prop_log']=='nonlog':
        ssfr_label = r'\boldmath $\textrm{ssfr}$'
    # Dictionary
    prop_labels = { 'g_r':r'\boldmath$g-r$',
                    'sersic':'sersic',
                    'logssfr':ssfr_label}
    n_prop = param_dict['n_prop']
    ##
    ## Type of galaxy pairs - Dictionary
    corr_pair_dict = {'cen_sat':'Cens - Sats', 'sat_sat':'Sats - Sats',
    'all':'All Pairs', 'cen_cen':'Cens - Cens'}
    corr_pair_str = corr_pair_dict[param_dict['corr_pair_type']]
    ##
    ## Group mass limits for Plotting
    Mg_lims = [param_dict['mg_min'],param_dict['mg_max']]
    Mg_keys = cu.Bins_array_create(Mg_lims, param_dict['Mg_bin'])
    Mg_keys_str = ['{0:.2f}_{1:.2f}'.format(Mg_keys[xx], Mg_keys[xx+1]) \
        for xx in range(len(Mg_keys)-1)]
    Mg_keys_str = num.sort( num.array(Mg_keys_str) )
    Mg_keys_str = num.intersect1d(Mg_keys_str, param_dict['mgroup_keys'])
    n_Mgroup    = len(Mg_keys_str)
    ##
    ## Group or Halo mass - String
    if param_dict['perf_opt']:
        mg_str = r'$M_{\textrm{halo}}$'
    else:
        mg_str = r'$M_{\textrm{group}}$'
    ##
    ## Shaded regions label
    if param_dict['catl_kind']=='data':
        shaded_str = 'Shuffles'
    elif param_dict['catl_kind']=='mocks':
        shaded_str = 'Mocks'
    ##
    ## Colormaps
    cm_dict = {'logssfr':'red', 'sersic':'royalblue', 'g_r':'green'}
    ##
    ## Figure Details
    ncols = int(param_dict['n_prop'])
    nrows = len(Mg_keys_str)
    # Choosing Figure size and fontsize
    if ncols == 1:
        figsize     = figsize_1
        size_label  = 17
        size_legend = 11
        size_text   = 14
    else:
        figsize     = figsize_2
        size_label  = 20
        size_legend = 10
        size_text   = 14
    # Dashes formatting
    dashes      = (5,5)
    # Initializing figure
    plt.clf()
    plt.close()
    propssfr = dict(boxstyle='round', facecolor='white', alpha=0.7)
    fig = plt.figure(figsize=figsize)
    gs_prop = gridspec.GridSpec(nrows, ncols, hspace=0.05, wspace=0.1)
    ## Plotting MCF for each galaxy property and group mass bin
    gs_ii = int(0)
    # Looping over Group mass bins
    for ii, gm in enumerate(Mg_keys_str):
        gm_str = param_dict['mgroup_dict'][gm]
        # Looping over galaxy properties
        for jj, prop in enumerate(param_dict['prop_keys']):
            ## Creating axis for element in `gs_prop`
            nrows_ax = int(2)
            ncols_ax = int(1)
            gs_prop_axes = gridspec.GridSpecFromSubplotSpec(nrows_ax, ncols_ax, 
                gs_prop[gs_ii], height_ratios=[2,1], hspace=0 )
            ##
            ## Defining axes for `gs_prop_axes`
            ax_data  = plt.Subplot(fig, gs_prop_axes[0,:])
            ax_sigma = plt.Subplot(fig, gs_prop_axes[1,:], sharex=ax_data)
            fig.add_subplot(ax_data)
            fig.add_subplot(ax_sigma)
            ax_data.set_facecolor('white')
            ax_sigma.set_facecolor('white')
            ## Galaxy Property - Color
            color_sh = cm_dict[prop]
            ##
            ## Determining `rp` axes limits
            mcf_lim = num.isfinite(prop_catl_dict[gm][prop]['mcf_conf'])
            rp_idx  = param_dict['rpbins_cens_unlog'][mcf_lim]
            rp_lim  = [rp_idx.min(),rp_idx.max()]
            ##
            ## Plotting in `ax-data`
            # Sigmas
            for zz in range(3):
                if zz == 0:
                    ax_data.fill_between(
                        param_dict['rpbins_cens_unlog'][mcf_lim],
                        prop_catl_dict[gm][prop]['mcf_conf_sig'][zz][0][mcf_lim],
                        prop_catl_dict[gm][prop]['mcf_conf_sig'][zz][1][mcf_lim],
                        facecolor=color_sh,
                        alpha=alpha_arr[zz],
                        zorder=zz+1,
                        label=shaded_str)
                else:
                    ax_data.fill_between(
                        param_dict['rpbins_cens_unlog'][mcf_lim],
                        prop_catl_dict[gm][prop]['mcf_conf_sig'][zz][0][mcf_lim],
                        prop_catl_dict[gm][prop]['mcf_conf_sig'][zz][1][mcf_lim],
                        facecolor=color_sh,
                        alpha=alpha_arr[zz],
                        zorder=zz+1)
            # MCF - Conformity Only
            if (ii==0):
                ax_data.plot(
                    param_dict['rpbins_cens_unlog'][mcf_lim],
                    prop_catl_dict[gm][prop]['mcf_conf'][mcf_lim],
                    color=color_prop,
                    marker='o',
                    linestyle='-',
                    zorder=4,
                    label = 'SDSS - Conf. Only')
            else:
                ax_data.plot(
                    param_dict['rpbins_cens_unlog'][mcf_lim],
                    prop_catl_dict[gm][prop]['mcf_conf'][mcf_lim],
                    color=color_prop,
                    marker='o',
                    linestyle='-',
                    zorder=4)
            #
            # MCF - Conformity + Segregation
            if (ii==0):
                ax_data.plot(
                    param_dict['rpbins_cens_unlog'][mcf_lim],
                    prop_catl_dict[gm][prop]['mcf_conf_seg'][mcf_lim],
                    color=color_prop_seg,
                    marker='o',
                    linestyle='--',
                    zorder=4,
                    label = 'SDSS - Conf + Seg',
                    dashes=dashes)
            else:
                ax_data.plot(
                    param_dict['rpbins_cens_unlog'][mcf_lim],
                    prop_catl_dict[gm][prop]['mcf_conf_seg'][mcf_lim],
                    color=color_prop_seg,
                    marker='o',
                    linestyle='--',
                    zorder=4,
                    dashes=dashes)
            ##
            ## Plotting in `ax_sigma`
            ## MCF - Residuals - Conformity Only
            if (ii==0):
                ax_sigma.plot(
                    param_dict['rpbins_cens_unlog'][mcf_lim],
                    prop_catl_dict[gm][prop]['conf_res'][mcf_lim],
                    color=color_prop,
                    marker='o',
                    linestyle='-',
                    zorder=4,
                    label = 'SDSS - Conf. Only')
            else:
                ax_sigma.plot(
                    param_dict['rpbins_cens_unlog'][mcf_lim],
                    prop_catl_dict[gm][prop]['conf_res'][mcf_lim],
                    color=color_prop,
                    marker='o',
                    linestyle='-',
                    zorder=4)
            ##
            ## MCF - Residuals - Conformity + Segregation
            if (ii==0):
                ax_sigma.plot(
                    param_dict['rpbins_cens_unlog'][mcf_lim],
                    prop_catl_dict[gm][prop]['conf_seg_res'][mcf_lim],
                    color=color_prop_seg,
                    marker='o',
                    linestyle='--',
                    zorder=4,
                    label = 'SDSS - Conf + Seg')
            else:
                ax_sigma.plot(
                    param_dict['rpbins_cens_unlog'][mcf_lim],
                    prop_catl_dict[gm][prop]['conf_seg_res'][mcf_lim],
                    color=color_prop_seg,
                    marker='o',
                    linestyle='--',
                    zorder=4)
            ##
            ## Sigma Lines in `ax_sigma`
            for zz in reversed(range(3)):
                ax_sigma.fill_between(
                    num.linspace( rp_lim[0], rp_lim[1], 10),
                    (zz+1)*num.ones(10),
                    -(zz+1)*num.ones(10),
                    facecolor=color_sh,
                    alpha=alpha_arr[zz],
                    zorder=1)
            ##
            ## Extra options
            # Hiding 'y-axis' tickmarks
            if n_prop!=1:
                if (jj != 0):
                    plt.setp(ax_data.get_yticklabels(), visible=False)
                    plt.setp(ax_sigma.get_yticklabels(), visible=False)
            ##
            ## Axes labels
            if n_prop==1:
                ax_data.set_ylabel(ylabel, fontsize=size_label )
                ax_sigma.set_ylabel(sigma_ylabel, fontsize=size_label )
            else:
                if (jj == 0):
                    ax_data.set_ylabel(ylabel, fontsize=size_label )
                    ax_sigma.set_ylabel(sigma_ylabel, fontsize=size_label )
            ##
            ## Hiding `x-ticks` for `ax_data`
            plt.setp(ax_data.get_xticklabels(), visible=False)
            #
            # Showing `xlabel` when necessary
            if gs_ii in range(n_prop * (nrows-1), ncols*nrows):
                ax_sigma.set_xlabel( xlabel, fontsize=size_label)
            else:
                plt.setp(ax_sigma.get_xticklabels(), visible=False)
            ##
            ## Adding text labels
            if (gs_ii == 0):
                ax_data.text(0.05, 0.30, corr_pair_str, 
                    transform=ax_data.transAxes,
                    verticalalignment='top', color='black',
                    bbox=propssfr, weight='bold', fontsize=size_text)
            ##
            ## Galaxy Property - label
            if (ii == 0):
                ax_data.text(0.05, 0.95, prop_labels[prop],
                    transform=ax_data.transAxes,
                    verticalalignment='top', color=color_sh,
                    bbox=propssfr, weight='bold', fontsize=size_text)
            ##
            ## Group mass - label
            if (jj == 0):
                ax_data.text(0.05, 0.15, gm_str,
                    transform=ax_data.transAxes,
                    verticalalignment='top', color='#BE0081',
                    bbox=propssfr, weight='bold', fontsize=size_text)
            ##
            ## Changing scales to 'log'
            ax_data.set_xscale('log')
            ax_sigma.set_xscale('log')
            ##
            ## Sigma Lines
            med_line_color = 'black'
            med_linewidth  = 1
            med_linestyle  = '--'
            med_yline      = 1
            ax_data.axhline(
                y=med_yline, 
                linestyle=med_linestyle, 
                color=med_line_color, 
                linewidth=med_linewidth,
                zorder=4,
                dashes=dashes)
            # Color sigma - Lines
            ax_sigma.axhline(
                y=0, 
                linestyle=med_linestyle,
                color=med_line_color, 
                linewidth=med_linewidth,
                zorder=4,
                dashes=dashes)
            ##
            ## Sigma Lines - `ax_sigma` axis
            shade_color     = 'grey'
            sigma_lines_arr = num.arange(5, 10.1, 5)
            for sig in sigma_lines_arr:
                ax_sigma.axhline(y = sig, linestyle='--', color=shade_color,
                    zorder=0, dashes=dashes, linewidth=med_linewidth)
                ax_sigma.axhline(y = -sig, linestyle='--', color=shade_color,
                    zorder=0, dashes=dashes, linewidth=med_linewidth)
            ##
            ## Tickmarks
            if param_dict['catl_kind']=='data':
                ##
                ## y-axis limits
                ylim_data      = [0.8, 1.25]
                ylim_sigma     = [-10, 9.8]
                ##
                ## Tickmarks
                ax_data_major  = 0.1
                ax_data_minor  = 0.05
                ax_sigma_major = 5.
                ax_sigma_minor = 1.
            elif param_dict['catl_kind']=='mocks':
                ## y-axis limits
                ylim_data      = [0.9, 1.1]
                ylim_sigma     = [-5, 9.8]
                ## Tickmarks
                ax_data_major  = 0.05
                ax_data_minor  = 0.01
                ax_sigma_major = 5.
                ax_sigma_minor = 1.
            ##
            ## Axes limits and sigma lines
            if (param_dict['rpmin'] == 0.05) and (param_dict['rpmax']==10):
                xlim_data  = [0.9*param_dict['rpmin'], 4.]
            else:    
                xlim_data  = [0.9*param_dict['rpmin'], 1.1*param_dict['rpmax']]
            ax_data.set_xlim(xlim_data)
            ax_data.set_ylim(ylim_data)
            ax_sigma.set_ylim(ylim_sigma)
            #
            # Major and Minor locators
            ax_data_major_loc  = ticker.MultipleLocator(ax_data_major)
            ax_data_minor_loc  = ticker.MultipleLocator(ax_data_minor)
            ax_sigma_major_loc = ticker.MultipleLocator(ax_sigma_major)
            ax_sigma_minor_loc = ticker.MultipleLocator(ax_sigma_minor)
            #
            # Setting minor and major in axes
            ax_data.yaxis.set_major_locator(ax_data_major_loc)
            ax_data.yaxis.set_minor_locator(ax_data_minor_loc)
            ax_sigma.yaxis.set_major_locator(ax_sigma_major_loc)
            ax_sigma.yaxis.set_minor_locator(ax_sigma_minor_loc)
            # Making the format of the y-axis scalar
            ax_sigma.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda y,pos: ('{{:.{:1d}f}}'.format(
                        int(num.maximum(-num.log10(y),0)))).format(y)))
            ##
            ## Legend
            if (gs_ii==0):
                leg = ax_data.legend(loc='upper right',
                    prop={'size':size_legend}) 
                leg_frame = leg.get_frame()
                leg_frame.set_facecolor('white')
            ##
            ## Adding ticks to both sides of the y-axis
            ax_data.yaxis.set_ticks_position('both')
            ax_sigma.yaxis.set_ticks_position('both')
            ##
            ## Increasing `gs_ii` by 1
            gs_ii += int(1)
    ##
    ## Saving figure
    if fig_fmt=='pdf':
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
    plt.clf()
    plt.close()
    ##
    ## Copying figure to `fig_paper_dir` path and renaming file
    if param_dict['catl_kind']=='data':
        fname_new = 'Fig3_1_halo_mcf_data.{0}'.format(fig_fmt)
    elif param_dict['catl_kind']=='mocks':
        fname_new = 'Fig4_1_halo_mcf_mocks.{0}'.format(fig_fmt)
    ## Executing commands
    cmd  = '\ncp {0} {1} ; '.format(fname, proj_dict['fig_paper_dir'])
    cmd += '\n\nmv {0}/{1} {0}/{2};'.format( proj_dict['fig_paper_dir'],
                                        fname_prefix,
                                        fname_new)
    if param_dict['verbose']:
        print(cmd)
    subprocess.call(cmd, shell=True)

## --------- Main Function ------------##

def main():
    """
    Produces the plots for the 1-halo MCF conformity results
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    ## ---- Adding to `param_dict` ---- 
    param_dict = add_to_dict(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ##
    ## Creating Folder Structure
    proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    if param_dict['verbose']:
        print('\n'+50*'='+'\n')
        for key, key_val in sorted(param_dict.items()):
            if key !='Prog_msg':
                print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
        print('\n'+50*'='+'\n')
    ## Running the analysis
    # Choosing which kind of plots to produce
    if param_dict['catl_kind'] == 'data':
        ## Analyzing data
        prop_catl_dict = data_shuffles_extraction(param_dict, proj_dict)
    elif param_dict['catl_kind'] == 'mocks':
        ## Analyzing data
        prop_catl_dict = mocks_data_extraction(param_dict, proj_dict)
    ##
    ## Plotting 1-halo MCF
    MCF_one_halo_plotting(prop_catl_dict, param_dict, proj_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main()
