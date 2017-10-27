#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 10/24/2017
# Last Modified: 10/24/2017
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Computes the 1-halo `Quenched` fractions for SDSS DR7
"""
# Importing Modules
import custom_utilities_python as cu
import numpy as num
import math
import os
import sys
import pandas as pd
import pickle
#sns.set()
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import copy
from datetime import datetime
from multiprocessing import Pool, Process, cpu_count

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
    description_msg = 'Script to evaluate 1-halo conformity quenched fractions \
                        on SDSS DR7'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    # 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
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
    ## Type of correlation function to perform
    parser.add_argument('-corrtype',
                        dest='corr_type',
                        help='Type of correlation function to perform',
                        type=str,
                        choices=['galgal'],
                        default='galgal')
    ## Shuffling Marks
    parser.add_argument('-shuffle',
                        dest='shuffle_marks',
                        help='Option for shuffling marks of Cens. and Sats.',
                        choices=['cen_sh', 'sat_sh', 'censat_sh'],
                        default='censat_sh')
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help='Delete pickle file containing pair counts',
                        type=_str2bool,
                        default=False)
    ## Type of error estimation
    parser.add_argument('-sigma',
                        dest='type_sigma',
                        help='Type of error to use. Percentiles or St. Dev.',
                        type=str,
                        choices=['std','perc'],
                        default='std')
    ## Statistics for evaluating conformity
    parser.add_argument('-frac_stat',
                        dest='frac_stat',
                        help='Statistics to use to evaluate the conformity signal',
                        type=str,
                        choices=['diff', 'ratio'],
                        default='diff')
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.75)
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
    ### Sample - Int
    sample_s = str(param_dict['sample'])
    ### Perfect Catalogue
    if param_dict['perf_opt']:
        perf_str = 'haloperf'
    else:
        perf_str = ''
    ### Figure
    fig_idx = 24
    ### Survey Details
    sample_title = r'\boldmath$M_{r}< -%d$' %(param_dict['sample'])
    ## Project Details
    # String for Main directories
    param_str_arr = [   param_dict['catl_kind'] , param_dict['catl_type']    ,
                        param_dict['sample']    , param_dict['prop_log']     ,
                        param_dict['Mg_bin']    , param_dict['itern_tot']    ,
                        param_dict['ngals_min' ], param_dict['shuffle_marks'],
                        param_dict['type_sigma'], param_dict['frac_stat']    ,
                        perf_str  ]
    param_str_p  = 'kind_{0}_type_{1}_sample_{2}_proplog_{3}_Mgbin_{4}_'
    param_str_p += 'itern_{5}_nmin_{6}_sh_marks_{7}_type_sigma_{8}_'
    param_str_p += 'fracstat_{9}_perf_str_{10}'
    param_str    = param_str_p.format(*param_str_arr)
    ###
    ### To dictionary
    param_dict['sample_s'     ] = sample_s
    param_dict['perf_str'     ] = perf_str
    param_dict['fig_idx'      ] = fig_idx
    param_dict['sample_title' ] = sample_title
    param_dict['param_str'    ] = param_str
    
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
    ## Checking `cpu_frac` range
    if (param_dict['cpu_frac'] > 0) and (param_dict['cpu_frac'] <= 1):
        pass
    else:
        msg = '{0} `cpu_frac` ({1}) must be between (0,1]'.format(
            param_dict['Prog_msg'],
            param_dict['cpu_frac'])
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
    path_prefix = 'SDSS/{0}/{1}/Mr{2}/Frac_results'.format(
                        param_dict['catl_kind'],
                        param_dict['catl_type'],
                        param_dict['sample'   ])
    ### MCF Output directory - Results
    pickdir = '{0}/processed/{1}/{2}/catl_pickle_files/{3}/'.format(
                    proj_dict['data_dir'],
                    path_prefix          ,
                    param_dict['corr_type'],
                    param_dict['param_str'])
    # Creating Folders
    cu.Path_Folder(pickdir)
    ## Adding to `proj_dict`
    proj_dict['pickdir'   ] = pickdir

    return proj_dict

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

## --------- Analysis functions ------------##

def frac_prop_calc(df_bin_org, prop, param_dict, catl_keys_dict):
    """
    Computes the quenched fractions of satellites in a given mass bin.

    Parameters
    ----------
    df_bin_org: pandas DataFrame
        Dataframe for the selected group/halo mass bin

    prop: string
        galaxy property being evaluated
    
    param_dict: python dictionary
        dictionary with input parameters and values

    catl_keys_dict: python dictionary
        dictionary containing keys for the galaxy properties in catalogue

    Returns
    ----------
    """
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ## Constants
    Cens     = int(1)
    Sats     = int(0)
    itern    = param_dict['itern_tot']
    ## Catalogue Variables for galaxy properties
    gm_key      = catl_keys_dict['gm_key']
    id_key      = catl_keys_dict['id_key']
    galtype_key = catl_keys_dict['galtype_key']
    ## Group statistics
    groupid_unq = df_bin_org[id_key].unique()
    ngroups     = groupid_unq.shape[0]
    ##
    ## Selecting columns
    df_bin_mod = df_bin_org.copy()[[galtype_key, id_key, prop]]
    ##
    ## Normalizing `prop` by the `prop_lim`
    df_bin_mod.loc[:,prop] /= param_dict['prop_lim'][prop]
    ##
    ## Determining galaxy fractions for `df_bin_mod`
    cens_pd_org = df_bin_mod.loc[(df_bin_mod[galtype_key]==Cens)].copy().reset_index()
    sats_pd_org = df_bin_mod.loc[(df_bin_mod[galtype_key]==Sats)].copy().reset_index()
    ##
    ## Quench Satellite fraction
    sat_quenched_frac = frac_stat_calc( cens_pd_org   ,
                                        sats_pd_org   ,
                                        prop          ,
                                        catl_keys_dict,
                                        param_dict    ,
                                        frac_stat=param_dict['frac_stat'])
    ##
    ## Converting `sat_quenched_frac` to numpy array
    sat_quenched_frac = sat_quenched_frac
    ##
    ## Calculation fractions for Shuffles
    sat_quenched_frac_sh = num.zeros((param_dict['itern_tot'], 1))
    # ProgressBar properties
    if param_dict['prog_bar']:
        widgets   = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
        pbar_mock = ProgressBar( widgets=widgets, maxval= 10 * itern).start()
    ## Iterating `itern` times and calculating quenched fractions
    for ii in range(itern):
        ## Quenched fractions
        sat_quenched_frac_sh[ii] = frac_stat_calc(cens_pd_org ,
                                                sats_pd_org   ,
                                                prop          ,
                                                catl_keys_dict,
                                                param_dict    ,
                                                shuffle=True  ,
                                                frac_stat=param_dict['frac_stat'])
        if param_dict['prog_bar']:
            pbar_mock.update(ii)
    if param_dict['prog_bar']:
        pbar_mock.finish()

    return sat_quenched_frac, sat_quenched_frac_sh.T

def frac_stat_calc(cens_pd_org, sats_pd_org, prop, catl_keys_dict, param_dict, 
    frac_stat='diff', shuffle=False):
    """
    Computes quenched fractions of satellites for a given galaxy property 
    `prop` in a given mass bin.

    Parameters
    ----------
    cens_pd_org: pandas DataFrame
        dataframe with only central galaxies in the given group mass bin.
        Centrals belong to groups with galaxies >= `param_dict['ngals_min']`

    sats_pd_org: pandas DataFrame
        dataframe with only satellite galaxies in the given group mass bin.
        Satellites belong to groups with galaxies >= `param_dict['ngals_min']`

    prop: string
        galaxy property being analyzed

    catl_keys_dict: python dictionary
        dictionary containing keys for the galaxy properties in catalogue

    param_dict: python dictionary
        dictionary with input parameters and values

    frac_stat: string, optional (default = 'diff')
        statistics to use to evaluate the conformity signal
        Options:
            - 'diff' : Takes the difference between P(sat=q|cen=q) and 
                        P(sat=q|cen=act)
            - 'ratio': Takes the ratio between P(sat=q|cen=q) and 
                        P(sat=q|cen=act)

    shuffle: boolean, optional (default = False)
        option for shuffling the galaxies' properties among themselves, i.e. 
        centrals among centrals, and satellites among satellites.

    Returns
    -------
    frac_sat_pas_cen_act: float
        number of `passive` satellites around `active` centrals over the 
        total number of satelltes around `active` centrals

    frac_sat_pas_cen_pas: float
        number of `passive` satellites around `passive` centrals over the 
        total number of satelltes around `passive` centrals

    frac_stat: float
        'ratio' or 'difference' of between P(sat=q|cen=q) and P(sat=q|cen=act)

    """
    ## Keys for group/halo ID, mass, and galaxy type
    gm_key      = catl_keys_dict['gm_key']
    id_key      = catl_keys_dict['id_key']
    galtype_key = catl_keys_dict['galtype_key']
    ## Copies of `cens_pd_org` and `sats_pd_org`
    cens_pd = cens_pd_org.copy()
    sats_pd = sats_pd_org.copy()
    ## Choosing if to shuffle `prop`
    if shuffle:
        ## Choosing which kind of shuffle to use
        # Shuffling only Centrals
        if param_dict['shuffle_marks'] == 'cen_sh':
            cens_prop_sh = cens_pd[prop].copy().values
            num.random.shuffle(cens_prop_sh)
            cens_pd.loc[:,prop] = cens_prop_sh
        # Shuffling only Satellites
        if param_dict['shuffle_marks'] == 'sat_sh':
            sats_prop_sh = sats_pd[prop].copy().values
            num.random.shuffle(sats_prop_sh)
            sats_pd.loc[:,prop] = sats_prop_sh
        # Shuffling both Centrals and Satellites
        if param_dict['shuffle_marks'] == 'censat_sh':
            # Centrals
            cens_prop_sh = cens_pd[prop].copy().values
            num.random.shuffle(cens_prop_sh)
            cens_pd.loc[:,prop] = cens_prop_sh
            # Satellites
            sats_prop_sh = sats_pd[prop].copy().values
            num.random.shuffle(sats_prop_sh)
            sats_pd.loc[:,prop] = sats_prop_sh
    ##
    ## Separating fractions for Centrals and Satellites
    cens_act = cens_pd.loc[(cens_pd[prop]) <= 1]
    cens_pas = cens_pd.loc[(cens_pd[prop]) >  1]
    ##
    ## Groups for each `act` and `pas` centrals
    cens_act_groups = cens_act[id_key].values
    cens_pas_groups = cens_pas[id_key].values
    ##
    ## Satellites residing in groups/halos with `act` and `pas` centrals
    sats_c_act     = sats_pd.loc[(sats_pd[id_key].isin(cens_act_groups))]
    sats_c_pas     = sats_pd.loc[(sats_pd[id_key].isin(cens_pas_groups))]
    ## Total number of satellites in around each type of central
    sats_c_act_tot = len(sats_c_act)
    sats_c_pas_tot = len(sats_c_pas)
    ##
    ## Number of quenched satellites around each type of centrals
    sats_pas_c_act     = sats_c_act.loc[sats_c_act[prop] > 1]
    sats_pas_c_pas     = sats_c_pas.loc[sats_c_pas[prop] > 1]
    sats_pas_c_act_tot = len(sats_pas_c_act)
    sats_pas_c_pas_tot = len(sats_pas_c_pas)
    ##
    ## Quenched fractions of satellites
    # Passive Satellites around `active` centrals
    if sats_c_act_tot != 0:
        frac_sat_pas_cen_act = sats_pas_c_act_tot/float(sats_c_act_tot)
    else:
        frac_sat_pas_cen_act = num.nan
    # Passive Satellites around `passive` centrals
    if sats_c_pas_tot != 0:
        frac_sat_pas_cen_pas = sats_pas_c_pas_tot/float(sats_c_pas_tot)
    else:
        frac_sat_pas_cen_pas = num.nan
    ##
    ## Evaluating `frac_stat`
    # Taking the difference of fractions
    if frac_stat=='diff':
        frac_res = (frac_sat_pas_cen_pas - frac_sat_pas_cen_act)
    # Taking the ratio of fractions
    if frac_stat == 'ratio':
        frac_res = (frac_sat_pas_cen_pas / frac_sat_pas_cen_act)

    return frac_res

def gm_fractions_calc(catl_pd, catl_name, param_dict, proj_dict):
    """
    Computes the 'quenched' satellite fractions for galaxy groups.
    Splits the sample into group mass bins, and determines the quenched 
    fraction of satellite for a given galaxy property.

    Parameters
    ----------
    catl_pd: pandas DataFrame
        DataFrame with information on catalogue

    catl_name: string
        name of the `catl_pd`

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    """
    Prog_msg = param_dict['Prog_msg']
    ### Catalogue Variables
    # `Group mass`, `groupid`, and `galtype` keys
    gm_key, id_key, galtype_key = cu.catl_keys(catl_kind=param_dict['catl_kind'],
                                                return_type='list',
                                                perf_opt=param_dict['perf_opt'])
    catl_keys_dict = cu.catl_keys(  catl_kind=param_dict['catl_kind'],
                                    return_type='dict',
                                    perf_opt=param_dict['perf_opt'])
    gm_key      = catl_keys_dict['gm_key']
    id_key      = catl_keys_dict['id_key']
    galtype_key = catl_keys_dict['galtype_key']
    # ssfr and mstar keys
    ssfr_key, mstar_key = cu.catl_keys_prop(catl_kind=param_dict['catl_kind'], 
                                                catl_info='members')
    # Galaxy Properties
    if param_dict['catl_kind']=='data':
        pd_keys     = ['logssfr', 'g_r', 'sersic']
    elif param_dict['catl_kind']=='mocks':
        pd_keys = ['logssfr']
    # Limits for each galaxy property
    prop_lim = {'logssfr':-11,
                'sersic':3,
                'g_r':0.75}
    param_dict['prop_lim'] = prop_lim
    # Cleaning catalogue with groups of N > `ngals_min`
    catl_pd_clean = cu.sdss_catl_clean_nmin(catl_pd, param_dict['catl_kind'],
        nmin=param_dict['ngals_min'])
    ### Mass limits
    GM_min  = catl_pd_clean[gm_key].min()
    GM_max  = catl_pd_clean[gm_key].max()
    GM_arr  = cu.Bins_array_create([GM_min,GM_max], param_dict['Mg_bin'])
    GM_bins = [[GM_arr[ii],GM_arr[ii+1]] for ii in range(GM_arr.shape[0]-1)]
    GM_bins = num.asarray(GM_bins)
    GM_keys = ['{0:.2f}_{1:.2f}'.format(GM_arr[xx],GM_arr[xx+1])\
                    for xx in range(len(GM_arr)-1)]
    ### Pickle file
    p_file = [  proj_dict['pickdir']       , param_dict['fig_idx']  ,
                catl_name                  , param_dict['sample']   ,
                param_dict['corr_type']    , param_dict['prop_log'] ,
                param_dict['shuffle_marks'], param_dict['ngals_min'],
                param_dict['perf_str']     ]
    p_fname = '{0}/{1}_{2}_Mr{3}_{4}_{5}_{6}_{7}_{8}.p'
    p_fname = p_fname.format(*p_file)
    ##
    ## Checking if file exists
    if (os.path.isfile(p_fname)) and (param_dict['remove_files']):
        os.remove(p_fname)
        print('{0} `p_fname` ({1}) removed! Calculating MCF statistics!'.format(
            Prog_msg, p_fname))
    ## Dictionary for storing results for each GM bin and galaxy property
    frac_gm_dict = dict(zip(GM_keys, [[] for x in range(len(GM_keys))]))
    stat_vals    = [num.zeros((len(GM_keys))), copy.deepcopy(frac_gm_dict)]
    GM_prop_dict = dict(zip(pd_keys,[list(stat_vals) for x in range(len(pd_keys))]))
    ## Looping ovr each GM bin (if file does not exist for the given catalogue)
    if not (os.path.isfile(p_fname)):
        ## Looping over mass bins
        for ii, GM_ii in enumerate(GM_bins):
            # GM Key label
            GM_key = GM_keys[ii]
            # GM string
            GMbin_min, GMbin_max = GM_ii
            if param_dict['perf_opt']:
                print('\n{0} Halo Mass range: {1}'.format(Prog_msg, GM_keys))
            else:
                print('\n{0} Group Mass range: {1}'.format(Prog_msg, GM_key))
            df_bin_org = catl_pd_clean.loc[ (catl_pd_clean[gm_key] >= GMbin_min) &\
                                        (catl_pd_clean[gm_key] <  GMbin_max)].copy()
            df_bin_org.reset_index(inplace=True, drop=True)
            ##
            ## Looping over galaxy properties
            for jj, prop in enumerate(pd_keys):
                print('{0} >> Galaxy Prop: {1}'.format(Prog_msg, prop))
                (   sat_quenched_frac,
                    sat_quenched_frac_sh) = frac_prop_calc( df_bin_org    ,
                                                            prop          ,
                                                            param_dict    ,
                                                            catl_keys_dict)
                ##
                ## Saving results to dictionary
                GM_prop_dict[prop][0][ii]     = sat_quenched_frac
                GM_prop_dict[prop][1][GM_key] = sat_quenched_frac_sh
        ##
        ## Saving `GM_prop_dict` to Pickle file
        print('{0} Saving data to Pickle: \n\t{1}\n'.format(Prog_msg, p_fname))
        p_data = [param_dict, GM_prop_dict, GM_arr, GM_bins, GM_keys]
        pickle.dump(p_data, open(p_fname,'wb'))
    ##
    ## Showing path to file
    print('{0} Data saved to Pickle: \n\t{1}\n'.format(Prog_msg, p_fname))

## --------- Multiprocessing ------------##

def multiprocessing_catls(catl_arr, param_dict, proj_dict, memb_tuples_ii):
    """
    Distributes the analysis of the catalogues into more than 1 processor

    Parameters:
    -----------
    catl_arr: numpy.ndarray, shape(n_catls,)
        array of paths to the catalogues to analyze

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    memb_tuples_ii: tuple
        tuple of catalogue indices to be analyzed
    """
    ## Program Message
    Prog_msg = param_dict['Prog_msg']
    ## Reading in Catalogue IDs
    start_ii, end_ii = memb_tuples_ii
    ##
    ## Looping the desired catalogues
    for ii, catl_ii in enumerate(catl_arr[start_ii : end_ii]):
        ## Choosing 1st catalogue
        print('{0} Analyzing `{1}`\n'.format(Prog_msg, catl_ii))
        ## Extracting `name` of the catalogue
        catl_name = os.path.splitext(os.path.split(catl_ii)[1])[0]
        ## Converting to pandas DataFrame
        catl_pd   = cu.read_hdf5_file_to_pandas_DF(catl_ii)
        ## Quenched Fraction calculations
        gm_fractions_calc(catl_pd, catl_name, param_dict, proj_dict)

## --------- Main Function ------------##

def main():
    """

    """
    ## Starting time
    start_time = datetime.now()
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
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ###
    ### ---- Analysis ---- ###
    ## Reading catalogues
    catl_arr_all = cu.extract_catls(catl_kind=param_dict['catl_kind'],
                                    catl_type=param_dict['catl_type'],
                                    sample_s =param_dict['sample_s'],
                                    perf_opt =param_dict['perf_opt'],
                                    catl_info='members',
                                    print_filedir=False)
    ##
    ## Only reading desired number of catalogues
    catl_arr = catl_arr_all[param_dict['catl_start']:param_dict['catl_finish']]
    ##
    ## Number of catalogues to analyze
    ncatls = len(catl_arr)
    ##
    ## Choosing whether or not to use multiprocessing for the analysis
    if ncatls == 1:
        ## Choosing 1st catalogue
        catl_ii = catl_arr[0]
        print('{0} Analyzing `{1}`\n'.format(Prog_msg, catl_ii))
        ## Extracting `name` of the catalogue
        catl_name = os.path.splitext(os.path.split(catl_ii)[1])[0]
        ## Converting to pandas DataFrame
        catl_pd   = cu.read_hdf5_file_to_pandas_DF(catl_ii)
        ## Quenched Fraction calculations
        gm_fractions_calc(catl_pd, catl_name, param_dict, proj_dict)
    else:
        ###
        ## Changing `prog_bar` to 'False'
        param_dict['prog_bar'] = False
        ### Using multiprocessing to analyze catalogues
        ## Number of CPUs to use
        cpu_number = int(cpu_count() * param_dict['cpu_frac'])
        ## Defining step-size for each CPU
        if cpu_number <= ncatls:
            catl_step  = int(ncatls / cpu_number)
        else:
            catl_step  = int((ncatls / cpu_number)**-1)
        ## Array with designanted catalogue numbers for each CPU
        memb_arr     = num.arange(0, ncatls+1, catl_step)
        memb_arr[-1] = ncatls
        ## Tuples of the ID of each catalogue
        memb_tuples  = num.asarray([(memb_arr[xx], memb_arr[xx+1])
                                for xx in range(memb_arr.size-1)])
        ## Assigning `memb_tuples` to function `multiprocessing_catls`
        procs = []
        for ii in range(len(memb_tuples)):
            # Defining `proc` element
            proc = Process(target=multiprocessing_catls, 
                            args=(catl_arr, param_dict, proj_dict, 
                                    memb_tuples[ii]))
            # Appending to main `procs` list
            procs.append(proc)
            proc.start()
        ##
        ## Joining `procs`
        for proc in procs:
            proc.join()
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
    main()













