#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# DATE
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, 1-halo Mark Correlation"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Computes the 1-halo Mark correlation function for SDSS DR7
"""
# Importing Modules
import custom_utilities_python as cu
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
import argparse
from argparse import ArgumentParser
from datetime import datetime

## Functions
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
    parser = ArgumentParser(description=description_msg)
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
                        dest='sdss_kind',
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
                        choices=['cen_sat', 'all'],
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
                        default=10)
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
                        default=20)
    ## Logarithm of the galaxy property
    parser.add_argument('-log',
                        dest='prop_log',
                        help='Use `log` or `non-log` units for `M*` and `sSFR`',
                        type=str,
                        choices=['log', 'nonlog'],
                        default='log')
    ## Mock Start
    parser.add_argument('-start',
                        dest='catl_start',
                        help='Starting index of mock catalogues to use',
                        type=int,
                        choices=range(100),
                        metavar='[0-99]',
                        default=0)
    ## Mock Finish
    parser.add_argument('-finish',
                        dest='catl_finish',
                        help='Finishing index of mock catalogues to use',
                        type=int,
                        choices=range(100),
                        metavar='[0-99]',
                        default=1)
    ## `Perfect Catalogue` Option
    parser.add_argument('-perf',
                        dest='perf_opt',
                        help='Option for using a `Perfect` catalogue',
                        type=_str2bool,
                        default=False)
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help='Delete pickle file containing pair counts',
                        type=_str2bool,
                        default=False)
    ## Type of correlation funciton to perform
    parser.add_argument('-corrtype',
                        dest='corr_type',
                        help='Type of correlation function to perform',
                        type=str,
                        choices=['galgal'],
                        default='galgal')
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
    fig_idx = 20
    ### Projected distance `rp` bins
    logrpmin    = num.log10(param_dict['rpmin'])
    logrpmax    = num.log10(param_dict['rpmax'])
    dlogrp      = (logrpmax - logrpmin)/float(param_dict['nrpbins'])
    rpbin_arr   = num.linspace(logrpmin, logrpmax, param_dict['nrpbins']+1)
    rpbins_cens = rpbin_arr[:-1]+0.5*(rpbin_arr[1:]-rpbin_arr[:-1])
    ### Survey Details
    sample_title = r'\boldmath$M_{r}< -%d$' %(param_dict['sample'])
    ## Project Details
    # String for Main directories
    param_str_arr = [   param_dict['rpmin']         , param_dict['rpmax']    ,
                        param_dict['nrpbins']       , param_dict['Mg_bin']   ,
                        param_dict['pimax' ]        , param_dict['itern_tot'],
                        param_dict['corr_pair_type'], param_dict['prop_log'] ,
                        param_dict['shuffle_marks'] , perf_str ]
    param_str  = 'rpmin_{0}_rpmax_{1}_nrpbins_{2}_Mgbin_{3}_pimax_{4}_'
    param_str += 'itern_{5}_corrpair_type_{6}_proplog_{7}_shuffle_{8}'
    if param_dict['perf_opt']:
        param_str += '_perf_opt_str_{9}/'
    else:
        param_str += '{9}/'
    param_str  = param_str.format(*param_str_arr)
    # String for Main Figures
    param_str_pic_arr = [param_dict['rpmin']  , param_dict['rpmax'] ,
                         param_dict['nrpbins'], param_dict['Mg_bin'],
                         param_dict['pimax']  , perf_str ]
    param_str_pic = 'rpmin_{0}_rpmax_{1}_nrpbins_{2}_Mgbin_{3}_pimax_{4}'
    if param_dict['perf_opt']:
        param_str_pic += '_perf_opt_str_{5}/'
    else:
        param_str_pic += '{5}/'
    param_str_pic = param_str_pic.format(*param_str_pic_arr)
    ###
    ### To dictionary
    param_dict['sample_s'     ] = sample_s
    param_dict['perf_str'     ] = perf_str
    param_dict['fig_idx'      ] = fig_idx
    param_dict['logrpmin'     ] = logrpmin
    param_dict['logrpmax'     ] = logrpmax
    param_dict['dlogrp'       ] = dlogrp
    param_dict['rpbin_arr'    ] = rpbin_arr
    param_dict['rpbins_cens'  ] = rpbins_cens
    param_dict['sample_title' ] = sample_title
    param_dict['param_str'    ] = param_str
    param_dict['param_str_pic'] = param_str_pic

    return param_dict

def directory_skeleton(param_dict, proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict: python dictionary

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    """
    ### Output Directory
    outdir = '{0}/interim/SDSS/{1}/{2}/Mr{3}/conformity_output'.format(
        proj_dict['data_dir'], param_dict['sdss_kind'],
        param_dict['catl_type'], param_dict['sample'])
    ### Directory of `Pickle files` with input parameters
    pickdir = '{0}/pickle_files/{1}/{2}'.format(
        outdir, param_dict['corr_type'], param_dict['param_str'])
    ### Directories for MCF
    corrdir = 'wq_{0}_idx_calc/'.format(param_dict['corr_type'])
    # Indices from wp(rp)
    out_idx = '{0}/{1}/{2}/indices'.format(
        outdir, param_dict['corr_type'], param_dict['param_str_pic'])
    # Results from wp(rp)
    out_res = '{0}/{1}/{2}/results'.format(
        outdir, param_dict['corr_type'], param_dict['param_str_pic'])
    # Results from DDrppi
    out_ddrp = '{0}/{1}/{2}/DDrppi_res'.format(
        outdir, param_dict['corr_type'], param_dict['param_str_pic'])
    # Output for catalogues - Pickle
    out_catl_p = '{0}/{1}/{2}/catl_pickle_pairs'.format(
        outdir, param_dict['corr_type'], param_dict['param_str_pic'])
    # Creating Folders
    cu.Path_Folder(outdir)
    cu.Path_Folder(pickdir)
    cu.Path_Folder(out_idx)
    cu.Path_Folder(out_res)
    cu.Path_Folder(out_ddrp)
    cu.Path_Folder(out_catl_p)
    ## Adding to `proj_dict`
    proj_dict['outdir'    ] = outdir
    proj_dict['pickdir'   ] = pickdir
    proj_dict['out_idx'   ] = out_idx
    proj_dict['out_res'   ] = out_res
    proj_dict['out_ddrp'  ] = out_ddrp
    proj_dict['out_catl_p'] = out_catl_p

    return proj_dict

def halo_corr(catl_pd, catl_name, param_dict, proj_dict):
    """
    1-halo mark correlation function for galaxy groups in each group mass bin.

    Parameters
    ----------
    catl_pd: 

    """

def main(args, Prog_msg = '1 >>>  '):#,
    # Prog_msg = cu.Program_Msg(__file__)):
    """
    Computes the 1-halo galactic conformity results on SDSS DR7

    Parameters
    ----------
    args: argparse.Namespace
        input parameters for script

    Notes
    ----------
    To see how to use the code run:
        >>>> python one_halo_mark_correlation.py -h [--help]
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## ---- Adding to `param_dict` ---- 
    param_dict = add_to_dict(param_dict)
    ## Creating Folder Structure
    # proj_dict  = cu.cookiecutter_paths(__file__)
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths('./'))
    ## Running analysis
    # Reading catalogues
    catl_arr_all = cu.extract_catls(catl_kind=param_dict['sdss_kind'],
                                    catl_type=param_dict['catl_type'],
                                    sample_s =param_dict['sample_s'],
                                    perf_opt =param_dict['perf_opt'],
                                    catl_info='members',
                                    print_filedir=False)
    catl_arr = catl_arr_all[param_dict['catl_start']:param_dict['catl_finish']]
    # Looping over catalogues
    for ii, catl_ii in enumerate(catl_arr):
        print('{0} Analyzing {1}\n'.format(Prog_msg, catl_ii))
        catl_name = os.path.splitext(os.path.split(catl_ii)[1])[0]
        catl_pd   = cu.read_hdf5_file_to_pandas_DF(catl_ii)
        # MCF Calculations
        halo_corr(catl_pd, catl_name, param_dict, proj_dict)




# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
