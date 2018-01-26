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

## Functions
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
    ## Random Seed
    parser.add_argument('-seed',
                        dest='seed',
                        help='Random seed to be used for the analysis',
                        type=int,
                        metavar='[0-4294967295]',
                        default=1)
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
    cu.url_checker(url_rand)
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
    ## Creating directories
    cu.Path_Folder(figdir)
    ##
    ## Adding to dictionary
    proj_dict['figdir'] = figdir

    return proj_dict

def galprop_distr(data_cl_pd, mocks_pd, param_dict, proj_dict):
    """
    Plots the distribution of color, ssfr, and morphology for the 
    SDSS DR7 dataset

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    """



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


# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
