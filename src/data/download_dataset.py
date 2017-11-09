#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 11/08/2017
# Last Modified: 11/08/2017
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Downloads the necessary galaxy catalogues from the web to perform the 
1- and 2-halo conformity analyses.
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
import os
import sys
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import subprocess

### ----| Common Functions |--- ###

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
    description_msg = 'Downloads the necessary catalogues from the web'
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
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cu.Program_Msg(__file__))
    ## Parsing Objects
    args = parser.parse_args()

    return args

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
    ###
    ### URL to download catalogues
    url_catl = 'http://vpac00.phy.vanderbilt.edu/~caldervf/GIFs/'
    ###
    ### To dictionary
    param_dict['sample_s'] = sample_s
    param_dict['url_catl'] = url_catl

    return param_dict

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
    ## Directory for Catalogues
    for kind_ii in ['data', 'mocks']:
        catl_dir = os.path.join(proj_dict['data_dir'],
                                'external',
                                'SDSS',
                                kind_ii,
                                param_dict['catl_type'],
                                'Mr{0}'.format(param_dict['sample']))
        ## Making sure directory exists
        cu.Path_Folder(catl_dir)
        ##
        ## Saving path to `param_dict`
        param_dict['{0}_out'.format(kind_ii)] = catl_dir

    return proj_dict

### ----| Downloading Data |--- ###

def download_directory(param_dict, cut_dirs=100):
    """
    Downloads the necessary catalogues to perform the analysis

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with input parameters and values

    cut_dirs: int, optional (default = 100)
        number of directories to skip.
        See `wget` documentation for more details.

    """
    ###
    ## Creating command to execute download
    for catl_kind in ['data', 'mocks']:
        ## Downloading directories from 
        calt_kind_url = '{0}/{1}/{2}/{3}'.format(param_dict['url_catl'],
                                                 catl_kind,
                                                 param_dict['catl_type'],
                                                 'Mr'+param_dict['sample_s'])
        ## String to be executed
        cmd_dw = 'wget -r -nH --cut_dirs={0} -np -R "index.html*" {1}'
        cmd_dw = cmd_dw.format(cut_dirs, calt_kind_url)
        ## Executing command
        print('{0} Downloading Dataset......'.format(param_dict['Prog_msg']))
        print(cmd_dw)
        subprocess.call(cmd_dw, shell=True, cwd=param_dict[catl_kind+'_out'])
        ## Deleting `robots.txt`
        os.remove('{0}/robots.txt'.format(param_dict))

### ----| Main Function |--- ###

def main():
    """
    Downloads the necessary catalogues to perform the 1- and 2-halo 
    conformity analysis
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
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ##
    ## Downloading necessary data
    download_directory(param_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main()
