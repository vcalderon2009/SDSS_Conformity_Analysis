#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 11/08/2017
# Last Modified: 04/05/2018
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
# from path_variables import git_root_dir
# sys.path.insert(0, os.path.realpath(git_root_dir(__file__)))

# Importing Modules
# import src.data.utilities_python as cu
import custom_utilities_lss as cu
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
import requests

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

def url_checker(url_str):
    """
    Checks if the `url_str` is a valid URL

    Parameters
    ----------
    url_str: string
        url of the website to probe
    """
    request = requests.get(url_str)
    if request.status_code != 200:
        msg = '`url_str` ({0}) does not exist'.format(url_str)
        raise ValueError(msg)
    else:
        pass

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
    description_msg = 'Downloads the necessary catalogues from the web'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
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
    ## SDSS Type
    parser.add_argument('-abopt',
                        dest='catl_type',
                        help='Type of Abund. Matching used in catalogue',
                        type=str,
                        choices=['mr'],
                        default='mr')
    ## Random Seed for CLF
    parser.add_argument('-clf_seed',
                        dest='clf_seed',
                        help='Random seed to be used for CLF',
                        type=int,
                        metavar='[0-4294967295]',
                        default=0)
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cu.Program_Msg(__file__))
    ## `Perfect Catalogue` Option
    parser.add_argument('-perf',
                        dest='perf_opt',
                        help='Option for downloading a `Perfect` catalogue for `mocks`',
                        type=_str2bool,
                        default=False)
    ## Verbose
    parser.add_argument('-v','--verbose',
                        dest='verbose',
                        help='Option to print out project parameters',
                        action="store_true")

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
    ### Sample - Mr
    sample_Mr = 'Mr{0}'.format(param_dict['sample'])
    ###
    ### URL to download catalogues
    url_catl = 'http://lss.phy.vanderbilt.edu/groups/data_vc/DR7/sdss_catalogues/'
    url_checker(url_catl)
    ###
    ### To dictionary
    param_dict['sample_s' ] = sample_s
    param_dict['sample_Mr'] = sample_Mr
    param_dict['url_catl' ] = url_catl

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
    for catl_kind in ['data', 'mocks']:
        # Data
        if catl_kind == 'data':
            catl_dir = os.path.join(    proj_dict['data_dir'],
                                        'external',
                                        'SDSS',
                                        catl_kind,
                                        param_dict['catl_type'],
                                        param_dict['sample_Mr'])
        # Mocks
        if catl_kind == 'mocks':
            catl_dir = os.path.join(    proj_dict['data_dir'],
                                        'external',
                                        'SDSS',
                                        catl_kind,
                                        'halos_{0}'.format(param_dict['halotype']),
                                        'hod_model_{0}'.format(param_dict['hod_n']),
                                        'clf_seed_{0}'.format(param_dict['clf_seed']),
                                        'clf_method_{0}'.format(param_dict['clf_method']),
                                        param_dict['catl_type'],
                                        param_dict['sample_Mr'])
        ##
        ## Extra folders
        # Member galaxy directory
        member_dir = os.path.join(catl_dir, 'member_galaxy_catalogues')
        cu.Path_Folder(member_dir)
        proj_dict['{0}_out_memb'.format(catl_kind)] = member_dir
        ##
        ## Perfect galaxy directory
        if (catl_kind == 'mocks') and (param_dict['perf_opt']):
            perf_member_dir = os.path.join( catl_dir,
                                            'perfect_member_galaxy_catalogues')
            cu.Path_Folder(perf_member_dir)
            proj_dict['{0}_out_perf_memb'.format(catl_kind)] = perf_member_dir

    return proj_dict

### ----| Downloading Data |--- ###

def download_directory(param_dict, proj_dict, cut_dirs=8):
    """
    Downloads the necessary catalogues to perform the analysis

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with input parameters and values

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    cut_dirs: int, optional (default = 8)
        number of directories to skip.
        See `wget` documentation for more details.

    """
    ###
    ## Creating command to execute download
    for catl_kind in ['data', 'mocks']:
        ## Downloading directories from the web
        # Data
        if catl_kind == 'data':
            calt_kind_url = os.path.join(param_dict['url_catl'],
                                        catl_kind,
                                        param_dict['catl_type'],
                                        'Mr'+param_dict['sample_s'],
                                        'member_galaxy_catalogues/')
            # Number of directories to cut
            cut_dirs = 8
        # Mocks
        if catl_kind == 'mocks':
            calt_kind_url = os.path.join(param_dict['url_catl'],
                                        catl_kind,
                                        'halos_{0}'.format(param_dict['halotype']),
                                        'hod_model_{0}'.format(param_dict['hod_n']),
                                        'clf_seed_{0}'.format(param_dict['clf_seed']),
                                        'clf_method_{0}'.format(param_dict['clf_method']),
                                        param_dict['catl_type'],
                                        param_dict['sample_Mr'],
                                        'member_galaxy_catalogues/')
            # Number of directories to cut
            cut_dirs = 12
        ## Checking if URL exists
        url_checker(calt_kind_url)
        ## String to be executed
        if param_dict['verbose']:
            cmd_dw = 'wget -m -nH -x -np -r -c --accept=*.hdf5 --cut-dirs={0} --reject="index.html*" {1}'
        else:
            cmd_dw = 'wget -m -nH -x -np -r -c -nv --accept=*.hdf5 --cut-dirs={0} --reject="index.html*" {1}'
        cmd_dw = cmd_dw.format(cut_dirs, calt_kind_url)
        ## Executing command
        print('{0} Downloading Dataset......'.format(param_dict['Prog_msg']))
        print(cmd_dw)
        subprocess.call(cmd_dw, shell=True, cwd=proj_dict[catl_kind+'_out_memb'])
        ## Deleting `robots.txt`
        os.remove('{0}/robots.txt'.format(proj_dict[catl_kind+'_out_memb']))
        ##
        ##
        print('\n\n{0} Catalogues were saved at: {1}\n\n'.format(
            param_dict['Prog_msg'], proj_dict[catl_kind+'_out_memb']))
        ##
        ## --- Perfect Catalogue -- Mocks
        if (catl_kind == 'mocks') and (param_dict['perf_opt']):
            ## Downloading directories from the web
            calt_kind_url = os.path.join(param_dict['url_catl'],
                                        catl_kind,
                                        param_dict['catl_type'],
                                        'Mr'+param_dict['sample_s'],
                                        'perfect_member_galaxy_catalogues/')
            url_checker(calt_kind_url)
            ## String to be executed
            cmd_dw = 'wget -r -nH -x -np -A *Mr{0}*.hdf5 --cut-dirs={1} -R "index.html*" {2}'
            cmd_dw = cmd_dw.format(param_dict['sample_s'], cut_dirs, calt_kind_url)
            ## Executing command
            print('{0} Downloading Dataset......'.format(param_dict['Prog_msg']))
            print(cmd_dw)
            subprocess.call(cmd_dw, shell=True, cwd=proj_dict['mocks_out_perf_memb'])
            ## Deleting `robots.txt`
            os.remove('{0}/robots.txt'.format(proj_dict['mocks_out_perf_memb']))
            ##
            ##
            print('\n\n{0} Catalogues were saved at: {1}\n\n'.format(
                param_dict['Prog_msg'], proj_dict['mocks_out_perf_memb']))

### ----| Main Function |--- ###

def main(args):
    """
    Downloads the necessary catalogues to perform the 1- and 2-halo 
    conformity analysis
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## ---- Adding to `param_dict` ---- 
    param_dict = add_to_dict(param_dict)
    ## Checking for correct input
    param_vals_test(param_dict)
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
    download_directory(param_dict, proj_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
