#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Oct 16, 2017
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
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import copy
from datetime import datetime
import subprocess
import astropy.cosmology as astrocosmo
import astropy.constants as ac
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
import warnings
from multiprocessing import Pool, Process, cpu_count

## Cython modules
from pair_counter_rp import pairwise_distance_rp

# Ignoring certain warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

## Functions
class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        """
        Modifier for `argparse` help parameters, that sorts them alphabetically
        """
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)

def cosmo_create(cosmo_choice='LasDamas', H0=100., Om0=0.25, Ob0=0.04,
    Tcmb0=2.7255):
    """
    Creates instance of the cosmology used throughout the project.

    Parameters
    ----------
    cosmo_choice: string, optional (default = 'Planck')
        choice of cosmology
        Options:
            - Planck: Cosmology from Planck 2015
            - LasDamas: Cosmology from LasDamas simulation

    h: float, optional (default = 1.0)
        value for small cosmological 'h'.

    Returns
    ----------                  
    cosmo_model: astropy cosmology object
        cosmology used throughout the project
    """
    ## Checking cosmology choices
    cosmo_choice_arr = ['Planck', 'LasDamas']
    assert(cosmo_choice in cosmo_choice_arr)
    ## Choosing cosmology
    if cosmo_choice == 'Planck':
        cosmo_model = astrocosmo.Planck15.clone(H0=H0)
    elif cosmo_choice == 'LasDamas':
        cosmo_model = astrocosmo.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, 
            Tcmb0=Tcmb0)
    ## Cosmo Paramters
    cosmo_params         = {}
    cosmo_params['H0'  ] = cosmo_model.H0.value
    cosmo_params['Om0' ] = cosmo_model.Om0
    cosmo_params['Ob0' ] = cosmo_model.Ob0
    cosmo_params['Ode0'] = cosmo_model.Ode0
    cosmo_params['Ok0' ] = cosmo_model.Ok0

    return cosmo_model

def spherical_to_cart(catl_pd, cosmo_model, method='astropy',
    dist_opt='astropy'):
    """
    Converts the spherical coordiates < ra dec cz > of galaxies 
    to cartesian coordinates

    Parameters
    ----------
    catl_pd: pandas DataFrame
        DataFrame with information on catalogue

    cosmo_model: astropy cosmology object
        cosmology used throughout the project

    method: string, optional (default = 'astropy')
        method to use to calculate cartesian coordinates
        Options:
            - 'astropy': uses `astropy.comoving_distance` to estimate the 
                        comoving distance to the galaxies
            - 'approx': uses approximation of `cz/H0` with an 
                        H0=100 km/s/Mpc to estimate comoving distances.
                        And it does not use `astropy` in any calculation.

    Returns
    ----------
    catl_pd: pandas DataFrame
        DataFrame with information on catalogue and with `new`
        cartesian coordinates of galaxies.
        In units of `astropy.units.Mpc` with h=1
    """
    if method=='astropy':
        ## Determining comoving distance
        c_units      = ac.c.to(u.km/u.s)
        gal_redshift = (catl_pd['cz']*(u.km/u.s))/(c_units)
        gal_dist     = cosmo_model.comoving_distance(gal_redshift).to(u.Mpc)
        ## Adding to `catl_pd`
        catl_pd.loc[:,'dist'] = gal_dist.value
        ## Spherical Coordinates
        gal_sph = SkyCoord( ra=catl_pd['ra'].values*u.degree, 
                            dec=catl_pd['dec'].values*u.degree,
                            distance=catl_pd['dist'].values*u.Mpc)
        ## Cartesian Coordinates
        gal_x, gal_y, gal_z = gal_sph.cartesian.xyz.value
    elif method=='approx':
        ## Determining comoving distance
        gal_dist     = (catl_pd['cz']*.01).values*u.Mpc
        ## Adding to `catl_pd`
        catl_pd.loc[:,'dist'] = gal_dist.value
        ## Spherical to Cartesian Coordinates
        ra_cen = dec_cen = dist_cen = 0.
        sph_dict, gal_cart = cu.Coord_Transformation(   catl_pd['ra'],
                                                        catl_pd['dec'],
                                                        catl_pd['dist'],
                                                        ra_cen,
                                                        dec_cen,
                                                        dist_cen,
                                                        trans_opt=1)
        gal_x, gal_y, gal_z = pd.DataFrame(gal_cart)[['X','Y','Z']].values.T
    ## Adding to `catl_pd`
    catl_pd.loc[:,'x'] = gal_x
    catl_pd.loc[:,'y'] = gal_y
    catl_pd.loc[:,'z'] = gal_z

    return catl_pd

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
    ## Cosmology Choice
    parser.add_argument('-cosmo',
                        dest='cosmo_choice',
                        help='Choice of Cosmology',
                        type=str,
                        choices=['LasDamas', 'Planck'],
                        default='LasDamas')
    ## Cartesian method
    parser.add_argument('-cart',
                        dest='cart_method',
                        help='Method for calculating distances to galaxies',
                        type=str,
                        choices=['astropy', 'approx'],
                        default='astropy')
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
        param_str_pic += '_perf_opt_str_{6}/'
    else:
        param_str_pic += '{6}/'
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
    ### Output Directory
    outdir = '{0}/interim/SDSS/{1}/{2}/Mr{3}/conformity_output'.format(
        proj_dict['data_dir'], param_dict['catl_kind'],
        param_dict['catl_type'], param_dict['sample'])
    ### Directory of `Pickle files` with input parameters
    pickdir = '{0}/catl_pickle_files/{1}/{2}'.format(
        outdir, param_dict['corr_type'], param_dict['param_str'])
    ### Directories for MCF
    corrdir = 'wq_{0}_idx_calc/'.format(param_dict['corr_type'])
    # Output for catalogues - Pickle
    out_catl_p = '{0}/DDrppi_results/{1}/{2}/catl_pickle_pairs'.format(
        outdir, param_dict['corr_type'], param_dict['param_str_pic'])
    # Creating Folders
    cu.Path_Folder(outdir)
    cu.Path_Folder(pickdir)
    cu.Path_Folder(out_catl_p)
    ## Adding to `proj_dict`
    proj_dict['outdir'    ] = outdir
    proj_dict['pickdir'   ] = pickdir
    proj_dict['out_catl_p'] = out_catl_p

    return proj_dict

def wp_idx_calc(group_df, param_dict):
    """
    Counts the pairs in each `rp` bins for each galaxy-pairs

    Parameters
    ----------
    group_df: pandas DataFrame
        DataFrame with info on galaxies from given group mass bin

    param_dict: python dictionary
        dictionary with project variables
    
    Returns
    ----------
    rp_idx: array-like, shape (len(param_dict['nrpbins']),[])
        multi-dimensional array for the i-th j-th elements for each pair

    rp_npairs: array_like, shape (len(param_dict['nrpbins']), [])
        array of the number of pairs in each `rp`-bin
    """
    ### Converting to cartesian coordinates
    coord_1 = group_df[['x','y','z']].values
    ### 
    rp_ith_arr = pairwise_distance_rp(  coord_1,
                                        coord_1,
                                        rpmin=param_dict['rpmin'],
                                        rpmax=param_dict['rpmax'],
                                        nrpbins=param_dict['nrpbins'])
    ### Converting to pandas DataFrame
    rp_ith_pd = pd.DataFrame(rp_ith_arr, columns=['rp','i','j'])
    ### Unique `rp` bins
    rp_idx = [rp_ith_pd.loc[rp_ith_pd['rp']==xx,['i','j']].values for xx in \
                range(param_dict['nrpbins'])]
    rp_idx = num.array(rp_idx)
    ## Array of total number of pairs in `rp`-bins
    rp_npairs = num.array([len(xx) for xx in rp_idx])

    return rp_idx, rp_npairs

def MCF_conf_seg(prop, df_bin_org, group_idx_arr, rpbins_npairs_tot, 
    param_dict, catl_keys_dict):
    """
    Marked correlation function calculation for the case,
    where `Conformity + Segregation` is considered
    In here, we include the segregation effect of satellites, and 
    thus, it does not show the "true" signal of galactic conformity 
    alone.

    Parameters
    -----------
    prop: string
        galaxy property being analyzed

    df_bin_org: pandas DataFrame
    
    group_idx_arr: numpy.ndarraym, shape (param_dict['nrpbins'])

    rpbins_npairs_tot: numpy.ndarray, shape (param_dict['nrpbins'])

    param_dict: python dictionary

    catl_keys_dict: python dictionary

    Returns
    -----------
    mcf_dict: python dictionary
        dictionary containing the statistics of the MCF and the `shuffled` MCF
        Keys:
            - 'mcf': Marked correlation function for `non-shuffled` scenario
            - 'npairs': Total number of pairs for the `selected` galaxy pairs
            - 'npairs_rp': Number of selected galaxy-pairs in each `rp`-bin
            - 'mcf_sh_mean': Mean of the Shuffled MCF.
            - 'mcf_sh_std': St. Dev. of the Shuffled MCF.
            - 'sigma': Dictionary of 1,2,3- sigma ranges for the `shuffled` MCF
            - 'mcf_sh': MCFs for all iterations for the shuffled scenario
    """
    ## Creating new DataFrame
    df_bin = df_bin_org.copy()
    ## Constants
    Cens  = int(1)
    Sats  = int(0)
    itern = param_dict['itern_tot']
    ## Galaxy Property Labels
    prop_orig = prop + '_orig'
    ## Catalogue Variables for galaxy properties
    gm_key      = catl_keys_dict['gm_key']
    id_key      = catl_keys_dict['id_key']
    galtype_key = catl_keys_dict['galtype_key']
    ##
    ## Type of correlation Galaxy Pair
    ## Adding the values for 'Cens' and 'Sats' and determining which 
    ## galaxy pair to keep for the analysis.
    if param_dict['corr_pair_type']=='cen_sat':
        ## Keeping 'Central-Satellite' pairs
        corr_pair_int = Cens + Sats
    elif param_dict['corr_pair_type']=='cen_cen':
        corr_pair_int = Cens + Cens
    ##
    ## Correctint for `log` in `prop`
    nonlog_arr = ['logssfr', 'logMstar_JHU']
    if (prop in nonlog_arr) and (param_dict['prop_log']=='nonlog'):
        df_bin.loc[:,prop] = 10**df_bin[prop].values
    ##
    ## Galaxy type, `prop`, and `original` arrays
    galtype_arr   = df_bin[galtype_key].copy().values
    galidx_arr    = df_bin.index.values.copy()
    prop_orig_arr = copy.deepcopy(df_bin[prop].values)
    ##
    ## Assigning original `prop` to new column in `df_bn`
    df_bin.loc[:,prop_orig] = prop_orig_arr.copy()
    ##
    ## Indices of `Centrals` and `Satellites
    cen_idx = df_bin.loc[df_bin[galtype_key]==Cens].index.values
    sat_idx = df_bin.loc[df_bin[galtype_key]==Sats].index.values
    ##
    ## Normalizing Centrals and Satellites by their mean values of `prop`
    cens_prop_mean = df_bin.loc[cen_idx, prop].mean()
    sats_prop_mean = df_bin.loc[sat_idx, prop].mean()
    df_bin.loc[cen_idx, prop] /= cens_prop_mean
    df_bin.loc[sat_idx, prop] /= sats_prop_mean
    ##
    ## Copy of `newly normalized` galaxy property `prop` array
    prop_arr = df_bin[prop].values
    ##
    ## Types of Galaxy Pairs - Analysis ##
    ##
    # Adding values for galaxy pairs
    galtype_sum = num.array([galtype_arr[xx.T[0]] + 
                             galtype_arr[xx.T[1]]
                             for xx in group_idx_arr])
    # Indices of galaxy pairs that match the criteria
    gal_pair_idx = num.array([num.where(xx==corr_pair_int)[0] 
                                for xx in galtype_sum])
    # Taking the product of `prop` for each galaxy pair
    galprop_prod = num.array([  prop_arr[xx.T[0]] *
                                prop_arr[xx.T[1]]
                                for xx in group_idx_arr])
    # Selecting indices of `selected galaxy pairs`
    galprop_prod_sel = num.array([galprop_prod[kk][gal_pair_idx[kk]]
                                for kk in range(len(gal_pair_idx))])
    # Total number of galaxy-pairs in each selected `rp`-bin
    rp_npairs_tot_sel = num.array([len(xx) for xx in galprop_prod_sel],
                                    dtype=float)
    ##
    ## Normalizing MCF
    # Summing over all `prop` products and normalizing by `counts`
    corrfunc = (num.array([num.sum(xx) for xx in galprop_prod_sel])/
                            rp_npairs_tot_sel)
    # Total number of counts in each `rp` bin
    npairs_tot = num.sum(rp_npairs_tot_sel)
    ##
    ## ---| Shuffles |--- ##
    ##
    # Selecting 2 columns only
    df_bin_sh = df_bin[[prop, galtype_key]].copy()
    #
    # Creating new column for `shuffled` case
    prop_sh = prop + '_sh'
    df_bin_sh.loc[:,prop_sh] = df_bin[prop].values.copy()
    ##
    ## Initializing array for the `shuffle` case
    corrfunc_sh_tot = num.zeros((param_dict['nrpbins'], 1))
    ##
    ## Looping over iterations to estimate the spread of the shuffles
    # ProgressBar properties
    if param_dict['prog_bar']:
        widgets   = [Bar('>'), 'MCF Conf+Seg Itern: ', ETA(), ' ', ReverseBar('<')]
        pbar_mock = ProgressBar( widgets=widgets, maxval= 10 * itern).start()
    for ii in range(param_dict['itern_tot']):
        ##
        ## Copying default `prop` array to DataFrame `df_bin_sh`
        df_bin_sh.loc[:,prop_sh] = copy.deepcopy(prop_arr)
        ##
        ## Shuffling based on `shuffle_marks`
        if param_dict['shuffle_marks']=='censat_sh':
            ## Shuffling `centrals` and `satellites` `prop` arrays
            # Centrals
            mark_sh_cen = df_bin_sh.loc[cen_idx, prop_sh].copy().values
            num.random.shuffle(mark_sh_cen)
            df_bin_sh.loc[cen_idx, prop_sh] = mark_sh_cen
            # Satellites
            mark_sh_sat = df_bin_sh.loc[sat_idx, prop_sh].copy().values
            num.random.shuffle(mark_sh_sat)
            df_bin_sh.loc[sat_idx, prop_sh] = mark_sh_sat
        ## Shuffling only centrals' `prop` array
        elif param_dict['shuffle_marks']=='cen_sh':
            # Centrals
            mark_sh_cen = df_bin_sh.loc[cen_idx, prop_sh].copy().values
            num.random.shuffle(mark_sh_cen)
            df_bin_sh.loc[cen_idx, prop_sh] = mark_sh_cen
        ## Shuffling only satellites' `prop` array
        elif param_dict['shuffle_marks']=='sat_sh':
            # Satellites
            mark_sh_sat = df_bin_sh.loc[sat_idx, prop_sh].copy().values
            num.random.shuffle(mark_sh_sat)
            df_bin_sh.loc[sat_idx, prop_sh] = mark_sh_sat
        ##
        ## Saving galaxy property `prop_sh` to array for shuffling
        prop_sh_arr = df_bin_sh[prop_sh].copy().values
        ##
        ## Product of marks
        galprop_prod_sh = num.array([   prop_sh_arr[xx.T[0]] *
                                        prop_sh_arr[xx.T[1]]
                                        for xx in group_idx_arr])
        ##
        ## Selecting indices of `selected galaxy pairs`
        galprop_prod_sh_sel = num.array([galprop_prod_sh[kk][gal_pair_idx[kk]]
                                        for kk in range(len(gal_pair_idx))])
        ##
        ## Normalizing MCF
        # Summing over all `prop` products and normalizing by `counts`
        corrfunc_sh = (num.array([num.sum(xx) for xx in galprop_prod_sh_sel])/
                                    rp_npairs_tot_sel)
        ##
        ## Appending to main MCF-Shuffles array
        corrfunc_sh_tot = num.insert(corrfunc_sh_tot,
                                    len(corrfunc_sh_tot.T),
                                    corrfunc_sh,
                                    1)
        if param_dict['prog_bar']:
            pbar_mock.update(10*ii)
    if param_dict['prog_bar']:
        pbar_mock.finish()
    ###
    ### ---| Statistics |--- ###
    ###
    # Removing first column of `zero's`
    corrfunc_sh_tot = num.delete(corrfunc_sh_tot, 0, axis=1)
    ##
    ## Percentiles
    perc_arr = [68., 95., 99.7]
    # Creating dictionary for calculating percentiles
    sigma_dict = {}
    for ii in range(len(perc_arr)):
        sigma_dict[ii] = []
    # Populating dictionary
    for ii, perc_ii in enumerate(perc_arr):
        mark_lower = num.nanpercentile(corrfunc_sh_tot, 50.-(perc_ii/2),axis=1)
        mark_upper = num.nanpercentile(corrfunc_sh_tot, 50.+(perc_ii/2),axis=1)
        # Saving to dictionary
        sigma_dict[ii] = num.column_stack((mark_lower, mark_upper)).T
    ## Mean and St. Dev.
    mark_mean = num.nanmean(corrfunc_sh_tot, axis=1)
    mark_std  = num.nanstd( corrfunc_sh_tot, axis=1)
    ##
    ## --| Saving everything to a dictionary
    ##
    mcf_dict = {}
    mcf_dict['mcf'         ] = corrfunc
    mcf_dict['npairs'      ] = npairs_tot
    mcf_dict['npairs_rp'   ] = rp_npairs_tot_sel
    mcf_dict['mcf_sh_mean' ] = mark_mean
    mcf_dict['mcf_sh_std'  ] = mark_std
    mcf_dict['sigma'       ] = sigma_dict
    mcf_dict['mcf_sh'      ] = corrfunc_sh_tot

    return mcf_dict

def MCF_conf(prop, df_bin_org, group_idx_arr, rpbins_npairs_tot, param_dict, 
    catl_keys_dict):
    """
    Marked correlation function calculation for the case,
    where `Conformity only` is considered
    In here, we 'remove' the segregation effect of satellites, and 
    thus, it does not show the "true" signal of galactic conformity 
    alone. This means that satellites are normalized by the mean 
    `prop` in each `rp`-bin, rather than by the total mean of `prop` 
    for satellites

    Parameters
    -----------
    prop: string
        galaxy property being analyzed

    df_bin_org: pandas DataFrame
    
    group_idx_arr: numpy.ndarraym, shape (param_dict['nrpbins'])

    rpbins_npairs_tot: numpy.ndarray, shape (param_dict['nrpbins'])

    param_dict: python dictionary

    catl_keys_dict: python dictionary

    Returns
    -----------
    mcf_dict: python dictionary
        dictionary containing the statistics of the MCF and the `shuffled` MCF
        Keys:
            - 'mcf': Marked correlation function for `non-shuffled` scenario
            - 'npairs': Total number of pairs for the `selected` galaxy pairs
            - 'npairs_rp': Number of selected galaxy-pairs in each `rp`-bin
            - 'mcf_sh_mean': Mean of the Shuffled MCF.
            - 'mcf_sh_std': St. Dev. of the Shuffled MCF.
            - 'sigma': Dictionary of 1,2,3- sigma ranges for the `shuffled` MCF
            - 'mcf_sh': MCFs for all iterations for the shuffled scenario
    """
    ## Creating new DataFrame
    df_bin = df_bin_org.copy()
    ## Constants
    Cens  = int(1)
    Sats  = int(0)
    itern = param_dict['itern_tot']
    ## Galaxy Property Labels
    prop_orig = prop + '_orig'
    ## Catalogue Variables for galaxy properties
    gm_key      = catl_keys_dict['gm_key']
    id_key      = catl_keys_dict['id_key']
    galtype_key = catl_keys_dict['galtype_key']
    ##
    ## Type of correlation Galaxy Pair
    ## Adding the values for 'Cens' and 'Sats' and determining which 
    ## galaxy pair to keep for the analysis.
    if param_dict['corr_pair_type']=='cen_sat':
        ## Keeping 'Central-Satellite' pairs
        corr_pair_int = Cens + Sats
    elif param_dict['corr_pair_type']=='cen_cen':
        corr_pair_int = Cens + Cens
    ##
    ## Correctint for `log` in `prop`
    nonlog_arr = ['logssfr', 'logMstar_JHU']
    if (prop in nonlog_arr) and (param_dict['prop_log']=='nonlog'):
        df_bin.loc[:,prop] = 10**df_bin[prop].values
    ##
    ## Galaxy type, `prop`, and `original` arrays
    galtype_arr   = df_bin[galtype_key].copy().values
    galidx_arr    = df_bin.index.values.copy()
    prop_orig_arr = copy.deepcopy(df_bin[prop].values)
    ##
    ## Assigning original `prop` to new column in `df_bn`
    df_bin.loc[:,prop_orig] = prop_orig_arr.copy()
    ##
    ## Indices of `Centrals` and `Satellites
    cen_idx = df_bin.loc[df_bin[galtype_key]==Cens].index.values
    sat_idx = df_bin.loc[df_bin[galtype_key]==Sats].index.values
    ##
    ## --| Normalization of Centrals and Satellites |--- ##
    ##
    # Normalizing `Central` galaxies
    cens_prop_mean = df_bin.loc[cen_idx, prop].mean()
    df_bin.loc[cen_idx, prop] /= cens_prop_mean
    ##
    ## Normalizing `Satellites`
    # Create a row for the `rp` of each Satellite
    df_bin.loc[:,'rpbin'] = -10*num.ones(df_bin.shape[0])
    # Assigning `rpbin` to centrals
    df_bin.loc[cen_idx, 'rpbin'] = -1
    ##
    ## Selecting the indices that match the criteria `corr_pair_int`
    # Sum of 'galaxy pairs'
    galtype_sum = num.array([galtype_arr[xx.T[0]] + 
                             galtype_arr[xx.T[1]]
                             for xx in group_idx_arr])
    #
    # Only choosing the galaxy-pairs that meet `corr_pair_int`
    galtype_sum_seg = num.array([num.where(xx==corr_pair_int)[0]
                                for xx in galtype_sum])
    #
    # Creating new array of galaxy-pair indices
    # This array contains the indices of galaxy-pairs that meet the 
    # criteria 'corr_pair_int'
    gal_pair_idx_seg = num.array([group_idx_arr[kk][galtype_sum_seg[kk]]
                                for kk in range(len(galtype_sum_seg))])
    # Array for saving indices of Satellites to change
    id_sats_arr = [[] for x in range(param_dict['nrpbins'])]
    # Assigning `rp` values to satellites
    for rp_ii in range(param_dict['nrpbins']):
        ## Getting id's of satellites in each rp_bin
        id_rp_ii           = num.unique(gal_pair_idx_seg[rp_ii])
        id_sats            = id_rp_ii[galtype_arr[id_rp_ii]==Sats]
        id_sats_arr[rp_ii] = id_sats
        # Assigning `rp` values
        df_bin.loc[id_sats, 'rpbin'] = rp_ii
    # Changing type of data for `rpbin` column
    df_bin      = df_bin.astype({'rpbin':int})
    id_sats_arr = num.array(id_sats_arr)
    #
    # Normalizing Satellites by each value of `prop` in each `rpbin`
    for ii, id_ii in enumerate(id_sats_arr):
        df_bin.loc[id_ii, prop] /= df_bin.loc[id_ii, prop].mean()
    ##
    ## ---| Computing MCF |--- ##
    ##
    # Copy of the 'newly normalized' galaxy property `prop` array
    prop_arr = df_bin[prop].values
    # Product of galaxy `prop` values
    galprop_prod_sel = num.array([  prop_arr[xx.T[0]]*
                                    prop_arr[xx.T[1]]
                                    for xx in gal_pair_idx_seg])
    #
    # Total number of galaxy-pairs in each selected `rp`-bin
    rp_npairs_tot_sel = num.array([len(xx) for xx in gal_pair_idx_seg],
                                    dtype=float)
    ##
    ## Normalizing MCF
    # Summing over all `prop` products and normalizing by `counts`
    corrfunc = (num.array([num.sum(xx) for xx in galprop_prod_sel])/
                            rp_npairs_tot_sel)
    # Total number of counts in each `rp` bin
    npairs_tot = num.sum(rp_npairs_tot_sel)
    ##
    ## ---| Shuffles |--- ##
    ##
    # Selecting 2 columns only
    df_bin_sh = df_bin_org[[prop, galtype_key]].copy()
    #
    # Creating new column for `shuffled` case with the original `prop` array
    prop_sh = prop + '_sh'
    df_bin_sh.loc[:, prop_sh] = df_bin_org[prop].values.copy()
    ##
    ## Initializing array for the `shuffle` case
    corrfunc_sh_tot = num.zeros((param_dict['nrpbins'], 1))
    ##
    ## Looping over iterations to estimate the spread of the shuffles
    # ProgressBar properties
    if param_dict['prog_bar']:
        widgets   = [Bar('>'), 'MCF Conf Itern: ', ETA(), ' ', ReverseBar('<')]
        pbar_mock = ProgressBar( widgets=widgets, maxval= 10 * itern).start()
    for ii in range(param_dict['itern_tot']):
        ##
        ## Copying default `prop` array to DataFrame `df_bin_sh`
        df_bin_sh.loc[:,prop_sh] = copy.deepcopy(prop_orig_arr)
        ##
        ## Shuffling based on `shuffle_marks`
        if param_dict['shuffle_marks']=='censat_sh':
            ## Shuffling `centrals` and `satellites` `prop` arrays
            # Centrals
            mark_sh_cen = df_bin_sh.loc[cen_idx, prop_sh].copy().values
            num.random.shuffle(mark_sh_cen)
            df_bin_sh.loc[cen_idx, prop_sh] = mark_sh_cen
            # Satellites
            mark_sh_sat = df_bin_sh.loc[sat_idx, prop_sh].copy().values
            num.random.shuffle(mark_sh_sat)
            df_bin_sh.loc[sat_idx, prop_sh] = mark_sh_sat
        ## Shuffling only centrals' `prop` array
        elif param_dict['shuffle_marks']=='cen_sh':
            # Centrals
            mark_sh_cen = df_bin_sh.loc[cen_idx, prop_sh].copy().values
            num.random.shuffle(mark_sh_cen)
            df_bin_sh.loc[cen_idx, prop_sh] = mark_sh_cen
        ## Shuffling only satellites' `prop` array
        elif param_dict['shuffle_marks']=='sat_sh':
            # Satellites
            mark_sh_sat = df_bin_sh.loc[sat_idx, prop_sh].copy().values
            num.random.shuffle(mark_sh_sat)
            df_bin_sh.loc[sat_idx, prop_sh] = mark_sh_sat
        ##
        ## --| Normalizing Central and Satellites in each `rp` bin |--- ##
        ##
        # Normalizing `Central` galaxies
        cen_prop_mean = df_bin_sh.loc[cen_idx, prop_sh].mean()
        df_bin_sh.loc[cen_idx, prop_sh] /= cens_prop_mean
        #
        # Normalizing `Satellite` galaxies by each value of `prop` in each 
        # `rp`-bin
        for zz, id_zz in enumerate(id_sats_arr):
            df_bin_sh.loc[id_zz, prop_sh] /= df_bin_sh.loc[id_zz, prop_sh].mean()
        ##
        ## Copy of the `newly normalized` galaxy property `prop_sh` in 
        ## each `rp` bin
        prop_sh_arr = df_bin_sh[prop_sh].copy().values
        ##
        ## Product of `marks`
        galprop_prod_sh = num.array([   prop_sh_arr[xx.T[0]]*
                                        prop_sh_arr[xx.T[1]]
                                        for xx in gal_pair_idx_seg])
        ##
        ## Normalizing MCF
        # Summing over all `prop` products and normalizing by `counts`
        corrfunc_sh = (num.array([num.sum(xx) for xx in galprop_prod_sh])/
                                    rp_npairs_tot_sel)
        ##
        ## Appending to main MCF-Shuffles array
        corrfunc_sh_tot = num.insert(corrfunc_sh_tot,
                                    len(corrfunc_sh_tot.T),
                                    corrfunc_sh,
                                    1)
        if param_dict['prog_bar']:
            pbar_mock.update(10*ii)
    if param_dict['prog_bar']:
        pbar_mock.finish()
    ###
    ### ---| Statistics |--- ###
    ###
    # Removing first column of `zero's`
    corrfunc_sh_tot = num.delete(corrfunc_sh_tot, 0, axis=1)
    ##
    ## Percentiles
    perc_arr = [68., 95., 99.7]
    # Creating dictionary for calculating percentiles
    sigma_dict = {}
    for ii in range(len(perc_arr)):
        sigma_dict[ii] = []
    # Populating dictionary
    for ii, perc_ii in enumerate(perc_arr):
        mark_lower = num.nanpercentile(corrfunc_sh_tot, 50.-(perc_ii/2),axis=1)
        mark_upper = num.nanpercentile(corrfunc_sh_tot, 50.+(perc_ii/2),axis=1)
        # Saving to dictionary
        sigma_dict[ii] = num.column_stack((mark_lower, mark_upper)).T
    ## Mean and St. Dev.
    mark_mean = num.nanmean(corrfunc_sh_tot, axis=1)
    mark_std  = num.nanstd( corrfunc_sh_tot, axis=1)
    ##
    ## --| Saving everything to a dictionary
    ##
    mcf_dict = {}
    mcf_dict['mcf'         ] = corrfunc
    mcf_dict['npairs'      ] = npairs_tot
    mcf_dict['npairs_rp'   ] = rp_npairs_tot_sel
    mcf_dict['mcf_sh_mean' ] = mark_mean
    mcf_dict['mcf_sh_std'  ] = mark_std
    mcf_dict['sigma'       ] = sigma_dict
    mcf_dict['mcf_sh'      ] = corrfunc_sh_tot

    return mcf_dict

def prop_sh_one_halo(df_bin_org, prop, GM_str, param_dict, proj_dict,
    catl_name, catl_keys_dict):
    """
    Shuffles the galaxy properties for the 1-halo term (same-halo pairs)

    Parameters
    ----------
    df_bin: pandas DataFrame
        Dataframe for the selected group/halo mass bin

    prop: string
        galaxy property being evaluated

    GM_str: string
        string for the corresponding group/halo mass bin limits

    param_dict: python dictionary
        dictionary with input parameters and values

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    catl_name: string
        prefix of the catalogue being analyzed

    catl_keys_dict: python dictionary
        dictionary containing keys for the galaxy properties in catalogue
    
    Returns
    ----------
    mcf_dict_conf: python dictionary
        dictionary with the statistics of the `shuffles` and the MCF 
        of the catalogue.
        This dictionary corresponds to the "Conformity Only" scenario

    mcf_dict_conf_seg: python dictionary
        dictionary with the statistics of the `shuffles` and the MCF 
        of the catalogue.
        This dictionary corresponds to the "Conformity + Segregation" scenario

    ngroups: int
        total number of groups in the desired group-mass bin

    """
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ## Constants
    Cens = int(1)
    Sats = int(0)
    ## Catalogue Variables for galaxy properties
    gm_key      = catl_keys_dict['gm_key']
    id_key      = catl_keys_dict['id_key']
    galtype_key = catl_keys_dict['galtype_key']
    ## Group statistics
    groupid_unq = df_bin_org[id_key].unique()
    ngroups     = groupid_unq.shape[0]
    ## Total number of galaxy pairs in `rp`
    rpbins_npairs_tot = num.zeros(param_dict['nrpbins'])
    ## Pickle file - name - for wp(rp)
    idx_arr = [ proj_dict['out_catl_p'], param_dict['sample'],
                GM_str                 , param_dict['rpmin'],
                param_dict['rpmax']    , param_dict['nrpbins'],
                param_dict['corr_type'], param_dict['pimax'],
                param_dict['Mg_bin']   , param_dict['perf_str'],
                param_dict['ngals_min'], catl_name]
    catl_idx_file = '{0}/Mr{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}.p'
    catl_idx_file = catl_idx_file.format(*idx_arr)
    ## Reading in file
    # Pair-counting for each galaxy group
    zz = 0
    # Removing file if needed
    if (os.path.isfile(catl_idx_file)) and (param_dict['remove_files']):
        os.remove(catl_idx_file)
        print('{0} Removing `catl_idx_file`: {1}'.format(
            Prog_msg, catl_idx_file))
    if (os.path.isfile(catl_idx_file)):
        try:
            catl_idx_pickle = pickle.load(open(catl_idx_file,'rb'))
            print('catl_idx_file: `{0}`'.format(catl_idx_file))
            group_idx_arr, rpbins_npairs_tot = catl_idx_pickle
        except ValueError:
            os.remove(catl_idx_file)
            print('{0} Removing `catl_idx_file`:{1}'.format(
                Prog_msg, catl_idx_file))
            ## Running `wp(rp)` for pair counting
            ## Looping over all galaxy groups
            # ProgressBar properties
            if param_dict['prog_bar']:
                widgets   = [Bar('>'), 'DDrppi Groups:', ETA(), ' ', ReverseBar('<')]
                pbar_mock = ProgressBar( widgets=widgets, maxval= 10*ngroups).start()
            for ii, group_ii in enumerate(groupid_unq):
                # DataFrame for `group_ii`
                group_df  = df_bin_org.loc[df_bin_org[id_key]==group_ii]
                group_idx = group_df.index.values
                ## Pair Counting
                gm_rp_idx, gm_rp_npairs = wp_idx_calc(group_df, param_dict)
                ## Converting to galaxy indices and checkin total number pairs
                if num.sum(gm_rp_npairs) != 0:
                    # Converting to galaxy indices
                    rp_idx_arr = num.array([group_idx[xx] for xx in gm_rp_idx])
                    # Total number of pairs
                    rpbins_npairs_tot += gm_rp_npairs
                    # Saving idx's and galaxy properties into arrays
                    if zz==0:
                        group_idx_arr = rp_idx_arr.copy()
                    else:
                        group_idx_arr = num.array([num.append(
                            group_idx_arr[x],rp_idx_arr[x],0) \
                            for x in range(len(gm_rp_idx))])
                    ## Increasing `zz`
                    if param_dict['prog_bar']:
                        pbar_mock.update(10*zz)
                    zz += int(1)
            if param_dict['prog_bar']:
                pbar_mock.finish()
            ## Saving indices into a Pickle file if file does not exist
            if num.sum(rpbins_npairs_tot) != 0:
                pickle.dump([group_idx_arr, rpbins_npairs_tot],
                    open(catl_idx_file,'wb'))
            else:
                corrfunc     = num.zeros(param_dict['nrpbins'])
                corrfunc [:] = num.nan
                corrfunc_seg = corrfunc.copy()
                npairs_tot   = num.sum(rpbins_npairs_tot)
                corrfunc_sh_tot = corrfunc.copy()
                mark_nanmean    = corrfunc.copy()
                mark_nanstd  = corrfunc.copy()
                sigma_arr       = num.zeros(2*param_dict['nrpbins']).reshape(
                    2, param_dict['nrpbins'])
                sigma_arr[:]    = num.nan
                sigma1_arr      = sigma_arr.copy()
                sigma2_arr      = sigma_arr.copy()
                sigma3_arr      = sigma_arr.copy()
                # Converting sigma's to dictionary
                sigma_dict = {}
                for jj in range(3):
                    sigma_dict[jj] = sigma_arr.copy()
                ## Converting to dictionaries
                # Conformity Only
                mcf_dict = {}
                mcf_dict['mcf'         ] = corrfunc
                mcf_dict['npairs'      ] = npairs_tot
                mcf_dict['npairs_rp'   ] = rpbins_npairs_tot
                mcf_dict['mcf_sh_mean' ] = mark_nanmean
                mcf_dict['mcf_sh_std'  ] = mark_nanstd
                mcf_dict['sigma'       ] = sigma_dict
                mcf_dict['mcf_sh'      ] = corrfunc_sh_tot
                # Conformity + Segregation
                mcf_dict_conf_seg = {}
                mcf_dict_conf_seg['mcf'         ] = corrfunc
                mcf_dict_conf_seg['npairs'      ] = npairs_tot
                mcf_dict_conf_seg['npairs_rp'   ] = rpbins_npairs_tot
                mcf_dict_conf_seg['mcf_sh_mean' ] = mark_nanmean
                mcf_dict_conf_seg['mcf_sh_std'  ] = mark_nanstd
                mcf_dict_conf_seg['sigma'       ] = sigma_dict
                mcf_dict_conf_seg['mcf_sh'      ] = corrfunc_sh_tot

                return mcf_dict_conf, mcf_dict_conf_seg, ngroups
    else:
        ## Running complete analysis
        ## Running `wp_idx_calc` for pair counting
        ## Looping over all galaxy groups
        # ProgressBar properties
        if param_dict['prog_bar']:
            widgets   = [Bar('>'), 'DDrppi Groups: ', ETA(), ' ', ReverseBar('<')]
            pbar_mock = ProgressBar( widgets=widgets, maxval= 10*ngroups).start()
        for ii, group_ii in enumerate(groupid_unq):
            # DataFrame for `group_ii`
            group_df  = df_bin_org.loc[df_bin_org[id_key]==group_ii]
            group_idx = group_df.index.values
            ## Pair Counting
            gm_rp_idx, gm_rp_npairs = wp_idx_calc(group_df, param_dict)
            ## Converting to galaxy indices and checkin total number pairs
            if num.sum(gm_rp_npairs) != 0:
                # Converting to galaxy indices
                rp_idx_arr = num.array([group_idx[xx] for xx in gm_rp_idx])
                # Total number of pairs
                rpbins_npairs_tot += gm_rp_npairs
                # Saving idx's and galaxy properties into arrays
                if zz==0:
                    group_idx_arr = rp_idx_arr.copy()
                else:
                    group_idx_arr = num.array([num.append(
                        group_idx_arr[x], rp_idx_arr[x],0) \
                        for x in range(len(gm_rp_idx))])
                ## Increasing `zz`
                if param_dict['prog_bar']:
                    pbar_mock.update(10*zz)
                zz += int(1)
        if param_dict['prog_bar']:
            pbar_mock.finish()
        ## Saving indices into a Pickle file if file does not exist
        if num.sum(rpbins_npairs_tot) != 0:
            pickle.dump([group_idx_arr, rpbins_npairs_tot],
                open(catl_idx_file,'wb'))
        else:
            corrfunc     = num.zeros(param_dict['nrpbins'])
            corrfunc [:] = num.nan
            corrfunc_seg = corrfunc.copy()
            npairs_tot   = num.sum(rpbins_npairs_tot)
            corrfunc_sh_tot = corrfunc.copy()
            mark_nanmean    = corrfunc.copy()
            mark_nanstd  = corrfunc.copy()
            sigma_arr       = num.zeros(2*param_dict['nrpbins']).reshape(
                2, param_dict['nrpbins'])
            sigma_arr[:]    = num.nan
            sigma1_arr      = sigma_arr.copy()
            sigma2_arr      = sigma_arr.copy()
            sigma3_arr      = sigma_arr.copy()
            # Converting sigma's to dictionary
            sigma_dict = {}
            for jj in range(3):
                sigma_dict[jj] = sigma_arr.copy()
            ## Converting to dictionaries
            # Conformity Only
            mcf_dict_conf = {}
            mcf_dict_conf['mcf'         ] = corrfunc
            mcf_dict_conf['npairs'      ] = npairs_tot
            mcf_dict_conf['npairs_rp'   ] = rpbins_npairs_tot
            mcf_dict_conf['mcf_sh_mean' ] = mark_nanmean
            mcf_dict_conf['mcf_sh_std'  ] = mark_nanstd
            mcf_dict_conf['sigma'       ] = sigma_dict
            mcf_dict_conf['mcf_sh'      ] = corrfunc_sh_tot
            # Conformity + Segregation
            mcf_dict_conf_seg = {}
            mcf_dict_conf_seg['mcf'         ] = corrfunc
            mcf_dict_conf_seg['npairs'      ] = npairs_tot
            mcf_dict_conf_seg['npairs_rp'   ] = rpbins_npairs_tot
            mcf_dict_conf_seg['mcf_sh_mean' ] = mark_nanmean
            mcf_dict_conf_seg['mcf_sh_std'  ] = mark_nanstd
            mcf_dict_conf_seg['sigma'       ] = sigma_dict
            mcf_dict_conf_seg['mcf_sh'      ] = corrfunc_sh_tot

            return mcf_dict_conf, mcf_dict_conf_seg, ngroups
    ###
    ### --- | Marked Correlation Function - Calculations | --- ###
    ###
    ##
    ## MCF for `Conformity Only`
    mcf_dict_conf     = MCF_conf(   prop,
                                    df_bin_org,
                                    group_idx_arr,
                                    rpbins_npairs_tot,
                                    param_dict,
                                    catl_keys_dict)
    ##
    ## MCF for `Conformity + Segregation`
    mcf_dict_conf_seg = MCF_conf_seg(prop,
                                    df_bin_org,
                                    group_idx_arr,
                                    rpbins_npairs_tot,
                                    param_dict,
                                    catl_keys_dict)
    ##
    ## Saving to dictionaries to new variables

    return mcf_dict_conf, mcf_dict_conf_seg, ngroups

def halo_corr(catl_pd, catl_name, param_dict, proj_dict):
    """
    1-halo mark correlation function for galaxy groups in each group mass bin.

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
    ## Program message
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
    # Cleaning catalogue with groups of N > `ngals_min`
    catl_pd_clean = cu.sdss_catl_clean_nmin(catl_pd, param_dict['catl_kind'],
        nmin=param_dict['ngals_min'])
    ### Mass limits
    GM_min  = catl_pd_clean[gm_key].min()
    GM_max  = catl_pd_clean[gm_key].max()
    GM_arr  = cu.Bins_array_create([GM_min,GM_max], param_dict['Mg_bin'])
    GM_bins = [[GM_arr[ii],GM_arr[ii+1]] for ii in range(GM_arr.shape[0]-1)]
    GM_bins = num.asarray(GM_bins)
    ## Pickle file
    p_arr = [   proj_dict['pickdir']   , param_dict['fig_idx']       ,
                catl_name              , 
                param_dict['corr_type'], param_dict['sample']        ,
                param_dict['itern_tot'], param_dict['rpmin']         ,
                param_dict['rpmax']    , param_dict['nrpbins']       ,
                param_dict['pimax']    , param_dict['corr_pair_type'],
                param_dict['prop_log'] , param_dict['shuffle_marks'] ,
                param_dict['ngals_min'], param_dict['perf_str'] ]
    p_fname = '{0}{1}_{2}_{3}_corr_Mr{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}.p'
    p_fname = p_fname.format(*p_arr)
    ##
    ## Checking if file exists
    if (os.path.isfile(p_fname)) and (param_dict['remove_files']):
        os.remove(p_fname)
        print('{0} `p_fname` ({1}) removed! Calculating MCF statistics!'.format(
            Prog_msg, p_fname))
    ## Dictionary for storing results for each GM bin
    GM_prop_dict = {}
    # Looping over each GM bin (if file doesn't exist)
    if not (os.path.isfile(p_fname)):
        ## Executing MCF calculations if file does not exist for the give 
        ## catalogue.
        for ii, GM_ii in enumerate(GM_bins):
            # GM string
            GMbin_min, GMbin_max = GM_ii
            GM_str = '{0:.2f}_{1:.2f}'.format(GMbin_min, GMbin_max)
            if param_dict['perf_opt']:
                print('\n{0} Halo Mass range: {1}'.format(Prog_msg, GM_str))
            else:
                print('\n{0} Group Mass range: {1}'.format(Prog_msg, GM_str))
            ## Galaxies in Group-mass bin
            df_bin_org = catl_pd_clean.loc[ (catl_pd_clean[gm_key] >= GMbin_min) &\
                                        (catl_pd_clean[gm_key] <  GMbin_max)].copy()
            df_bin_org.reset_index(inplace=True, drop=True)
            ##
            ## Creating dictionary that contains results for all galaxy properties
            stat_vals = [[] for x in range(3)]
            prop_dict = dict(zip(pd_keys,[list(stat_vals) for x in range(len(pd_keys))]))
            ## Looping over galaxy properties
            for jj, prop in enumerate(pd_keys):
                print('{0} >> Galaxy Prop: {1}'.format(Prog_msg, prop))
                mcf_dict_conf, \
                mcf_dict_conf_seg, \
                ngroups = prop_sh_one_halo(df_bin_org,
                                prop,
                                GM_str,
                                param_dict,
                                proj_dict,
                                catl_name,
                                catl_keys_dict)
                ##
                ## Saving results to dictionary
                prop_dict[prop][0] = mcf_dict_conf
                prop_dict[prop][1] = mcf_dict_conf_seg
                prop_dict[prop][2] = ngroups
            ##
            ## Saving results to final dictionary
            GM_prop_dict[GM_str] = prop_dict
    ##
    ## saving data to Pickle file
    print('{0} Saving data to Pickle: \n\t{1}\n'.format(Prog_msg, p_fname))
    p_data = [param_dict, proj_dict, GM_prop_dict, catl_name, GM_arr]
    pickle.dump(p_data, open(p_fname,'wb'))
    print('{0} Data saved to Pickle: \n\t{1}\n'.format(Prog_msg, p_fname))

def main(args):
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
    ## Checking for correct input
    param_vals_test(param_dict)
    ## ---- Adding to `param_dict` ---- 
    param_dict = add_to_dict(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ## Creating Folder Structure
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths('./'))
    ## Choosing cosmological model
    cosmo_model = cosmo_create(cosmo_choice=param_dict['cosmo_choice'])
    # Assigning the cosmological model to `param_dict`
    param_dict['cosmo_model'] = cosmo_model
    # Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ## Running analysis
    # Reading catalogues
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
    if ncatls==1:
        # Looping over catalogues
        catl_ii = catl_arr[0]     
        print('{0} Analyzing `{1}`\n'.format(Prog_msg, catl_ii))
        ## Extracting `name` of the catalogue
        catl_name = os.path.splitext(os.path.split(catl_ii)[1])[0]
        ## Converting to pandas DataFrame
        catl_pd   = cu.read_hdf5_file_to_pandas_DF(catl_ii)
        ## Computing cartesian coordinates
        catl_pd = spherical_to_cart(catl_pd,
                                    param_dict['cosmo_model'],
                                    method=param_dict['cart_method'])
        # MCF Calculations
        halo_corr(catl_pd, catl_name, param_dict, proj_dict)
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
        start_time = datetime.now()
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
        end_time = datetime.now()
        total_time = end_time - start_time
        print('{0} Total Time taken (Create): {1}'.format(Prog_msg, total_time))

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
        print('{0} Analyzing `{1}`\n'.format(Prog_msg, catl_ii))
        catl_name = os.path.splitext(os.path.split(catl_ii)[1])[0]
        catl_pd   = cu.read_hdf5_file_to_pandas_DF(catl_ii)
        ## Computing cartesian coordinates
        catl_pd = spherical_to_cart(catl_pd,
                                    param_dict['cosmo_model'], 
                                    method=param_dict['cart_method'])
        # MCF Calculations
        halo_corr(catl_pd, catl_name, param_dict, proj_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
