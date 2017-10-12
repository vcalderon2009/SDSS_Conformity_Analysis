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
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)

# Extra-modules
import argparse
from argparse import ArgumentParser
import copy
from datetime import datetime
import subprocess
import astropy.cosmology as astrocosmo
import astropy.constants as ac
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance

## Cython modules
from pair_counter_rp import pairwise_distance_rp

## Functions
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

def halo_corr(catl_pd, catl_name, param_dict, proj_dict, nmin=2,
    Prog_msg = '1 >>>  '):
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
    elif param_dict['catl_kind']==mocks:
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
                param_dict['perf_str'] ]
    p_fname = '{0}{1}_{2}_{3}_corr_Mr{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}.p'
    p_fname = p_fname.format(*p_arr)
    ## Dictionary for storing results for each GM bin
    GM_prop_dict = {}
    # Looping over each GM bin
    for ii, GM_ii in enumerate(GM_bins):
        # GM string
        GMbin_min, GMbin_max = GM_ii
        GM_str = '{0:.2f}_{1:.2f}'.format(GMbin_min, GMbin_max)
        if param_dict['perf_opt']:
            print('{0} Halo Mass range: {1}'.format(Prog_msg, GM_str))
        else:
            print('{0} Group Mass range: {1}'.format(Prog_msg, GM_str))
        ## Galaxies in Group-mass bin
        df_bin_org = catl_pd_clean.loc[ (catl_pd_clean[gm_key] >= GMbin_min) &\
                                    (catl_pd_clean[gm_key] <  GMbin_max)].copy()
        df_bin_org.reset_index(inplace=True, drop=True)
        ## Looping over galaxy properties
        for jj, prop in enumerate(pd_keys):
            print('{0} >> Galaxy Prop: {1}'.format(Prog_msg, prop))
            prop_sh_one_halo()

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
        
def MCF_conf_seg(df_bin, group_idx_arr, rpbins_npairs_tot, 
    param_dict, catl_keys_dict):
    """
    Marked correlation function calculation for the case,
    where `Conformity + Segregation` is considered
    In here, we include the segregation effect of satellites, and 
    thus, it does not show the "true" signal of galactic conformity 
    alone.

    Parameters
    -----------
    df_bin: pandas DataFrame
    
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
    widgets   = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
    pbar_mock = ProgressBar( widgets=widgets, maxval= 10 * itern).start()
    for ii in range(param_dict['itern_tot']):
        ##
        ## Copying default `prop` array to DataFrame df_bin_sh`
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
        pbar_mock.update(10*ii)
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



    
    


def prop_sh_one_halo(df_bin_org, prop, GM_str, catl_name, catl_keys_dict,
    Prog_msg = '1 >>>  '):
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

    catl_name: string
        prefix of the catalogue being analyzed

    catl_keys_dict: python dictionary
        dictionary containing keys for the galaxy properties in catalogue
    
    Returns
    ----------

    """
    ## Creating new DataFrame
    df_bin = df_bin_org.copy()
    ## Constants
    Cens = int(1)
    Sats = int(0)
    ## Galaxy Property Labels
    prop_orig = prop + '_orig'
    prop_seg  = prop + '_seg'
    ## Catalogue Variables for galaxy properties
    gm_key      = catl_keys_dict['gm_key']
    id_key      = catl_keys_dict['id_key']
    galtype_key = catl_keys_dict['galtype_key']
    ## Group statistics
    groupid_unq = df_bin[id_key].unique()
    ngroups     = groupid_unq.shape[0]
    ## Total number of galaxy pairs in `rp`
    rpbins_npairs_tot = num.zeros(param_dict['nrpbins'])
    ## Pickle file - name - for wp(rp)
    idx_arr = [ proj_dict['out_catl_p'], param_dict['sample'],
                GM_str                 , param_dict['rpmin'],
                param_dict['rpmax']    , param_dict['nrpbins'],
                param_dict['corr_type'], param_dict['pimax'],
                param_dict['Mg_bin']   , param_dict['perf_str'],
                catl_name]
    catl_idx_file = '{0}/Mr{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}.p'
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
            widgets   = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
            pbar_mock = ProgressBar( widgets=widgets, maxval= 10*ngroups).start()
            for ii, group_ii in enumerate(groupid_unq):
                # DataFrame for `group_ii`
                group_df  = df_bin.loc[df_bin[id_key]==group_ii]
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
                    pbar_mock.update(10*zz)
                    zz += int(1)
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

                return (corrfunc, sigma1_arr, sigma2_arr, sigma3_arr, ngroups,
                        npairs_tot, rpbins_npairs_tot, corrfunc_sh_tot, 
                        mark_nanmean, mark_nanstd, corrfunc_seg, sigma1_arr, 
                        sigma2_arr, sigma3_arr)
    else:
        ## Running complete analysis
        ## Running `wp_idx_calc` for pair counting
        ## Looping over all galaxy groups
        # ProgressBar properties
        widgets   = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
        pbar_mock = ProgressBar( widgets=widgets, maxval= 10*ngroups).start()
        for ii, group_ii in enumerate(groupid_unq):
            # DataFrame for `group_ii`
            group_df  = df_bin.loc[df_bin[id_key]==group_ii]
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
                pbar_mock.update(10*zz)
                zz += int(1)
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

            return (corrfunc, sigma1_arr, sigma2_arr, sigma3_arr, ngroups,
                    npairs_tot, rpbins_npairs_tot, corrfunc_sh_tot, 
                    mark_nanmean, mark_nanstd, corrfunc_seg, sigma1_arr, 
                    sigma2_arr, sigma3_arr)
    ###
    ### --- | Marked Correlation Function - Calculations | --- ###
    ###
    ## Type of Correlation pair
    if param_dict['corr_pair_type']=='cen_sat':
        corr_pair_int = Cens + Sats
    elif param_dict['corr_pair_type']=='sat_sat':
        corr_pair_int = Sats + Sats
    ## Correcting for `log` in `prop`
    nonlog_arr = ['logssfr', 'logMstar_JHU']
    if (prop in nonlog_arr) and (param_dict['prop_log']=='nonlog'):
        df_bin.loc[:,prop] = 10**df_bin[prop].values
    ## Galaxy type, `prop`, and `original` arrays
    df_galtype_arr  = df_bin[galtype_key].values.copy()
    df_gal_idx_arr  = df_bin.index.values.copy()
    df_galprop_orig = copy.deepcopy(df_bin[prop].copy().values)
    # Assigning to DataFrame
    df_bin.loc[:,prop_orig] = df_galprop_orig.copy()
    df_bin.loc[:,prop_seg ] = df_galprop_orig.copy()
    ## Normalizing Cens and Sats - with Segregation
    cens_prop_mean = df_bin.loc[df_bin[galtype_key]==Cens,prop].mean()
    sats_prop_mean = df_bin.loc[df_bin[galtype_key]==Sats,prop].mean()
    df_bin.loc[df_bin[galtype_key]==Cens, prop] /= cens_prop_mean
    df_bin.loc[df_bin[galtype_key]==Sats, prop] /= sats_prop_mean
    ## New normalized `gal_prop_arr`
    gal_prop_arr = df_bin.loc[:,prop].copy().values
    ## Determining indices of centrals and satellites
    cen_idx = df_bin.loc[df_bin[galtype_key]==Cens].index
    sat_idx = df_bin.loc[df_bin[galtype_key]==Sats].index
    ###
    ### --- Types of Galaxy Pairs to Keep - Shuffles--- ###
    ###
    ## Type of Galaxy Pair to analyze
    if param_dict['corr_pair_type']=='all':
        ## Selecting all types of galaxy pairs, i.e. `cen-sat`, `sat-sat`
        ## Note: There are no `cen-cen` since it's only 1-halo term
        #
        # Getting the average of the galaxy property `prop` for all 
        # galaxy pairs
        corrfunc = num.array([num.sum(  gal_prop_arr[xx.T[0]]*\
                                        gal_prop_arr[xx.T[1]])\
                                        for xx in group_idx_arr])/\
                                        rpbins_npairs_tot
        npairs_tot = num.sum(rpbins_npairs_tot).astype(int)
        ###
        ### ---| Shuffling galaxy properties |--- ###
        corr_sh_tot = num.zeros((param_dict['nrpbins'],1))
        ## Only selecting 2 columns of original DataFrame
        df_bin_sh   = df_bin[[prop, galtype_key]].copy()
        # ProgressBar properties
        widgets   = [Bar('>'), ' ', ETA(), '1-halo Shuffling ', ReverseBar('<')]
        pbar_mock = ProgressBar( widgets=widgets, 
                                maxval= 10*param_dict['itern_tot']).start()
        ## Iterating `itern_tot` times
        for kk in range(0,param_dict['itern_tot']):
            ## Resetting `prop` column to default
            df_bin_sh.loc[:,prop] = copy.deepcopy(gal_prop_arr)
            ## Shuffling galaxy `marks`
            if param_dict['shuffle_marks']=='censat_sh':
                ## Shuffling centrals' and satellites' `prop` arrays
                # Centrals
                mark_sh_cen = df_bin_sh.loc[cen_idx, prop].values
                num.random.shuffle(mark_sh_cen)
                df_bin_sh.loc[cen_idx, prop] = mark_sh_cen
                # Satellites
                mark_sh_sat = df_bin_sh.loc[sat_idx, prop].values
                num.random.shuffle(mark_sh_sat)
                df_bin_sh.loc[sat_idx, prop] = mark_sh_sat
            ## Shuffling only centrals' `prop` array
            if param_dict['shuffle_marks']=='cen_sh':
                # Centrals
                mark_sh_cen = df_bin_sh.loc[cen_idx, prop].values
                num.random.shuffle(mark_sh_cen)
                df_bin_sh.loc[cen_idx, prop] = mark_sh_cen
            ## Shuffling only satellites' `prop` array
            if param_dict['shuffle_marks']=='sat_sh':
                # Satellites
                mark_sh_sat = df_bin_sh.loc[sat_idx, prop].values
                num.random.shuffle(mark_sh_sat)
                df_bin_sh.loc[sat_idx, prop] = mark_sh_sat
            ####
            #### Normalizing Cens and Sats by mean of the cens or sats
            gal_prop_sh = df_bin_sh.loc[:,prop].values
            ## Average of galaxy property
            corr_sh = num.array([gal_prop_sh[xx.T[0]]*
                                 gal_prop_sh[xx.T[1]] 
                                 for xx in group_idx_arr])/\
                                 rpbins_npairs_tot
            ## Inserting mark values into array
            corrfunc_sh_tot = num.insert(corrfunc_sh_tot,
                                         len(corrfunc_sh_tot.T),
                                         corr_sh,
                                         1)
            ## Updating progress bar
            pbar_mock.update(10*kk)
        pbar_mock.finish()
        ## Creating `corrfunc` for segregation
        corrfunc_seg = corrfunc.copy()
    else:
        ## Selecting the chosen type of galaxy pairs, i.e. `cen-sat`, `sat-sat`
        ## Note: There are no `cen-cen` since it's only 1-halo term
        ##
        ## ---| Conformity + Segregation |---
        # Addition of Galaxy Pairs Integers
        galtype_sum_arr = num.array([df_galtype_arr[xx.T[0]]+
                                     df_galtype_arr[xx.T[1]]
                                     for xx in group_idx_arr])
        # Product of Galxy Pairs `prop` arrays
        corrfunc_prod = num.array([ gal_prop_arr[xx.T[0]]*
                                    gal_prop_arr[xx.T[1]]
                                    for xx in group_idx_arr])
        # Selecting only galaxy pairs based on `corr_pair_type`
        corrfunc_prod_sel = num.array([
                            corrfunc_prod[kk][galtype_sum_arr[kk]==corr_pair_int]
                            for kk in range(len(group_idx_arr))])
        # Total number of galaxy pairs in `rp`-bins
        rpbins_npairs_tot_sel = num.array(
                                    [len(xx) for xx in corrfunc_prod_sel]
                                    ).astype(float)
        # Normalizing MCF for selected galaxy pairs
        corrfunc   = num.array([num.sum(xx) for xx in corrfunc_prod_sel])
        corrfunc  /= rpbins_npairs_tot_sel
        npairs_tot = num.sum(rpbins_npairs_tot_sel)
        ###
        ### -- Shuffling of Galaxy `prop` -- ###
        ###
        ## Initializing arrays for the `shuffle` case
        corrfunc_sh_tot     = num.zeros((param_dict['nrpbins'],1))
        ## Only selecting 2 columns 
        df_bin_sh     = df_bin[[prop, prop_seg, galtype_key]].copy()
        ##
        ## Looping over iterations to estimate the spread of the shuffles
        for ii in range(param_dict['itern_tot']):
            ###
            ### ---| Conformity + Segregation |---
            ###
            # Copying default `prop` array to DataFrame
            df_bin_sh.loc[:,prop] = copy.deepcopy(gal_prop_arr)
            # Types of Shuffles
            if param_dict['shuffle_marks']=='censat_sh':
                ## Shuffling centrals' and satellites' `prop` arrays
                # Centrals
                mark_sh_cen = df_bin_sh.loc[cen_idx, prop].copy().values
                num.random.shuffle(mark_sh_cen)
                df_bin_sh.loc[cen_idx, prop] = mark_sh_cen
                # Satellites
                mark_sh_sat = df_bin_sh.loc[sat_idx, prop].copy().values
                num.random.shuffle(mark_sh_sat)
                df_bin_sh.loc[sat_idx, prop] = mark_sh_sat
            ## Shuffling only centrals' `prop` array
            if param_dict['shuffle_marks']=='cen_sh':
                # Centrals
                mark_sh_cen = df_bin_sh.loc[cen_idx, prop].copy().values
                num.random.shuffle(mark_sh_cen)
                df_bin_sh.loc[cen_idx, prop] = mark_sh_cen
            ## Shuffling only satellites' `prop` array
            if param_dict['shuffle_marks']=='sat_sh':
                # Satellites
                mark_sh_sat = df_bin_sh.loc[sat_idx, prop].copy().values
                num.random.shuffle(mark_sh_sat)
                df_bin_sh.loc[sat_idx, prop] = mark_sh_sat
            ##
            ## Saving new galaxy properties array for shuffles
            gal_prop_sh = df_bin_sh[prop].copy().values
            ##
            ## Product of `marks`
            corrfunc_prod_sh = num.array([  gal_prop_sh[xx.T[0]]*
                                            gal_prop_sh[xx.T[1]]
                                            for xx in group_idx_arr])
            ## Choosing only selected `Galaxy Pairs`
            corrfunc_prod_sel_sh = num.array([  corrfunc_prod_sh[xx.T[0]]*
                                                corrfunc_prod_sh[xx.T[1]]
                                                for xx in galtype_sum_arr])
            ## Normalizing MCF
            corrfunc_sh  = num.array([num.sum(xx) for xx in corrfunc_prod_sel_sh])
            corrfunc_sh /= rpbins_npairs_tot_sel
            ## Inserting `corrfunc_sh` to MCF
            corrfunc_sh_tot = num.insert(corrfunc_sh_tot,
                                         len(corrfunc_sh_tot.T),
                                         corrfunc_sh,
                                         1)



        ##
        ## ---| Only Conformity (Segregation effect removed) |---
        ##
        # Selecting the indices that match the criteria `corr_pair_int`
        galtype_sum_seg = num.array([num.where(xx==corr_pair_int)[0] 
                                    for xx in galtype_sum_arr])
        # Galaxy indices of matching pairs
        gal_idx_seg = num.array([group_idx_arr[kk][galtype_sum_seg[kk]]
                                for kk in range(len(group_idx_arr))])
        # Unique galaxy indices of matching pairs
        gal_idx_seg_unq = num.array([num.unique(xx.T) for xx in gal_idx_seg])
        # Selecting the `galtype` of each unique galaxy in `gal_idx_seg_unq`
        galtype_corrpair_seg = num.array([df_galtype_arr[xx]
                                        for xx in gal_idx_seg_unq])
        ## -- Centrals
        # Normalizing centrals
        prop_cens_mean_seg = df_bin.loc[df_bin[galtype_key]==Cens,prop_seg].mean()
        df_bin.loc[df_bin[galtype_key]==Cens,prop_seg] /= prop_cens_mean_seg
        ## -- Satellites
        # Indices of satellite galaxies in matched galaxy pairs
        gal_idx_sats_seg = num.array([
            gal_idx_seg_unq[kk][galtype_corrpair_seg[kk]==Sats]
            for kk in range(len(gal_idx_seg_unq))])
        # Mean of galaxy `prop` for only Satellites
        prop_sats_mean_seg = num.array([num.mean(df_galprop_orig[xx]) 
                                        for xx in gal_idx_sats_seg])
        # Normalizing Satellites in each `rp`bin
        for rp_ii in range(len(gal_idx_seg_unq)):
            df_bin.loc[gal_idx_sats_seg[rp_ii],prop_seg] /= prop_sats_mean_seg[rp_ii]
        ##
        ## Computing MCF with `segregation removed`
        # Making copy of `prop` for segregation case
        galprop_arr_seg = df_bin[prop_seg].copy().values
        ## Computing MCF products
        corrfunc_prod_seg = num.array([ galprop_arr_seg[xx.T[0]]*
                                        galprop_arr_seg[xx.T[1]]
                                        for xx in gal_idx_seg])
        ## Total number of pairs in each `rp` bin
        rpbins_npairs_tot_sel_seg = num.array([len(xx) for xx in corrfunc_prod_seg])
        rpbins_npairs_tot_sel_seg = rpbins_npairs_tot_sel_seg.astype(float)
        ##
        ## Normalizing MCF
        ##
        corrfunc_seg = (num.array([num.sum(xx) for xx in corrfunc_prod_seg])/
                                    rpbins_npairs_tot_sel_seg)
        npairs_tot_seg = num.sum(rpbins_npairs_tot_sel_seg)



        ###
        ### -- Shuffling of Galaxy `prop` -- ###
        ###
        ## Initializing arrays for the `shuffle` case
        corrfunc_sh_tot     = num.zeros((param_dict['nrpbins'],1))
        corrfunc_sh_tot_seg = num.zeros((param_dict['nrpbins'],1))
        ## Only selecting 2 columns 
        df_bin_sh     = df_bin[[prop, prop_seg, galtype_key]].copy()
        ##
        ## Looping over iterations to estimate the spread of the shuffles
        for ii in range(param_dict['itern_tot']):
            ###
            ### ---| Conformity + Segregation |---
            ###
            # Copying default `prop` array to DataFrame
            df_bin_sh.loc[:,prop] = copy.deepcopy(gal_prop_arr)
            # Types of Shuffles
            if param_dict['shuffle_marks']=='censat_sh':
                ## Shuffling centrals' and satellites' `prop` arrays
                # Centrals
                mark_sh_cen = df_bin_sh.loc[cen_idx, prop].copy().values
                num.random.shuffle(mark_sh_cen)
                df_bin_sh.loc[cen_idx, prop] = mark_sh_cen
                # Satellites
                mark_sh_sat = df_bin_sh.loc[sat_idx, prop].copy().values
                num.random.shuffle(mark_sh_sat)
                df_bin_sh.loc[sat_idx, prop] = mark_sh_sat
            ## Shuffling only centrals' `prop` array
            if param_dict['shuffle_marks']=='cen_sh':
                # Centrals
                mark_sh_cen = df_bin_sh.loc[cen_idx, prop].copy().values
                num.random.shuffle(mark_sh_cen)
                df_bin_sh.loc[cen_idx, prop] = mark_sh_cen
            ## Shuffling only satellites' `prop` array
            if param_dict['shuffle_marks']=='sat_sh':
                # Satellites
                mark_sh_sat = df_bin_sh.loc[sat_idx, prop].copy().values
                num.random.shuffle(mark_sh_sat)
                df_bin_sh.loc[sat_idx, prop] = mark_sh_sat
            ##
            ## Saving new galaxy properties array for shuffles
            gal_prop_sh = df_bin_sh[prop].copy().values
            ##
            ## Product of `marks`
            corrfunc_prod_sh = num.array([  gal_prop_sh[xx.T[0]]*
                                            gal_prop_sh[xx.T[1]]
                                            for xx in group_idx_arr])
            ## Choosing only selected `Galaxy Pairs`
            corrfunc_prod_sel_sh = num.array([  corrfunc_prod_sh[xx.T[0]]*
                                                corrfunc_prod_sh[xx.T[1]]
                                                for xx in galtype_sum_arr])
            ## Normalizing MCF
            corrfunc_sh  = num.array([num.sum(xx) for xx in corrfunc_prod_sel_sh])
            corrfunc_sh /= rpbins_npairs_tot_sel
            ## Inserting `corrfunc_sh` to MCF
            corrfunc_sh_tot = num.insert(corrfunc_sh_tot,
                                         len(corrfunc_sh_tot.T),
                                         corrfunc_sh,
                                         1)
            ###
            ### ---| Only Conformity (Segregation effect removed) |---
            ###
            ## Restoring `prop_seg` to original values
            ## Each time, the `prop_seg` with be normalized, after having 
            ## removed the `segregation` signature in satellites










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
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths('./'))
    ## Choosing cosmological model
    cosmo_model = cosmo_create(cosmo_choice=param_dict['cosmo_choice'])
    ## Running analysis
    # Reading catalogues
    catl_arr_all = cu.extract_catls(catl_kind=param_dict['catl_kind'],
                                    catl_type=param_dict['catl_type'],
                                    sample_s =param_dict['sample_s'],
                                    perf_opt =param_dict['perf_opt'],
                                    catl_info='members',
                                    print_filedir=False)
    catl_arr = catl_arr_all[param_dict['catl_start']:param_dict['catl_finish']]
    # Looping over catalogues
    for ii, catl_ii in enumerate(catl_arr):
        print('{0} Analyzing `{1}`\n'.format(Prog_msg, catl_ii))
        catl_name = os.path.splitext(os.path.split(catl_ii)[1])[0]
        catl_pd   = cu.read_hdf5_file_to_pandas_DF(catl_ii)
        ## Computing cartesian coordinates
        catl_pd = spherical_to_cart(catl_pd,
                                    cosmo_model, 
                                    method=param_dict['cart_method'])
        # MCF Calculations
        halo_corr(catl_pd, catl_name, param_dict, proj_dict,
                    nmin=param_dict['ngals_min'])




# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)

























