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
import subprocess

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
        df_bin = catl_pd_clean.loc[ (catl_pd_clean[gm_key] >= GMbin_min) &\
                                    (catl_pd_clean[gm_key] <  GMbin_max)]
        df_bin.reset_index(inplace=True, drop=True)
        ## Looping over galaxy properties
        for jj, prop in enumerate(pd_keys):
            print('{0} >> Galaxy Prop: {1}'.format(Prog_msg, prop))
            prop_sh_one_halo()

def wp_idx_calc(group_df, GM_str, catl_name, group_ii, param_dict, proj_dict,
    ext='txt', return_rpbin=False):
    """
    Computes the 2-point wp(rp) for the set of galaxies in `group_df`

    Parameters
    -----------
    group_df: pandas DataFrame
        DataFrame with info on galaxies from given group mass bin

    GM_str: string
        string for the current group/halo mass bin

    catl_name: string
        basename of the current catalogue being analyzed

    group_ii: int
        group number being analyzed

    param_dict: python dictionary
        dictionary with project variables

    proj_dict: python dictionary
        dictionary with paths to the project directory

    ext: string, optional (default = 'ext')
        extension of output files

    return_rpbin: boolean, optional (default = False)
        Option for returning an array of the rpbin of each galaxy

    Returns
    -----------
    rp_idx: array-like, shape (len(param_dict['nrpbins']),[])
        multi-dimensional array for the i-th and j-th elements for each pair

    rp_npairs: array_like, shape (len(param_dict['nrpbins']),[])
        array of the number of pairs in each rp-bin
    """
    ### wprp variables
    # Cosmology
    lasdamas_cosmo = int(1)
    ### Files for wp(rp)
    files_prefix_arr = [    param_dict['sample'] ,
                            GM_str               , group_ii            ,
                            param_dict['rpmin']  , param_dict['rpmax'] ,
                            param_dict['nrpbins'], param_dict['pimax'],
                            catl_name            , ext]
    files_prefix = 'Mr{0}_{1}_gal_radeccz_{2}_{3}_{4}_{5}_{6}_{7}.{8}'.format(
        *files_prefix_arr)
    # File with < ra dec cz> values for galaxies in group
    radeccz_file = '{0}/{1}'.format(proj_dict['out_ddrp'], files_prefix)
    # File with ith and jth indices of galaxy pairs
    galidx_file  = '{0}/{1}'.format(proj_dict['out_idx'], files_prefix)
    # File with the results from wp(rp)
    res_file     = '{0}/{1}'.format(proj_dict['out_res'], files_prefix)
    ## Checking if files exists
    if (not os.path.isfile(radeccz_file)) or (not os.path.isfile(galidx_file)):
        ## File with <ra dec cz>
        group_df[['ra','dec','cz']].to_csv(radeccz_file,sep=' ', index=False,
            header=None)
        cu.File_Exists(radeccz_file)
        ## wp(rp) Executable
        ddrp_exe = cu.get_code_c()+'corrfuncs/VC/DDrp_indices'
        cu.File_Exists(ddrp_exe)
        ## Command for executable and executing it
        ddrp_cmd = '{0} {1} {2} {3} {4} > {5}'.format(ddrp_exe,
            param_dict['rpmin'], param_dict['rpmax'], param_dict['nrpbins'],
            radeccz_file       , galidx_file )
        # Executing `ddrp_cmd` command
        print('\n{0}\n'.format(ddrp_cmd))
        subprocess.call(ddrp_cmd, shell=True)
        cu.File_Exists(galidx_file)
    ## Reading in output file
    DDrp_pd = pd.read_csv(galidx_file, delimiter='\s+', names=['rpbin','p','j'])
    # Checking lenghts
    if (DDrp_pd.shape[0]==0):
        os.remove(galidx_file)
    ## Unique `rp`-bins
    rpbins_unq = num.unique(DDrp_pd['rpbin'])
    ## Array of p-th and j-th indices
    rp_idx = [DDrp_pd.loc[DDrp_pd['rpbin']==xx,['p','j']].values for xx in \
                range(param_dict['nrpbins'])]
    rp_idx = num.array(rp_idx)
    ## Array of total number of pairs in `rp`-bins
    rp_npairs = num.array([len(xx) for xx in rp_idx])

    return rp_idx, rp_npairs

    
def prop_sh_one_halo(df_bin, prop, GM_str, catl_name, catl_keys_dict,
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
            gm_rp_idx, rpbins_npairs_tot = catl_idx_pickle
        except ValueError:
            os.remove(catl_idx_file)
            print('{0} Removing `catl_idx_file`:{1}'.format(
                Prog_msg, catl_idx_file))
            ## Running `wp(rp)` for pair counting
            ## Looping over all galaxy groups
            for ii, group_ii in enumerate(groupid_unq):
                # DataFrame for `group_ii`
                group_df  = df_bin.loc[df_bin[id_key]==group_ii]
                group_idx = group_df.index.values
                ## Pair Counting
                gm_rp_idx, gm_rp_npairs = wp_idx_calc(
                    group_df, GM_str, catl_name, group_ii, param_dict, 
                    proj_dict)
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
                    zz += int(1)
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
        ## Running `wp(rp)` for pair counting
        ## Looping over all galaxy groups
        for ii, group_ii in enumerate(groupid_unq):
            # DataFrame for `group_ii`
            group_df  = df_bin.loc[df_bin[id_key]==group_ii]
            group_idx = group_df.index.values
            ## Pair Counting
            gm_rp_idx, gm_rp_npairs = wp_idx_calc(
                group_df, GM_str, catl_name, group_ii, param_dict, 
                proj_dict)
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
                zz += int(1)
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
        # MCF Calculations
        halo_corr(catl_pd, catl_name, param_dict, proj_dict)




# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
