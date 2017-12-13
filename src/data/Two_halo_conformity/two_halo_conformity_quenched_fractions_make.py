#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 10/31/2017
# Last Modified: 11/07/2017
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, 1-halo MCF Make"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Script that runs the 2-halo conformity results and plots.
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
import pandas as pd

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import datetime

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

def get_parser():
    """
    Get parser object for `eco_mocks_create.py` script.

    Returns
    -------
    args: 
        input arguments to the script
    """
    ## Define parser object
    description_msg = 'Script that runs the 1-halo conformity results and plots.'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('-a', '--analysis',
                        dest='analysis_type',
                        help='Type of analysis to make',
                        type=str,
                        choices=['calc', 'plots'],
                        default='calc')
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cu.Program_Msg(__file__))
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help='Delete pickle files containing pair counts',
                        type=_str2bool,
                        default=False)
    ## Option for removing file from DDrp
    parser.add_argument('-remove-wp',
                        dest='remove_wp_files',
                        help='Delete pickle file containing pair counts from wp',
                        type=_str2bool,
                        default=False)
    ## CPU to use
    parser.add_argument('-cpu_frac',
                        dest='cpu_frac',
                        help='Fraction of CPUs to use',
                        type=float,
                        default=0.7)
    ## CPU to use
    parser.add_argument('-pimax',
                        dest='pimax',
                        help='Maximum parallel distance to find galaxy pairs',
                        type=float,
                        default=20.)
    ## Verbose
    parser.add_argument('-v','--verbose',
                        dest='verbose',
                        help='Option to print out project parameters',
                        action="store_true")
    ## Type of Statistics
    parser.add_argument('-frac_stat',
                        dest='frac_stat',
                        help='Statistics to use to evaluate the conformity signal',
                        type=str,
                        choices=['diff', 'ratio'],
                        default='diff')
    ## Type of Shuffling
    parser.add_argument('-shuffle_type',
                        dest='shuffle_type',
                        help='Type of shuffling to do for the centrals',
                        type=str,
                        choices=['normal', 'unique', 'not_unique'],
                        default='normal')
    ## Random Seed
    parser.add_argument('-seed',
                        dest='seed',
                        help='Random seed to be used for the analysis. Def: 1',
                        type=int,
                        metavar='[0-4294967295]',
                        default=1)
    ## Parsing Objects
    args = parser.parse_args()

    return args

def get_analysis_params(param_dict):
    """
    Parameters for the 1-halo conformity analysis

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with project variables

    Returns
    --------
    params_pd: pandas DataFrame
        DataFrame with necessary parameters to run 1-halo conformity analysis
    """
    ##
    ## Array of values used for the analysis.
    ## Format: (name of variable, flag, value)
    #
    # For Calculations
    if param_dict['analysis_type'] == 'calc':
        params_arr = num.array([('sample'         ,'-sample'       ,19),
                                ('catl_type'      ,'-abopt'        ,'mr'),
                                ('corr_pair_type' ,'-pairtype'     ,'cen_cen'),
                                ('shuffle_marks'  ,'-shuffle'      ,'cen_sh'),
                                ('rpmin'          ,'-rpmin'        ,0.09),
                                ('rpmax'          ,'-rpmax'        ,20.),
                                ('nrpbins'        ,'-nrp'          ,10),
                                ('itern_tot'      ,'-itern'        ,1000),
                                ('ngals_min'      ,'-nmin'         ,1),
                                ('Mg_bin'         ,'-mg'           ,0.4),
                                ('prop_log'       ,'-log'          ,'log'),
                                ('catl_start'     ,'-catl_start'   ,0),
                                ('catl_finish'    ,'-catl_finish'  ,100),
                                ('perf_opt'       ,'-perf'         ,'False'),
                                ('cosmo_choice'   ,'-cosmo'        ,'LasDamas'),
                                ('cpu_frac'       ,'-cpu'          ,0.7),
                                ('remove_files'   ,'-remove'       ,'False'),
                                ('remove_wp_files','-remove-wp'    ,'False'),
                                ('type_sigma'     ,'-sigma'        ,'std'),
                                ('pimax'          ,'-pimax'        ,20.  ),
                                ('frac_stat'      ,'-frac_stat'    ,'diff'),
                                ('verbose'        ,'-v'            ,'False'),
                                ('seed'           ,'-seed'         ,1     ),
                                ('shuffle_type'   ,'-shuffle_type' ,'normal')])
    #
    # Variables for Plotting
    if param_dict['analysis_type'] == 'plots':
        params_arr = num.array([('sample'         ,'-sample'       ,19),
                                ('catl_type'      ,'-abopt'        ,'mr'),
                                ('corr_pair_type' ,'-pairtype'     ,'cen_cen'),
                                ('shuffle_marks'  ,'-shuffle'      ,'cen_sh'),
                                ('rpmin'          ,'-rpmin'        ,0.09),
                                ('rpmax'          ,'-rpmax'        ,20.),
                                ('nrpbins'        ,'-nrp'          ,10),
                                ('itern_tot'      ,'-itern'        ,1000),
                                ('ngals_min'      ,'-nmin'         ,1),
                                ('Mg_bin'         ,'-mg'           ,0.4),
                                ('prop_log'       ,'-log'          ,'log'),
                                ('catl_start'     ,'-catl_start'   ,0),
                                ('catl_finish'    ,'-catl_finish'  ,100),
                                ('perf_opt'       ,'-perf'         ,'False'),
                                ('type_sigma'     ,'-sigma'        ,'std'),
                                ('mg_min'         ,'-mg_min'       ,11.61),
                                ('mg_max'         ,'-mg_max'       ,13.2),
                                ('verbose'        ,'-v'            ,'False'),
                                ('pimax'          ,'-pimax'        ,20.  ),
                                ('frac_stat'      ,'-frac_stat'    ,'diff')])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name','Flag','Value']
    params_pd = pd.DataFrame(params_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    params_pd = params_pd.sort_values(by='Name').reset_index(drop=True)
    ##
    ## Options for `Calculations`
    if (param_dict['analysis_type'] == 'calc'):
        ##
        ## Choosing if to delete files -- DDrp
        if param_dict['remove_wp_files']:
            ## Overwriting `remove_files` from `params_pd`
            params_pd.loc[params_pd['Name']=='remove_wp_files','Value'] = 'True'
        ##
        ## Choosing if to delete files -- Final result of MCF
        if param_dict['remove_files']:
            ## Overwriting `remove_files` from `params_pd`
            params_pd.loc[params_pd['Name']=='remove_files','Value'] = 'True'
        ##
        ## Changing `pimax` value
        params_pd.loc[params_pd['Name']=='pimax','Value'] = param_dict['pimax']
        ##
        ## Choosing the amount of CPUs
        params_pd.loc[params_pd['Name']=='cpu_frac','Value'] = param_dict['cpu_frac']
        ##
        ## Choosing types of shuffles to do on Centrals
        params_pd.loc[params_pd['Name']=='shuffle_type','Value'] = param_dict['shuffle_type']
        ##
        ## Random seed
        params_pd.loc[params_pd['Name']=='seed','Value'] = param_dict['seed']
    ##
    ## Options for `Plotting`
    if (param_dict['analysis_type'] == 'plots'):
        ##
        ## Option for verbose output
        if param_dict['verbose']:
            ## Overwriting `verbose` from `params_pd`
            params_pd.loc[params_pd['Name']=='verbose','Value'] = 'True'
        ##
        ## Chaning `pimax` value
        params_pd.loc[params_pd['Name']=='pimax','Value'] = param_dict['pimax']

    return params_pd

def get_exec_string(params_pd, param_dict):
    """
    Produces string be executed in the bash file

    Parameters
    -----------
    params_pd: pandas DataFrame
        DataFrame with necessary parameters to run 1-halo conformity analysis

    param_dict: python dictionary
        dictionary with project variables

    Returns
    -----------
    string_dict: python dictionary
        dictionary containing strings for `data` and `mocks`
    """
    ## Current directory
    working_dir = os.path.abspath(os.path.dirname(__file__))
    ## Choosing which file to run
    if param_dict['analysis_type']=='calc':
        FRAC_file = 'two_halo_conformity_quenched_fractions.py'
    elif param_dict['analysis_type']=='plots':
        FRAC_file = 'two_halo_conformity_quenched_fractions_plots.py'
    ##
    ## Getting path to `FRAC_file`
    file_path = os.path.join(working_dir, FRAC_file)
    ##
    ## Check if File exists
    if os.path.isfile(file_path):
        pass
    else:
        msg = '{0} `FRAC_file` ({1}) not found!! Exiting!'.format(
            param_dict['Prog_msg'], file_path)
        raise ValueError(msg)
    ##
    ## Constructing string
    MCF_string = 'python {0} '.format(file_path)
    for ii in range(params_pd.shape[0]):
        ## Appending to string
        MCF_string += ' {0} {1}'.format(    params_pd['Flag'][ii],
                                            params_pd['Value'][ii])
    ##
    ## Creating strings for `mocks` and `data`
    MCF_data  = MCF_string + ' {0} {1}'.format('-kind','data')
    MCF_mocks = MCF_string + ' {0} {1}'.format('-kind','mocks')
    ##
    ##
    string_dict = {'data':MCF_data, 'mocks':MCF_mocks}

    return string_dict

def file_construction_and_execution(params_pd, param_dict):
    """
    1) Creates file that has shell commands to run executable
    2) Executes the file, which creates a screen session with the executables

    Parameters:
    -----------
    params_pd: pandas DataFrame
        DataFrame with necessary parameters to run 1-halo conformity analysis

    param_dict: python dictionary
        dictionary with project variables
    
    """
    ##
    ## Getting today's date
    now_str = datetime.datetime.now().strftime("%x %X")
    ##
    ## Obtain MCF strings
    string_dict = get_exec_string(params_pd, param_dict)
    ##
    ## Parsing text that will go in file
    # Working directory
    working_dir = os.path.abspath(os.path.dirname(__file__))
    ## Obtaining path to file
    outfile_name = 'two_halo_conformity_quenched_fractions_{0}_run.sh'.format(param_dict['analysis_type'])
    outfile_path = os.path.join(working_dir, outfile_name)
    ##
    ## Opening file
    with open(outfile_path, 'wb') as out_f:
        out_f.write(b"""#!/usr/bin/env bash\n\n""")
        out_f.write(b"""## Author: Victor Calderon\n\n""")
        out_f.write( """## Last Edited: {0}\n\n""".format(now_str).encode())
        out_f.write(b"""### --- Variables\n""")
        out_f.write(b"""ENV_NAME="conformity"\n""")
        out_f.write( """WINDOW_NAME="Two_Halo_FRAC_STAT_conformity_{0}"\n""".format(param_dict['analysis_type']).encode())
        out_f.write(b"""WINDOW_DATA="data"\n""")
        out_f.write(b"""WINDOW_MOCKS="mocks"\n""")
        out_f.write(b"""# Home Directory\n""")
        out_f.write(b"""home_dir=`eval echo "~$different_user"`\n""")
        out_f.write(b"""# Type of OS\n""")
        out_f.write(b"""ostype=`uname`\n""")
        out_f.write(b"""# Sourcing profile\n""")
        out_f.write(b"""if [[ $ostype == "Linux" ]]; then\n""")
        out_f.write(b"""    source $home_dir/.bashrc\n""")
        out_f.write(b"""else\n""")
        out_f.write(b"""    source $home_dir/.bash_profile\n""")
        out_f.write(b"""fi\n""")
        out_f.write(b"""# Activating Environment\n""")
        out_f.write(b"""activate=`which activate`\n""")
        out_f.write(b"""source $activate ${ENV_NAME}\n""")
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Python Strings\n""")
        out_f.write( """FRAC_data="{0}"\n""".format(string_dict['data']).encode())
        out_f.write( """FRAC_mocks="{0}"\n""".format(string_dict['mocks']).encode())
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Deleting previous Screen Session\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X quit\n""")
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Screen Session\n""")
        out_f.write(b"""screen -mdS ${WINDOW_NAME}\n""")
        out_f.write(b"""##\n""")
        out_f.write(b"""## Data\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X screen -t ${WINDOW_DATA}\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_DATA} -X stuff $"source $activate ${ENV_NAME};"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_DATA} -X stuff $"${FRAC_data}"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_DATA} -X stuff $'\\n'\n""")
        out_f.write(b"""##\n""")
        out_f.write(b"""## Mocks\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X screen -t ${WINDOW_MOCKS}\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_MOCKS} -X stuff $"source $activate ${ENV_NAME};"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_MOCKS} -X stuff $"${FRAC_mocks}"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_MOCKS} -X stuff $'\\n'\n""")
        out_f.write(b"""\n""")
    ##
    ## Check if File exists
    if os.path.isfile(outfile_path):
        pass
    else:
        msg = '{0} `outfile_path` ({1}) not found!! Exiting!'.format(
            param_dict['Prog_msg'], outfile_path)
        raise ValueError(msg)
    ##
    ## Make file executable
    print(".>>> Making file executable....")
    print("     chmod +x {0}".format(outfile_path))
    os.system("chmod +x {0}".format(outfile_path))
    ##
    ## Running script
    print(".>>> Running Script....")
    os.system("{0}".format(outfile_path))

def main():
    """
    Computes the analysis and 
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ##
    ## Parameters for the analysis
    params_pd = get_analysis_params(param_dict)
    ##
    ## Running analysis
    file_construction_and_execution(params_pd, param_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main()
