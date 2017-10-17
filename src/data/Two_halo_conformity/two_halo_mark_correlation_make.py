#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : DATE
# Last Modified: DATE
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, 1-halo MCF Make"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Script that runs the 1-halo conformity results and plots.
"""
# Importing Modules
import custom_utilities_python as cu
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
    ## Parsing Objects
    args = parser.parse_args()

    return args

def get_analysis_params():
    """
    Parameters for the 1-halo conformity analysis

    Returns
    --------
    params_pd: pandas DataFrame
        DataFrame with necessary parameters to run 1-halo conformity analysis
    """
    ##
    ## Array of values used for the analysis.
    ## Format: (name of variable, flag, value)
    params_arr = num.array([('sample'        ,'-sample'       ,19),
                            ('catl_type'     ,'-abopt'        ,'mr'),
                            ('corr_pair_type','-pairtype'     ,'cen_cen'),
                            ('shuffle_marks' ,'-shuffle'      ,'cen_sh'),
                            ('rpmin'         ,'-rpmin'        ,0.09),
                            ('rpmax'         ,'-rpmax'        ,20.),
                            ('nrpbins'       ,'-nrp'          ,10),
                            ('itern_tot'     ,'-itern'        ,1000),
                            ('ngals_min'     ,'-nmin'         ,1),
                            ('Mg_bin'        ,'-mg'           ,0.4),
                            ('prop_log'      ,'-log'          ,'log'),
                            ('catl_start'    ,'-catl_start'   ,0),
                            ('catl_finish'   ,'-catl_finish'  ,100),
                            ('perf_opt'      ,'-perf'         ,'False'),
                            ('cosmo_choice'  ,'-cosmo'        ,'LasDamas'),
                            ('cpu_frac'      ,'-cpu'          ,0.7)])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name','Flag','Value']
    params_pd = pd.DataFrame(params_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    params_pd = params_pd.sort_values(by='Name').reset_index(drop=True)


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
    ## Choosing which file to run
    if param_dict['analysis_type']=='calc':
        MCF_file = 'two_halo_mark_correlation.py'
    elif param_dict['analysis_type']=='plots':
        MCF_file = 'two_halo_mark_correlation_plots.py'
    ##
    ## Getting path to `MCF_file`
    file_path = os.path.abspath(MCF_file)
    ##
    ## Check if File exists
    if os.path.isfile(file_path):
        pass
    else:
        msg = '{0} `MCF_file` ({1}) not found!! Exiting!'.format(
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
    outfile_name = 'two_halo_mark_correlation_{0}_run.sh'.format(param_dict['analysis_type'])
    outfile_path = os.path.abspath('./'+outfile_name)
    ##
    ## Opening file
    with open(outfile_path, 'wb') as out_f:
        out_f.write(b"""#!/usr/bin/env bash\n\n""")
        out_f.write(b"""## Author: Victor Calderon\n\n""")
        out_f.write( """## Last Edited: {0}\n\n""".format(now_str).encode())
        out_f.write(b"""### --- Variables\n""")
        out_f.write(b"""ENV_NAME="conformity"\n""")
        out_f.write( """WINDOW_NAME="Two_Halo_MCF_conformity_{0}"\n""".format(param_dict['analysis_type']).encode())
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
        out_f.write( """MCF_data="{0}"\n""".format(string_dict['data']).encode())
        out_f.write( """MCF_mocks="{0}"\n""".format(string_dict['mocks']).encode())
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Screen Session\n""")
        out_f.write(b"""screen -mdS ${WINDOW_NAME}\n""")
        out_f.write(b"""##\n""")
        out_f.write(b"""## Data\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X screen -t ${WINDOW_DATA}\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_DATA} -X stuff $"source $activate ${ENV_NAME};"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_DATA} -X stuff $"${MCF_data}"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_DATA} -X stuff $'\\n'\n""")
        out_f.write(b"""##\n""")
        out_f.write(b"""## Mocks\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X screen -t ${WINDOW_MOCKS}\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_MOCKS} -X stuff $"source $activate ${ENV_NAME};"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_MOCKS} -X stuff $"${MCF_mocks}"\n""")
        # out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_MOCKS} -X stuff $'\\n'\n""")
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
    params_pd = get_analysis_params()
    ##
    ## Running analysis
    file_construction_and_execution(params_pd, param_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main()