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
import seaborn.apionly as sns
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)

# Extra-modules
def get_parser():
    """
    Get parser object for `eco_mocks_create.py` script.

    Returns
    -------
    args: 
        input arguments to the script
    """
    ## Define parser object
    description_msg = 'Script to Create ECO, Resolve A and B Catalogues'
    parser = ArgumentParser(description=description_msg)
    # 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('-namevar', '--long-name',
                        dest='variable_name',
                        help='Description of variable',
                        type=float,
                        default=0)
    ## Parsing Objects
    args = parser.parse_args()

    return args


def main():
    """

    """


# Main function
if __name__=='__main__':
    from argparse import ArgumentParser

    ## Input arguments
    args = get_parser()
    # Main Function
    main()
