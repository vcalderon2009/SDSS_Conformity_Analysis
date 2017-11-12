#! /usr/bin/env python

# Victor Calderon
# February 22, 2016
# Vanderbilt University

"""
Tools for converting pandas DataFrames to .hdf5 files, and converting from 
one type of hdf5 file to `pandas_hdf5` file format
"""
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, geometry"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
__all__        =["read_hdf5_file_to_pandas_DF"]

import numpy as num
import pandas as pd
import h5py
from . import file_dir_check as fd
import os

def read_hdf5_file_to_pandas_DF(hdf5_file, key=None, ret=None):
	"""
	Reads content of HDF5 file and converts it to Pandas DataFrame.
	The content of the HDF5 was saved using `pd.to_hdf`. See 
	documentations for further information.

	Parameters
	----------
	hdf5_file: string
		Path to HDF5 file with the data

	key: string, default=None
		Key or path in hdf5 file for the pandas DataFrame and the normal hddf5 
		file

	ret: boolean, optional (default = False)
		option to return both the `pandas.DataFrame` and `key`.
		If True, returns both.
		If False, only returns `pd_df`

	Returns
	-------
	pd_df: Pandas DataFrame
		DataFrame from `hdf5_file` with under the `key` directory.
	"""
	Program_Message = fd.Program_Msg(__file__)
	##
	## Checking if file exists
	fd.File_Exists(hdf5_file)
	##
	## Using pandas to load file
	pd_df = pd.read_hdf(hdf5_file, key=key)

	if ret:
		return pd_df, key
	else:
		return pd_df
