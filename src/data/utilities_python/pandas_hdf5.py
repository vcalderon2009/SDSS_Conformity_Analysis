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
__all__        =["read_pandas_hdf5","read_hdf5_file_to_pandas_DF"]

import numpy as num
import pandas as pd
import h5py
from . import file_dir_check as fd
import os

def read_pandas_hdf5(hdf5_file, key=None, ret=False):
	"""
	Reads a `.hdf5` file that contains one or many datasets, and converts into 
	a pandas DataFrame. It assumes that the file is a PyTable

	Parameters
	----------
	hdf5_file: string
		Path to `.hdf5` file containing one or many pandas DataFrame(s).

	key: string
		If provided, it will extract `key` as a pandas DataFrame

	ret: boolean, (default=False)
		Option to return key of the file. 

	Returns
	-------
	pd_dataframe: pandas DataFrame object
		DataFrame from `hdf5_file` with under the `key` directory.
	"""
	Program_Message = fd.Program_Msg(__file__)
	fd.File_Exists(hdf5_file)
	# Checking number of keys
	hdf5_file_obj  = pd.HDFStore(hdf5_file)
	hdf5_file_keys = [key_ii for key_ii in hdf5_file_obj.keys()]
	hdf5_file_obj.close()
	if key==None:
		try:
			pd_dataframe = pd.read_hdf(hdf5_file)
			if ret:	return pd_dataframe, hdf5_file_keys[0]
			else: return pd_dataframe
		except:
			print('{0} Must specify which key to use:'.format(Program_Message))
			print('Possible keys: \n')
			for key1, name in enumerate(hdf5_file_keys):
				print('\t Key {0}:   {1}'.format(key1,name))
	else:
		if key not in hdf5_file_keys:
			print('{0} Key not in the file:'.format(Program_Message))
			print('Possible keys: \n')
			for key1, name in enumerate(hdf5_file_keys):
				print('\t Key {0}:   {1}'.format(key1,name))
		else:
			pd_dataframe = pd.read_hdf(hdf5_file, key=key)
			if ret:	return pd_dataframe, key
			else: return pd_dataframe

def read_hdf5_file_to_pandas_DF(hdf5_file, key=None):
	"""
	Reads content of HDF5 file and converts it to Pandas DataFrame

	Parameters
	----------
	hdf5_file: string
		Path to HDF5 file with the data

	key: string, default=None
		Key or path in hdf5 file for the pandas DataFrame and the normal hddf5 
		file

	Returns
	-------
	pd_dataframe: Pandas DataFrame
		DataFrame from `hdf5_file` with under the `key` directory.
	"""
	Program_Message = fd.Program_Msg(__file__)
	fd.File_Exists(hdf5_file)
	data_file = h5py.File(hdf5_file,'r')
	pd_dict = {}
	if not key:
		data_keys = num.array([key_ii for key_ii in data_file.keys()])
		if len(data_keys)==1:
			data = data_file.get(data_keys[0])
			for name in data.dtype.names:
				name1 = name.replace('-','_')
				pd_dict[name1] = data[name]
			pd_dataframe = pd.DataFrame(pd_dict)
			return pd_dataframe
		else:
			print('{0} Must specify which key to use:'.format(Program_Message))
			print('Possible keys: \n')
			for key, name in enumerate(data_keys):
				print('\t Key {0}:   {1}'.format(key,name))
	else:
		data_keys = num.array([key_ii for key_ii in data_file.keys()])
		if key not in data_keys:
			print('{0} Key not in the file:'.format(Program_Message))
			print('Possible keys: \n')
			for key, name in enumerate(data_keys):
				print('\t Key {0}:   {1}'.format(key,name))
		else:
			data = data_file.get(key)
			for name in data.dtype.names:
				name1 = name.replace('-','_')
				pd_dict[name1] = data[name]
			pd_dataframe = pd.DataFrame(pd_dict)
			return pd_dataframe
	data_file.close()
