#!/usr/bin/env python
'''
Description:
    Setup file paths for systems that I usually work on.
'''

# Author: Victor Calderon
# Created: 11/10/2017
# Vanderbilt University
# Setup file paths for common systems I work on.

from __future__ import absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, get_path"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
__all__        =["get_base_path", "get_output_path", "git_root_dir",\
                 "cookiecutter_paths"]

# Importing modules
import git
import os
from .file_dir_check import Path_Folder as PF

def get_base_path(path='./'):
    """
    get the base path for the system
    """
    base_path = git_root_dir(path) + '/'

    return base_path

def get_output_path(path='./'):
    """
    get the base path to get_output_path storage for the system
    """
    proj_dict = cookiecutter_paths(path)
    ## Output Path
    output_path = os.path.join(proj_dict['data_dir'], 'external/')

    return output_path

## Based on the `Data Science` Cookiecutter Template
def git_root_dir(path='./'):
    """
    Determines the path to the main `.git` folder of the project.
    Taken from:
    https://goo.gl/46y9v1

    Parameters
    -----------
    path: string, optional (default = './')
        path to the file within the `.git` repository

    Returns
    -----------
    git_root: string
        path to the main `.git` project repository
    """
    # Creating instance of Git Repo
    git_repo = git.Repo(os.path.abspath(path), search_parent_directories=True)
    # Root path
    git_root = git_repo.git.rev_parse("--show-toplevel")

    return git_root

def cookiecutter_paths(path='./'):
    """
    Paths to main folders in the `Data Science` Cookiecutter template.
    
    Parameters
    -----------
    path: string, optional (default = './')
        path to the file within the `.git` repository

    Returns
    ------------
    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.
    """
    # Base path
    base_dir = git_root_dir(path) + '/'
    assert(os.path.exists(base_dir))
    # Plot Path
    plot_dir = base_dir + 'reports/figures/'
    PF(plot_dir)
    # Source Code Path
    src_dir = base_dir + 'src/data/'
    PF(src_dir)
    # Data Path
    data_dir = base_dir + 'data/'
    PF(data_dir)
    # Saving into dictionary
    proj_dict = {}
    proj_dict['base_dir'] = base_dir
    proj_dict['plot_dir'] = plot_dir
    proj_dict['src_dir' ] = src_dir
    proj_dict['data_dir'] = data_dir

    return proj_dict
