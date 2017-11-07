#!/bin/bash
path_pkg='src/data/utilities_python/pair_counter_rp/'
name_pkg='src'
name_mod='pair_counter_rp'
## Removing previous *.so files
rm ./*.so
(rm -rf ./build && rm -rf ./${name_pkg}) || (printf "")
## Running Cython compilation
python setup.py build_ext --inplace
##
## Compiling Cython module
##
(   ## Copying python module to current folder
    cp ./${path_pkg}/*.so ./ &&
    ## Changing name
    mv ./${name_mod}*.so  ./${name_mod}.so &&
    ## Deleting extra folders
    rm -rf ./${name_pkg} &&
    rm -rf ./build) || (
    
    ## Renaming python module
    mv ./${name_mod}*.so ./${name_mod}.so &&
    ## Deleting extra folders produced by Cython
    rm -rf ./build )
