#!/bin/bash
name_mod='pair_counter_rp'
## Removing previous *.so files
rm ./*.so
(rm -r ./build && rm -r ./${name_mod}) || (printf "")
## Running Cython compilation
python setup.py build_ext --inplace
##
## Compiling Cython module
##
(   ## Copying python module to current folder
    cp ./${name_mod}/*.so ./ &&
    ## Changing name
    mv ./${name_mod}*.so  ./${name_mod}.so &&
    ## Deleting extra folders
    rm -rf ./${name_mod} &&
    rm -rf ./build) || (
    
    ## Renaming python module
    mv ./${name_mod}*.so ./${name_mod}.so &&
    ## Deleting extra folders produced by Cython
    rm -rf ./build )
