#!/bin/bash
name_mod='pair_counter_rp'
## Removing previous *.so files
rm ./*.so
## Running Cython compilation
python setup.py build_ext --inplace
## Copying python module
cp ./${name_mod}/${pair_counter_rp}*.so ./
mv ./${name_mod}*.so ./${name_mod}.so
## Deleting extra folders
rm -r ./${name_mod}
rm -r ./build