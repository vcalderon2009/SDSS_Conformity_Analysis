#!/bin/bash
name_mod='pair_counter_rp'
## Removing previous *.so files
rm ./*.so
(rm -r ./build && rm -r ./${name_mod}) || (printf "")
## Running Cython compilation
python setup.py build_ext --inplace
## Copying python module
(   cp ./${name_mod}/*.so ./ &&
    mv ./${name_mod}*.so  ./${name_mod}.so &&
    rm -rf ./${name_mod} &&
    rm -rf ./build) || (
    printf "\n\n\nAhora si no se puede\n\n\n")




## Running Cython compilation
# python setup.py build_ext --inplace
## Copying python module
# printf "\n\n"
# (echo "Test" && printf "\n\n Otro test\n" && rm ./${name_mod}_Testing) || (
#     printf "Esta cosa no resulto\n\n\n")
# (echo"cp ./${name_mod}/*so ./ && mv ./${name_mod}/*.so ./${name_mod}.so") || (
    # echo"mv ./${name_mod}*.so ./${name_mod}.so" )
## Deleting extra folders
# rm -r ./${name_mod}
# rm -r ./build