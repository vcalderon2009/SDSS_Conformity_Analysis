#!/bin/bash

python setup.py build_ext --inplace
cp ./pair_counter/pair_counter_rp*.so ./
mv ./pair_counter_rp*.so ./pair_counter_rp.so
rm -r ./pair_counter
rm -r ./build