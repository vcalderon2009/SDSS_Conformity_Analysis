# Victor Calderon
# October 31st, 2017
# Vanderbilt University

from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, 1-halo Mark Correlation"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
__all__        =['pairwise_distance_rp']
"""
Computes the number of pairs in perpendicular projected distance `rp` bins
for a set of galaxies with <ra dec cz> positions
"""
# Importing Modules
cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, log10, fabs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

## Functions

## Pair Counter - MCF
def pairwise_distance_rp(coord_1, coord_2, rpmin=0.01, rpmax=10, 
    nrpbins=10, pimax=20., double_count=False):
    """
    Cython engine for returning pairs of points separated in 
    projected radial bins with an observer at (0,0,0)

    Parameters
    ----------

    coord_1: array_lke, shape (N,3)
        arrays of < x | y | z > of sample 1

    coord_2: array_lke, shape (N,3)
        arrays of < x | y | z > of sample 2

    rpmin: float, optional (default = 0.01)
        minimum `rp` (perpendicular distance) to search for and return pairs

    rpmax: float, optional (default = 10.)
        maximum `rp` (perpendicular distance) to search for and return pairs

    nrpbins: int, optional (default = 10)
        total number of `rp` bins

    pimax: float, optional (default = 20.)
        maximum parallel separation distance to serach for and return pairs

    double_count: boolean, optional (default = False)
        option to decide whether or not to double count each galaxy pairs.
        Useful for the `2-halo Quenched Fractions` calculations

    Returns
    ----------
    rp_ith_arr: array-like, shape (M,3)
        three-dimensional array of M-elements containing:
        - `rp` bin number, to which galaxy pair belongs
        - i_ind: indices of 0-indexed indices in sample 1
        - j_ind: indices of 0-indexed indices in sample 2
    """
    ## -- Output Lists -- 
    rp_arr  = []
    ith_arr = []
    jth_arr = []
    ## Number of Elements
    cdef int Ni, Nj, i, j
    cdef cnp.float64_t rpbin
    Ni = len(coord_1)
    Nj = len(coord_2)
    ## -- Constants --
    cdef cnp.float64_t PI180 = 3.141592653589793/180.
    ## -- Count Pairs variables --
    cdef cnp.float64_t sx, sy, sz, lx, ly, lz, l2, ll, spar, s2, sperp
    ## -- `rp` constants
    cdef int nrpbins_p          = nrpbins
    cdef cnp.float64_t rp_min_p = rpmin
    cdef cnp.float64_t rp_max_p = rpmax
    cdef cnp.float64_t pi_max_p = pimax
    cdef cnp.float64_t logrpmin = log10(rpmin)
    cdef cnp.float64_t logrpmax = log10(rpmax)
    cdef cnp.float64_t dlogrp   = (logrpmax-logrpmin)/nrpbins
    ## -- Cartesian Coordinates --
    # Sample 1
    cdef cnp.float64_t[:] x1 = coord_1.astype(float).T[0]
    cdef cnp.float64_t[:] y1 = coord_1.astype(float).T[1]
    cdef cnp.float64_t[:] z1 = coord_1.astype(float).T[2]
    # Sample 2
    cdef cnp.float64_t[:] x2 = coord_2.astype(float).T[0]
    cdef cnp.float64_t[:] y2 = coord_2.astype(float).T[1]
    cdef cnp.float64_t[:] z2 = coord_2.astype(float).T[2]
    ## Determining if to double count or not
    if double_count:
        ## Looping over points in `coord_1`
        for i in range(0,Ni):
            ## Looping over points in `coord_2`
            for j in range(0,Nj):
                # Calculate the square distances
                sx    = x1[i] - x2[j]
                sy    = y1[i] - y2[j]
                sz    = z1[i] - z2[j]
                lx    = 0.5*(x1[i] + x2[j])
                ly    = 0.5*(y1[i] + y2[j])
                lz    = 0.5*(z1[i] + z2[j])
                l2    = (lx * lx) + (ly * ly) + (lz * lz)
                ll    = sqrt(l2)
                spar  = abs(((sx * lx) + (sy * ly) + (sz * lz)) / ll)
                s2    = (sx * sx) + (sy * sy) + (sz * sz)
                sperp = sqrt(s2 - spar * spar)
                ## Criteria for `projected separation`
                if (spar <= pi_max_p) & (sperp > rp_min_p) & (sperp < rp_max_p):
                    # `rp` bin of pair
                    rpbin = (log10(sperp) - logrpmin)/dlogrp
                    # Appending to lists
                    rp_arr.append(rpbin)
                    ith_arr.append(i)
                    jth_arr.append(j)
    else:
        ## Looping over points in `coord_1`
        for i in range(Ni):
            ## Looping over points in `coord_2`
            for j in range(i+1,Nj):
                # Calculate the square distances
                sx    = x1[i] - x2[j]
                sy    = y1[i] - y2[j]
                sz    = z1[i] - z2[j]
                lx    = 0.5*(x1[i] + x2[j])
                ly    = 0.5*(y1[i] + y2[j])
                lz    = 0.5*(z1[i] + z2[j])
                l2    = (lx * lx) + (ly * ly) + (lz * lz)
                ll    = sqrt(l2)
                spar  = abs(((sx * lx) + (sy * ly) + (sz * lz)) / ll)
                s2    = (sx * sx) + (sy * sy) + (sz * sz)
                sperp = sqrt(s2 - spar * spar)
                ## Criteria for `projected separation`
                if (spar <= pi_max_p) & (sperp > rp_min_p) & (sperp < rp_max_p):
                    # `rp` bin of pair
                    rpbin = (log10(sperp) - logrpmin)/dlogrp
                    # Appending to lists
                    rp_arr.append(rpbin)
                    ith_arr.append(i)
                    jth_arr.append(j)
    ## Converting to Numpy arrays
    rp_arr     = np.array(rp_arr).astype(int)
    ith_arr    = np.array(ith_arr).astype(int)
    jth_arr    = np.array(jth_arr).astype(int)
    # Combining arrays into a single array
    rp_ith_arr = np.column_stack((rp_arr, ith_arr, jth_arr))

    return rp_ith_arr
