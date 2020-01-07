"""
    gauss_mod_p.py

    This module implements gaussian elimination mod a prime number p.
"""
import numpy as np
from .arithmetic_mod_p import *

##################################################################################
# Index searching function    

def _index_pivot(l):
    """
    Given a 1D array of integers, return the index of the last nonzero entry. 
    Returns -1 if the list is zero. 
    INPUT:
    -l: list on integers.
    OUTPUT:
    -index of last nonzero entry.
    """
    l_bool = np.nonzero(l)
    if len(l_bool[0]) > 0:
        return l_bool[0][-1]
    
    return -1

assert _index_pivot(np.array([0,1,0,1,0]))==3
assert _index_pivot(np.array([0,0,0]))==-1

###################################################################################
# Gaussian elimination procedure 

def gauss_col(A, p):
    """
    This function implements the gaussian elimination by columns. 
    INPUT:
    -A: matrix to be reduced
    -p: prime number. The corresponding field will be Z mod p.
    OUTPUT:
    -R: reduced matrix by left to right column additions. 
    -T: matrix recording additions performed, so that AT = R
    """
    if np.size(A,0) == 0:
        return np.array([]), np.array([])

    # number of columns in A
    N = np.size(A,1) 
    # copy of matrix to be reduced 
    # The matrix is transposed for more computational efficiency
    R = np.copy(np.transpose(A)) 
    T = np.identity(N)
    # iterate over all columns
    for j in range(N):
        pivot = _index_pivot(R[j])
        # Assume that the j-column is not reduced
        reduced = False
        while (pivot > -1) & (not reduced): 
            reduced = True 
            # look for previous columns to j
            for k in range(j):
                # if the pivots coincide, substract column k to column j
                # multiplied by a suitable coefficient q
                if _index_pivot(R[k]) == pivot:
                    q = (R[j][pivot] * inv_mod_p(R[k][pivot], p)) % p
                    R[j] = add_arrays_mod_c(R[j], -q * R[k], p)
                    T[j] = add_arrays_mod_c(T[j], -q * T[k], p) 
                    # reset pivot
                    if np.any(R[j]):
                        pivot = _index_pivot(R[j])
                        reduced = False

                    break
                # end if
            # end for
        # end while
    # end for
    return np.transpose(R), np.transpose(T) 

