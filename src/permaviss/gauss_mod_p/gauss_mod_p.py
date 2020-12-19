"""gauss_mod_p.py

This module implements Gaussian elimination by columns modulo a prime
number p.
"""
import numpy as np
from .arithmetic_mod_p import add_arrays_mod_c, inv_mod_p

###############################################################################
# Index searching function


def _index_pivot(l):
    """Returns the pivot of a 1D array

    Parameters
    ----------
    l : :obj:`list(int)`
        List of integers to compute pivot from.

    Returns
    -------
    int
        Index of last nonzero entry on `l`. Returns -1 if the list is zero.
    """
    l_bool = np.nonzero(l)
    if len(l_bool[0]) > 0:
        return l_bool[0][-1]

    return -1


assert _index_pivot(np.array([0, 1, 0, 1, 0])) == 3
assert _index_pivot(np.array([0, 0, 0])) == -1

###############################################################################
# Gaussian elimination procedure


def gauss_col(A, p):
    """This function implements the Gaussian elimination by columns.

    A is reduced by left to right column additions. The reduced matrix has
    unique column pivots.

    Parameters
    ----------
    A : :obj:`Numpy Array`
        Matrix to be reduced
    p : `int(prime)`
        Prime number. The corresponding field will be Z mod p.

    Returns
    -------
    R : :obj:`Numpy Array`
        Reduced matrix by left to right column additions.
    T : :obj:`Numpy Array`
        Matrix recording additions performed, so that AT = R

    """
    if np.size(A, 0) == 0:
        return np.array([]), np.array([])

    # number of columns in A
    N = np.size(A, 1)
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
                # if the pivots coincide, subtract column k to column j
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

def gauss_col_rad(A, R, start_index, p):
    """This function implements the Gaussian elimination by columns,
    but specialized for columns with birth radius.

    A is reduced by left to right column additions starting
    from  start_index. Only columns from a lower index are
    added to columns with a higher index.


    Parameters
    ----------
    A : :obj:`Numpy Array`
        Matrix to be reduced
    R: :obj:`Numpy Array`
        Vector with radius
    start_index:`int`
        Index at which reduction starts
    p : `int(prime)`
        Prime number. The corresponding field will be Z mod p.

    Returns
    -------
    T : :obj:`Numpy Array`
        Matrix recording additions performed, so that
        we obtain the lifts and coefficients.

    Raises
    ------
    ValueError
        If reduced columns do not vanish.
    """
    # TO DO: check that trivial matrices are not sent here.
    # number of columns in A
    N = np.size(A, 1)
    # copy of matrix to be reduced
    # The matrix is transposed for indexing convenience
    Red = np.copy(np.transpose(A))
    T = np.identity(N)
    # iterate over all columns
    for j in range(start_index, N):
        pivot = _index_pivot(Red[j])
        # Assume that the j-column is not reduced
        reduced = False
        while (pivot > -1) & (not reduced):
            reduced = True
            # look for previous columns to j
            for k in range(start_index):
                # check the radius
                if R[k] <= R[j]:
                    # if the pivots coincide, subtract column k to column j
                    # multiplied by a suitable coefficient q
                    if _index_pivot(Red[k]) == pivot:
                        q = (Red[j][pivot] * inv_mod_p(Red[k][pivot], p)) % p
                        Red[j] = add_arrays_mod_c(Red[j], -q * Red[k], p)
                        T[j] = add_arrays_mod_c(T[j], -q * T[k], p)
                        # reset pivot and check for nullity
                        if np.any(Red[j]):
                            pivot = _index_pivot(Red[j])
                            reduced = False
                            break
                        # end if
                    # end if
                # end if
            # end for
        # end while
    # end for
    return np.transpose(Red), np.transpose(T)

def gauss_barcodes(A, row_barcode, col_barcode, start_index, p):
    """This function implements the Gaussian elimination by columns,
    but specialized for columns and rows with arbitrary finite barcodes.

    A is reduced by left to right column additions starting
    from  start_index. Only columns from a lower index are
    added to columns with a higher index.


    Parameters
    ----------
    A : :obj:`Numpy Array`
        Matrix to be reduced
    row_R: :obj:`Numpy Array`
        Vector with radius of rows
    col_R: :obj:`Numpy Array`
        Vector with radius of columns
    start_index:`int`
        Index at which reduction starts
    p : `int(prime)`
        Prime number. The corresponding field will be Z mod p.

    Returns
    -------
    T : :obj:`Numpy Array`
        Matrix recording additions performed, so that
        we obtain the lifts and coefficients.

    Raises
    ------
    ValueError
        If reduced columns do not vanish.
    """
    # TO DO: check that trivial matrices are not sent here.
    # number of columns in A
    N_col = np.size(A, 1)
    # copy of matrix to be reduced
    # Matrix to reduce
    Red = np.copy(A)
    T = np.identity(N_col)
    # iterate over all columns
    for j in range(start_index, N_col):
        active_rows = np.array(range(np.size(A, 0)))[np.logical_and(
            row_barcode[:,0] <= col_barcode[j, 0],
            col_barcode[j,0] < row_barcode[:,1])]
        # pivot relative to active rows
        active_pivot = _index_pivot(Red[:,j][active_rows])
        # pivot relative to number of rows in A
        real_pivot = active_rows[active_pivot]
        # Assume that the j-column is not reduced
        reduced = False
        # active_pivot = -1 when active column is 0
        while (active_pivot > -1) & (not reduced):
            reduced = True
            # look for previous columns to j
            for k in range(start_index):
                # check the radius
                if col_barcode[k,0] <= col_barcode[j,0]:
                    # if the pivots coincide, subtract column k to column j
                    # multiplied by a suitable coefficient q
                    if _index_pivot(Red[:,k][active_rows]) == active_pivot:
                        q = (Red[real_pivot,j] * inv_mod_p(
                            Red[real_pivot,k], p)) % p
                        Red[:,j] = add_arrays_mod_c(Red[:,j], -q * Red[:,k], p)
                        T[:,j] = add_arrays_mod_c(T[:,j], -q * T[:,k], p)
                        # reset pivot and check for nullity
                        if np.any(Red[:,j][active_rows]):
                            active_pivot = _index_pivot(Red[:,j][active_rows])
                            real_pivot = active_rows[active_pivot]
                            reduced = False
                            break
                        # end if
                    # end if
                # end if
            # end for
        # end while
    # end for
    return Red, T
