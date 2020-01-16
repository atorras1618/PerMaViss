"""
    functions.py

    This code implements multiplication mod p and solving a linear
    equation mod p.
"""
import numpy as np

from .gauss_mod_p import gauss_col

###############################################################################
# Multiply two matrices mod p


def multiply_mod_p(A, B, p):
    """
        Multiply matrices mod p.
    """
    return np.matmul(A, B) % p


###############################################################################
# solve_mod_p
# Function to solve linear equations mod p

def solve_mod_p(A, b, p):
    """
    Find the vector x such that A * x = b (mod p)

    This method assumes that a solution exists to the equation
    A * x = b (mod p). If a solution does not exist, it raises a ValueError
    exception.

    Parameters
    ----------
    A : :obj:`Numpy Array`
        2D array
    b : :obj:`Numpy Array`
        1D array
    p : int(prime)
        Number to mod out by.

    Returns
    -------
    x : :obj:`Numpy Array`
        1D array. Solution to equation.

    Raises
    ------
    ValueError
        If a solution to the equation does not exist.
    """
    R, T = gauss_col(np.append(A, np.array([b]).T, axis=1), p)
    if np.any(R[:, -1]):
        print("Linear equation has no solution.")
        raise ValueError

    # Return last column from T without the last row
    return -T[:, -1][:-1] % p


###############################################################################
# solve_matrix_mod_p
#

def solve_matrix_mod_p(A, B, p):
    """
    Same as :meth:`solve_mod_p`, but with B and X being matrices.

    That is, given two matrices A and B, we want to find a matrix X
    such that A * X = B (mod p)

    Parameters
    ----------
    A : :obj:`Numpy Array`
        2D array
    B : :obj:`Numpy Array`
        2D array
    p : int(prime)

    Returns
    -------
    X : :obj:`Numpy Array`
        2D array solution.

    Raises
    ------
    ValueError
        There is no solution to the given equation
    """
    R, T = gauss_col(np.append(A, B.T, axis=1), p)
    if np.any(R[:, np.size(A, 1):]):
        print("Linear matrix equation has no solution.")
        raise ValueError

    # Return matrix X
    return - T[:, np.size(A, 1):][:np.size(A, 1)] % p
