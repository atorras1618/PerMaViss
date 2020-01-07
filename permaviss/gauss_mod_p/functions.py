"""
    functions.py
    
    This code implements multiplication mod p and solving a linear equation mod p
"""
import numpy as np

from .gauss_mod_p import gauss_col

###############################################################################
# Multiply two matrices mod p

def multiply_mod_p( A, B, p):
    """
        Multiply matrices mod p.
    """
    return np.matmul(A,B) % p


#################################################################################
# solve_mod_p
# Function to solve linear equations mod p

def solve_mod_p(A, b, p):
    """
    This method assumes that a solution exists to the equation A * x = b (mod p)
    If a solution does not exist, it raises a ValueError exception.  
    INPUT:
    -A: 2D numpy.array 
    -b: 1D numpy.array
    -p: prime number
    OUTPUT:
    -x: 1D numpy.array
    """
    R, T = gauss_col(np.append(A, np.array([b]).T , axis=1), p)
    if np.any(R[:,-1]):
        print("Linear equation has no solution.")
        raise ValueError

    # Return last column from T without the last row
    return -T[:,-1][:-1] % p


###############################################################################
# solve_matrix_mod_p
#
# Basically, this is the same as the previous function, but with multiple equations to be
# solved simultanously. Alternatively, we are looking at a solution for the matrix
# equation A * X = B (mod p)
#
def solve_matrix_mod_p(A, B, p):
    """
    This method assumes that a solution exists to the matrix equation A * X = B (mod p)
    If a solution does not exist, it raises a ValueError exception.  
    INPUT:
    -A: 2D numpy.array 
    -B: 2D numpy.array
    -p: prime number
    OUTPUT:
    -X: 2D numpy.array
    """
    R, T = gauss_col(np.append(A, B.T , axis=1), p)
    if np.any(R[:, np.size(A,1):]):
        print("Linear matrix equation has no solution.")
        raise ValueError

    # Return matrix X
    return - T[:, np.size(A,1):][:np.size(A,1)] % p

