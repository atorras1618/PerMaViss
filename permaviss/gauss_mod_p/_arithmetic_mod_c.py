"""
arithmetic_mod_c.py

Arithmetic functions for working mod c
Also includes function for inverses mod a prime number p
"""
import numpy as np

def add_mod_c(a, b, c):
    """
    Add two integers mod c. 
    INPUT:
    -a,b: integers to be added
    -c: integer to mod out by
    OUTPUT:
    -s: a + b (mod c)
    """
    s = a + b
    return s % c

assert add_mod_c(4,3,5)==2


def add_arrays_mod_c(A, B, c):
    """
    Add two arrays mod c. 
    INPUT:
    -a,b: integers to be added
    -c: integer to mod out by
    OUTPUT:
    -s: a + b (mod c)
    """
    if len(A) != len(B):
        print("Trying to add two arrays with different lengths!")
        raise ValueError

    N = len(A)
    C = np.zeros(N)
    # C = (A + B) % np.array(c)
    for i in range(N):
        C[i] = add_mod_c(A[i], B[i], c)
    
    return C

A1 = np.array([1,2,3])
B1 = np.array([4,0,4])
C1 = np.array([2,2,1])
assert np.array_equal(add_arrays_mod_c(A1, B1, 3), C1)

###############################################################################
# Inverse mod p

def inv_mod_p(a, p):
    """
    Returns the inverse of a mod p
    INPUT:
    -a, p: integers
    OUTPUT:
    -m: integer such that m * a = 1 (mod p)
    """
    x0, x1 = 0, 1
    y0, y1 = 1, 0
    b = p
    while a != 0:
        q, b, a = b // a, a, b % a
        y0, y1 = y1, y0 - q * y1
        x0, x1 = x1, x0 - q * x1
    
    if b != 1:
        print("Error:{} is not a prime number".format(p))
        raise ValueError

    if x0 < 0:
        x0 = x0 + p
    return x0

assert inv_mod_p(2,7) == 4 
assert inv_mod_p(9,5) == 4

