"""
arithmetic_mod_c.py

Arithmetic functions for working mod c
Also includes function for inverses mod a prime number p
"""
import numpy as np


def add_mod_c(a, b, c):
    """
    Integer addition mod c.

    Parameters
    ----------

    a,b : int
        Integers to be added
    c : int
        Integer to mod out by

    Returns
    -------
    s : a + b (mod c)
    """
    s = a + b
    return s % c


def add_arrays_mod_c(A, B, c):
    """
    Adds two arrays mod c.

    Parameters
    ----------
    a,b : :obj:`Numpy Array(int)`
        Two integer arrays to be added.
    c : int
        Integer to mod out by.

    Returns
    -------
    C : :obj:`Numpy Array(int)`
        a + b (mod c)

    """
    return np.mod(A + B, c)


def inv_mod_p(a, p):
    """
    Returns the inverse of a mod p

    Parameters
    ----------
    a : int
        Number to compute the inverse mod p
    p : int(prime)

    Returns
    -------
    m : int
        Integer such that m * a = 1 (mod p)

    Raises
    ------
    ValueError
        If p is not a prime number
    """
    if a%p == 0:
        print("cannot invert 0!")
        raise ValueError

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
