import numpy as np

from permaviss.gauss_mod_p.arithmetic_mod_p import (
    add_mod_c, add_arrays_mod_c, inv_mod_p)


def test_add_mod_c():
    assert add_mod_c(4, 3, 5) == 2


def test_add_arrays_mod_c():
    A1 = np.array([1, 2, 3])
    B1 = np.array([4, 0, 4])
    C1 = np.array([2, 2, 1])
    assert np.array_equal(add_arrays_mod_c(A1, B1, 3), C1)


def test_inv_mod_p():
    assert inv_mod_p(2, 7) == 4
    assert inv_mod_p(9, 5) == 4
