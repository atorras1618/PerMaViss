"""
test_functions.py
"""

import numpy as np

from permaviss.gauss_mod_p.functions import (
    multiply_mod_p, solve_mod_p, solve_matrix_mod_p)


def test_multiply_mod_p():
    A = np.array([[2, 3], [4, 5]])
    B = np.array([[1, 1], [2, 3]])
    assert np.array_equal(multiply_mod_p(A, B, 5),
                          np.array([[3, 1], [4, 4]]))
    A = np.array([[1, 0], [1, 1]])
    B = np.array([[1, 1], [0, 1]])
    assert np.array_equal(multiply_mod_p(A, B, 2),
                          np.array([[1, 1], [1, 0]]))
    A = np.array([
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1]])
    B = np.array([1, 1, 1])
    assert np.array_equal(multiply_mod_p(A, B, 5),
                          np.array([3, 2, 1]))


def test_solve_mod_p():
    A = np.array([
        [1.,  1.,  1.,  0.],
        [0.,  4.,  0.,  1.],
        [0.,  0.,  4.,  0.],
        [0.,  0.,  0.,  4.]])
    b = np.array([0.,  0.,  1.,  4.])
    p = 5
    solution = solve_mod_p(A, b, p)
    correct_sol = np.array([0, 1, 4, 1])
    assert np.allclose(solution, correct_sol, rtol=1e-05, atol=1e-08)


def test_solve_matrix_mod_p():
    A = np.array([
        [1.,  1.,  1.,  0.],
        [0.,  4.,  0.,  1.],
        [0.,  0.,  4.,  0.],
        [0.,  0.,  0.,  4.]])
    B = np.array([
        [0.,  0.,  1.,  4.],
        [4.,  0.,  0.,  3.]])
    p = 5
    solution = solve_matrix_mod_p(A, B, p)
    correct_sol = np.array([
        [0, 1, 4, 1],
        [2, 2, 0, 2]]).T
    assert np.allclose(solution, correct_sol, rtol=1e-05, atol=1e-08)
