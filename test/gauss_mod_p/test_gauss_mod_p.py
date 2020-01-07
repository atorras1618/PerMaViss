
import numpy as np

from permaviss.gauss_mod_p.gauss_mod_p import *

def test_gauss_col():
    A1 = np.array([[0,1,1], [1,1,0], [1,1,0]])
    R1 = np.array([[0,1,0], [1,0,0], [1,0,0]])
    T1 = np.array([[1,1,1], [0,1,1], [0,0,1]])
    eq1 = np.array([R1, T1])
    eq2 = np.array(gauss_col(A1, 2))
    assert np.array_equal(eq1, eq2)
    
    A2 = np.array([[2,1,0,1], [3,1,2,3], [4,1,4,3], [2,1,1,0]])
    R2 = np.array([[2,0,4,3], [3,2,2,0], [4,4,0,0], [2,0,0,0]])
    T2 = np.array([[1,2,1,4], [0,1,2,4], [0,0,1,3], [0,0,0,1]])
    eq1 = np.array([R2,T2])
    eq2 = np.array(gauss_col(A2, 5))
    assert np.array_equal(eq1, eq2)

