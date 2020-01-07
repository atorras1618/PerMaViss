
import numpy as np

from permaviss.simplicial_complexes.flag_complex import *

def test_flag_complex():
    graph = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
    expected_complex = [4, 
        np.array([[0, 1],[0, 2],[0, 3],[1, 2],[1, 3],[2, 3]]), 
        np.array([[0, 1, 2],[0, 1, 3],[0, 2, 3],[1, 2, 3]]), 
        np.array([[0, 1, 2, 3]]), 
        np.array([])
    ]
    result_complex =  flag_complex(graph, 4, 4)
    for dim, simplex in enumerate(result_complex):
        print(simplex)
        print(expected_complex[dim])
        assert np.array_equal(expected_complex[dim],simplex)

