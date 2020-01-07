import numpy as np

import scipy.spatial.distance as dist 

from permaviss.simplicial_complexes.vietoris_rips import *

def test_vietoris_rips():
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    Dist = dist.squareform(dist.pdist(X))
    ### Expected values
    expected_complex = [4, 
        np.array([[0, 1],[0, 2],[1, 3],[2, 3],[1, 2],[0, 3]]), 
        np.array([[0, 1, 3],[0, 2, 3],[1, 2, 3],[0, 1, 2]]), 
        np.array([[0, 1, 2, 3]]), 
        np.array([])
    ]
    expected_R = [np.zeros(4), 
        np.array([1,1,1,1,np.sqrt(2),np.sqrt(2)]),
        np.array([np.sqrt(2),np.sqrt(2),np.sqrt(2),np.sqrt(2)]),
        np.array([np.sqrt(2)]),
        np.array([])
    ] 
    ### Calculate vietoris_rips complex
    viRip , R = vietoris_rips(Dist, 4,4)
    print(viRip)
    assert viRip[0]==expected_complex[0]
    for dim, simplices in enumerate(viRip[1:]):
        assert np.array_equal(simplices, expected_complex[dim+1])

    for dim, rad in enumerate(R):
        assert np.array_equal(rad, expected_R[dim])
    

