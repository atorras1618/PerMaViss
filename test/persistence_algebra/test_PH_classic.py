
import numpy as np
import scipy.spatial.distance as dist

from permaviss.simplicial_complexes.differentials import complex_differentials
from permaviss.simplicial_complexes.vietoris_rips import vietoris_rips

from permaviss.persistence_algebra.PH_classic import persistent_homology


def test_persistent_homology():
    # Test 1 "Two bars"
    max_r = 2.4
    max_dim = 3
    p = 5
    X = np.array([
        [0, 0], [0, 1], [0, 2],
        [1, 0], [1, 2],
        [2, 0], [2, 0.4], [2, 1.6], [2, 2],
        [3, 0], [3, 2],
        [4, 0], [4, 1], [4, 2]])
    Dist = dist.squareform(dist.pdist(X))
    compx, R = vietoris_rips(Dist, max_r, max_dim)
    differentials = complex_differentials(compx, p)
    Hom, Im, PreIm = persistent_homology(differentials, R, max_r, p)
    cycle_bars = np.array([[1, 2], [1.2, 2]])
    assert np.allclose(np.copy(Hom[1].barcode), cycle_bars,
                       rtol=1e-05, atol=1e-08)
    # Test 2 "Four points"
    max_r = 2
    max_dim = 3
    p = 5
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    Dist = dist.squareform(dist.pdist(X))
    compx, R = vietoris_rips(Dist, max_r, max_dim)
    differentials = complex_differentials(compx, p)
    Hom, Im, PreIm = persistent_homology(differentials, R, max_r, p)
    Im_1_res = np.array([
        [1.,  4.,  0.],
        [0.,  1.,  0.],
        [1.,  4.,  4.],
        [0.,  1.,  1.],
        [0.,  0.,  1.],
        [4.,  0.,  0.]])
    PreIm_2_res = np.array([
        [1.,  4.,  0.],
        [0.,  1.,  0.],
        [0.,  0.,  1.],
        [0.,  0.,  0.]])
    assert np.allclose(Im[1].coordinates, Im_1_res, rtol=1e-05, atol=1e-08)
    assert np.allclose(PreIm[2], PreIm_2_res, rtol=1e-05,  atol=1e-08)
