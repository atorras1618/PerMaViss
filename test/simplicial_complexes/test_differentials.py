
import numpy as np

import scipy.spatial.distance as dist

from permaviss.simplicial_complexes.vietoris_rips import vietoris_rips

from permaviss.simplicial_complexes.differentials import (
    complex_differentials)


def test_complex_differentials():
    X = np.array([
        [1.91580552,  0.57418571],
        [-0.72993636,  1.86203999],
        [1.97700111,  0.30243449],
        [1.99699445,  0.10960461],
        [-1.58839255,  1.21532264]])
    Dist = dist.squareform(dist.pdist(X))
    max_r = 3
    max_dim = 4
    p = 5
    C, _ = vietoris_rips(Dist, max_r, max_dim)
    Diff = complex_differentials(C, p)
    res = np.matmul(Diff[1], Diff[2]) % p
    assert np.array_equal(res, np.zeros(
        (np.size(Diff[1], 0), np.size(Diff[2], 1))))
    res = np.matmul(Diff[2], Diff[3]) % p
    assert np.array_equal(res, np.zeros(
        (np.size(Diff[2], 0), np.size(Diff[3], 1))))
