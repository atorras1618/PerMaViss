import numpy as np
import scipy.spatial.distance as dist
from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss
from permaviss.simplicial_complexes.vietoris_rips import vietoris_rips
from permaviss.simplicial_complexes.differentials import complex_differentials
from permaviss.persistence_algebra.PH_classic import persistent_homology

#def test_second_diff():
#    saved_data = np.loadtxt(
#        "test/spectral_sequence/second_differential_ex_1.txt")
#    no_points = int(np.size(saved_data,0) / 3)
#    X = np.reshape(saved_data, (no_points, 3))
#    max_r = 0.39
#    max_dim = 4
#    max_div = 2
#    overlap = max_r
#    p = 5
#    # compute ordinary persistent homology
#    Dist = dist.squareform(dist.pdist(X))
#    C, R = vietoris_rips(Dist, max_r, max_dim)
#    Diff = complex_differentials(C, p)
#    PerHom, _, _ = persistent_homology(Diff, R, max_r, p)
#    # compute spectral sequence
#    MV_ss = create_MV_ss(X, max_r, max_dim, max_div, overlap, p)
#    # Check that computed barcodes coincide
#    for it, PH in enumerate(MV_ss.persistent_homology):
#        assert np.array_equal(PH.barcode, PerHom[it].barcode)
