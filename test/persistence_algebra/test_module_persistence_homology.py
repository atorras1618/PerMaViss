
import numpy as np

from permaviss.persistence_algebra.barcode_bases import barcode_basis
from permaviss.persistence_algebra.module_persistence_homology import (
    module_persistence_homology)


def test_module_persistence_homology():
    A_bas = barcode_basis([[1, 4], [2, 4], [3, 5]])
    B_bas = barcode_basis([[0, 4], [0, 2], [1, 4], [1, 3]])
    C_bas = barcode_basis([[0, 3], [0, 2], [0, 2]])
    Base = [C_bas, B_bas, A_bas]
    f = np.array([[0, 1, 1], [1, 0, 0], [0, 4, 1], [0, 0, 0]])
    g = np.array([[0, 0, 0, 1], [4, 0, 1, 3], [1, 0, 1, 4]])
    D = [0, g, f]
    Hom, Im, PreIm = module_persistence_homology(D, Base, 5)
    H0 = np.array([[0, 1], [0, 1]])
    H1 = np.array([[0, 1], [2, 3]])
    H2 = np.array([[2, 4], [4, 5]])
    H = [H0, H1, H2]
    PreIm0 = []
    PreIm1 = np.array([[1, 2, 4], [0, 0, 0], [0, 4, 1], [0, 1, 0]])
    PreIm2 = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    PreImages = [PreIm0, PreIm1, PreIm2]
    for j in range(3):
        assert np.array_equal(Hom[j].barcode, H[j])
        assert np.array_equal(PreIm[j], PreImages[j])
