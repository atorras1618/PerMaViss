"""
    differentials.py

"""
import numpy as np


def complex_differentials(C, p):
    """Given a simplicial complex C, it returns a list D with its
    differentials mod p

    Parameters
    ----------
    C : :obj:`list(int, Numpy Array, ...)`
        The `0` entry stores the number of vertices. For a higher entry `i`,
        `C[i]` stores a :obj:`Numpy Array` matrix with the `i` simplices
        from `C`.
    D : :obj:`list(Numpy Array)`
        List of differentials of `C`. The `i` entry contains a :obj:`Numpy
        Array` of the `i` differential for the simplicial complex.

    """
    dim = len(C) - 1  # get the dimension of the simplicial complex
    D = []
    for d in range(dim + 1):
        D.append([])

    D[0] = C[0]  # number of points
    # create 1D boundary
    D[1] = np.zeros((C[0], np.size(C[1], 0)))
    for i, c in enumerate(C[1]):
        D[1][int(c[0])][i] = 1
        D[1][int(c[1])][i] = p - 1

    # We proceed to compute all the differentials for all dimensions
    for d in range(2, dim + 1):
        """
            Go through all the d-simplexes of C, computing their images
            in (d-1)-simplex basis
        """
        D[d] = np.zeros((np.size(C[d-1], 0), np.size(C[d], 0)))
        for i, s in enumerate(C[d]):
            im = []  # initialize image of s
            signs = []  # also initialize signs of faces
            for j, c in enumerate(C[d - 1]):
                if len(np.intersect1d(c, s)) == d:
                    face_idx = np.setdiff1d(s, c)[0]
                    im.append(j)
                    signs.append((-1) ** np.searchsorted(s, face_idx))

            for j, v in enumerate(im):
                D[d][v][i] = signs[j] % p

    return D
