"""
    module_persistence_homology.py

    This module implements the persistence module homology
"""
import numpy as np

from .barcode_bases import barcode_basis

from .image_kernel import image_kernel

###############################################################################
# Find pivot of array


def _pivot(l):
    """Given a 1D array of integers, return the index of the last nonzero entry.

    Parameters
    ----------
    l : :obj:`list`
        1D array of integers.

    Returns
    -------
    int
        Index of last nonzero entry. If `l` is zero, returns -1.

    """
    l_bool = np.nonzero(l)
    if len(l_bool[0]) > 0:
        return l_bool[0][-1]

    return -1


assert _pivot(np.array([0, 1, 0, 1, 0])) == 3
assert _pivot(np.array([0, 0, 0])) == -1


###############################################################################
# Quotient of barcode bases


def quotient(M, N, p):
    """Assuming that N generates a submodule of M, we compute
    a barcode basis for the quotient M / N.

    Parameters
    ----------
    M : :obj:`barcode_basis`
        Basis for module

    N : :obj:`barcode_basis`
        Basis for submodule of N

    p : int(prime)

    Returns
    -------
    Q : :obj:`barcode_basis`
        Barcode basis for the quotient M / N

    """
    if N.dim == 0:
        return M

    # Check that the barcode bases coincide
    if M.prev_basis != N.prev_basis:
        raise ValueError

    # We create N_M, which is not a basis, but will be used as the domain
    # basis in image_kernel so that the quotient mod N can be performed
    N_M_barcode = np.concatenate((N.barcode, M.barcode), axis=0)
    N_M = barcode_basis(N_M_barcode)
    # create a matrix (N|M) in M.prev_basis
    matrix_N_M = np.concatenate((N.coordinates, M.coordinates), axis=1)
    # Compute quotient barcodes for M mod N
    Q, Ker, PreIm = image_kernel(N_M, M.prev_basis, matrix_N_M, p,
                                 start_index=N.dim, prev_basis=M)
    return Q


###############################################################################
# Persistence module homology mod p


def module_persistence_homology(D, Base, p):
    """Given the differentials of a chain of tame persistence modules, we
    compute barcode bases for the homology of the chain.

    Parameters
    ----------
    D : :obj:`list(Numpy Array)`
        List of differentials of the chain complex.
    Base : :obj:`Numpy Array`
        List containing barcode bases for each dimension
    p : int(prime)
        Prime number to perform arithmetic mod p

    Returns
    -------
    Hom : :obj:`list(barcode_basis)`
        Cycles mod boundaries of differentials, starting with: birth rad,
        death rad. If a cycle does not die we put max_rad as death radius.
    Im : :obj:`list(barcode_basis)`
        List storing bases for the images of differentials
    PreIm : :obj:`list(Numpy Array)`
        List storing bases for the preimages of the differentials. That is,
        which generators produce each image generator. This leads to how to
        `go back` from boundaries to preimages.

    """
    dim = len(Base)
    # lists of homology, boundaries and preboundaries
    Hom = []
    Im = []
    PreIm = []
    for d in range(dim):
        Hom.append([])
        Im.append([])
        PreIm.append([])

    Im[dim - 1] = barcode_basis([], Base[d])
    # Find bases for Kernel and Image, starting from D[dim-1], ending on D[1].
    for d in range(dim - 1, 0, -1):
        # Handle trivial cases
        if Base[d].dim == 0:
            Im[d-1] = barcode_basis([], Base[d-1])
            Hom[d] = barcode_basis([], Base[d])
        elif Base[d-1].dim == 0:
            Im[d-1] = Base[d-1]
            Hom[d] = Base[d]
        else:
            # compute barcode bases for image and kernel
            Im[d - 1], Ker, PreIm[d] = image_kernel(Base[d], Base[d-1], D[d],
                                                    p)
            # Perform quotient of Ker by Im[d]
            Hom[d] = quotient(Ker, Im[d], p)

    # Trivial case dim 0
    if Base[0].dim == 0:
        Hom[0] = Base[0]
    else:
        # All the generators from Base[0] lie in the kernel, since D[0]=0
        Ker = barcode_basis(Base[0].barcode, Base[0], np.identity(Base[0].dim))
        Ker.sort()
        if Ker.prev_basis != Im[0].prev_basis:
            print(Ker.prev_basis)
            print(Im[0].prev_basis)
            assert False

        Hom[0] = quotient(Ker, Im[0], p)

    # Return lists of barcode bases for Hom, Im and PreIm
    return Hom, Im, PreIm
