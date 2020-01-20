"""
    PH_classic.py

    This module implements a function which computes bases for the image and
    kernel of morphisms between persistence modules.
"""
import numpy as np

from ..gauss_mod_p import gauss_mod_p

from .barcode_bases import barcode_basis


###############################################################################
# Find pivot of array


def _pivot(l):
    """
    Compute pivot of a list of integers.

    Parameters
    ----------
    l : :obj:`list(int)`

    Returns
    -------
    index : int
        Index of last nonzero entry.
        Returns -1 if the list is zero.

    """
    l_bool = np.nonzero(l)
    if len(l_bool[0]) > 0:
        return l_bool[0][-1]

    return -1


assert _pivot(np.array([0, 1, 0, 1, 0])) == 3
assert _pivot(np.array([0, 0, 0])) == -1


###############################################################################
# Persistent homology mod p


def persistent_homology(D, R, max_rad, p):
    """
    Given the differentials of a filtered simplicial complex `X`,
    we compute its homology.

    In this function, the domain is on the columns and the range on the rows.
    Coordinates are stored as columns in an array.
    Barcode ranges are stored as pairs in an array of two columns.

    Parameters
    ----------
    D : :obj:`list(Numpy Array)`
        The ith entry stores the ith differential of the simplicial complex.
    R : :obj:`list(int)`
        The ith entry contains the radii of filtration for the ith skeleton
        of `X`. For example, the 1st entry contains, in order, the radii
        of each edge in `X`. In dimension 0 we have an empty list.
    p : int(prime)
        Chosen prime to perform arithmetic mod `p`.

    Returns
    -------
    Hom : :obj:`list(barcode_bases)`
        The `i` entry contains the `i` Persistent Homology classes for `X`.
        These are stored as :obj:`barcode_bases`. If a cycle does not die
        we put max_rad as death radius. Additionally, each entry is ordered
        according to the standard barcode order.
    Im : :obj:`list(barcode_bases)`
        The `i` entry contains the image of the `i+1` differential as
        :obj:`barcode_bases`.
    PreIm: :obj:`list(Numpy Array (len(R[*]), Im[*].dim)`
        Preimage matrices, 'how to go back from boundaries'

    Example
    -------
        >>> from permaviss.sample_point_clouds.examples import circle
        >>> from permaviss.simplicial_complexes.differentials import
        ... complex_differentials
        >>> from permaviss.simplicial_complexes.vietoris_rips import
        ... vietoris_rips
        >>> import scipy.spatial.distance as dist
        >>> point_cloud = circle(10, 1)
        >>> max_rad = 1
        >>> p = 5
        >>> max_dim = 3
        >>> Dist = dist.squareform(dist.pdist(point_cloud))
        >>> compx, R = vietoris_rips(Dist, max_rad, max_dim)
        >>> differentials = complex_differentials(compx, p)
        >>> Hom, Im, PreIm = persistent_homology(differentials, R, max_rad, p)
        >>> print(Hom[0])
        Barcode basis
        [[ 0.          1.        ]
         [ 0.          0.61803399]
         [ 0.          0.61803399]
         [ 0.          0.61803399]
         [ 0.          0.61803399]
         [ 0.          0.61803399]
         [ 0.          0.61803399]
         [ 0.          0.61803399]
         [ 0.          0.61803399]
         [ 0.          0.61803399]]
        [[ 1.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  4.  1.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  4.  1.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  4.  1.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  4.  0.  0.  1.  0.  0.]
         [ 0.  0.  0.  0.  0.  1.  0.  4.  0.  0.]
         [ 0.  0.  0.  0.  0.  4.  0.  0.  1.  0.]
         [ 0.  0.  0.  0.  0.  0.  1.  0.  4.  0.]
         [ 0.  0.  0.  0.  0.  0.  4.  0.  0.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  4.]]
        >>> print(Hom[1])
        Barcode basis
        [[ 0.61803399  1.        ]]
        [[ 4.]
         [ 4.]
         [ 4.]
         [ 4.]
         [ 4.]
         [ 4.]
         [ 4.]
         [ 4.]
         [ 4.]
         [ 1.]]
        >>> print(Im[0])
        Barcode basis
        [[ 0.61803399  1.        ]
         [ 0.61803399  1.        ]
         [ 0.61803399  1.        ]
         [ 0.61803399  1.        ]
         [ 0.61803399  1.        ]
         [ 0.61803399  1.        ]
         [ 0.61803399  1.        ]
         [ 0.61803399  1.        ]
         [ 0.61803399  1.        ]]
        [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 4.  1.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  4.  1.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  4.  1.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  4.  0.  0.  1.  0.  0.]
         [ 0.  0.  0.  0.  1.  0.  4.  0.  0.]
         [ 0.  0.  0.  0.  4.  0.  0.  1.  0.]
         [ 0.  0.  0.  0.  0.  1.  0.  4.  0.]
         [ 0.  0.  0.  0.  0.  4.  0.  0.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  4.]]
        >>> print(Im[1])
        Barcode basis
        []
        >>> print(PreIm[0])
        []
        >>> print(PreIm[1])
        [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  1.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  1.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  1.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  1.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  1.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  1.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  1.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]

    """
    dim = len(D)

    # Introduce list of Hom_bars, Hom_coord, Im_bars, Im_coord, and Preim
    Hom_bars = []
    Hom_coord = []
    Im_bars = []
    Im_coord = []
    PreIm = []
    for d in range(dim):
        Hom_bars.append([])
        Hom_coord.append([])
        Im_bars.append([])
        Im_coord.append([])
        PreIm.append([])

    # preallocate space for dim - 1
    domain_dim = np.size(D[dim - 1], 1)
    Hom_bars[dim - 1] = np.zeros((domain_dim, 2))
    Hom_coord[dim - 1] = np.zeros((domain_dim, domain_dim))
    Hom_dim_next = 0
    # Perform Gaussian eliminations, starting from D[dim-1] and ending on D[1].
    pivots = []
    for d in range(dim - 1, 0, -1):
        # Compute dimensions for domain and range of D[d]
        domain_dim = np.size(D[d], 1)
        range_dim = np.size(D[d], 0)
        Daux = np.copy(D[d])
        # Set pivot columns automatically to zero (clear optimization)
        Daux[:, pivots] = np.zeros((range_dim, len(pivots)))
        # Compute pivots complement and reset pivots to zero
        non_pivots = np.setdiff1d(range(domain_dim), pivots)
        pivots = []
        # preallocate space for the dimension d - 1
        Hom_bars[d - 1] = np.zeros((range_dim, 2))
        Hom_coord[d - 1] = np.zeros((range_dim, range_dim))
        Im_bars[d - 1] = np.zeros((len(non_pivots), 2))
        Im_coord[d - 1] = np.zeros((range_dim, len(non_pivots)))
        PreIm[d] = np.zeros((domain_dim, len(non_pivots)))
        # Perform Gaussian reduction by left to right column additions mod p
        Im_aux, T = gauss_mod_p.gauss_col(Daux, p)
        # Reset dimension variables
        Hom_dim_current = Hom_dim_next
        Hom_dim_next = 0
        Im_dim_next = 0
        # Go through all reduced columns of Im_aux which are not pivots
        for k in non_pivots:
            reduced_im = Im_aux[:, k]
            # If column is nonzero
            if np.any(reduced_im):
                pivots.append(_pivot(reduced_im))
                birth_rad = R[d - 1][_pivot(reduced_im)]
                death_rad = R[d][k]
                if birth_rad < death_rad:
                    Hom_bars[d - 1][Hom_dim_next] = [birth_rad, death_rad]
                    Hom_coord[d - 1][:, Hom_dim_next] = reduced_im
                    Hom_dim_next += 1
                # end if

                Im_bars[d - 1][Im_dim_next] = [death_rad, max_rad]
                Im_coord[d - 1][:, Im_dim_next] = reduced_im
                PreIm[d][:, Im_dim_next] = T[:, k]
                Im_dim_next += 1
            # end if
            # If column is zero
            else:
                birth_rad = R[d][k]
                death_rad = max_rad
                Hom_bars[d][Hom_dim_current] = [birth_rad, death_rad]
                Hom_coord[d][:, Hom_dim_current] = T[:, k]
                Hom_dim_current += 1
            # end else
        # end for

        # Get rid of extra preallocated storage
        Hom_bars[d] = Hom_bars[d][:Hom_dim_current]
        Hom_coord[d] = Hom_coord[d][:, :Hom_dim_current]
        Im_bars[d - 1] = Im_bars[d - 1][:Im_dim_next]
        Im_coord[d - 1] = Im_coord[d - 1][:, :Im_dim_next]
        PreIm[d] = PreIm[d][:, :Im_dim_next]
    # end for

    # Get infinite barcodes of Z[0]
    Hom_dim_current = Hom_dim_next
    non_pivots = np.setdiff1d(range(D[0]), pivots)
    for k in non_pivots:
        Hom_bars[0][Hom_dim_current] = [0, max_rad]
        Hom_coord[0][k, Hom_dim_current] = 1
        Hom_dim_current += 1
    # end for

    # free extra preallocated space
    Hom_bars[0] = Hom_bars[0][:Hom_dim_current]
    Hom_coord[0] = Hom_coord[0][:, :Hom_dim_current]

    # Store as persistence bases
    Hom = []
    Im = []
    # Reference to corresponding dimension R.
    # Here, we create barcode basis for the underlying complexes,
    # the case dim=0 is special.
    for d in range(dim):
        barcode_simplices = np.append([R[d]], [max_rad * np.ones(len(R[d]))],
                                      axis=0).T
        basis = barcode_basis(barcode_simplices)
        if d == 0:
            basis.dim = D[0]
        # end if
        if np.size(Hom_bars[d], 0) > 0:
            Hom.append(barcode_basis(Hom_bars[d], basis, Hom_coord[d]))
            Hom[d].sort()
        else:
            Hom.append(barcode_basis([]))
        # end else
        if len(Im_bars[d]) > 0:
            # Order PreIm according to order of Im
            Im.append(barcode_basis(Im_bars[d], basis, Im_coord[d],
                                    store_well_defined=True))
            PreIm[d+1] = PreIm[d+1][:, Im[-1].well_defined]
            order = Im[d].sort(send_order=True)
            PreIm[d+1] = PreIm[d+1][:, order]
        else:
            Im.append(barcode_basis([]))
        # end else
    # end for

    """Return everything."""
    return Hom, Im, PreIm
