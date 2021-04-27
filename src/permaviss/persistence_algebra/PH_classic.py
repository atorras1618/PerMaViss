"""
    PH_classic.py

    This module implements a function which computes bases for the image and
    kernel of morphisms between persistence modules.
"""
from functools import lru_cache

import numpy as np
from numba import njit
from numba import types
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.unsafe.ndarray import to_fixed_tuple
from numba.typed import List, Dict

from .barcode_bases import barcode_basis
from ..gauss_mod_p import gauss_mod_p


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


def sort_filtration_by_dim(simplex_tree, maxdim=None):
    if maxdim is None:
        maxdim = simplex_tree.dimension()

    filtration_by_dim = [[] for _ in range(maxdim + 1)]
    for idx, (spx, value) in enumerate(simplex_tree.get_filtration()):
        spx_t = tuple(sorted(spx))
        dim = len(spx_t) - 1
        if dim <= maxdim:
            filtration_by_dim[dim].append([idx, spx_t, value])

    for dim, filtr in enumerate(filtration_by_dim):
        filtration_by_dim[dim] = [np.asarray(x)
                                  for i, x in enumerate(zip(*filtr))]

    return filtration_by_dim


@njit
def _twist_reduction(boundary, triangular, pivots_lookup):
    """R = MV"""
    n = len(boundary)

    pos_idxs_to_clear = []
    for j in range(n):
        lowest_one = boundary[j][-1] if boundary[j] else -1
        pivot_col = pivots_lookup[lowest_one]
        while (lowest_one != -1) and (pivot_col != -1):
            boundary[j] = _symm_diff(boundary[j][:-1],
                                     boundary[pivot_col][:-1])
            triangular[j] = _symm_diff(triangular[j],
                                       triangular[pivot_col])
            lowest_one = boundary[j][-1] if boundary[j] else -1
            pivot_col = pivots_lookup[lowest_one]
        if lowest_one != -1:
            pivots_lookup[lowest_one] = j
            pos_idxs_to_clear.append(lowest_one)

    return np.asarray(pos_idxs_to_clear, dtype=np.int64)


@lru_cache
def _reduce_single_dim(dim):
    len_tups_dim = dim + 1
    len_tups_next_dim = dim
    tuple_typ_next_dim = types.UniTuple(types.int64, len_tups_next_dim)
    int64_list_typ = types.List(types.int64)

    @njit
    def _inner_reduce_single_dim(tups_dim, pos_idxs_to_clear,
                                 tups_next_dim=None):
        """R = MV"""
        # Initialize reduced matrix as the (cleared) boundary matrix
        reduced = List.empty_list(int64_list_typ)
        triangular = List.empty_list(int64_list_typ)
        if tups_next_dim is not None:
            spx2idx_next_dim = Dict.empty(tuple_typ_next_dim, types.int64)
            for j in range(len(tups_next_dim)):
                spx = to_fixed_tuple(tups_next_dim[j], len_tups_next_dim)
                spx2idx_next_dim[spx] = j

            if not pos_idxs_to_clear.size:
                for i in range(len(tups_dim)):
                    spx = to_fixed_tuple(tups_dim[i], len_tups_dim)
                    reduced.append(sorted([spx2idx_next_dim[face]
                                           for face in _drop_elements(spx)]))
                    triangular.append([i])
            else:
                to_clear_bool = np.zeros(len(tups_dim), dtype=np.bool_)
                to_clear_bool[pos_idxs_to_clear] = True
                for i in range(len(tups_dim)):
                    spx = to_fixed_tuple(tups_dim[i], len_tups_dim)
                    if to_clear_bool[i]:
                        reduced.append([types.int64(x) for x in range(0)])
                    else:
                        reduced.append(
                            sorted([spx2idx_next_dim[face]
                                    for face in _drop_elements(spx)])
                            )
                    triangular.append([i])

            pivots_lookup = np.full(len(tups_next_dim), -1, dtype=np.int64)

            pos_idxs_to_clear = _twist_reduction(reduced,
                                                 triangular,
                                                 pivots_lookup)

        else:
            for i in range(len(tups_dim)):
                reduced.append([types.int64(x) for x in range(0)])
                triangular.append([i])

        return reduced, triangular, pos_idxs_to_clear

    return _inner_reduce_single_dim


def get_reduced_triangular(filtr_by_dim):
    maxdim = len(filtr_by_dim) - 1
    pos_idxs_to_clear = np.empty(0, dtype=np.int64)
    reduced_triangular = []  # WARNING: Populated in reverse order
    for dim in range(maxdim, 0, -1):
        reduction_dim = _reduce_single_dim(dim)
        _, tups_dim, _ = filtr_by_dim[dim]
        _, tups_next_dim, _ = filtr_by_dim[dim - 1]
        reduced, triangular, pos_idxs_to_clear = reduction_dim(
            tups_dim,
            pos_idxs_to_clear=pos_idxs_to_clear,
            tups_next_dim=tups_next_dim
            )
        reduced_triangular.append((reduced, triangular))

    reduction_dim = _reduce_single_dim(0)
    _, tups_dim, _ = filtr_by_dim[0]
    reduced, triangular, _ = reduction_dim(tups_dim,
                                           pos_idxs_to_clear=pos_idxs_to_clear)
    reduced_triangular.append((reduced, triangular))

    return reduced_triangular[::-1]


def get_barcode(filtr_by_dim):
    reduced_triangular = get_reduced_triangular(filtr_by_dim)
    maxdim = len(filtr_by_dim) - 1

    pairs = [list()] * len(filtr_by_dim)

    _, _, values_maxdim = filtr_by_dim[maxdim]
    reduced_maxdim, _ = reduced_triangular[maxdim]
    pairs_maxdim = []
    for i in range(len(values_maxdim)):
        if not reduced_maxdim[i]:
            pairs_maxdim.append((values_maxdim[i], np.inf))
    pairs[maxdim] = sorted(pairs_maxdim)

    for dim in range(maxdim - 1, -1, -1):
        all_birth_indices = set()
        _, _, values_dim = filtr_by_dim[dim]
        reduced_dim, _ = reduced_triangular[dim]
        _, _, values_prev_dim = filtr_by_dim[dim + 1]
        reduced_prev_dim, _ = reduced_triangular[dim + 1]

        pairs_dim = []
        for j in range(len(values_prev_dim)):
            if reduced_prev_dim[j]:
                i = reduced_prev_dim[j][-1]
                b, d = values_dim[i], values_prev_dim[j]
                if b != d:
                    pairs_dim.append((b, d))
                all_birth_indices.add(i)

        for i in range(len(values_dim)):
            if i not in all_birth_indices:
                if not reduced_dim[i]:
                    pairs_dim.append((values_dim[i], np.inf))

        pairs[dim] = sorted(pairs_dim)

    return pairs


@njit
def _symm_diff(x, y):
    n = len(x)
    m = len(y)
    result = []
    i = 0
    j = 0
    while (i < n) and (j < m):
        if x[i] < y[j]:
            result.append(x[i])
            i += 1
        elif y[j] < x[i]:
            result.append(y[j])
            j += 1
        else:
            i += 1
            j += 1

    while i < n:
        result.append(x[i])
        i += 1

    while j < m:
        result.append(y[j])
        j += 1

    return result


@njit
def _drop_elements(tup: tuple):
    for x in range(len(tup)):
        empty = tup[:-1]  # Not empty, but the right size and will be mutated
        idx = 0
        for i in range(len(tup)):
            if i != x:
                empty = tuple_setitem(empty, idx, tup[i])
                idx += 1
        yield empty


def check_agreement_with_gudhi(gudhi_barcode, barcode):
    max_dimension_gudhi = max([pers_info[0] for pers_info in gudhi_barcode])
    assert max_dimension_gudhi <= len(barcode) - 1

    for dim, barcode_dim in enumerate(barcode):
        gudhi_barcode_dim = sorted([
            pers_info[1] for pers_info in gudhi_barcode if pers_info[0] == dim
            ])
        assert gudhi_barcode_dim == sorted(barcode_dim), \
            f"Disagreement in degree {dim}"
