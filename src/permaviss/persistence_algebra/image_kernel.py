"""
    image_kernel.py

    This module implements a function which computes bases for the image and
    kernel of morphisms between persistence modules.
"""
import numpy as np

from .barcode_bases import barcode_basis

from ..gauss_mod_p.gauss_mod_p import index_pivot
from ..gauss_mod_p.arithmetic_mod_p import inv_mod_p, add_mod_c
from ..gauss_mod_p.functions import multiply_mod_p


###############################################################################
# Find barcode generators for image and kernel of a persistence morphism.

def image_kernel(A, B, F, p, start_index=0, prev_basis=None):
    """
    This computes basis for the image and kernel of a persistence morphism.
        f: A --> B

    This is the algorithm described in https://arxiv.org/abs/1907.05228.
    Recall that for such an algorithm to work the `A` and `B` must be ordered.
    This is why the function first orders the barcode generators from
    start_index until A.dim.
    Additionally, it also orders the barcodes from B. By 'ordered' we mean that
    the barcodes are sorted according to the standard order of barcodes.

    It can also compute relative barcode bases for the image. This is used when
    computing quotients. The optional argument start_index indicates the
    minimum index from which we want to compute barcodes relative to the
    previous generators. That is, given start_index, the function will return
    image barcodes for

            <F[start_dim, ..., A.dim]>  mod <F[0,1,..., start_dim-1]>.

    At the end the bases for the image and kernel are
    returned in terms of the original ordering.

    Additionally, this handles the case for when B is a broken barcode basis.

    *Notice that in both relative or broken barcodes only the barcode basis of
    the image will be computed.

    Parameters
    ----------
    A : :class:`barcode_basis` object
        Basis of domain.
    B : :class:`barcode_basis` object
        Basis of range.  This can be a `broken` barcode basis.
    F : Numpy Array (`B`.dim, `A`.dim)
        Matrix associated to the considered persistence morphism.
    p : int
        prime number of finite field.
    start_index : int, default is 0
        Index from which we get a barcode basis for the image.
    prev_basis : int, default is None
        If  `start_index` > 0, we need to also give a reference to a basis
        of barcodes from A[start_dim] until A[A.dim].

    Returns
    -------
    Ker : :class:`barcode_basis` object
        Absolute/relative basis of kernel of f.
    Im : :class:`barcode_basis` object
        Absolute/relative basis of image of f.
    PreIm : Numpy Array (A.dim, Im.dim)
        Absolute/relative preimage coordinates of f. That is, each column
        stores the sums that generate the corresponding Image barcode.

    Examples
    --------

        >>> import numpy as np
        >>> from permaviss.persistence_algebra.barcode_bases import
        ... barcode_basis
        >>> A = barcode_basis([[1,8],[1,5],[2,5], [4,8]])
        >>> B = barcode_basis([[-1,3],[0,4],[0,3.5],[2,5],[2,4],[3,8]])
        >>> F = np.array([[4,1,1,0],[1,4,1,0],[1,1,4,0],[0,0,1,4],[0,0,4,1],
        ... [0,0,0,1]])
        >>> p = 5
        >>> Im, Ker, PreIm = image_kernel(A,B,F,p)
        >>> print(Im)
        Barcode basis
        [[ 1.   4. ]
         [ 1.   3.5]
         [ 2.   5. ]
         [ 4.   8. ]]
        [[ 4.  0.  1.  0.]
         [ 1.  0.  1.  0.]
         [ 1.  2.  4.  0.]
         [ 0.  0.  1.  4.]
         [ 0.  0.  4.  1.]
         [ 0.  0.  0.  1.]]
        >>> print(Ker)
        Barcode basis
        [[ 3.5  8. ]
         [ 4.   5. ]]
        [[ 1.  0.]
         [ 1.  4.]
         [ 0.  0.]
         [ 0.  0.]]
        >>> print(PreIm)
        [[ 1.  1.  0.  0.]
         [ 0.  1.  0.  0.]
         [ 0.  0.  1.  0.]
         [ 0.  0.  0.  1.]]

    Note
    ----
    This algorithm will only work if the matrix of the persistence morphism
    is well defined. That is, a generator can only map to generators that have
    been born and have not yet died.

    """
    # Throw ValueError if dimensions of A, B and F do not fit
    if B.dim != np.size(F, 0) or A.dim != np.size(F, 1):
        raise ValueError
    # If the basis B is not ordered, we order it
    B_origin = B
    return_order_B = range(B.dim)
    if not B.sorted:
        B = barcode_basis(B.barcode, broken_basis=B.broken_basis,
                          broken_differentials=B.broken_differentials)
        order = B.sort(send_order=True)
        F = np.copy(F[order])
        return_order_B = np.argsort(order)

    # Order A from start_index until A.dim
    A_origin = A
    return_order_A = range(A.dim)
    if not A.sorted:
        A_rel = barcode_basis(A.barcode[start_index:])
        order = A_rel.sort(send_order=True)
        shifted_order = order + start_index
        shifted_order = np.append(range(start_index), shifted_order,
                                  axis=0).astype(int)
        return_order_A = np.argsort(order)
        F = np.copy(F[:, shifted_order])
        A = barcode_basis(np.append(A.barcode[:start_index], A_rel.barcode,
                                    axis=0))
    # end if
    # Get sorted radii where changes happen
    a = np.sort(np.unique(np.append(A.changes_list(), B.changes_list())))
    # Save space for Im barcode, coordinates and pivots
    Im_barcode = np.concatenate((
        [A.barcode[:, 0]], [np.zeros(A.dim)]), axis=0).T
    # Initialize a copy of F which will be changing as we iterate over a
    Im_coordinates = np.copy(F)
    T = np.identity(A.dim)
    # compute last nonzero entries of columns in Im_coordinates
    pivots_Im = np.ones(A.dim) * (-2)
    pivots_Im = pivots_Im.astype(int)
    # Save space for Ker barcode, coordinates and pivots (if necessary)
    if (not B.broken_basis) and (start_index == 0):
        Ker_barcode = np.zeros((A.dim, 2))
        Ker_coordinates = np.zeros((A.dim, A.dim))
        kernel_dim = 0
        # value -2 means that the corresponding column has not yet a pivot
        pivots_Ker = np.ones(A.dim) * (-2)
        pivots_Ker = pivots_Ker.astype(int)
    # Go through all radius where a change happens
    # do a first reduction if a[0]==0 if necessary
    for i, rad in enumerate(a):
        active_A = A.active(rad)
        active_B = B.active(rad)
        dying_A = A.death(rad)
        birth_A = A.birth(rad)
        dying_B = B.death(rad)
        if (not B.broken_basis) and (start_index == 0):
            new_pivots = []
            for col_idx, piv in enumerate(np.copy(pivots_Ker)):
                if (piv in dying_A) or (piv in new_pivots):
                    pivots_Ker[col_idx] = pivot_check_reduce(
                        Ker_coordinates, active_A, col_idx,
                        pivots_Ker[:col_idx], p)
                    if pivots_Ker[col_idx] != -1:
                        new_pivots.append(pivots_Ker[col_idx])
                        Im_barcode[pivots_Ker[col_idx], 1] = rad
                        cycle_coord = Ker_coordinates[:, col_idx]
                        new_coord_Im = multiply_mod_p(
                            F[:, active_A],
                            np.transpose([cycle_coord[active_A]]), p)[:, 0]
                        if np.any(new_coord_Im[active_B]):
                            raise ValueError
                        elif pivots_Im[pivots_Ker[col_idx]] != -1:
                            Im_coordinates[
                                :, pivots_Ker[col_idx]] = new_coord_Im
                            Im_barcode[pivots_Ker[col_idx], 1] = rad
                            pivots_Im[pivots_Ker[col_idx]] = int(-1)
                            T[:, pivots_Ker[col_idx]][
                                active_A] = cycle_coord[active_A]
                        # end if else
                    else:
                        Ker_barcode[col_idx, 1] = rad
                    # end if else
                # end if
            # end for
        else:
            for col_idx in dying_A:
                if (col_idx >= start_index) and (pivots_Im[col_idx] != -1):
                    Im_barcode[col_idx, 1] = rad
                    pivots_Im[col_idx] = int(-1)
                # end if
            # end for
            # update broken barcodes
            if B.broken_basis:
                Im_coordinates = B.update_broken(Im_coordinates, rad, p)
        # end if else
        new_pivots = []
        for col_idx, piv in enumerate(np.copy(pivots_Im)):
            if (piv >= 0 and Im_coordinates[piv, col_idx] % p == 0) or (
                    piv in dying_B) or (piv in new_pivots) or (col_idx in birth_A):
                pivots_Im[col_idx] = pivot_check_reduce(
                    Im_coordinates, active_B, col_idx, pivots_Im[:col_idx],
                    p, T)
                if pivots_Im[col_idx] != -1:
                    new_pivots.append(pivots_Im[col_idx])
                else:
                    Im_barcode[col_idx, 1] = rad
                    # add if some bar is being born in kernel
                    if (col_idx not in dying_A) and (not B.broken_basis) and (
                            start_index == 0):
                        Ker_barcode[kernel_dim, 0] = rad
                        Ker_coordinates[:, kernel_dim] = T[:, col_idx]
                        pivots_Ker[kernel_dim] = int(col_idx)
                        kernel_dim += 1
                    if start_index > 0:
                        T[:, col_idx][:start_index] *= 0
                        Im_coordinates[:, col_idx] = multiply_mod_p(
                            F, np.transpose([T[:, col_idx]]), p)[:, 0]
                # end if else
            # end if
        # end for
    # end for
    # Kill bars that might be still alive
    for col_idx, piv in enumerate(pivots_Im):
        if piv != -1:
            Im_barcode[col_idx, 1] = a[-1]
        # end if
    # end for
    if (not B.broken_basis) and (start_index == 0):
        for col_idx, piv in enumerate(pivots_Ker[:kernel_dim]):
            if piv != -1:
                Ker_barcode[col_idx, 1] = a[-1]
            # end if
        # end for
    # end if
    Im_coordinates = Im_coordinates[return_order_B][:, start_index:]
    Im_basis = barcode_basis(Im_barcode[start_index:], B_origin,
                             Im_coordinates, store_well_defined=True)

    # Return to normal order in A
    PreIm = T[start_index:][return_order_A][:, start_index:]
    PreIm = PreIm[:, Im_basis.well_defined]
    # Order Im and store the order in PreIm
    order = Im_basis.sort(send_order=True)
    PreIm = PreIm[:, order]
    if B.broken_basis or (start_index > 0):
        return Im_basis, PreIm
    Ker_coordinates = Ker_coordinates[return_order_A]
    # Create a barcode basis for the kernel and sort it.
    Ker_basis = barcode_basis(
        Ker_barcode[:kernel_dim], A_origin,  Ker_coordinates[:, :kernel_dim])
    Ker_basis.sort()
    return Im_basis, Ker_basis, PreIm
# end image kernel


def pivot_check_reduce(M, active_rows, col_idx, pivots_prev, p, T=None):
    new_pivot = index_pivot(M[active_rows, col_idx] % p)
    if new_pivot != -1:
        new_pivot = int(active_rows[new_pivot])
        if (new_pivot != -1) and (new_pivot in pivots_prev):
            prev_idx = list(pivots_prev).index(new_pivot)
            M[:, col_idx] = add_mod_c(
                inv_mod_p(M[new_pivot, prev_idx], p) * M[:, prev_idx],
                M[:, col_idx], p)
            if np.any(T):
                T[:, col_idx] = add_mod_c(
                    inv_mod_p(M[new_pivot, prev_idx], p) * T[:, prev_idx],
                    T[:, col_idx], p)
            # end if not None
            new_pivot = pivot_check_reduce(
                M, active_rows, col_idx, pivots_prev, p, T)
        # end if
    # end if
    return int(new_pivot)
# end def
