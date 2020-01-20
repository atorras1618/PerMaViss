"""
    image_kernel.py

    This module implements a function which computes bases for the image and
    kernel of morphisms between persistence modules.
"""
import numpy as np

from .barcode_bases import barcode_basis

from ..gauss_mod_p import gauss_mod_p
from ..gauss_mod_p.functions import multiply_mod_p

###############################################################################
# Find pivot of array


def _pivot(l):
    """
    Parameters
    ----------
    l : :obj:`list(int)`
        List of integers to compute pivot from.

    Returns
    -------
    int
        Index of last nonzero entry on `l`. Returns -1 if the list is zero.
    """
    l_bool = np.nonzero(l)
    if len(l_bool[0]) > 0:
        return l_bool[0][-1]

    return -1


assert _pivot(np.array([0, 1, 0, 1, 0])) == 3
assert _pivot(np.array([0, 0, 0])) == -1


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
    Notice that in such a case, only the barcode basis of the image will be
    computed

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
    # If the basis B is not ordered, we order it
    B_origin = B
    return_order_B = range(B.dim)
    if not B.sorted:
        B = barcode_basis(B.barcode)
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

    # Get sorted radii where changes happen
    a = np.sort(np.unique(np.append(A.changes_list(), B.changes_list())))
    rel_A_dim = A.dim - start_index
    # Save space for Im barcode and coordinates
    Im_barcode = np.concatenate(
        ([A.barcode[start_index:, 0]], [np.zeros(rel_A_dim)]),
        axis=0).T
    Im_rel_coordinates = np.copy(F[:, start_index:])
    # Save space for Ker barcode and coordinates
    Ker_barcode = np.zeros((rel_A_dim, 2))
    Ker_coordinates = np.zeros((rel_A_dim, rel_A_dim))
    kernel_dim = 0
    # Save space for preimage
    PreIm = np.identity(rel_A_dim)
    # Tracking variables for deaths of image and kernel barcodes
    # These store indices from 0 to rel_A_dim
    dead_images = []
    dead_kernels = []
    # Initialize a copy of F which will be changing as we iterate over a
    Im_coordinates = np.copy(F)
    # Go through all radius where a change happens
    for i, rad in enumerate(a):
        if B.broken_basis:
            Im_coordinates = B.update_broken(Im_coordinates, rad)

        # Update I0 from submatrix of Im
        I0 = Im_coordinates[B.active(rad)][:, A.active(rad)]
        # Get the active indices in A and the relative active indices in A
        active_indices = A.active(rad)
        active_rel_indices = A.active(rad, start=start_index)
        # Take care of special case when I0 = []
        if len(I0) == 0:
            # Find active domain elements that have not died yet
            active_not_dead = np.setdiff1d(active_rel_indices, dead_images)
            for j in active_not_dead:
                Im_barcode[j, 1] = rad
                Ker_barcode[kernel_dim, 0] = rad
                Ker_coordinates[j, kernel_dim] = 1
                kernel_dim += 1
                dead_images.append(j)
            # end for
        # end if

        # Find dying domain generators whose image has not died yet
        dying_generators = A.death(rad, start=start_index)
        alive_dying = np.setdiff1d(dying_generators, dead_images)
        # Set the death radius for the image barcodes
        for j in alive_dying:
            Im_barcode[j, 1] = rad
            dead_images.append(j)
        # end for

        if kernel_dim > 0 and not B.broken_basis:
            # Update K0 from submatrix of Ker
            K0 = Ker_coordinates[:, :kernel_dim]
            K0 = K0[active_rel_indices]
            if len(K0) == 0:
                dying_kernels = np.setdiff1d(range(kernel_dim), dead_kernels)
                for j in dying_kernels:
                    dead_kernels.append(j)
                    Ker_barcode[j, 1] = rad
                # end for
            # end if
            else:
                # Reduce K0 by column additions
                K0, Q0 = gauss_mod_p.gauss_col(K0, p)
                # Perform the same reductions on Ker, if Q0 is not empty
                if np.any(Q0):
                    Ker_coordinates[:, :kernel_dim] = multiply_mod_p(
                        Ker_coordinates[:, :kernel_dim], Q0, p)
                # Compute the start_index of active barcodes
                start_index_active = len(active_indices) - len(
                    active_rel_indices)
                # Look at K0, eliminating linear dependencies and killing
                # barcodes on image.
                for j, c in enumerate(K0.T):
                    # Find pivot in terms of whole basis A, keep also the
                    # active pivot.
                    active_pivot = _pivot(c)
                    piv = active_rel_indices[active_pivot]
                    # If the pivot is new and the column is nonzero
                    if (piv not in dead_images) and (active_pivot > -1):
                        # Add new coordinates to Image, set barcode endpoint,
                        # store preimage.
                        A_rel_coord = A.trans_active_coord(
                            c, rad, start=start_index)
                        image_A_rel = multiply_mod_p(
                            F[:, start_index:], np.transpose([A_rel_coord]),
                            p)[:, 0]
                        Im_rel_coordinates[:, piv] = image_A_rel
                        Im_coordinates[
                            :, start_index + piv] = np.copy(image_A_rel)
                        Im_barcode[piv, 1] = rad
                        PreIm[:, piv] = A_rel_coord
                        # Set corresponding column in I0 to zero and add piv
                        # to dead images
                        I0[
                            :, active_pivot + start_index_active
                            ] = np.zeros(np.size(I0, 0))
                        dead_images.append(piv)
                    # end if
                    # If a barcode is dying in the kernel, add death radius
                    # to Ker_barcode
                    if (active_pivot == -1) and (j not in dead_kernels):
                        dead_kernels.append(j)
                        Ker_barcode[j, 1] = rad
                    # end if
                # end for
            # end else
        # end if
        # Reduce I0, adding new generators to the kernel and adjusting
        # Im_rel_coordinates.
        if I0.size > 0:
            I0, T0 = gauss_mod_p.gauss_col(I0, p)
            # Adapt T0 to the whole basis A
            Q0 = np.zeros((A.dim, np.size(T0, 1)))
            Q0[active_indices] = T0
            Q = np.identity(A.dim)
            Q[:, active_indices] = Q0
            # Perform same additions in Im_coordinates
            Im_coordinates = multiply_mod_p(Im_coordinates, Q, p)
            # Then, adapt Q to the relative basis of A
            Q = Q[start_index:, start_index:]
            # Then perform the additions in PreIm and Im
            PreIm = multiply_mod_p(PreIm, Q, p)
            Im_rel_coordinates = multiply_mod_p(Im_rel_coordinates, Q, p)
            # Now, we check for 0 columns in I0, iterating over
            # active_indices >= start_index.
            rel_I0_T = I0.T[active_indices >= start_index]
            start_kernel_dim = kernel_dim
            for j, c in enumerate(rel_I0_T):
                piv = active_rel_indices[j]
                if (not np.any(c)) and (piv not in dead_images):
                    Ker_coordinates[:, kernel_dim] = PreIm[:, piv]
                    Ker_barcode[kernel_dim, 0] = rad
                    kernel_dim += 1
                    Im_barcode[piv, 1] = rad
                    dead_images.append(piv)
                # end if
            # end for
            # Reduce new additions to kernel
            if (kernel_dim > start_kernel_dim) and not B.broken_basis:
                # Update K0 from submatrix of Ker
                K0 = Ker_coordinates[:, :kernel_dim]
                K0 = K0[active_rel_indices]
                # Reduce K0 by column additions
                K0, Q0 = gauss_mod_p.gauss_col(K0, p)
                # Perform the same reductions on Ker, if Q0 is not empty
                if np.any(Q0):
                    Ker_coordinates[
                        :, :kernel_dim
                        ] = multiply_mod_p(Ker_coordinates[:, :kernel_dim],
                                           Q0, p)
                # Look at K0, eliminating linear dependencies
                for j, c in enumerate(K0.T):
                    if (_pivot(c) == -1) and (j not in dead_kernels):
                        dead_kernels.append(j)
                        Ker_barcode[j, 1] = rad
                    # end if
                # end for
            # end if
        # end if
        # Go to next radius in a

    # end for

    # Store the barcode endpoints for the image and kernel generators that are
    # still alive.
    for j in range(rel_A_dim):
        if j not in dead_images:
            Im_barcode[j, 1] = a[-1]
        # end if
    # end for
    for j in range(kernel_dim):
        if j not in dead_kernels:
            Ker_barcode[j, 1] = a[-1]
        # end if
    # end for
    # Return to normal order in B
    Im_rel_coordinates = Im_rel_coordinates[return_order_B]
    # Return to normal order in A
    PreIm = PreIm[return_order_A]
    Ker_coordinates = Ker_coordinates[return_order_A]
    # Create barcode basis for image, and adjust PreIm according to well
    # defined barcodes.
    Im_basis = barcode_basis(Im_barcode, B_origin, Im_rel_coordinates,
                             store_well_defined=True)
    PreIm = PreIm[:, Im_basis.well_defined]
    # Order Im and store the order in PreIm
    order = Im_basis.sort(send_order=True)
    PreIm = PreIm[:, order]
    # If the basis of B is broken, return Im_basis
    if B.broken_basis:
        return Im_basis

    # Create a barcode basis for the kernel and sort it.
    if start_index > 0:
        Ker_basis = barcode_basis(Ker_barcode[:kernel_dim], prev_basis,
                                  Ker_coordinates[:, :kernel_dim])
    else:
        Ker_basis = barcode_basis(Ker_barcode[:kernel_dim], A_origin,
                                  Ker_coordinates[:, :kernel_dim])

    Ker_basis.sort()

    return Im_basis, Ker_basis, PreIm
