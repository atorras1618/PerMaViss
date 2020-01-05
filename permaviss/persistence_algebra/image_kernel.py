"""
    image_kernel.py

    This module implements a function which computes bases for the image and 
    kernel of morphisms between persistence modules. 
"""
import numpy as np

from .barcode_bases import *

from ..gauss_mod_p import gauss_mod_p
from ..gauss_mod_p.functions import multiply_mod_p 

###############################################################################
# Find pivot of array

def _pivot(l):
    """ 
    Given a 1D array of integers, return the index of the last nonzero entry.  
    Returns -1 if the list is zero.  
    INPUT: 
    -l: list on integers. 
    OUTPUT: 
    -index of last nonzero entry. 
    """ 
    l_bool = np.nonzero(l) 
    if len(l_bool[0]) > 0: 
        return l_bool[0][-1] 
     
    return -1 

assert _pivot(np.array([0,1,0,1,0]))==3
assert _pivot(np.array([0,0,0]))==-1


################################################################################
# Algorithm to find barcode generators for image and kernel of a persistence morphism.

def image_kernel(A, B, F, p, start_index=0, prev_basis=None):
    """
    This computes basis for the image and kernel of a persistence morphism.
        f: A --> B
    It can also compute relative barcode bases for the image and kernel. The optional argument
    start_index indicates the minimum index from which we want to compute barcodes
    relative to the previous generators. That is, given start_index, the function will return 
    image barcodes for <F[start_dim, ..., A.dim]>  mod <F[0,1,..., start_dim-1]>.
    Also, it returns ker(f) mod A[0, ..., start_index - 1] 
    The function first orders the barcode generators from start_index until A.dim.
    Additionally, it also orders the barcodes from B. Both these preprocessing steps are necessary
    in order for the algorithm to work properly. At the end the bases for the image and kernel are 
    returned in terms of the original ordering.  By 'ordered' we mean that
    the barcodes are sorted according to the standard order of barcodes.
    Also, this handles the case for when B is a broken barcode basis. 
    Notice that in such a case, only the barcode basis of the imate will be computed
    INPUT:
    -A: barcode basis of domain.  
    -B: barcode basis of range.  This can be a broken barcode basis.
    -F: matrix associated to the considered persistence morphism.
    -p: prime number of finite field.
    -start_index: (default=0) index from which we get a barcode basis for the image.
    -prev_basis:(default=None) if the start_index > 0, we need to also give a reference to a basis
            of barcodes from A[start_dim] until A[A.dim].
    OUTPUT:
    -Ker: absolute/relative basis of kernel of f.
    -Im: absolute/relative basis of image of f.
    -PreIm: absolute/relative preimage coordinates of f. That is, each column stores the
            sums that generate the corresponding Image barcode. 
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
        shifted_order = np.append(range(start_index), shifted_order, axis=0).astype(int)
        return_order_A = np.argsort(order)
        F = np.copy(F[:, shifted_order])
        A = barcode_basis(np.append(A.barcode[:start_index], A_rel.barcode, axis=0))

    # Get sorted radii where changes happen
    a = np.sort(np.unique(np.append(A.changes_list(), B.changes_list())))
    rel_A_dim = A.dim - start_index
    # Save space for Im barcode and coordinates
    Im_barcode = np.concatenate(([A.barcode[start_index:,0]], [np.zeros(rel_A_dim)]), axis=0).T
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
            Im_barcode[j,1] = rad
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
                    Ker_barcode[j,1] = rad        
                # end for
            # end if
            else:
                # Reduce K0 by column additions
                K0, Q0 = gauss_mod_p.gauss_col(K0, p)
                # Perform the same reductions on Ker, if Q0 is not empty
                if np.any(Q0):
                    Ker_coordinates[:, :kernel_dim] = multiply_mod_p(Ker_coordinates[:, :kernel_dim], Q0, p)
                # Compute the start_index of active barcodes
                start_index_active = len(active_indices) - len(active_rel_indices)
                # Look at K0, eliminating linear dependencies and killing barcodes on image 
                for j, c in enumerate(K0.T):
                    # Find pivot in terms of whole basis A, keep also the active pivot
                    active_pivot = _pivot(c)
                    piv = active_rel_indices[active_pivot]
                    # If the pivot is new and the column is nonzero
                    if (piv not in dead_images) and (active_pivot > -1):
                        # Add new coordinates to Image, set barcode endpoint, store preimage
                        A_rel_coord = A.trans_active_coord(c, rad, start=start_index)
                        image_A_rel =  multiply_mod_p(F[:,start_index:], 
                                                            np.transpose([A_rel_coord]), p)[:,0]
                        Im_rel_coordinates[:, piv] = image_A_rel
                        Im_coordinates[:, start_index + piv] = np.copy(image_A_rel)
                        Im_barcode[piv, 1] = rad
                        PreIm[:, piv] = A_rel_coord
                        # Set corresponding column in I0 to zero and add piv to dead images
                        I0[:,active_pivot + start_index_active] = np.zeros(np.size(I0, 0))
                        dead_images.append(piv)
                    # end if
                    # If a barcode is dying in the kernel, add death radius to Ker_barcode
                    if (active_pivot == -1) and (j not in dead_kernels):
                        dead_kernels.append(j)
                        Ker_barcode[j,1] = rad
                    # end if
                # end for
            # end else
        # end if 
        # Reduce I0, adding new generators to the kernel and adjusting Im_rel_coordinates
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
            # Now, we check for 0 columns in I0, iterating over active_indices >= start_index
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
                    Ker_coordinates[:, :kernel_dim] = multiply_mod_p(Ker_coordinates[:, :kernel_dim], Q0, p)
                # Look at K0, eliminating linear dependencies 
                for j, c in enumerate(K0.T):
                    if ( _pivot(c) == -1) and (j not in dead_kernels):
                        dead_kernels.append(j)
                        Ker_barcode[j,1] = rad
                    # end if
                # end for
            # end if
        # end if
        # Go to next radius in a

    # end for

    # Store the barcode endpoints for the image and kernel generators that are still alive.
    for j in range(rel_A_dim):
        if j not in dead_images:
            Im_barcode[j,1] = a[-1] 
        # end if
    # end for
    for j in range(kernel_dim):
        if j not in dead_kernels:
            Ker_barcode[j,1] = a[-1] 
        # end if
    # end for
    # Return to normal order in B
    Im_rel_coordinates = Im_rel_coordinates[return_order_B]
    # Return to normal order in A
    PreIm = PreIm[return_order_A]
    Ker_coordinates = Ker_coordinates[return_order_A]
    # Create barcode basis for image, and adjust PreIm according to well defined barcodes
    Im_basis = barcode_basis(Im_barcode, B_origin, Im_rel_coordinates, store_well_defined=True)
    PreIm = PreIm[:, Im_basis.well_defined]
    # Order Im and store the order in PreIm
    order = Im_basis.sort(send_order=True)
    PreIm = PreIm[:, order]
    # If the basis of B is broken, return Im_basis
    if B.broken_basis:
        return Im_basis

    # Create a barcode basis for the kernel and sort it. 
    if start_index > 0:
        Ker_basis = barcode_basis(Ker_barcode[:kernel_dim], prev_basis, Ker_coordinates[:, :kernel_dim])
    else: 
        Ker_basis = barcode_basis(Ker_barcode[:kernel_dim], A_origin, Ker_coordinates[:, :kernel_dim])

    Ker_basis.sort()

    return Im_basis, Ker_basis, PreIm


################################################################################
# Test 1
A = barcode_basis([[1,5],[1,4],[2,5]])
B = barcode_basis([[0,5],[0,3],[1,4]])
F = np.array([[0,0,1],[1,0,0],[1,1,1]])
Im = np.array([[0,0,1],[4,4,0],[4,0,0]])
Ker = np.array([[4,1,0]]).T
PreIm = np.array([[4,4,0],[0,1,4],[0,0,1]])
res_Im, res_Ker, res_PreIm = image_kernel(A,B,F,5)
assert np.array_equal(res_Im.coordinates, Im)
assert np.array_equal(res_Ker.coordinates, Ker)
assert np.array_equal(res_PreIm, PreIm)

# Test 2
A = barcode_basis([[1,8],[1,5],[2,5], [4,8]])
B = barcode_basis([[-1,3],[0,4],[0,3.5],[2,5],[2,4],[3,8]])
F = np.array([[4,1,1,0],[1,4,1,0],[1,1,4,0],[0,0,1,4],[0,0,4,1],[0,0,0,1]])
res_Im, res_Ker, res_PreIm = image_kernel(A,B,F,5)
Im = np.array([
 [4, 0, 1, 0],
 [1, 0, 1, 0],
 [1, 2, 4, 0],
 [0, 0, 1, 4],
 [0, 0, 4, 1],
 [0, 0, 0, 1]])
Ker = np.array([
 [1, 0],
 [1, 4],
 [0, 0],
 [0, 0]])
PreIm= np.array([[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
assert np.array_equal(res_Im.coordinates, Im)
assert np.array_equal(res_Ker.coordinates, Ker)
assert np.array_equal(res_PreIm, PreIm)

# Test 3
A = barcode_basis([[0,4],[0,2],[1,4],[1,3]])
B = barcode_basis([[0,3],[0,2],[0,2]])
F = np.array([[0,0,0,1],[4,0,1,3],[1,0,1,4]])
res_Im, res_Ker, PreIm = image_kernel(A,B,F,5)
Ker_basis = np.array([[0,2],[2,4], [2,4]])
Im_basis = np.array([[0,2],[1,3],[1,2]])
res_PreIm = np.array([[1,2,4],[0,0,0],[0,4,1],[0,1,0]])
assert np.array_equal(Im_basis, res_Im.barcode)
assert np.array_equal(Ker_basis, res_Ker.barcode)
assert np.array_equal(res_PreIm, PreIm)

# Test 4
A = barcode_basis([[1,4],[2,4],[3,5]])
B = barcode_basis([[0,4],[0,2],[1,4],[1,3]])
F = np.array([[0,1,1],[1,0,0],[0,4,1],[0,0,0]])
res_Im, res_Ker, PreIm = image_kernel(A,B,F,5)
Ker_basis = np.array([[2,4],[4,5]])
Im_basis = np.array([[1,2],[2,4],[3,4]])
assert np.array_equal(Im_basis, res_Im.barcode)
assert np.array_equal(Ker_basis, res_Ker.barcode)

# Test 5 (same as test 3, but with relative bases)
A = barcode_basis([[0,4],[0,2],[1,4],[1,3]])
A_rel = barcode_basis([[0,2],[1,4],[1,3]])
B = barcode_basis([[0,3],[0,2],[0,2]])
F = np.array([[0,0,0,1],[4,0,1,3],[1,0,1,4]])
res_Im, res_Ker, PreIm = image_kernel(A,B,F,5, start_index=1, prev_basis=A_rel)
Ker_basis = np.array([[0,2], [2,4]])
Im_basis = np.array([[1,3],[1,2]])
res_PreIm = np.array([[0,0],[4,1],[1,0]])
assert np.array_equal(Im_basis, res_Im.barcode)
assert np.array_equal(Ker_basis, res_Ker.barcode)
assert np.array_equal(res_PreIm, PreIm)

# Test for unordered bases
start_idx = 4
A = np.array(
[[ 0.,   2. ],
 [ 0.,   1. ],
 [ 0.,   0.4],
 [ 0.,   0.4],
 [ 0.,   2. ],
 [ 0.,   2. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   0.4],
 [ 0.,   0.4],
 [ 0.,   0.4],
 [ 0.,   0.4]])
A_dim = np.size(A,0)
shuffleA = np.append(range(4, A_dim, 2), range(5, A_dim,2), axis=0)
shifted_shuffleA = np.append(range(4), shuffleA, axis=0)
A_shuffle = barcode_basis(A[shifted_shuffleA])
A_rel = barcode_basis(A[start_idx:])
A = barcode_basis(A)
B = np.array(
[[ 0.,   2. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   0.4],
 [ 0.,   0.4],
 [ 0.,   2. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   0.4],
 [ 0.,   0.4]])
B_dim = np.size(B, 0)
shuffleB = np.append(range(0, B_dim, 2), range(1, B_dim, 2), axis=0) 
B_shuffle = barcode_basis(B[shuffleB])
B = barcode_basis(B)
F = np.array(
[[ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 4.,  4.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 4.,  4.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
 [ 4.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
 [ 0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
F_shuffle = F[shuffleB]
F_shuffle = F_shuffle[:, shifted_shuffleA]
Im = np.array(
[[ 0.,   2. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   0.4],
 [ 0.,   0.4]])
res_Im_sh, res_Ker_sh, PreIm_sh = image_kernel(A_shuffle,B_shuffle,F_shuffle,5, start_index=start_idx, prev_basis=A_rel)
res_Im, res_Ker, PreIm = image_kernel(A,B,F,5, start_index=start_idx, prev_basis=A_rel)
shuffleA = shuffleA - start_idx
assert np.array_equal(res_Im_sh.barcode, Im)
assert np.array_equal(res_Im.barcode, Im)
assert np.array_equal(res_Ker_sh.coordinates, res_Ker.coordinates[shuffleA])

# Extra test
A = barcode_basis(np.array(
[[ 0.20065389,  0.21080812],
 [ 0.25135865,  0.29062369],
 [ 0.25135865,  0.27869416],
 [ 0.25403887,  0.3       ],
 [ 0.28256212,  0.3       ],
 [ 0.29679942,  0.3       ],
 [ 0.24715565,  0.29062369]]))
B = barcode_basis(np.array(
[[ 0.20065389,  0.21080812],
 [ 0.25135865,  0.29062369],
 [ 0.25135865,  0.27869416],
 [ 0.25403887,  0.3       ],
 [ 0.28256212,  0.3       ],
 [ 0.29679942,  0.3       ],
 [ 0.24715565,  0.25135865]]))
F = np.array(
[[ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  1.,  0.,  0.,  0.,  0.,  3.],
 [ 0.,  0.,  1.,  0.,  0.,  0.,  3.],
 [ 0.,  0.,  0.,  1.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  1.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  1.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  1.]])
res_Im, res_Ker, PreIm = image_kernel(A,B,F,5)
Im_bar = np.array(
[[ 0.20065389,  0.21080812] ,
 [ 0.24715565,  0.29062369] ,
 [ 0.25135865,  0.27869416] ,
 [ 0.25403887,  0.3       ] ,
 [ 0.28256212,  0.3       ] ,
 [ 0.29679942,  0.3       ]])
assert np.array_equal(Im_bar, res_Im.barcode)

# Test for quotients
A = barcode_basis(np.array(
[[2.5,5],[0,5],[1,4],[2,3]]))
B = barcode_basis(np.array(
[[0,5],[1,4],[2,3]]))
F = np.array(
[[1,1,0,0],
 [1,0,1,0],
 [1,0,0,1]])
res_Im, res_Ker, PreIm = image_kernel(A,B,F,5, start_index=1)
Im_bar = np.array(
[[ 0.,   4. ],
 [ 1.,   3. ],
 [ 2.,   2.5]])
Im_coord = np.array(
[[ 1.,  1.,  1.],
 [ 0.,  1.,  1.],
 [ 0.,  0.,  1.]])
assert np.array_equal(Im_bar, res_Im.barcode)
assert np.array_equal(Im_coord, res_Im.coordinates)

# Test for image and kernel
A = barcode_basis(
[[ 0.32001485,  0.33148304],
 [ 0.32104403,  0.35      ],
 [ 0.34418382,  0.35      ],
 [ 0.34863464,  0.35      ],
 [ 0.29818462,  0.35      ],
 [ 0.31456264,  0.35      ],
 [ 0.32001485,  0.33148304],
 [ 0.32001485,  0.33148304],
 [ 0.32104403,  0.35      ],
 [ 0.32104403,  0.35      ],
 [ 0.3239026 ,  0.35      ],
 [ 0.32849817,  0.35      ],
 [ 0.33184894,  0.35      ],
 [ 0.33578525,  0.35      ],
 [ 0.33627334,  0.35      ],
 [ 0.33937523,  0.35      ],
 [ 0.34020755,  0.35      ],
 [ 0.34085129,  0.35      ],
 [ 0.34132324,  0.35      ],
 [ 0.34418382,  0.3445235 ],
 [ 0.34659024,  0.35      ],
 [ 0.34863464,  0.35      ],
 [ 0.34863464,  0.35      ],
 [ 0.34863464,  0.35      ]])
B = barcode_basis(
[[ 0.3239026 ,  0.35      ],
 [ 0.34863464,  0.35      ],
 [ 0.29818462,  0.35      ],
 [ 0.32001485,  0.33148304],
 [ 0.34020755,  0.35      ],
 [ 0.34132324,  0.35      ],
 [ 0.33184894,  0.35      ],
 [ 0.34085129,  0.35      ],
 [ 0.34418382,  0.3445235 ],
 [ 0.34863464,  0.35      ],
 [ 0.32849817,  0.35      ],
 [ 0.33578525,  0.35      ],
 [ 0.34659024,  0.35      ],
 [ 0.31456264,  0.35      ],
 [ 0.32001485,  0.33148304],
 [ 0.32104403,  0.35      ],
 [ 0.33627334,  0.35      ],
 [ 0.34863464,  0.35      ],
 [ 0.32104403,  0.35      ],
 [ 0.33937523,  0.35      ]])
Im = np.array(
[[ 0.,  0.,  4.,  1.],
 [ 0.,  0.,  0.,  4.],
 [ 0.,  0.,  0.,  0.],
 [ 4.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  4.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  1.,  0.],
 [ 0.,  0.,  0.,  1.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 1.,  0.,  0.,  0.],
 [ 0.,  4.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  1.,  0.,  0.],
 [ 0.,  0.,  0.,  0.]])
Ker = np.array(
[[ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
 [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
 [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
 [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
F = np.append(Im, Ker, axis=1)
res_Im, res_Ker, PreIm = image_kernel(A,B,F,5, start_index=4)
# Test for broken barcodes
# broken_differential= np.array([
# [0,0,1,0],
# [4,1,0,0],
# [0,0,1,0],
# [0,4,0,0],
# [0,1,0,0],
# [0,0,0,1],
# [0,0,0,1]])
# diagonal_barcode=np.array([
# [0,5],[0,5],[0,5],[0,5],[0,3], [1,5],])
# direct_sum_barcode=np.array([
# [0,5],[1,5],[1,5],[1,4],[2,3],[3,5],[3,4]])

