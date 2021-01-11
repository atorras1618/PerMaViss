"""
This module implements the Mayer-Vietoris spectral sequence management.
"""

import numpy as np
import scipy.spatial.distance as dist

from multiprocessing import Pool
from functools import partial

from ..covers import cubical_cover

from ..simplicial_complexes.differentials import complex_differentials
from ..simplicial_complexes.vietoris_rips import vietoris_rips

from ..persistence_algebra.PH_classic import persistent_homology
from ..persistence_algebra.image_kernel import image_kernel
from ..persistence_algebra.module_persistence_homology import (
     module_persistence_homology)
from ..persistence_algebra.barcode_bases import barcode_basis

from .spectral_sequence_class import spectral_sequence


def create_MV_ss(point_cloud, max_r, max_dim, max_div, overlap, p):
    """ This function creates a Mayer Vietoris spectral sequence with the
    given parameters. The procedure has four main steps:

    1) Obtain a cover and a nerve associated to it.

    2) Compute the persistent homology on each cover, intersections, and so on.

    3) Compute spectral sequence pages until they collapse.

    4) Solve the extension problem

    Parameters
    ----------
    point_cloud : Numpy Array
        Coordinates for given points. Each row corresponds to a point.
    max_r : float
        Maximum radius of persistence.
    max_dim : int
        Maximum dimension of simplexes in Vietoris-Rips complex.
    max_div : int
        Number of division hypercubes on the dimension with maximum length on
        point cloud.
    overlap : float
        Overlap between adjacent covers.

    Returns
    -------
    MV_ss : :class:`spectral_sequence` object containing all the information.

    Example
    -------
        >>> from permaviss.sample_point_clouds.examples import random_cube,
        ... take_sample
        >>> X = random_cube(1000,3)
        >>> point_cloud = take_sample(X,130)
        >>> max_r = 0.36
        >>> max_dim = 3
        >>> p = 3
        >>> max_div = 2
        >>> overlap = max_r*1.01
        >>> MV_ss = create_MV_ss(point_cloud, max_r, max_dim, max_div, overlap,
        ... p)
        PAGE: 1
        [[ 25   7   4   1   0   0   0   0   0]
         [160 118 128 144 112  56  16   2   0]
         [310 380 436 445 336 168  48   6   0]]
        PAGE: 2
        [[ 21   0   0   0   0   0   0   0   0]
         [ 98   2   0   0   0   0   0   0   0]
         [131   5   1   0   0   0   0   0   0]]
        PAGE: 3
        [[ 21   0   0   0   0   0   0   0   0]
         [ 97   2   0   0   0   0   0   0   0]
         [131   5   0   0   0   0   0   0   0]]
        PAGE: 4
        [[ 21   0   0   0   0   0   0   0   0]
         [ 97   2   0   0   0   0   0   0   0]
         [131   5   0   0   0   0   0   0   0]]
        >>> print(MV_ss.persistent_homology[0].dim)
        131
        >>> print(MV_ss.persistent_homology[1].dim)
        97
        >>> print(MV_ss.persistent_homology[2].dim)
        21

    """
    # Divide point cloud using hypercube cover
    # Use points_IN to build Nerve in future more general version
    nerve_point_cloud, points_IN, nerve = cubical_cover.generate_cover(
        max_div, overlap, point_cloud)

    nerve_dim = len(nerve)
    # Count maximum points on hypercube cover
    max_points = 0
    for hyp_pc in nerve_point_cloud[0]:
        if max_points < np.size(hyp_pc, 0):
            max_points = np.size(hyp_pc, 0)

    # initialize the spectral sequence, compute the maximum number of pages
    no_pages = min(max_dim + 2, nerve_dim)
    MV_ss = spectral_sequence(nerve, nerve_point_cloud, points_IN, max_dim,
                              max_r, no_pages, p)

    # 0 PAGE ###################################################################

    for n_dim in range(0, nerve_dim):
        if n_dim > 0:
            n_spx_number = np.size(nerve[n_dim], 0)
        else:
            n_spx_number = nerve[0]

        partial_persistent_homology = partial(local_persistent_homology,
                                              nerve_point_cloud,
                                              max_r, max_dim, p, n_dim)

        workers_pool = Pool()
        output = workers_pool.map(partial_persistent_homology,
                                  range(n_spx_number))
        workers_pool.close()

        MV_ss.add_output_first(output, n_dim)

    # test for cech differential
    # MV_ss.test_local_cech_matrix()

    # 1 PAGE ###################################################################

    # Print page
    print("PAGE: 1")
    flip = np.array(range(MV_ss.no_rows))
    flip = -flip
    print(MV_ss.page_dim_matrix[1][np.argsort(flip)])
    # Loop through all rows of spectral sequence
    for deg in range(max_dim):
        base = [barcode_basis(MV_ss.first_page_barcodes[0][deg])]
        differentials = [0]
        for n_dim in range(1, nerve_dim):
            base.append(barcode_basis(MV_ss.first_page_barcodes[n_dim][deg]))
            differentials.append((MV_ss.first_differential(n_dim, deg)).T)
        # end for
        Hom, Im, PreIm = module_persistence_homology(differentials, base, p)
        MV_ss.add_output_higher(Hom, Im, PreIm, 0, deg, 1)
        # compute total complex representatives for second page classes
        for n_dim in range(nerve_dim):
            if MV_ss.Hom[1][n_dim][deg].dim > 0:
                MV_ss.compute_two_page_representatives(n_dim, deg)
            # end if
        # end for
    # end for


    # PAGES => 2 ###############################################################

    for current_page in range(2, no_pages):
        # Print page
        print("PAGE: {}".format(current_page))
        flip = np.array(range(MV_ss.no_rows))
        flip = -flip
        print(MV_ss.page_dim_matrix[current_page][np.argsort(flip)])

        # Loop through sequences of possibly nontrivial differentials
        # current_page columns

        print("higher reps, current_page:{}".format(current_page))
        for start_n_dim in range(current_page):
            for start_deg in range(max_dim):
                deg = start_deg
                n_dim = start_n_dim
                differentials = [0]
                base = [MV_ss.Hom[current_page - 1][n_dim][deg]]
                # While (n_dim, deg) lies within the spectral sequence
                # boundaries
                while deg >= 0 and n_dim < nerve_dim:
                    # Advance to differential domain and compute differential
                    deg += 1 - current_page
                    n_dim += current_page
                    if deg >= 0 and n_dim < nerve_dim:
                        differentials.append(MV_ss.high_differential(n_dim,
                                deg, current_page).T)
                        base.append(MV_ss.Hom[current_page - 1][n_dim][deg])
                    # end if
                # end while
                Hom, Im, PreIm = module_persistence_homology(differentials,
                                                             base, p)
                MV_ss.add_output_higher(Hom, Im, PreIm, start_n_dim, start_deg,
                                        current_page)

                # adjust reps of im and create reps for next hom
                deg = start_deg
                n_dim = start_n_dim
                while deg >= 0 and n_dim < nerve_dim:
                    # compute total complex reps for next page classes
                    print("hr deg:{}, n_dim:{}".format(deg, n_dim))
                    MV_ss.compute_higher_representatives(
                        n_dim, deg, current_page)
                    deg += 1 - current_page
                    n_dim += current_page
                    # end if
                # end while
            # end for
        # end for
    # end for



    # EXTENSION PROBLEM ########################################################
    # store 0 dim persistent homology
    MV_ss.persistent_homology.append(MV_ss.Hom[MV_ss.no_pages - 1][0][0])
    # Go through each diagonal
    for deg in range(1, MV_ss.no_rows):
        # compute dimension of diagonal deg
        dim_PH = [0]
        # Cumulative dimensions for persistent homology along diagonals
        start_deg = deg
        for start_n_dim in range(min(deg+1, MV_ss.no_columns)):
            dim_PH.append(MV_ss.page_dim_matrix[no_pages][
                start_deg, start_n_dim] + dim_PH[-1])
            start_deg -= 1
        # end for
        if dim_PH[-1] > 0:
            # Save space for extension matrix
            ext_mat = np.zeros((dim_PH[-1], dim_PH[-1]))
            # compute extension matrix
            start_deg = deg-1
            for start_n_dim in range(1, min(deg+1, MV_ss.no_columns)):
                if MV_ss.page_dim_matrix[no_pages-1][
                        start_deg, start_n_dim] > 0:
                    MV_ss.extension(start_n_dim, start_deg)
                    extensions = MV_ss.extensions[start_n_dim][start_deg]
                    column_range = ext_mat[:, dim_PH[start_n_dim]: dim_PH[
                        start_n_dim+1]]
                    for i, blk in enumerate(extensions):
                        column_range[dim_PH[start_n_dim - i]: dim_PH[
                            start_n_dim + 1 - i]] = np.copy(extensions[i])
                    # end for
                # end if
                start_deg -= 1
            # end for
            #
            # add up barcodes in diagonal
            barcode = []
            start_deg = deg
            for start_n_dim in range(min(deg+1, MV_ss.no_columns)):
                if len(barcode) == 0 and \
                        MV_ss.Hom[no_pages-1][start_n_dim][start_deg].dim > 0:
                    barcode = MV_ss.Hom[no_pages-1][start_n_dim][
                        start_deg].barcode
                else:
                    if MV_ss.Hom[no_pages-1][start_n_dim][start_deg].dim > 0:
                        barcode = np.append(barcode, MV_ss.Hom[no_pages-1][
                            start_n_dim][start_deg].barcode, axis=0)
                # end else
                start_deg -= 1
            # end for
            # Create a new barcode basis, based on the death radius of the
            # extension matrix column
            diagonal_basis = barcode_basis(np.copy(barcode))
            # Create new basis, which is the direct sum matrix. This is a
            # broken barcode basis with broken differentials given by the
            # extension matrix.
            direct_sum_basis = barcode_basis(barcode, broken_basis=True,
                                             broken_differentials=ext_mat)
            # Create an identity matrix as a morphism from diagonal_basis
            # to direct_sum_basis.
            diag_diff = np.identity(diagonal_basis.dim)
            # Order the barcodes for domain and range
            MV_ss.order_diagonal_basis.append(diagonal_basis.sort(
                                              send_order=True))
            order = direct_sum_basis.sort(send_order=True)
            # Reorder associated matrix accordingly
            diag_diff = diag_diff[:, MV_ss.order_diagonal_basis[-1]]
            diag_diff = diag_diff[order]
            # set all barcode endpoints to max_r in diagonal_basis
            diagonal_basis.barcode[:, 1] = max_r * np.ones(diagonal_basis.dim)
            PH = image_kernel(diagonal_basis, direct_sum_basis,
                              diag_diff, MV_ss.p)
            MV_ss.persistent_homology.append(PH)
        # end if
    # end for
    return MV_ss

###############################################################################
# Local persistent homology
# Method to be parallelized


def local_persistent_homology(nerve_point_cloud, max_r, max_dim, p,  n_dim,
                              spx_idx):
    """ This function computes the Vietoris Rips complex and persistent
    homology of a covering region.

    It is meant to be run in parallel.

    Parameters
    ----------
    nerve_point_cloud : :obj:`list(list(Numpy Array))`
        Local point cloud coordinates indexed by nerve. The first entry
        contains a list of the point cloud coordinates for each covering
        region. The second entry contains a list of the point cloud coordinates
        for each double intersection of covering regions. And so on.
    points_IN : :obj:`list(list(Numpy Array))`
        Local Identification Numbers (IN) indexed by nerve. That is, this is
        the same as `nerve_point_cloud`, but containing IN instead of
        coordinates for each point.
    max_r : `float`
        Maximum radius for computing persistent homology.
    max_dim : int
        Maximum dimension for complexes
    n_dim : int
        Current dimension in Nerve of cover
    spx_idx : int
        Index of n_dim simplex of the covering nerve.

    Returns
    -------
    local_complex : :obj:`list(Numpy Array)`
        See :mod:`permaviss.simplicial_complexes.vietoris_rips`
    local_differentials : :obj:`list(Numpy Array)`
        See :mod:`permaviss.simplicial_complexes.differentials`
    Hom, Im, PreIm : :obj:`list(barcode_basis)`, :obj:`list(barcode_basis)`,
                     :obj:`list(Numpy Array)`
        See
        :meth:`permaviss.persistence_algebra.PH_classic.persistent_homology`

    """
    local_point_cloud = nerve_point_cloud[n_dim][spx_idx]
    # Compute local Vietoris Rips complex and differentials
    if len(local_point_cloud) == 0:
        local_Dist = []
    else:
        local_Dist = dist.squareform(dist.pdist(local_point_cloud))

    local_complex, local_R = vietoris_rips(local_Dist, max_r, max_dim)
    local_differentials = complex_differentials(local_complex, p)

    # Persistent Homology
    Hom, Im, PreIm = persistent_homology(local_differentials, local_R,
                                         max_r, p)

    # check that Hom are indeed cycles
    for idx, hom in enumerate(Hom):
        if hom.dim > 0 and idx > 0:
            trivial_image = np.matmul(local_differentials[idx], hom.coordinates)
            if np.any(trivial_image % p):
                print(trivial_image % p)
                raise(RuntimeError)

    return local_complex, local_differentials, Hom, Im, PreIm
