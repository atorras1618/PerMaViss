"""
This should be called PerMaViss

Created on Wed Jul 25 10:35:48 2018

@author: C1736188

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
from ..persistence_algebra.module_persistence_homology import module_persistence_homology
from ..persistence_algebra.persistence_algebra import barcode_basis

from ..gauss_mod_p.gauss_mod_p import gauss_col

from .spectral_sequence_class import spectral_sequence


def create_MV_ss(point_cloud, max_r, max_dim, max_div, overlap, p):
    """
    This function creates a mayer Vietoris spectral sequence with the given parameters.
    The procedure has four main steps:
        1) Obtain a cover and a nerve associated to it. 
        2) Compute the persistent homology on each cover, intersections, and so on. 
        3) Compute spectral sequence pages until they collapse. 
        4) Solve the extension problem
    INPUT:
        - point_cloud: list of lists containing points.
        - max_r: maximum radius of persistence.
        - max_dim: maximum dimension of simplexes in Vietoris-Rips complex.
        - max_div: number of division hypercubes on the dimension with
            maximum length on point cloud.
        - overlap: how much different regions should overlap each other.
    OUTPUT:
        - MV_ss: spectral_sequence object containing all the information. 
    """
    # Divide point cloud using hypercube cover
    # Use points_IN to build Nerve in future more general version
    nerve_point_cloud, points_IN, nerve = cubical_cover.generate_cover(
            max_div, overlap, point_cloud)

    nerve_dim = len(nerve)
    # Count maximum points on hypercube cover
    max_points = 0
    for hyp_pc in nerve_point_cloud[0]:
        if max_points < np.size(hyp_pc,0):
            max_points = np.size(hyp_pc,0)
    
    # initialize the spectral sequence, compute the maximum number of pages
    no_pages = min(max_dim + 2, nerve_dim)
    MV_ss = spectral_sequence(nerve, nerve_point_cloud, points_IN, max_dim, max_r, no_pages, p)
    
    # 0 PAGE

    for n_dim in range(0,nerve_dim):
        if n_dim > 0:
            n_spx_number = np.size(nerve[n_dim],0)
        else:
            n_spx_number = nerve[0]

        partial_persistent_homology = partial(local_persistent_homology,
                                              nerve_point_cloud,
                                              max_r, max_dim, p, n_dim)
      
 
        workers_pool = Pool()
        output = workers_pool.map(partial_persistent_homology, range(n_spx_number))
        workers_pool.close()
   
        MV_ss.add_output_first(output, n_dim)

    # # test for double complex well defined
    # # check for Cech differential to be an actual differential
    # for deg in range(max_dim):
    #     for start_n in range(2, nerve_dim):
    #         for nerve_spx in range(len(nerve[start_n])): 
    #             
    #     # end for
    # # end for
    # MV_ss.double_complex_check()
        

    # PAGES => 1 
    for current_page in range(1, no_pages):
        # Print page
        print("PAGE: {}".format(current_page))
        flip = np.array(range(MV_ss.no_rows))
        flip = -flip
        print(MV_ss.page_dim_matrix[current_page][np.argsort(flip)])
        # Loop through sequences of differentials ending in the first current_page columns
        for start_n_dim in range(current_page):
            for start_deg in range(max_dim):
                base = []
                differentials = [MV_ss.page_dim_matrix[current_page, start_deg, start_n_dim]]
                deg = start_deg
                n_dim = start_n_dim
                # While (n_dim, deg) lies within the spectral sequence boundaries
                while deg >= 0 and n_dim < nerve_dim:
                    # Generate barcode bases for sending to module_persistence_homology 
                    if current_page == 1:
                        if MV_ss.page_dim_matrix[1][deg][n_dim] == 0:
                            barcode = []
                        else:
                            barcode = np.zeros((MV_ss.page_dim_matrix[1][deg][n_dim], 2))
                            prev_cycle_dim = 0
                            for nerv_spx_index, cycle_dim in enumerate(MV_ss.cycle_dimensions[n_dim][deg]):
                                if cycle_dim - prev_cycle_dim > 0:
                                    barcode[prev_cycle_dim: cycle_dim] = MV_ss.Hom[0][
                                                                n_dim][nerv_spx_index][deg].barcode
                                # end if
                                prev_cycle_dim = cycle_dim
                            # end for 
                        # end else
                        base.append(barcode_basis(barcode))
                    else:
                        if type(MV_ss.Hom[current_page -1][n_dim][deg])==list:
                            MV_ss.Hom[current_page-1][n_dim][deg] = barcode_basis([])
                        # end if 
                        base.append(MV_ss.Hom[current_page -1][n_dim][deg])
                    # end else
                    # Advance to differential domain and compute differential
                    deg += 1 - current_page
                    n_dim += current_page
                    if deg >= 0 and n_dim < nerve_dim:
                        differentials.append(MV_ss.zig_zag(n_dim, deg, current_page).T)
                    # end if
                # end while
                Hom, Im, PreIm = module_persistence_homology(differentials, base, p)
                MV_ss.add_higher_output(Hom, Im, PreIm, start_n_dim, start_deg, current_page) 
            # end for
        # end for
    # end for
    
    # compute representatives for cycles
    for n_dim in range(MV_ss.no_columns):
        for deg in range(MV_ss.no_rows):
            if MV_ss.page_dim_matrix[no_pages-1, deg, n_dim] > 0:
                _ = MV_ss.zig_zag(n_dim, deg, n_dim + deg + 2, store_reps=True) 
                if(MV_ss.page_dim_matrix[no_pages-1, deg, n_dim] !=len(MV_ss.tot_complex_reps[n_dim][deg][0])):
                    raise ValueError
            # end if
        # end for
    # end for


    # EXTENSION PROBLEM
    MV_ss.persistent_homology.append(MV_ss.Hom[no_pages-1][0][0])
    # Go through each diagonal
    for deg in range(1, MV_ss.no_rows):
        # compute dimension of diagonal deg
        dim_PH = [0]
        # Cumulative dimensions for persistent homology along diagonals
        start_deg = deg
        for start_n_dim in range(min(deg+1, MV_ss.no_columns)):
            dim_PH.append(MV_ss.page_dim_matrix[no_pages][start_deg, start_n_dim] + dim_PH[-1])
            start_deg -= 1
        # end for        
        if dim_PH[-1] > 0:
            # Save space for extension matrix
            extension_matrix = np.zeros((dim_PH[-1], dim_PH[-1]))
            # compute extension matrix
            start_deg = deg-1
            for start_n_dim in range(1, min(deg+1, MV_ss.no_columns)):
                if MV_ss.page_dim_matrix[no_pages-1][start_deg, start_n_dim] > 0:
                    extensions = MV_ss.extension(start_n_dim, start_deg)
                    column_range = extension_matrix[:, dim_PH[start_n_dim]: dim_PH[start_n_dim+1]]
                    for i, blk in enumerate(extensions):
                        column_range[dim_PH[start_n_dim - i]: dim_PH[start_n_dim + 1 - i]] = extensions[i]
                    # end for
                # end if
                start_deg -= 1
            # end for
            #
            # add up barcodes in diagonal
            barcode = []
            start_deg = deg
            for start_n_dim in range(min(deg+1, MV_ss.no_columns)):
                if len(barcode) == 0 and MV_ss.Hom[no_pages-1][start_n_dim][start_deg].dim>0:
                    barcode = MV_ss.Hom[no_pages-1][start_n_dim][start_deg].barcode
                else:
                    if MV_ss.Hom[no_pages-1][start_n_dim][start_deg].dim > 0:
                        barcode = np.append(barcode, MV_ss.Hom[no_pages-1][start_n_dim][
                                            start_deg].barcode, axis=0)
                # end else
                start_deg -= 1
            # end for
            # Create a new barcode basis, based on the death radius of the extension matrix column
            diagonal_basis = barcode_basis(np.copy(barcode))
            # Create new basis, which is the direct sum matrix. This is a broken barcode basis 
            # with broken differentials given by the extension matrix.
            direct_sum_basis = barcode_basis(barcode, broken_basis=True, broken_differentials= extension_matrix)
            # Create an identity matrix as a morphism from diagonal_basis to direct_sum_basis.
            diagonal_differential = np.identity(diagonal_basis.dim)
            # Order the barcodes for domain and range
            MV_ss.order_diagonal_basis.append(diagonal_basis.sort(send_order=True))
            order = direct_sum_basis.sort(send_order=True)
            # Reorder associated matrix accordingly
            diagonal_differential = diagonal_differential[:, MV_ss.order_diagonal_basis[-1]]
            diagonal_differential = diagonal_differential[order]
            # set all barcode endpoints to max_r in diagonal_basis
            diagonal_basis.barcode[:,1] = max_r * np.ones(diagonal_basis.dim)
            births = MV_ss.Hom[4][1][0].barcode[:,0]
            deaths = MV_ss.Hom[4][1][0].barcode[:,1]
            rel_dom = []
            rel_range = []
            PH = image_kernel(diagonal_basis, direct_sum_basis, diagonal_differential, MV_ss.p)
            MV_ss.persistent_homology.append(PH)
        # end if
    # end for 
    return MV_ss

###############################################################################
# Local persistent homology
# Method to be parallelized

def local_persistent_homology(nerve_point_cloud, max_r, max_dim, p,  n_dim, spx_idx):
    """
    This function computes the Vietoris Rips complex and persistent
        homology  of a cover element. 
    It is meant to be run in parallel.
    INPUT:
        - nerve_point_cloud : point clouds indexed by nerve
        - points_IN : Identification Numbers for points indexed by nerve
        - max_r : maximum radius for computing 
        - max_dim : maximum dimension for complexes
        - n_dim : dimension in nerve
        - spx_idx : index of current index
    OUTPUT:
        - local_complex
        - Hom, Im, PreIm
    """
    local_point_cloud = nerve_point_cloud[n_dim][spx_idx]
    # Compute local Vietoris Rips complex and differentials
    if len(local_point_cloud)==0:
        local_Dist = []
    else:
        local_Dist = dist.squareform(dist.pdist(local_point_cloud))

    local_complex, local_R = vietoris_rips(local_Dist, max_r, max_dim)
    local_differentials = complex_differentials(local_complex, p)

    # Persistent Homology
    Hom, Im, PreIm = persistent_homology(local_differentials, local_R, max_r, p)
    
    return local_complex, local_differentials, Hom, Im, PreIm


