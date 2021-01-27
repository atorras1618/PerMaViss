import numpy as np
import scipy.spatial.distance as dist

from permaviss.sample_point_clouds.examples import random_cube, take_sample
from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss

from permaviss.simplicial_complexes.vietoris_rips import vietoris_rips
from permaviss.simplicial_complexes.differentials import complex_differentials
from permaviss.persistence_algebra.PH_classic import persistent_homology

def test_double_complex():
    # creating and saving new point cloud ############
    X = random_cube(500, 3)
    point_cloud = take_sample(X, 50)
    output_file = open("test/spectral_sequence/random_cube.txt", "w")
    for row in point_cloud:
        np.savetxt(output_file, row)
    output_file.close()
    # using old point cloud ###################
    # saved_data = np.loadtxt("test/spectral_sequence/failing_1.txt")
    # no_points = int(np.size(saved_data,0) / 3)
    # point_cloud = np.reshape(saved_data, (no_points, 3))
    max_r = 0.4
    max_dim = 3
    max_div = 2
    overlap = max_r * 1.01
    p = 3
    # compute ordinary persistent homology
    Dist = dist.squareform(dist.pdist(point_cloud))
    C, R = vietoris_rips(Dist, max_r, max_dim)
    Diff = complex_differentials(C, p)
    PerHom, _, _ = persistent_homology(Diff, R, max_r, p)
    ###########################################################################
    # compute spectral sequence
    MV_ss = create_MV_ss(point_cloud, max_r, max_dim, max_div, overlap, p)

    ###########################################################################
    # check that first page representatives are cycles
    for n_deg in range(MV_ss.no_columns):
        if n_deg == 0:
            no_covers = MV_ss.nerve[0]
        else:
            no_covers = len(MV_ss.nerve[n_deg])
        for nerve_spx_index in range(no_covers):
            local_differentials = complex_differentials(
                MV_ss.subcomplexes[n_deg][nerve_spx_index], p)
            for mdeg, hom in enumerate(MV_ss.Hom[0][n_deg][nerve_spx_index][1:]):
                if type(hom) != type([]) and hom.dim > 0:
                    trivial_image = np.matmul(
                        local_differentials[mdeg+1],
                        hom.coordinates)
                    if np.any(trivial_image % p):
                        raise(RuntimeError)
                    # end if
                # end if
            # end for
        # end for
    # end for

    ###########################################################################
    # compute cech differential twice and check that it vanishes

    """Iterate over test_local_cech_matrix"""
    for n_dim in range(2, MV_ss.no_columns):
        for deg in range(MV_ss.no_rows):
            if MV_ss.page_dim_matrix[1][deg][n_dim] > 0:
                cech_differential_twice(MV_ss, n_dim, deg)
            # end if
        # end for
    # end for

    ############################################################################
    # check that classic PH coincides with result
    for it, PH in enumerate(MV_ss.persistent_homology):
        assert np.array_equal(PH.barcode, PerHom[it].barcode)
# end test_double complex


def cech_differential_twice(MV_ss, n_dim, deg):
    """ Check that the local check matrices add up to form a
    cech differential. Check this on generators starting in position
    (n_dim, deg)  from first page.

    We assume that the given position is not trivial.
    """
    # compute coordinates for all generators in positoin (n_dim, deg)
    ref_preim = []
    coord_preim = []
    prev = 0
    for nerv_idx, next in enumerate(MV_ss.cycle_dimensions[
            n_dim][deg][:-1]):
        if prev < next:
            ref_preim.append(range(prev, next))
            coord_preim.append(MV_ss.Hom[0][n_dim][nerv_idx][
                deg].coordinates.T)
        else:
            ref_preim.append(np.array([]))
            coord_preim.append(np.array([]))
        # end if else
        prev = next
    # end for
    # IMAGE OF CECH DIFFERENTIAL twice #############################
    for k in [1,2]:
        ref_im = []
        coord_im = []
        for nerve_face_index, coboundary in enumerate(
                MV_ss.nerve_differentials[n_dim-k+1]):
            ref_loc, coord_loc = MV_ss.cech_diff_local(
                ref_preim, coord_preim, n_dim-k,
                deg, nerve_face_index
            )
            ref_im.append(ref_loc)
            if len(coord_loc) > 0:
                coord_im.append(coord_loc.T)
            else:
                coord_im.append([])
        # end for
        ref_preim = ref_im
        coord_preim = coord_im
    # twice cech differential
    # Check that images are zero
    for _, A in enumerate(coord_im):
        assert np.any(A) == False
    # end for
# end cech_differential

#####
def multprint(A,B):
    print("multiplying ({},{}) times ({},{})".format(
        np.size(A,0), np.size(A,1), np.size(B,0), np.size(B,1)
    ))
    return np.matmul(A,B)
