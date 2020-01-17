import numpy as np
import scipy.spatial.distance as dist

from permaviss.sample_point_clouds.examples import random_cube, take_sample
from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss
from permaviss.spectral_sequence.spectral_sequence_class import (
    add_dictionaries)

from permaviss.simplicial_complexes.vietoris_rips import vietoris_rips
from permaviss.simplicial_complexes.differentials import complex_differentials
from permaviss.persistence_algebra.PH_classic import persistent_homology


def test_double_complex():
    X = random_cube(500, 3)
    point_cloud = take_sample(X, 50)
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
    # compute spectral sequence
    MV_ss = create_MV_ss(point_cloud, max_r, max_dim, max_div, overlap, p)
    # Check that computed barcodes coincide
    for it, PH in enumerate(MV_ss.persistent_homology):
        assert np.array_equal(PH.barcode, PerHom[it].barcode)
    # Vertical Differentials
    for n_dim in range(MV_ss.no_columns):
        # check for vertical differential
        if n_dim == 0:
            rk_nerve = MV_ss.nerve[n_dim]
        else:
            rk_nerve = len(MV_ss.nerve[n_dim])
        for nerv_spx in range(rk_nerve):
            for deg in range(2, MV_ss.no_rows):
                zero_coordinates = []
                coord_matrix = np.identity(len(
                    MV_ss.subcomplexes[n_dim][nerv_spx][deg]))
                for spx_idx in range(len(
                        MV_ss.subcomplexes[n_dim][nerv_spx][deg])):
                    zero_coordinates.append({nerv_spx: coord_matrix[spx_idx]})
                # end for
                im_vert = MV_ss.vert_image(zero_coordinates, n_dim, deg)
                im_vert = MV_ss.vert_image(im_vert, n_dim, deg-1)
                for i, coord in enumerate(im_vert):
                    for n in iter(coord):
                        if np.any(coord[n]):
                            assert False
                        # end if
                    # end for
                # end for
            # end for
        # end for
    # end for
    # check for Cech differentials
    for deg in range(MV_ss.no_rows):
        for n_dim in range(2, MV_ss.no_columns):
            for nerv_spx in range(len(MV_ss.nerve[n_dim])):
                zero_coordinates = []
                if deg == 0:
                    no_simplexes = MV_ss.subcomplexes[n_dim][nerv_spx][deg]
                else:
                    no_simplexes = len(
                        MV_ss.subcomplexes[n_dim][nerv_spx][deg])
                coord_matrix = np.identity(no_simplexes)
                for spx_idx in range(no_simplexes):
                    zero_coordinates.append({nerv_spx: coord_matrix[spx_idx]})
                # end for
                cech_diff = MV_ss.cech_differential(
                    zero_coordinates, n_dim, deg)
                cech_diff = MV_ss.cech_differential(cech_diff, n_dim-1, deg)
                for i, coord in enumerate(cech_diff):
                    for n in iter(coord):
                        if np.any(coord[n]):
                            assert False
                        # end if
                    # end for
                # end for
            # end for
        # end for
    # end for

    # check for anticommutativity
    for n_dim in range(1, MV_ss.no_columns):
        for nerv_spx in range(len(MV_ss.nerve[n_dim])):
            for deg in range(1, MV_ss.no_rows):
                zero_coordinates = []
                coord_matrix = np.identity(
                    len(MV_ss.subcomplexes[n_dim][nerv_spx][deg]))
                for spx_idx in range(len(MV_ss.subcomplexes[n_dim][
                        nerv_spx][deg])):
                    zero_coordinates.append({nerv_spx: coord_matrix[spx_idx]})
                # end for
                im_left = MV_ss.cech_differential(zero_coordinates, n_dim, deg)
                im_left = MV_ss.vert_image(im_left, n_dim - 1, deg)
                im_right = MV_ss.vert_image(zero_coordinates, n_dim, deg)
                im_right = MV_ss.cech_differential(im_right, n_dim, deg-1)
                for i, coord in enumerate(im_left):
                    result = add_dictionaries(
                        [1, 1], [coord, im_right[i]], MV_ss.p)
                    for n in iter(result):
                        if np.any(result[n]):
                            assert False
                        # end if
                    # end for
                # end for
            # end for
        # end for
    # end for
