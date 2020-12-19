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
    # test cech_differential
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

    ###########################################################################
    # test local_cech_matrix

    def test_local_cech_matrix(self):
        """Iterate over test_local_cech_matrix"""
        for n_dim in range(2, self.no_columns):
            for deg in range(self.no_rows):
                if self.page_dim_matrix[1][deg][n_dim] > 0:
                    self.test_cech_differential(n_dim, deg)

    def test_cech_differential(self, n_dim, deg):
        """ Check that the local check matrices add up to form a
        cech differential. Check this on generators starting in position
        (n_dim, deg)  from first page.

        We assume that the given position is not trivial.
        """
        # compute coordinates for all generators in positoin (n_dim, deg)
        ref_preim = []
        coord_preim = []
        prev = 0
        for nerv_idx, no_cycles in enumerate(self.cycle_dimensions[
                n_dim][deg][:-1]):
            if prev < no_cycles:
                ref_preim.append(range(prev, no_cycles))
                coord_preim.append(self.Hom[0][n_dim][nerv_idx][
                    deg].coordinates.T)
            else:
                ref_preim.append(np.array([]))
                coord_preim.append(np.array([]))
            # end if else
            prev = no_cycles
        # end for
        # IMAGE OF CECH DIFFERENTIAL twice #############################
        for k in [1,2]:
            ref_im = []
            coord_im = []
            for nerve_face_index, coboundary in enumerate(
                    self.nerve_differentials[n_dim-k+1]):
                self.cech_differential_loc(
                    ref_preim, coord_preim, ref_im, coord_im, n_dim-k,
                    deg, nerve_face_index, coboundary
                )
            # end for
            ref_preim = ref_im
            coord_preim = coord_im
        # twice cech differential
        # Check that images are zero
        for _, A in enumerate(coord_im):
            assert np.any(A) == False
        # end for

    def cech_differential_loc(self, ref_preim, coord_preim, ref_im, coord_im,
                              n_dim, deg, nerve_face_index, coboundary):
        """Given preimage references and coordinates, computes the image of the
        Cech differential restricted to nerve_face_index component.

        Coboundary stores the simplices in the coboundary of nerve_face_index,
        together with the differential coefficients.
        Results are stored in ref_im and coord_im which are assumed to be
        given as empty lists []

        Returns im_matrix.T
        """
        # size of local complex
        if deg == 0:
            cpx_size = self.subcomplexes[n_dim][nerve_face_index][deg]
        else:
            cpx_size = len(self.subcomplexes[n_dim][nerve_face_index][deg])

        cofaces = np.nonzero(coboundary)[0]
        coefficients = coboundary[cofaces]
        # indices of generators that are nontrivial by cech diff
        generators = ref_preim[cofaces[0]]
        for coface_index in cofaces[1:]:
            generators = np.append(generators, ref_preim[
                coface_index]).astype(int)
        # end for
        generators = np.unique(generators)
        # store references and space for Coordinates
        ref_im.append(generators)
        im_matrix = np.zeros((cpx_size, len(generators)))
        # compute cech
        for coface_index, nerve_coeff in zip(cofaces, coefficients):
            # check non-trivial
            if self.Hom[0][n_dim+1][coface_index][deg].dim > 0:
                # generate boundary matrix
                cech_local = self.local_cech_matrix(
                    n_dim+1, deg, coface_index, nerve_face_index,
                    nerve_coeff)
                active_generators = np.where(np.in1d(
                    generators, ref_preim[coface_index])
                    )[0]
                if np.any(active_generators):
                    im_matrix[:, active_generators] += np.matmul(
                        cech_local, coord_preim[coface_index].T
                        )
                    im_matrix = im_matrix % self.p
                # end if
            # end if
        # end for cofaces

        coord_im.append(im_matrix.T)
