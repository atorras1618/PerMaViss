import numpy as np
import scipy.spatial.distance as dist

from permaviss.sample_point_clouds.examples import random_cube, take_sample
from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss
from permaviss.spectral_sequence.spectral_sequence_class import (
    add_local_chains)

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
            for mdeg, hom in enumerate(
                    MV_ss.Hom[0][n_deg][nerve_spx_index][1:]):
                if hom.dim > 0:
                    trivial_image = np.matmul(
                        MV_ss.zero_diff[n_deg][nerve_spx_index][mdeg+1],
                        hom.coordinates)
                    if np.any(trivial_image % p):
                        raise(RuntimeError)
                    # end if
                # end if
            # end for
        # end for
    # end for

    # TEST commutativity of zig-zags
    ########################################################################
    for start_n_dim in range(1, MV_ss.no_columns):
        for start_deg in range(MV_ss.no_rows):
            Sn_dim, Sdeg = start_n_dim, start_deg
            for k, chains in enumerate(
                    MV_ss.Hom_reps[MV_ss.no_pages - 1][start_n_dim][
                        start_deg][:-1]):
                # calculate Cech differential of chains in (Sn_dim, Sdeg)
                diff_im = MV_ss.cech_diff(Sn_dim - 1, Sdeg, chains)
                # calculate vertical differential of (Sn_dim - 1, Sdeg + 1)
                Sn_dim, Sdeg = Sn_dim - 1, Sdeg + 1
                next_chains = MV_ss.Hom_reps[MV_ss.no_pages - 1][
                    start_n_dim][start_deg][k + 1]
                vert_im = [next_chains[0], []]
                for nerve_spx_index, ref in enumerate(next_chains[0]):
                    if len(ref) > 0:
                        # compute vertical image of next_chains locally
                        vert_im[1].append(np.matmul(
                            MV_ss.zero_diff[Sn_dim][nerve_spx_index][Sdeg],
                            next_chains[1][nerve_spx_index].T).T % p)
                    else:
                        vert_im[1].append([])
                    # end if
                # end for
                # check that d_v(next_chains) + d_cech(chains) = 0
                zero_chains = add_local_chains(vert_im, diff_im)
                for coord in zero_chains[1]:
                    if len(coord) > 0:
                        if np.any(coord % p):
                            raise(ValueError)
            # end for
        # end for
    # end for

    # TEST that image of Cech differential is a cycle
    ###########################################################################
    for start_n_dim in range(1, MV_ss.no_columns):
        for start_deg in range(1, MV_ss.no_rows):
            Sn_dim, Sdeg = start_n_dim - 1, start_deg
            for k, chains in enumerate(
                    MV_ss.Hom_reps[MV_ss.no_pages - 1][
                        start_n_dim][start_deg][:-1]):
                # calculate Cech differential of chains in (Sn_dim, Sdeg)
                diff_im = MV_ss.cech_diff(Sn_dim, Sdeg, chains)
                # initialize image vertical diff of cech diff
                zero_chains = [diff_im[0], []]
                for nerve_spx_index, ref in enumerate(diff_im[0]):
                    if len(ref) > 0:
                        # compute vertical image of next_chains locally
                        zero_chains[1].append(np.matmul(
                            MV_ss.zero_diff[Sn_dim][nerve_spx_index][Sdeg],
                            diff_im[1][nerve_spx_index].T).T % p)
                    else:
                        zero_chains.append([])
                # check that result is indeed zero
                for coord in zero_chains[1]:
                    if len(coord) > 0:
                        print(coord)
                        if np.any(coord % p):
                            raise(ValueError)
                # end for
            # end for
        # end for
    # end for
