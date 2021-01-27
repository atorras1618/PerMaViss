import numpy as np
import scipy.spatial.distance as dist

from permaviss.sample_point_clouds.examples import random_cube, take_sample
from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss
from permaviss.spectral_sequence.spectral_sequence_class import add_local_chains

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
            for mdeg, hom in enumerate(MV_ss.Hom[0][n_deg][nerve_spx_index][1:]):
                if type(hom) != type([]) and hom.dim > 0:
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
                    MV_ss.Hom_reps[MV_ss.no_pages -1
                        ][start_n_dim][start_deg][:-1]):
                # calculate Cech differential of (Sn_dim, Sdeg)
                diff_im = MV_ss.cech_diff(Sn_dim, Sdeg, chains)
                # calculate vertical differential of (Sn_dim - 1, Sdeg + 1)
                Sn_dim, Sdeg = Sn_dim - 1, Sdeg + 1
                next_chains = MV_ss.Hom_reps[MV_ss.no_pages -1
                    ][Sn_dim][Sdeg][k+1]
                vert_im = [next_chains[0],[]]
                for nerve_spx_index, ref in enumerate(next_chains[0]):
                    if len(ref) > 0:
                        vert_im[1].append(np.matmul(
                            MV_ss.zero_diff[n_deg][nerve_spx_index][Sdeg],
                            -next_chains[1][nerve_spx_index].T).T % p)
                    else:
                        vert_im[1].append([])
                    # end if
                # end for
                zero_chains = add_local_chains(vert_im, diff_im)
                for coord in zero_chains:
                    if np.any(coord % p):
                        raise(ValueError)
            # end for
        # end for
    # end for

#
#
#    # check that chains[-1] is zero through  first vertical then horizontal
#    if n_dim == 2 and deg == 0:
#        # compute vertical differential
#        chains_new = [[],[]]
#        for idx, ref in enumerate(chains[0]):
#            chains_new[0].append(ref)
#            if len(ref) > 0:
#                chains_new[1].append(np.matmul(
#                    self.zero_diff[1][idx][1],
#                    chains[1][idx].T
#                ).T)
#            else:
#                chains_new[1].append([])
#        # check that representative is correct
#        # compute horizontal differential of first entry
#        horiz_image = self.cech_diff(1,0, self.Hom_reps[current_page-1][n_dim][deg][0])
#        for idx, local_ch in enumerate(horiz_image[1]):
#            if np.any(local_ch % self.p):
#                if np.any((local_ch[chains[0][idx]] + chains_new[1][idx]) % self.p):
#                    print((local_ch[chains[0][idx]] + chains_new[1][idx]) % self.p)
#                    raise(RuntimeError)
#        #compute horizontal differential
#        trivial_image = self.cech_diff(0, 0, chains_new)
#        for triv in trivial_image[1]:
#            if len(triv) > 0:
#                if np.any(triv % self.p):
#                    print(triv % self.p)
#                    raise(RuntimeError)
#        image_chains = self.cech_diff(0,1,chains)
#        trivial_im = [[],[]]
#        for idx, ref in enumerate(image_chains[0]):
#            trivial_im[0].append(ref)
#            if len(ref) > 0:
#                trivial_im[1].append(np.matmul(
#                    self.zero_diff[0][idx][1],
#                    image_chains[1][idx].T
#                ).T)
#                if np.any(trivial_im[1][-1] % self.p):
#                    print(trivial_im[1][-1] % self.p)
#                    raise(RuntimeError)
#            else:
#                chains_new[1].append([])
#
#    ########################################################################
#
#
#
#
# # TEST lift to page: whether it lifts hom coordinates (inside high_differential)
# ########################################################################
#
#
# ########################################################################
#
#
# # TEST commumativity of second page reps (in compute_two_page_representatives)
# ########################################################################
# if n_dim > 0:
#     cech_diff_im = self.cech_diff(n_dim-1, deg, chains)
#     betas, _ = self.first_page_lift(n_dim-1, deg, cech_diff_im, R)
#     if np.any(betas):
#         print(betas)
#         raise(RuntimeError)
#     for idx, ref in enumerate(lift[0]):
#         if len(ref) > 0:
#             vert_im = np.matmul(
#                 self.zero_diff[n_dim-1][idx][deg+1],
#                 lift[1][idx].T
#             ).T
#             cech_local_nontrivial = cech_diff_im[1][idx][
#                 cech_diff_im[1][idx].any(axis=1)]
#             if np.any((vert_im + cech_local_nontrivial)%self.p):
#                 print((vert_im + cech_local_nontrivial)%self.p)
#                 raise(RuntimeError)
#         else:
#             if len(cech_diff_im[0][idx]) > 0 and np.any(cech_diff_im[1][idx]):
#                 print("here")
#                 print("local_lift")
#                 print(lift[1][idx])
#                 print("local_cech_im")
#                 print(cech_diff_im[1][idx])
#                 raise(ValueError)
# ########################################################################
#
# # TEST that image of Cech differential is a cycle (in cech_diff_and_lift_local)
# ########################################################################
# if deg > 0:
#     trivial_image = np.matmul(
#         self.zero_diff[n_dim][nerve_spx_index][deg],
#         local_chains) % self.p
#     if np.any(trivial_image):
#         print("Image of Cech diff not a cycle")
#         raise(RuntimeError)
# ########################################################################
#
# # TEST radius nonzero entries in betas_aux (in cech_diff_and_lift_local)
# ########################################################################
# for g in generators:
#     nonzero_coeff_bars = self.Hom[0][n_dim][nerve_spx_index][
#         deg].barcode[np.nonzero(betas_aux[g])[0]]
#     if len(nonzero_coeff_bars) > 0:
#         if R[g] < np.max(nonzero_coeff_bars[:,0]):
#             raise(ValueError)
# ########################################################################
#
