#
#    spectral_sequence_class.py
#

import numpy as np

from multiprocessing import Pool
from functools import partial

from ..simplicial_complexes.differentials import complex_differentials
from ..gauss_mod_p.functions import solve_mod_p, multiply_mod_p
from ..gauss_mod_p.gauss_mod_p import gauss_col_rad


class spectral_sequence(object):
    """Space and methods for Mayer-Vietoris spectral sequences

    Parameters
    ----------
    nerve : :obj:`list(Numpy Array)`
        Simplicial complex storing the nerve of the covering. This is stored as
        a list, where the ith entry contains a Numpy Array storing all the
        ith simplices.
    nerve_point_cloud : :obj:`list(list(Numpy Array))`
        Point clouds indexed by nerve of the cover, see
        :mod:`permaviss.covers.cubical_cover`
    points_IN : :obj:`list(list(Numpy Array))`
        Point Identification Numbers (IN) indexed by nerve of the cover, see
        :mod:`permaviss.covers.cubical_cover`
    max_dim : int
        Maximum dimension of simplices.
    max_r : float
        Maximum persistence radius.
    no_pages : int
        Number of pages of the spectral sequence
    p : int(prime)
        The prime number so that our computations are mod p

    Attributes
    ----------
    nerve, nerve_point_cloud, points_IN, max_dim, max_r, no_pages, p :
        as described above
    no_rows, no_columns : int, int
        Number of rows and columns in each page
    nerve_differentials : :obj:`list(Numpy Array)`
        List storing the differentials of the Nerve. The ith entry stores the
        matrix of the ith differential.
    subcomplexes : :obj:`list(list(list(Numpy Array)))`
        List storing the simplicial complex on each cover element. For integers
        `n_dim`, `k` and `dim` the variable `subcomplexes[n_dim][k][dim]`
        stores the `dim`-simplices on the cover indexed by the `k` simplex of
        dimension `n_dim` in the nerve.
    zero_differentials : :obj:`list(list(list(Numpy Array)))`
        List storing the vertical differential matrices on the 0 page of the
        spectral sequence. For  integers `n_dim`, `k` and `dim` the variable
        `zero_differentials[n_dim][k][dim]` stores the `dim` differential of
        the complex on the cover indexed by the `k` simplex of dimension
        `n_dim` in the nerve.
    cycle_dimensions : :obj:`list(list(list(int)))`
        List storing the number of bars on each local persistent homology.
        Given two integers `n_dim` and `dim`, the variable
        `cycle_dimensions[n_dim][dim]` contains a list where each entry
        corresponds to an `n_dim` simplex in the nerve. For each such entry,
        we store the number of nontrivial persistent homology classes of
        dimension `dim` in the corresponding cover.
    Hom : :obj:`list(...(list(barcode_basis)))`
        Homology for each page of the spectral sequence. Given three integers
        which we denote `n_dim`, `nerv_spx` and `deg` we have that
        `Hom[0][n_dim][nerv_spx][deg]` stores a :obj:`barcode_basis` with the
        `deg`-persistent homology of the covering indexed by
        `nerve[n_dim][nerv_spx]`. All these store the homology on the `0` page
        of the spectral sequence. Additionally, for integers `k > 0`, `n_dim`
        and `deg`, we store in `Hom[k][n_dim][deg]` the :obj:`barcode_basis`
        for the homology on the `(deg, n_dim)` entry in the `k` page of the
        spectral sequence.
    Im : :obj:`list(...(list(barcode_basis)))`
        Image for each page of the spectral sequence. Given three integers
        which we denote `n_dim`, `nerv_spx` and `deg` we have that
        `Im[0][n_dim][nerv_spx][deg]` stores a :obj:`barcode_basis` for the
        image of the `deg+1`-differential of the covering indexed by
        `nerve[n_dim][nerv_spx]`. All these store the images on the `0` page
        of the spectral sequence. Additionally, for integers `k > 0`, `n_dim`
        and `deg`, we store in `Im[k][n_dim][deg]` the :obj:`barcode_basis` for
        the image on the `(deg, n_dim)` entry in the `k` page of the spectral
        sequence.
    PreIm : :obj:`list(...(list(Numpy Array)))`
        Preimages for each page of the spectral sequence. Given three integers
        which we denote `n_dim`, `nerv_spx` and `deg` we have that
        `PreIm[0][n_dim][nerv_spx][deg]` stores a :obj:`Numpy Array` for the
        Preimage of the `deg+1`-differential of the covering indexed by
        `nerve[n_dim][nerv_spx]`. Additionally, for integers `k > 0`, `n_dim`
        and `deg`, we store in `PreIm[k][n_dim][deg]` a :obj:`Numpy Array` for
        the preimages of the differential images in the `(deg, n_dim)` entry in
        the `k` page of the spectral sequence.
    tot_complex_reps : :obj:`list(list(*))`
        The asterisc `*` on the type can be either [] or
        :obj:`list(Numpy Array)`. This is used for storing complex
        representatives for the cycles.
    page_dim_matrix : :obj:`Numpy Array(no_pages+1, max_dim, no_columns)`
        Array storing the dimensions of the entries in each page. Notice that
        the order in which we store columns and rows differs from all the
        previous attributes.
    persistent_homology : :obj:`list(barcode_basis)`
        List storing the persistent homology generated by the spectral
        sequence. The `i` entry contains the `i` dimensional persistent
        homology.
    order_diagonal_basis : `list`
        This intends to store the original order of `persistent_homology`
        before applying the standard order.
    extensions : :obj:`list(list(list(Numpy Array)))`
        Nested lists, where the first two indices are for the column and row.
        The last index indicates the corresponding extension matrix.

    Notes
    -----
    The indexing on the 0 page is different from that of the next pages. This
    is because we do not want to store all the 0 page information on the same
    place.

    """
    def __init__(self, nerve, nerve_point_cloud, points_IN, max_dim,
                 max_r, no_pages, p):
        """Construction method
        """
        # dimensions of spectral sequence
        self.no_pages = no_pages
        self.no_rows = max_dim
        self.no_columns = len(nerve)
        self.max_r = max_r
        self.p = p
        # add nerve_point_cloud to spectral_sequence info
        self.nerve_point_cloud = nerve_point_cloud
        # add points IN to support Cech Differential
        self.points_IN = points_IN
        # add nerve and compute nerve differentials
        self.nerve = nerve
        self.nerve_differentials = complex_differentials(nerve, p)
        # list containing barcode bases for Hom, Im and PreIm
        # Hom and Im go through all pages, whereas
        # PreIm is only contained in the 0 page
        self.Hom = [[]]
        self.Im = [[]]
        self.PreIm = [[]]
        self.subcomplexes = []
        self.zero_differentials = []
        self.cycle_dimensions = []
        self.first_page_barcodes = []
        # vectors that translate local indices to global
        self.tot_complex_reps = []
        # higher page representatives
        self.optimal_reps = [[]]
        # store extension matrices
        self.extensions = []
        for n_dim in range(len(nerve)):
            self.Hom[0].append([])
            self.Im[0].append([])
            self.PreIm[0].append([])
            self.subcomplexes.append([])
            self.zero_differentials.append([])
            self.cycle_dimensions.append([])
            self.first_page_barcodes.append([])
            self.tot_complex_reps.append([])
            self.extensions.append([])
            for deg in range(self.no_rows):
                self.first_page_barcodes[n_dim].append([])
                self.tot_complex_reps[n_dim].append([])
                self.extensions[n_dim].append([])

        # make lists to store information in higher pages
        for k in range(1, no_pages):
            self.Hom.append([])
            self.Im.append([])
            self.PreIm.append([])
            self.optimal_reps.append([])
            for n_dim in range(self.no_columns):
                self.Hom[k].append([])
                self.Im[k].append([])
                self.PreIm[k].append([])
                self.optimal_reps[k].append([[]])
                for deg in range(self.no_rows):
                    self.Hom[k][n_dim].append([])
                    self.Im[k][n_dim].append([])
                    self.PreIm[k][n_dim].append([])
                    self.optimal_reps[k][n_dim].append([])

        # save space for dimension matrices for all pages
        # the order of variables is for printing the spectral sequence
        self.page_dim_matrix = np.zeros((no_pages+1, max_dim,
                                         self.no_columns)).astype(int)

        # define persistent homology and order of diagonal basis
        self.persistent_homology = []
        self.order_diagonal_basis = []

    ###########################################################################
    # add content to first page

    def add_output_first(self, output, n_dim):
        """Stores the 0 page data of `n_dim` column after it has been computed
        in parallel by `multiprocessing.pool`

        Parameters
        ----------
        output : :obj:`list`
            Result after using `multiprocessing.pool` on
            :meth:`permaviss.spectral_sequence.MV_spectral_seq.local_persistent_homology`
        n_dim : int
            Column of `0`-page whose data has been computed.

        """
        self.subcomplexes[n_dim] = [it[0] for it in output]
        self.zero_differentials[n_dim] = [it[1] for it in output]
        self.Hom[0][n_dim] = [it[2] for it in output]
        self.Im[0][n_dim] = [it[3] for it in output]
        self.PreIm[0][n_dim] = [it[4] for it in output]

        # Number of simplices in nerve[n_dim]
        if n_dim > 0:
            n_spx_number = np.size(self.nerve[n_dim], 0)
        else:
            n_spx_number = self.nerve[0]

        # add this to check that the level of intersection is not empty.
        # otherwise we might run out of index range
        if len(self.Hom[0][n_dim]) > 0:
            for deg in range(self.no_rows):
                no_cycles = 0
                # cumulative dimensions
                self.cycle_dimensions[n_dim].append(
                    np.zeros(n_spx_number+1).astype(int))
                for k in range(n_spx_number):
                    # Generate page dim matrix and local_coordinates info
                    cycles_in_cover = self.Hom[0][n_dim][k][deg].dim
                    no_cycles += cycles_in_cover
                    self.cycle_dimensions[n_dim][deg][k] = no_cycles
                # end for
                self.page_dim_matrix[1, deg, n_dim] = no_cycles
                # put together first page barcodes
                if no_cycles == 0:
                    self.first_page_barcodes[n_dim][deg] = []
                else:
                    self.first_page_barcodes[n_dim][deg] = np.zeros((
                    no_cycles, 2))
                    prev = 0
                    for k in range(n_spx_number):
                        # Generate page dim matrix and local_coordinates info
                        next = self.cycle_dimensions[n_dim][deg][k]
                        if prev < next:
                            self.first_page_barcodes[n_dim][deg][
                                prev:next] = self.Hom[0][n_dim][k][deg].barcode
                        # end if
                        prev = next
                    # end for
                # end else
            # end for
        # end if


    ###########################################################################
    # localize_coordinates

    def localize_coordinates(self, initial_coordinates, n_dim, deg):
        """
        Transforms matrix of coordinates in first page to a pair
        of references and local coordinates on the 0 page
        """
        reference = []
        local_coordinates = []
        prev = 0
        for k, dim_next in enumerate(self.cycle_dimensions[n_dim][deg][:-1]):
            print("K:{}  dim_next:{}".format(k, dim_next))
            local_hom_coord = initial_coordinates[:, prev:dim_next]
            prev = dim_next
            reference.append(np.nonzero(local_hom_coord)[0])
            local_coordinates.append(multiply_mod_p(
                self.Hom[0][n_dim][k][deg].coordinates,
                local_hom_coord[reference[-1]].T, self.p).T)
        # end for
        return reference, local_coordinates


    ############################################################################
    # local boundary matrix
    def local_cech_matrix(self, n_dim, deg, nerve_spx_index,
                              nerve_face_index, nerve_coeff):

        """
         (subcpx[n_dim-1], sbcpx[n_dim]) matrix
        """
        deg_sign = (-1)**deg
        if deg == 0:
            # save space for boundary matrix
            boundary = np.zeros((
                self.subcomplexes[n_dim-1][nerve_face_index][deg],
                self.subcomplexes[n_dim][nerve_spx_index][deg]))
            # inclusions for points
            for point_idx in range(
                    self.subcomplexes[n_dim][nerve_spx_index][0]):
                face_point_idx = np.argmax(self.points_IN[
                    n_dim-1][nerve_face_index] == self.points_IN[n_dim][
                        nerve_spx_index][point_idx])
                boundary[face_point_idx, point_idx] = nerve_coeff * deg_sign
                boundary[face_point_idx, point_idx] %= self.p
            # end for
        else:
            # save space for boundary matrix
            boundary = np.zeros((
                len(self.subcomplexes[n_dim-1][nerve_face_index][deg]),
                len(self.subcomplexes[n_dim][nerve_spx_index][deg])))
            # inclusions for edges, 2-simplices and higher
            # Iterate over nontrivial local simplices in domain
            for spx_index, simplex in enumerate(
                    self.subcomplexes[n_dim][nerve_spx_index][deg]):
                # Obtain IN for vertices of simplex
                vertices_spx = self.points_IN[n_dim][nerve_spx_index][simplex]
                for im_index, im_spx in enumerate(
                        self.subcomplexes[n_dim-1][
                            nerve_face_index][deg]):
                    vertices_face = self.points_IN[n_dim-1][
                        nerve_face_index][im_spx.astype(int)]
                    # When the vertices coincide, break the loop
                    if len(np.intersect1d(
                            vertices_spx,
                            vertices_face)) == deg + 1:
                        boundary[im_index, spx_index] = nerve_coeff * deg_sign
                        boundary[im_index, spx_index] %= self.p
                        break
                    # end if
                # end for
            # end for
        # end else
        return boundary

    #######################################################################
    # Cech chain plus lift of preimage
    def cech_diff_and_lift(
            self, R, reference_preimage, local_preimage, n_dim, deg,
            lift_references, lift_coordinates, first_page_image,
            nerve_spx_index
            ):
        # if nerve_spx_index==0, then prev=0
        prev = self.cycle_dimensions[n_dim][deg][nerve_spx_index-1]
        next = self.cycle_dimensions[n_dim][deg][nerve_spx_index]
        # if trivial cover skip
        if prev == next:
            return
        coboundary = self.nerve_differentials[n_dim+1][nerve_spx_index]
        # cofaces and coefficients on cech differential
        cofaces = np.nonzero(coboundary)[0]
        coefficients = coboundary[cofaces]
        # indices of generators that are nontrivial by cech diff
        generators = reference_preimage[cofaces[0]]
        for coface_index in cofaces[1:]:
            generators = np.append(generators, reference_preimage[
                coface_index]).astype(int)
        # end for
        generators = np.unique(generators)
        #if there are no images to compute, return
        if len(generators) == 0:
            return
        # size of local complex
        if deg == 0:
            cpx_size = self.subcomplexes[n_dim][nerve_spx_index][deg]
        else:
            cpx_size = len(self.subcomplexes[n_dim][nerve_spx_index][deg])

        local_chains = np.zeros((cpx_size, len(generators)))
        # IMAGE OF CECH DIFFERENTIAL #############################
        for coface_index, nerve_coeff in zip(cofaces, coefficients):
            # check local preimage is non-trivial
            if self.Hom[0][n_dim+1][coface_index][deg].dim > 0:
                # generate boundary matrix
                cech_local = self.local_cech_matrix(
                    n_dim+1, deg, coface_index, nerve_spx_index,
                    nerve_coeff)
                active_generators = np.where(np.in1d(
                    generators, reference_preimage[coface_index])
                    )[0]
                    # image of cech complex
                # print("cech_local")
                # print(cech_local)
                # print("local_preimage[coface_index].T")
                # print(local_preimage[coface_index].T)
                local_chains[:, active_generators] += np.matmul(
                    cech_local, local_preimage[coface_index].T
                    )
            # end if
        # end for
        # FIRST PAGE LIFT ######################################
        # create Im_Hom Matrix
        Im_Hom = np.append(
            self.Im[0][n_dim][nerve_spx_index][deg].coordinates,
            self.Hom[0][n_dim][nerve_spx_index][deg].coordinates,
            axis=1)

        start_index = np.size(Im_Hom,1)
        # Gaussian elimination of  M = (Im | Hom | local_chains)
        M = np.append(Im_Hom, local_chains, axis = 1)
        # R_M : vector of birth radii of columns in M
        R_M = self.Im[0][n_dim][nerve_spx_index][deg].barcode[:,0]
        R_M = np.concatenate([
            R_M, self.Hom[0][n_dim][nerve_spx_index][deg].barcode[:,0]],
            axis=None
            )
        R_M = np.concatenate([R_M, R[generators]], axis=None)
        _, T = gauss_col_rad(M, R_M, start_index, self.p)
        # look at reductions on generators
        T = T[:, start_index:]
        # compute vertical preimage and store
        gammas = T[0:self.Im[0][n_dim][nerve_spx_index][deg].dim]
        # look for indices of nonzero columns
        preimages = np.matmul(self.PreIm[0][n_dim][nerve_spx_index][deg+1],
                              gammas).T
        nonzero_idx = np.where(gammas.any(axis=0))[0]
        if np.any(nonzero_idx):
            lift_coordinates[nerve_spx_index] = preimages[nonzero_idx]
            lift_references[nerve_spx_index] = generators[nonzero_idx]

        # store first page coefficients
        betas = T[self.Im[0][n_dim][nerve_spx_index][
            deg].dim : start_index]
        betas_aux = np.zeros((len(R), next-prev))
        betas_aux[generators] = np.transpose(betas)
        first_page_image[:, prev:next] = betas_aux
    # end cech_diff_and_lift

    ###########################################################################
    # self.compute_differential()

    def compute_differential(self, n_dim, deg, current_page):
        """ Compute differential of spectral sequence starting on position
        (n_dim, deg) in current_page.
        """
        print("n_dim:{}, deg:{}".format(n_dim, deg))
        # handle trivial cases
        if self.page_dim_matrix[current_page, deg, n_dim] == 0:
            return np.array([])
        # generate refs and coordinates of current page
        # store as total complex
        if current_page == 1:
            R = np.zeros(self.page_dim_matrix[1, deg, n_dim])
            references = []
            local_coordinates = []
            prev = 0
            for nerv_idx, no_cycles in enumerate(self.cycle_dimensions[
                    n_dim][deg][:-1]):
                if prev < no_cycles:
                    R[prev:no_cycles] = self.Hom[0][n_dim][nerv_idx][
                        deg].barcode[:,0]
                    references.append(range(prev, no_cycles))
                    local_coordinates.append((self.Hom[0][n_dim][nerv_idx][
                        deg].coordinates).T)
                else:
                    references.append(np.array([]))
                    local_coordinates.append(np.array([]))
                # end if else
                prev = no_cycles
            # end for
            total_complex_chain = [[references, local_coordinates]]
            active_ref = range(self.page_dim_matrix[1, deg, n_dim])
        else:
            # compute coordinates of homology class
            R = self.Hom[current_page-1][n_dim][deg].barcode[:,0]
            print("R")
            print(R)
            print("Optimal Reps")
            print(self.optimal_reps[current_page-1][n_dim][deg][-1][0])
            print(self.optimal_reps[current_page-1][n_dim][deg][-1][1])
            print("Hom")
            print(self.Hom[current_page-1][n_dim][deg].coordinates.T)

            total_complex_chain = local_sums(
                self.optimal_reps[current_page-1][n_dim][deg],
                self.Hom[current_page-1][n_dim][deg].coordinates.T,
                )

            active_ref = np.array([])
            for idx_local, ref in enumerate(total_complex_chain[-1][0]):
                np.concatenate([active_ref, ref], axis=None)

            print("active_ref")
            print(active_ref)
            if len(total_complex_chain[-1][0]) != len(self.nerve[
                    n_dim - current_page + 1]):
                raise ValueError

        # number of simplices to parallelize over
        if n_dim == 1:
            n_spx_number = self.nerve[0]
        else:
            n_spx_number = len(self.nerve[n_dim-1])

        first_page_image = np.zeros((
            len(active_ref), self.page_dim_matrix[1, deg, n_dim-1]
            ))
        # store space for preimages
        lift_references = []
        lift_coordinates = []
        Betas = np.zeros((
            len(R), self.page_dim_matrix[current_page, deg, n_dim-1]
            ))
        if np.any(active_ref):
            for nerve_spx_index in range(n_spx_number):
                lift_references.append([])
                lift_coordinates.append([])

            partial_cech_diff_and_lift = partial(
                self.cech_diff_and_lift, R[active_ref], total_complex_chain[-1][0],
                total_complex_chain[-1][1], n_dim - current_page,
                deg + current_page - 1, lift_references, lift_coordinates,
                first_page_image
            )
            for idx in range(n_spx_number):
                partial_cech_diff_and_lift(idx)
            # workers_pool = Pool()
            # workers_pool.map(partial_cech_diff_and_lift, range(n_spx_number))
            # workers_pool.close()
            betas = first_page_image.T
            # lift to higher pages, modifying total complex chains
            for k in range(1, current_page):
                Im = self.Im[k][n_dim][deg]
                Im_dim = self.Im[k][n_dim][deg].dim
                Hom = self.Hom[k][n_dim][deg]
                Hom_dim = self.Hom[k][n_dim][deg].dim
                if Im_dim > 0 and Hom_dim > 0:
                    Im_Hom = np.append(Im, Hom, axis=1)
                    barcode_col = np.append(Im.barcode, Hom.barcode, axis=0)
                elif Im_dim > 0:
                    Im_Hom = Im
                    barcode_col = Im.barcode
                elif Hom_dim > 0:
                    Im_Hom = Hom
                    barcode_col = Im.barcode
                else:
                    break

                start_index = Im_dim + Hom_dim
                # add radii coordinates and radii
                barcode_col = np.append(barcode_col, coord_R, axis=0)
                A = np.append(Im_Hom, betas, axis=1)
                # gaussian reduction
                death_row = self.Hom[k][n_dim][deg].prev_basis.barcode[:,1]
                _, T = gauss_col_row_rad(A, death_row, barcode_col[:,0],
                                         barcode_col[:,1], start_index, p)
                T = T[:, start_index:]
                # use preimage coefficients to modify total_representatives
                gammas = T[:Im_dim]
                #### TO DO, modify total complex reps

                # next page coefficients
                betas = T[Im_dim:start_index]
            # end for over pages
            # store optimal reps
            self.optimal_reps[current_page][n_dim][deg] = total_complex_chain
            for local in total_complex_chain:
                refs = local[0]
                chains = local[1]
                for loc_idx, ref in enumerate(refs):
                    if len(ref) != len(chains[loc_idx]):
                        print(len(ref))
                        print(np.size(chains[loc_idx],0))
                        print(np.size(chains[loc_idx],1))
                        raise ValueError
                    # end if
                # end for
            # end for
            Betas[active_ref] = betas.T
        # end if
        total_complex_chain.append([lift_references, lift_coordinates])
        refs = total_complex_chain[-1][0]
        chains = total_complex_chain[-1][1]
        for loc_idx, ref in enumerate(refs):
            # print("loc_idx:{}".format(loc_idx))
            # print("refs")
            # print(ref)
            # print("chains")
            # print(chains[loc_idx])
            if len(ref) != len(chains[loc_idx]):
                print(len(ref))
                print(np.size(chains[loc_idx],0))
                print(np.size(chains[loc_idx],1))
                raise ValueError
        return Betas
    # end compute_differential


    ###########################################################################
    # self.optimal reps method for obtaining modified representatives



    ###########################################################################
    # solve higher page equations
    def solve_higher(self, first_page_coord, target_page, n_dim, deg):
        """Takes an array of coordinates on first page and lifts to target_page
        Returns arrays of preimage gamma coefficients
        """
        for k in range(1, target_page):
            prev_target_coordinates = target_coordinates
            # this should be simplified by the use of image_kernel, however,
            # since this corresponds to higher pages we leave it as it is
            target_coordinates = np.zeros((
                len(start_coordinates), self.page_dim_matrix[k+1, deg, n_dim]))
            for i, coord in enumerate(prev_target_coordinates):
                # solve (Im|Hom)
                Hom_dim = self.Hom[k][n_dim][deg].dim
                if Hom_dim > 0:
                    Hom = self.Hom[k][n_dim][deg].active_coordinates(R[i])

                Im_dim = self.Im[k][n_dim][deg].dim
                if Im_dim > 0:
                    Im = self.Im[k][n_dim][deg].active_coordinates(R[i])
                    Im_dim = np.size(Im, 1)
                    if Hom_dim > 0 and len(Hom) > 0:
                        Im_Hom = np.append(Im, Hom, axis=1)
                else:
                    Im_Hom = Hom
                # end else
                if len(Im_Hom) > 0:
                    lifted_coordinates = np.zeros(self.Hom[k][n_dim][deg].dim)
                    if np.size(Im_Hom, 1) > 0:
                        prev_active_coordinates = prev_target_coordinates[i][
                            self.Hom[k][n_dim][deg].prev_basis.active(R[i])]
                        if np.any(prev_active_coordinates):
                            reduction = solve_mod_p(
                                Im_Hom, prev_active_coordinates,
                                self.p)[Im_dim:]
                            gammas = reduction[:Im_dim] #modifiy further total complex
                            active_coordinates = reduction[Im_dim:]
                            lifted_coordinates[self.Hom[k][n_dim][deg].active(
                                R[i])] = active_coordinates
                        # end if
                    # end if
                    target_coordinates[i] = lifted_coordinates
                # end if
            # end for
        # end for
        return coefficients, lift_coefficients


    ###########################################################################
    # add higher page contents

    def add_output_higher(self, Hom, Im, PreIm, end_n_dim, end_deg,
                          current_page):
        """Stores higher page data that has been computed along a sequence of
        consecutive differentials.

        The studied sequence of differentials ends in

            `(end_n_dim, end_deg)`

        coming from

            `(end_n_dim + current_page, end_deg - current_page + 1)`

        and continuing until reaching an integer `r > 0` such that either

                end_n_dim + r * current_page > self.no_columns

        or

                end_deg - r * current_page + 1 > 0

        Parameters
        ----------
        Hom : :obj:`list(barcode_basis)`
            Homology of a sequence of differentials in the spectral sequence.
            This is computed using
            :mod:`permaviss.persistence_algebra.module_persistence_homology`.
        Im : :obj:`list(barcode_basis)`
            Images of a sequence of differentials in the spectral sequence.
        PreIm : :obj:`list(Numpy Array)`
            Preimages of a sequence of differentials in the spectral sequence.
        end_n_dim : int
            Integer specifying the column position where the sequence of
            differentials ends.
        end_deg : int
            Integer specifying the row position where the sequence of
            differentials ends.
        current_page : int
            Current page of the spectral sequence.

        """
        n_dim = end_n_dim
        deg = end_deg
        for i, h in enumerate(Hom):
            self.PreIm[current_page][n_dim][deg] = PreIm[i]
            self.Hom[current_page][n_dim][deg] = h
            self.page_dim_matrix[current_page+1, deg, n_dim] = h.dim
            self.Im[current_page][n_dim][deg] = Im[i]
            deg = deg - current_page + 1
            n_dim += current_page

    ###########################################################################
    # zig_zag

    def zig_zag(self, n_dim, deg, current_page, lift_sum=True,
                initial_sum=[], store_reps=False):
        """Computes the image of the differential of the spectral sequence
        applied to a matrix of coordinates.

        Given an array of coordinates, computes the image under the
        differential in the current page. If no coordinates are given in
        initial_sum, we compute the image of all the generators on the position
        (deg, n_dim) of the `current_page`-page of the spectral sequence.

        Parameters
        ----------
        n_dim : int
            Column in page.
        deg : int
            Row in page
        current_page : int
            Current page in spectral sequence
        lift_sum : bool, default is True
            Whether we return a lifted sum on current_page or we return an
            expression on the 0 page instead.
        initial_sum : :obj:`list`, default is []
            Expression for which we want to compute zig-zag. Coordinates are
            stored on each column and are given in terms of the basis stored in
            `Hom` for the position `(deg, n_dim)` in `current_page -1`. If this
            is not given, we set it to be the identity of size equal to the
            dimension of the considered position on the spectral sequence.
        store_reps : bool,  default is False.
            Whether we want to store total complex representatives.

        Returns
        -------
        :obj:`Numpy Array`, if `lift_sum==False` then returns :obj:`list(dict)`
            Coordinates for images. Each column stores the image by the
            `current_page` differential applied to the corresponding column in
            `initial_sum`. These coordinates are given in terms of the basis
            stored in `Hom` for the position `(deg, n_dim)` for the page
            `current_page - 1`. If `lift_sum` is set up to False, then this is a
            :obj:`list` where each entry stores a cochain. Each cochain is a
            dictionary :obj:`dict`, where entries are indexed by covering
            simplices, and these contain the local coordinates.

        """
        if len(initial_sum) == 0:
            initial_sum = np.identity(self.page_dim_matrix[current_page, deg,
                                                           n_dim])
        # end if

        # Save initial positions if we store reps
        if store_reps:
            start_n_dim = n_dim
            start_deg = deg
        # end if

        # Handle trivial arrows
        if self.page_dim_matrix[current_page, deg, n_dim] == 0:
            return np.array([])
        # end if

        # Load expression to zero page
        zero_coordinates, R = self.load_to_zero_page(initial_sum, n_dim, deg,
                                                     current_page)
        if store_reps:
            self.tot_complex_reps[start_n_dim][start_deg].append(
                zero_coordinates)
            if n_dim == 0:
                return
            # end if
        # end if

        # Compute zigzag for the zero_page expression
        zero_coordinates = self.cech_differential(zero_coordinates, n_dim, deg)

        n_dim -= 1
        for k in range(current_page - 1):
            zero_coordinates = self.lift_preimage(zero_coordinates,
                                                  R,  n_dim, deg)
            deg += 1
            if store_reps:
                self.tot_complex_reps[start_n_dim][start_deg].append(
                    zero_coordinates)
                if n_dim == 0:
                    return

            zero_coordinates = self.cech_differential(zero_coordinates,
                                                      n_dim, deg)
            n_dim -= 1
            # after this even though the class of zero_coordinates is zero
            # in current_page, this does not directly imply that these
            # can be lifted through the vertical differential. We adjust.
            if k < current_page - 2:
                self.liftable_0_class(zero_coordinates, R, n_dim, deg,
                                      current_page, store_reps,
                                      start_n_dim=start_n_dim,
                                      start_deg=start_deg)

        # Lift back to current_page
        if lift_sum:
            target_coordinates = self.lift_to_page(
                zero_coordinates, R,  n_dim, deg, current_page)
        else:
            target_coordinates = zero_coordinates

        return target_coordinates

    ###########################################################################
    # vert_image

    def vert_image(self, initial_sum, n_dim, deg):
        """Given an expression on zero page, computes the image under the
        vertical differentials.

        Parameters
        ----------
        initial_sum : :obj:`list(dict)`
            Coordinates of chains stored as a dictionary. Each dictionary entry
            corresponds to a `n_dim` simplex `s` on the covering nerve, and
            this corresponds to an intersection of covers `K_s`. On each such
            entry, there is a :obj:`Numpy Array` matrix storing the local
            coordinates for each chain. More precisely, the rows correspond to
            different cochains while the columns correspond `deg` simplices
            contained in `K_s`.

        """
        result = []
        for i, rep in enumerate(initial_sum):
            result.append({})
            for nerv_spx in iter(rep):
                image = multiply_mod_p(
                    self.zero_differentials[n_dim][nerv_spx][deg],
                    rep[nerv_spx], self.p)
                if np.any(image):
                    result[i][nerv_spx] = image
                # end if
            # end for
        # end for
        return result

    ###########################################################################
    # load_to_zero_page

    def load_to_zero_page(self, initial_sum, n_dim, deg, current_page):
        """Given an expression on the nth page of the spectral sequence, we
        pick a representative on the 0th page and return it.

        Parameters
        ----------
        initial_sum : :obj:`Numpy Array`
            Matrix storing the coordinates of different elements in terms of
            the basis stored in `Hom` at `(deg, n_dim)` and on the page
            `current_page`. Different element coordinates are on each column,
            while the basis corresponds to rows in the matrix.
        n_dim : int
            Column position on page.
        deg : int
            Row position on page
        current_page : int
            Current page number

        Returns
        -------
        target_coordinates : :obj:`list(dict)`
            List where each entry contains a dictionary storing cochains on
            zero page. Each dictionary entry corresponds to a `n_dim` simplex
            `s` on the covering nerve, and this contains the local coordinates
            for each cochain.
        R : :obj:`Numpy Array`
            1D array storing the radii of birth for each cochain stored in
            `target_coordinates`.

        """
        # Load sum to 1st page
        aux_sum = initial_sum
        for k in range(current_page-1, 0, -1):
            aux_sum = multiply_mod_p(self.Hom[k][n_dim][deg].coordinates,
                                     aux_sum, self.p)
        # end for

        # Load sum from 1 to 0 page
        # Transform coordinates to dictionary data types
        target_coordinates = []
        for i, coord in enumerate(aux_sum.T):
            target_coordinates.append({})
            for nerve_spx in range(len(self.cycle_dimensions[n_dim][deg])-1):
                local_coordinates = coord[
                    self.cycle_dimensions[n_dim][deg][nerve_spx-1]:
                    self.cycle_dimensions[n_dim][deg][nerve_spx]]

                # If the local coordinates are nontrivial, add them to
                # dictionary.
                if len(local_coordinates) > 0 and np.any(local_coordinates):
                    target_coordinates[i][nerve_spx] = multiply_mod_p(
                        self.Hom[0][n_dim][nerve_spx][deg].coordinates,
                        local_coordinates, self.p)
                # end if
            # end for
        # end for
        # Get initial radius of barcodes
        R = np.zeros(int(self.page_dim_matrix[current_page, deg, n_dim]))
        if current_page == 1:
            for i, coord in enumerate(initial_sum.T):
                for nerve_spx in range(len(self.cycle_dimensions[n_dim][
                        deg]) - 1):
                    local_coordinates = coord[self.cycle_dimensions[n_dim][
                        deg][nerve_spx - 1]: self.cycle_dimensions[
                            n_dim][deg][nerve_spx]]
                    # Take the maximum of birth radius over all nontrivial
                    # local coordinates.
                    if len(local_coordinates) > 0 and np.any(
                            local_coordinates):
                        R[i] = max(R[i], self.Hom[0][n_dim][nerve_spx][
                            deg].birth_radius(local_coordinates))
                    # end if
                # end for
            # end for
        else:
            for i, coord in enumerate(initial_sum.T):
                R[i] = self.Hom[current_page - 1][n_dim][
                    deg].birth_radius(coord)
            # end for
        # end else

        return target_coordinates, R

    ###########################################################################
    # cech_differential (copy of original method)

    def cech_differential(self, start_coordinates, n_dim, deg):
        """Computes the Cech Differential of various cochains.

        That is, given a few cochains on the position `(deg, n_dim)` of the
        `0` page, we compute the Cech differential. This leads to cochains on
        `(deg, n_dim - 1)`.

        Parameters
        ----------
        start_coordinates : :obj:`list(dict)`
            List where each entry contains a dictionary storing the
            coordinates of a cochain in `(n_dim, deg)`. Each dictionary key
            corresponds to a `n_dim` simplex `s` on the covering nerve, and
            this contains the local coordinates for each cochain.
        n_dim : int
            Column on page
        deg : int
            Row on page

        Returns
        -------
        target_coordinates : :obj:`list(dict)`
            Cech differential images of `start_coordinates`. These are stored
            as a list of cochains in `(n_dim-1, deg)` with the same format as
            `start_coordinates`

        """

        target_coordinates = []
        deg_sign = (-1)**deg
        # go through all dictionaries in input
        for i, coordinates in enumerate(start_coordinates):
            cech_image = {}
            # look at each nerve simplex on each dictionary
            for nerve_spx_index in iter(coordinates):
                points_IN = np.copy(self.points_IN[n_dim][nerve_spx_index])
                # Iterate over simplices in boundary of nerve simplex
                nerv_boundary_indx = np.nonzero(self.nerve_differentials[
                    n_dim][:, nerve_spx_index])[0]
                nerv_boundary_coeff = self.nerve_differentials[n_dim][
                    :, nerve_spx_index][nerv_boundary_indx]
                # go through all faces for each simplex in nerve
                for nerve_face_index, nerve_coeff in zip(
                        nerv_boundary_indx, nerv_boundary_coeff):
                    if deg == 0:
                        # inclusions for points
                        # Preallocate space
                        if nerve_face_index not in cech_image:
                            cech_image[nerve_face_index] = np.zeros(
                                self.subcomplexes[n_dim - 1][
                                    nerve_face_index][0])
                        # end if
                        for point_idx, point_coeff in enumerate(coordinates[
                                nerve_spx_index]):
                            if point_coeff != 0:
                                face_point_idx = np.argmax(self.points_IN[
                                    n_dim-1][nerve_face_index] == points_IN[
                                        point_idx])
                                cech_image[nerve_face_index][
                                    face_point_idx
                                    ] += nerve_coeff * point_coeff * deg_sign
                                cech_image[nerve_face_index][
                                    face_point_idx] %= self.p
                            # end if
                        # end for
                    else:
                        # inclusions for edges, 2-simplices and higher
                        # simplices. Preallocate space as well.
                        if nerve_face_index not in cech_image:
                            cech_image[nerve_face_index] = np.zeros(len(
                                self.subcomplexes[n_dim - 1][nerve_face_index][
                                    deg]))
                        # end if
                        # Iterate over nontrivial local simplices in domain
                        spx_indices = np.nonzero(coordinates[
                            nerve_spx_index])[0]
                        spx_coefficients = coordinates[nerve_spx_index][
                            spx_indices]
                        for spx_index, spx_coeff in zip(
                                spx_indices, spx_coefficients):
                            # Obtain IN for vertices of simplex
                            vertices_spx = points_IN[self.subcomplexes[n_dim][
                                nerve_spx_index][deg][spx_index]]
                            # Iterate over simplices in range to see which
                            # one has vertices_spx as vertices.
                            for im_indx, im_spx in enumerate(
                                    self.subcomplexes[n_dim-1][
                                        nerve_face_index][deg]):
                                vertices_face = self.points_IN[n_dim-1][
                                    nerve_face_index][im_spx.astype(int)]
                                # When the vertices coincide, break the loop
                                if len(np.intersect1d(
                                        vertices_spx,
                                        vertices_face)) == deg + 1:
                                    cech_image[nerve_face_index][im_indx] += \
                                        spx_coeff * nerve_coeff * deg_sign
                                    cech_image[nerve_face_index][im_indx] %= \
                                        self.p
                                    break
                                # end if
                            # end for
                        # end for
                    # end else
                # end for
            # end for
            target_coordinates.append(cech_image)
        # end for

        return target_coordinates

    ###########################################################################
    # cech_differential (new parallel method)

    def cech_differential_new(self, local_coordinates, reference_list, n_dim, deg):
        """Computes the Cech Differential of various cochains.

        That is, given a few cochains on the position `(deg, n_dim)` of the
        `0` page, we compute the Cech differential. This leads to cochains on
        `(deg, n_dim - 1)`.

        Parameters
        ----------
        start_coordinates : localized coordinates + refs
        n_dim : int
            Column on page
        deg : int
            Row on page

        Returns
        -------
        target_coordinates : localized coordinates + refs

        """

        target_coordinates = []
        deg_sign = (-1)**deg
        # go through covers on codomain of cech differentials
        # do this step in parallel
        for nerve_spx_index in range(len(self.nerve[n_dim-1])):
            call
        # end for
        # go through all dictionaries in input
        for i, coordinates in enumerate(start_coordinates):
            cech_image = {}
            # look at each nerve simplex on each dictionary
            for nerve_spx_index in iter(coordinates):
                points_IN = np.copy(self.points_IN[n_dim][nerve_spx_index])
                # Iterate over simplices in boundary of nerve simplex
                nerv_boundary_indx = np.nonzero(self.nerve_differentials[
                    n_dim][:, nerve_spx_index])[0]
                nerv_boundary_coeff = self.nerve_differentials[n_dim][
                    :, nerve_spx_index][nerv_boundary_indx]
                # go through all faces for each simplex in nerve
                for nerve_face_index, nerve_coeff in zip(
                        nerv_boundary_indx, nerv_boundary_coeff):
                    if deg == 0:
                        # inclusions for points
                        # Preallocate space
                        if nerve_face_index not in cech_image:
                            cech_image[nerve_face_index] = np.zeros(
                                self.subcomplexes[n_dim - 1][
                                    nerve_face_index][0])
                        # end if
                        for point_idx, point_coeff in enumerate(coordinates[
                                nerve_spx_index]):
                            if point_coeff != 0:
                                face_point_idx = np.argmax(self.points_IN[
                                    n_dim-1][nerve_face_index] == points_IN[
                                        point_idx])
                                cech_image[nerve_face_index][
                                    face_point_idx
                                    ] += nerve_coeff * point_coeff * deg_sign
                                cech_image[nerve_face_index][
                                    face_point_idx] %= self.p
                            # end if
                        # end for
                    else:
                        # inclusions for edges, 2-simplices and higher
                        # simplices. Preallocate space as well.
                        if nerve_face_index not in cech_image:
                            cech_image[nerve_face_index] = np.zeros(len(
                                self.subcomplexes[n_dim - 1][nerve_face_index][
                                    deg]))
                        # end if
                        # Iterate over nontrivial local simplices in domain
                        spx_indices = np.nonzero(coordinates[
                            nerve_spx_index])[0]
                        spx_coefficients = coordinates[nerve_spx_index][
                            spx_indices]
                        for spx_index, spx_coeff in zip(
                                spx_indices, spx_coefficients):
                            # Obtain IN for vertices of simplex
                            vertices_spx = points_IN[self.subcomplexes[n_dim][
                                nerve_spx_index][deg][spx_index]]
                            # Iterate over simplices in range to see which
                            # one has vertices_spx as vertices.
                            for im_indx, im_spx in enumerate(
                                    self.subcomplexes[n_dim-1][
                                        nerve_face_index][deg]):
                                vertices_face = self.points_IN[n_dim-1][
                                    nerve_face_index][im_spx.astype(int)]
                                # When the vertices coincide, break the loop
                                if len(np.intersect1d(
                                        vertices_spx,
                                        vertices_face)) == deg + 1:
                                    cech_image[nerve_face_index][im_indx] += \
                                        spx_coeff * nerve_coeff * deg_sign
                                    cech_image[nerve_face_index][im_indx] %= \
                                        self.p
                                    break
                                # end if
                            # end for
                        # end for
                    # end else
                # end for
            # end for
            target_coordinates.append(cech_image)
        # end for

        return target_coordinates

    ###########################################################################
    # lift_to_page

    def lift_to_page(self, start_coordinates, R,  n_dim, deg, target_page):
        """Lifts some zero page element to a target page.

        This is, in a way, opposite to
        :meth:`permaviss.spectral_sequence.spectral_sequence_class.load_to_zero_page`.
        More precisely, given a few cochains in `start_coordinates`, we compute
        their respective classes on `target_page`.

        Parameters
        ----------
        start_coordinates : :obj:`list(dict)`
            List where each entry contains a dictionary storing cochains on
            zero page. Each dictionary entry corresponds to a `n_dim` simplex
            `s` on the covering nerve, and this contains the local coordinates
            for each cochain.
        R : :obj:`Numpy Array`
            1D array storing the radii of birth for each cochain stored in
            `start_coordinates`.
        n_dim : int
            Column on page
        deg : int
            Row on page
        target_page : int
            Page to which we want to lift `start_coordinates`.

        Returns
        -------
        target_coordinates : :obj:`Numpy Array`
            Matrix storing the coordinates for each class corresponding to each
            cochain in `start_coordinates`. Rows correspond to the number of
            classes, while the columns correspond to the dimension of
            `Hom[n_dim][deg][target_page]`.

        """
        # Lift from 0 to 1 page
        # Transform from dictionary to np.array
        target_coordinates = np.zeros((len(start_coordinates),
                                       self.page_dim_matrix[1, deg, n_dim]))
        for i, coord in enumerate(start_coordinates):
            for nerve_spx_index in iter(coord):
                # solve (Im|Hom) locally
                # This is with respect to the active coordinates at R[i]
                Hom_dim = self.Hom[0][n_dim][nerve_spx_index][deg].dim
                if Hom_dim > 0:
                    Hom = self.Hom[0][n_dim][nerve_spx_index][
                        deg].active_domain(R[i])
                Im_dim = self.Im[0][n_dim][nerve_spx_index][deg].dim
                if Hom_dim > 0 and len(Hom) > 0:
                    if Im_dim > 0:
                        Im = self.Im[0][n_dim][nerve_spx_index][
                            deg].active_domain(R[i])
                        Im_dim = np.size(Im, 1)
                        Im_Hom = np.append(Im, Hom, axis=1)

                    else:
                        Im_Hom = Hom
                    # end else

                    active_coordinates = solve_mod_p(
                        Im_Hom, coord[nerve_spx_index], self.p)[Im_dim:]
                    # Save resulting coordinates as 1 page coordinates
                    local_lifted_coordinates = np.zeros(self.cycle_dimensions[
                        n_dim][deg][nerve_spx_index] - self.cycle_dimensions[
                            n_dim][deg][nerve_spx_index-1])
                    local_active = self.Hom[0][n_dim][nerve_spx_index][
                        deg].active(R[i])
                    local_lifted_coordinates[local_active] = active_coordinates
                    target_coordinates[i, self.cycle_dimensions[n_dim][deg][
                        nerve_spx_index-1]:self.cycle_dimensions[n_dim][deg][
                            nerve_spx_index]] = local_lifted_coordinates
                # end if
            # end for
        # end for

        # Lift from 1 to target page
        for k in range(1, target_page):
            prev_target_coordinates = target_coordinates
            target_coordinates = np.zeros((
                len(start_coordinates), self.page_dim_matrix[k+1, deg, n_dim]))
            for i, coord in enumerate(prev_target_coordinates):
                # solve (Im|Hom) locally
                Hom_dim = self.Hom[k][n_dim][deg].dim
                if Hom_dim > 0:
                    Hom = self.Hom[k][n_dim][deg].active_coordinates(R[i])

                Im_dim = self.Im[k][n_dim][deg].dim
                if Hom_dim > 0 and len(Hom) > 0:
                    if Im_dim > 0:
                        Im = self.Im[k][n_dim][deg].active_coordinates(R[i])
                        Im_dim = np.size(Im, 1)
                        Im_Hom = np.append(Im, Hom, axis=1)
                    else:
                        Im_Hom = Hom
                    # end else
                    lifted_coordinates = np.zeros(self.Hom[k][n_dim][deg].dim)
                    if np.size(Im_Hom, 1) > 0:
                        prev_active_coordinates = prev_target_coordinates[i][
                            self.Hom[k][n_dim][deg].prev_basis.active(R[i])]
                        if np.any(prev_active_coordinates):
                            active_coordinates = solve_mod_p(
                                Im_Hom, prev_active_coordinates,
                                self.p)[Im_dim:]
                            lifted_coordinates[self.Hom[k][n_dim][deg].active(
                                R[i])] = active_coordinates
                        # end if
                    # end if
                    target_coordinates[i] = lifted_coordinates
                # end if
            # end for
        # end for

        return target_coordinates

    ###########################################################################
    # lift_preimage

    def lift_preimage(self,  start_coordinates, R, n_dim, deg, sign=-1):
        """Given a zero page element, we lift it using the vertical
        differentials.

        Additionally, we multiply the lift by -1 mod self.p
        In this method we assume that this can be done.

        Parameters
        ----------
        start_coordinates : :obj:`list(dict)`
            These are the cochains on the `(deg, n_dim)` position that we want
            to lift through the vertical differential. List containing cochains
            on `(deg, n_dim)`. Each entry contains the coefficients of a
            cochain as a dictionary indexed by `n_dim` simplices on the nerve
            of the cover.
        R : :obj:`Numpy Array`
            1D array storing the radii of birth for each cochain stored in
            `start_coordinates`.
        n_dim : int
            Column on page
        deg : int
            Row on page

        Returns
        -------
        lifted_coordinates : :obj:`list(dict)`
            These are the cochains on the `(deg+1, n_dim)` position that we
            want to lift through the vertical differential. List containing
            cochains on `(deg+1, n_dim)`. Each entry contains the coefficients
            of a cochain as a dictionary indexed by `n_dim` simplices on the
            nerve of the cover.

        """
        lifted_coordinates = []
        for i, coord in enumerate(start_coordinates):
            lifted_coordinates.append({})
            for nerve_spx_index in iter(coord):
                # Only lift nontrivial expressions
                if np.any(coord[nerve_spx_index]):
                    Im = self.Im[0][n_dim][nerve_spx_index][
                        deg].active_domain(R[i])
                    solution = solve_mod_p(Im, coord[nerve_spx_index], self.p)
                    local_lift = multiply_mod_p(
                        self.PreIm[0][n_dim][nerve_spx_index][deg+1][
                            :, self.Im[0][n_dim][nerve_spx_index][
                                deg].active(R[i])],
                        solution, self.p)
                    # lifted coordinates in terms of all coordinates
                    lifted_coordinates[i][
                        nerve_spx_index] = sign * local_lift % self.p
                # end if
            # end for
        # end for

        return lifted_coordinates

    ###########################################################################
    # extension

    def extension(self, start_n_dim, start_deg):
        """Generate extension block matrices for given n_dim and deg

        Parameters
        ----------
        start_n_dim : int
            Column on page where we want to compute the extension coefficients.

        start_deg : int
            Row on page where we want to compute the extension coefficients.

        Stores
        ------
        extensions : :obj:`list(Numpy Array)`
            Extension matrices for the basis contained at `(start_deg,
            start_n_dim)` of the infinity page. More precisely, given an
            integer `ext_dim`, the entry `extension[ext_dim]` stores a
            :obj:`Numpy Array` matrix where the `i` column corresponds to the
            `i` generator in the position (start_deg, start_n_dim) from the
            infinity page. Each such column contains the extension
            coefficients in terms of the generators from the
            `(deg + ext_deg, n_dim - ext_deg)` entry on the infinity page.
            Recall that each slope -1 diagonal on the infinity page corresponds
            to a `broken basis` for the persistent homology. Then, each of the
            matrices in `extension_matrices` corresponds to a block in the
            `broken differentials`. Notice that the order in which we store
            columns and rows differs from all the previous attributes except
            page_dim_matrix.

        """
        extensions = self.extensions[start_n_dim][start_deg]
        death_R = self.Hom[self.no_pages-1][start_n_dim][
            start_deg].barcode[:, 1]
        dim_domain = self.Hom[self.no_pages-1][start_n_dim][start_deg].dim
        not_infty = death_R < self.max_r
        death_R = death_R[not_infty]
        not_infty = np.sort(np.nonzero(not_infty)[0])
        dim_ext = len(not_infty)
        # Take only the representatives that are not infinity
        total_representatives = self.tot_complex_reps[start_n_dim][start_deg]
        representatives = []
        for i, zero_coord in enumerate(total_representatives):
            representatives.append([
                copy_dictionary(zero_coord[i]) for i in not_infty])

        # Start in specified entry, and go backwards
        ext_deg = 0
        deg = start_deg
        for n_dim in range(start_n_dim, -1, -1):
            # find and store extension coefficients
            extension = np.zeros((self.page_dim_matrix[
                self.no_pages-1][deg][n_dim], dim_domain))
            if len(not_infty) > 0:
                extension_coefficients = self.lift_to_page(
                    representatives[ext_deg], death_R, n_dim, deg,
                    self.no_pages-1)
                extension[:, not_infty] = extension_coefficients.T
                # change total chain representatives accordingly
                current_representatives = self.tot_complex_reps[n_dim][deg]
                for i in range(dim_ext):
                    # subtract sum of coefficients along all total complex,
                    # starting at n_dim, deg.
                    for k, rep in enumerate(current_representatives):
                        substractor = add_dictionaries(
                            extension_coefficients[i], rep, self.p)
                        representatives[k+ext_deg][i] = add_dictionaries(
                            [1, -1],
                            [representatives[k+ext_deg][i], substractor],
                            self.p)
                    # end for
                # end for
                # subtract images to representatives, so that we can lift the
                # vertical differentials.
                if n_dim > 0:
                    # subtract images of differentials so that we can lift
                    self.liftable_0_class(representatives[ext_deg], death_R,
                                          n_dim, deg, self.no_pages - 1, False)
                    # lift through vertical differential the last nontrivial
                    # entry in total complex diagonal.
                    preimages_sums = self.lift_preimage(representatives[
                        ext_deg], death_R, n_dim, deg)
                    representatives[ext_deg] = {}
                    # compute the cech differential for preimage_sums
                    cech_images = self.cech_differential(preimages_sums, n_dim,
                                                         deg + 1)
                    # subtract cech differential of preimage
                    for i in range(dim_ext):
                        representatives[ext_deg + 1][i] = add_dictionaries([
                            1, 1], [representatives[ext_deg + 1][i],
                                    cech_images[i]], self.p)
                    # end for
                # end if
            # end if
            extensions.append(extension)
            deg += 1
            ext_deg += 1
        # end for

    ###########################################################################
    # liftable_0_class

    def liftable_0_class(self, start_coordinates, R, n_dim, deg, target_page,
                         store_reps, start_n_dim=None, start_deg=None):
        """ Given cochains on the `0`-page, and assuming that they are zero
        when lifted to `target_page`, we get representatives that are
        equivalent on `target_page - 1` and also lift through the vertical
        differential.

        Parameters
        ----------
        start_coordinates : :obj:`list(dict)`
            These are the cochains on the `(deg, n_dim)` position that we want
            to lift through the vertical differential. List containing cochains
            on `(deg, n_dim)`. Each entry contains the coefficients of a
            cochain as a dictionary indexed by `n_dim` simplices on the nerve
            of the cover. This is modified so that it can be lifted.
        R : :obj:`Numpy Array`
            1D array storing the radii of birth for each cochain stored in
            `start_coordinates`
        n_dim : int
            Column on page
        deg : int
            Row on page
        target_page : int
            Page of spectral sequence at which the given classes vanish.

        Returns
        -------
        liftable_coordinates : :obj:`list(dict)`
            Cochains on the `(deg, n_dim)` position whose classes on
            `target_page` are equivalent to those of `start_coordinates`.
            Furthermore, these can also be lifted through vertical
            differentials.

        """
        dim_reps = len(start_coordinates)
        for page in range(target_page - 1, 0, -1):
            image_coefficients = self.image_coordinates(start_coordinates, R,
                                                        n_dim, deg, page)
            # If image in position is nontrivial
            if len(image_coefficients) > 0 and np.any(image_coefficients):
                preimages_sums = multiply_mod_p(
                    image_coefficients,
                    self.PreIm[page][n_dim+page][deg-page+1].T,
                    self.p)
                # subtract images to zero page sums
                if np.any(preimages_sums):
                    images = self.zig_zag(
                        n_dim+page, deg-page+1, page,
                        lift_sum=False,
                        initial_sum=preimages_sums.T)
                    preimage_zeros, _ = self.load_to_zero_page(
                        preimages_sums.T, n_dim+page, deg-page+1, page)
                    tot_complex_reps = self.tot_complex_reps[
                        start_n_dim][start_deg][deg - page + 1 - start_deg]
                    for i in range(dim_reps):
                        start_coordinates[i] = add_dictionaries(
                            [1, -1], [start_coordinates[i], images[i]],
                            self.p)
                        if store_reps:
                            tot_complex_reps[i] = add_dictionaries(
                                [1, -1],
                                [tot_complex_reps[i], preimage_zeros[i]],
                                self.p)

                    # end for
                # end if
            # end if
        # end for

    ###########################################################################
    # image_coordinates

    def image_coordinates(self, start_coordinates, R, n_dim, deg,
                          target_page):
        """Given cochains on the `0`-page, we compute the coordinates of their
        persistent Homology classes in terms of the basis in
        `Im[target_page][n_dim][deg]`

        Parameters
        ----------
        start_coordinates : :obj:`list(dict)`
            These are the cochains on the `(deg, n_dim)` position that we want
            to lift through the vertical differential. List containing cochains
            on `(deg, n_dim)`. Each entry contains the coefficients of a
            cochain as a dictionary indexed by `n_dim` simplices on the nerve
            of the cover.
        R : :obj:`Numpy Array`
            1D array storing the radii of birth for each cochain stored in
            `start_coordinates`
        n_dim : int
            Column on page
        deg : int
            Row on page
        target_page : int
            Page of spectral sequence at which we want to solve the image
            equation.

        Returns
        -------
        image_coordinates : :obj:`Numpy Array`
            Start_coordinates classes in target_page written in terms of
            image of differential. Rows correspond to different vectors
            and columns correspond to the dimension of the image.

        Raises
        ------
        ValueError
            If the classes of the cochains are not contained in the images.

        """
        # Lift start_coordinates from 0 to target_page
        lifted_sum = self.lift_to_page(start_coordinates, R, n_dim, deg,
                                       target_page)

        # If image is trivial, return
        Im_dim = self.Im[target_page][n_dim][deg].dim
        if Im_dim == 0:
            return []

        # Compute each coordinate in terms of images
        image_coordinates = np.zeros((len(start_coordinates), Im_dim))
        for i, rad in enumerate(R):
            Im = self.Im[target_page][n_dim][deg].active_coordinates(rad)
            active_lifted_sum = lifted_sum[i][self.Im[target_page][n_dim][
                deg].prev_basis.active(rad)]
            if len(active_lifted_sum) > 0:
                image_coordinates[i, self.Im[target_page][n_dim][
                    deg].active(rad)] = solve_mod_p(
                        Im, active_lifted_sum, self.p)

        return image_coordinates
    # end def
# end spectral_sequence class

###############################################################################
# add_dictionaries
#
# This mainly supports extension. Does what its name says.
# That is, entries with the same key are added.


def add_dictionaries(coefficients, representatives, p):
    """ Computes a dictionary that is the linear combination of `coefficients`
    on `representatives`

    Parameters
    ----------
    coefficients : :obj:`Numpy Array`
        1D array with the same number of elements as `representatives`. Each
        entry is an integer mod p.
    representatives : :obj:`list(dict)`
        List where each entry is a dictionary. The keys on each dictionary are
        integers, and these might coincide with dictionaries on other entries.
    p : int(prime)

    Returns
    -------
    rep_sum : :obj:`dict`
        Result of adding the dictionaries on `representatives` with
        `coefficients`.

    Example
    -------
        >>> import numpy as np
        >>> p=5
        >>> coefficients = np.array([1,2,3])
        >>> representatives = [
        ... {0:np.array([1,3]), 3:np.array([0,0,1])},
        ... {0:np.array([4,3]),2:np.array([4,5])},
        ... {3:np.array([0,4,0])}]
        >>> add_dictionaries(coefficients, representatives, p)
        {0: array([4, 4]), 3: array([0, 2, 1]), 2: array([3, 0])}


    """
    rep_sum = {}
    for i, rep in enumerate(representatives):
        for spx_idx in iter(rep):
            if spx_idx not in rep_sum:
                rep_sum[spx_idx] = (coefficients[i] * rep[spx_idx]) % p
            else:
                rep_sum[spx_idx] = (rep_sum[spx_idx] + coefficients[i] * rep[
                    spx_idx]) % p
            # end else
        # end for
    # end for
    # Find simplices where expression is zero
    zero_simplices = []
    for spx_idx in iter(rep_sum):
        if not np.any(rep_sum[spx_idx]):
            zero_simplices.append(spx_idx)
        # end if
    # end for
    # If an entry is zero,  delete it
    for spx_idx in zero_simplices:
        del rep_sum[spx_idx]
    # end for

    return rep_sum


###############################################################################
# copy_dictionary
#

def copy_dictionary(original):
    """Copies a dictionary where each entry is a :obj:`Numpy Array`
    """
    copy = {}
    for spx_idx in iter(original):
        copy[spx_idx] = np.copy(original[spx_idx])

    return copy

###############################################################################
# add local_coordinates
#
#
#def add_local_coordinates(A_ref, A_coord, B_ref, B_coord):
#    """Obtains local coordinate representation for A + B, given local coordinate
#    representations of A and B
#    """
#    if len(A_ref != B_ref):
#        raise ValueError
#
#    return sum_ref, sum_coord

###############################################################################
# local_sums
#

def local_sums(local_chains, sums):
    """Sum local_chains as indicated by "sums"
    """
    no_sums = np.size(sums, axis=0)
    no_generators = np.size(sums, 1)
    new_chains = []
    for pair in local_chains:
        old_ref = pair[0]
        old_coord = pair[1]
        new_ref = []
        new_coord = []
        for local_idx, ref in enumerate(old_ref):
            new_ref.append([])
            new_coord.append([])
            for idx_sum, sum in enumerate(sums):
                generators = np.nonzero(sum)[0]
                active_local_generators = np.where(np.in1d(
                    ref, generators,))[0]
                if np.any(active_local_generators):
                    coefficients = np.array([sum[active_local_generators]])
                    new_ref[local_idx].append(idx_sum)
                    new_coord[local_idx].append(np.matmul(
                        coefficients, old_coord[local_idx][
                            active_local_generators])[0]) # 2d array to 1d array
                # end if
            # end for
            new_coord[local_idx] = np.array(new_coord[local_idx])
        # end for
        new_chains.append([new_ref, new_coord])
    # end for
    return new_chains
