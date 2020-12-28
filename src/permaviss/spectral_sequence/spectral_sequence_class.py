#
#    spectral_sequence_class.py
#

import numpy as np

from multiprocessing import Pool
from functools import partial

from ..simplicial_complexes.differentials import complex_differentials
from ..gauss_mod_p.functions import solve_mod_p, multiply_mod_p
from ..gauss_mod_p.gauss_mod_p import gauss_col_rad, gauss_barcodes

from ..persistence_algebra.barcode_bases import barcode_basis


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
        self.Hom_reps = [[]]
        self.Im_reps = [[]]
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
            self.Hom_reps.append([])
            self.Im_reps.append([])
            for n_dim in range(self.no_columns):
                self.Hom[k].append([])
                self.Im[k].append([])
                self.PreIm[k].append([])
                self.Hom_reps[k].append([[]])
                self.Im_reps[k].append([[]])
                for deg in range(self.no_rows):
                    self.Hom[k][n_dim].append(barcode_basis([]))
                    self.Im[k][n_dim].append([])
                    self.PreIm[k][n_dim].append([])
                    self.Hom_reps[k][n_dim].append([])
                    self.Im_reps[k][n_dim].append([])

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
    # self.first_differential(self, n_dim, deg, current_page):

    def first_differential(self, n_dim, deg, current_page):
        """ Compute first differential from (n_dim, deg)

        """
        # handle trivial cases
        if self.page_dim_matrix[current_page, deg, n_dim] == 0:
            return np.array([])
        # generate refs and coordinates of current page
        # store as total complex
        # compute array of initial radii
        R = np.zeros(self.page_dim_matrix[1, deg, n_dim])
        # birth radii and localized_coordinates for classes
        references = []
        local_coordinates = []
        prev = 0
        for nerve_spx_index, next in enumerate(
                self.cycle_dimensions[n_dim][deg][:-1]):
            if prev < next:
                references.append(np.array(range(prev, next)))
                local_coordinates.append((self.Hom[0][n_dim][nerve_spx_index][
                    deg].coordinates).T)
                R[prev:next] = self.Hom[0][n_dim][nerve_spx_index][
                    deg].barcode[:,0]
                prev = next
            else:
                references.append([])
                local_coordinates.append([])
        # end for
        chains = [references, local_coordinates]
        # call cech_diff_and_lift
        Betas, _ = self.cech_diff_and_lift(n_dim, deg, chains, R)
        return  Betas


    ###########################################################################
    # self.high_differential(self, n_dim, deg, current_page):

    def high_differential(self, n_dim, deg, current_page):
        """ Compute differential of spectral sequence starting on position
        (n_dim, deg) in current_page >= 2.

        """
        # handle trivial case
        if self.Hom[current_page-1][n_dim][deg].dim == 0:
            return np.array([])

        # take last total complex entry of Hom reps
        chains = self.Hom_reps[current_page-1][n_dim][deg][current_page-1]
        Hom_barcode = self.Hom[current_page-1][n_dim][deg].barcode
        # differential (n_dim, deg) --> (Sn_dim, Sdeg)
        Sn_dim = n_dim - current_page
        Sdeg = deg + current_page - 1
        Betas, _ = self.cech_diff_and_lift(Sn_dim + 1, Sdeg, chains,
                                           Hom_barcode[:,0])

        Betas, _ = self.lift_to_page(Sn_dim, Sdeg, current_page, Betas,
                                     Hom_barcode)
        return  Betas
    # end high_differential

    ###########################################################################
    # self.lift_to_page(self, n_dim, deg, page, chains):

    def lift_to_page(self, n_dim, deg, target_page, Betas, Beta_barcode):
        """ Lifts chains in position (n_dim, deg) from page 1 to target_page.

        Returns Betas and image coordinates.
        """
        Betas = Betas.T
        # lift up to current_page
        for k in range(1, target_page):
            Im = self.Im[k][n_dim][deg]
            Im_dim = self.Im[k][n_dim][deg].dim
            Hom = self.Hom[k][n_dim][deg]
            Hom_dim = self.Hom[k][n_dim][deg].dim
            if Hom_dim > 0:
                if Im_dim > 0:
                    Im_Hom = np.append(Im.coordinates, Hom.coordinates, axis=1)
                    barcode_col = np.append(Im.barcode, Hom.barcode, axis=0)
                else:
                    Im_Hom = Hom.coordinates
                    barcode_col = Hom.barcode
            else:
                Betas = np.array([])
                break

            start_index = Im_dim + Hom_dim
            # add radii coordinates and radii
            barcode_col = np.append(barcode_col, Beta_barcode, axis=0)
            A = np.append(Im_Hom, Betas, axis=1)
            # barcodes of rows
            barcode_row = self.Hom[k][n_dim][deg].prev_basis.barcode
            # gaussian reduction on matrix between persistence modules
            _, T = gauss_barcodes(A, barcode_row, barcode_col, start_index,
                                  self.p)
            T = T[:, start_index:]
            # next page coefficients
            Gammas = T[:Im_dim]
            Betas = T[Im_dim:start_index]
        # end for
        return  Betas.T, Gammas.T
    # end lift_to_page

    ###########################################################################
    # self.cech_diff_and_lift

    def cech_diff_and_lift(self, n_dim, deg, chains, R):
        """Given chains in position (n_dim, deg), computes horizontal
        differential followed by lift by vertical differential.

        Procedure:
        (1) take chains in position (n_dim, deg)
        (2) compute the Cech differential of these chains. We do this in
        parallel over the covers in (n_dim-1, deg)
        (3) Lift locally.
        Steps (2) and (3) are parallelized at the same time.

        Parameters
        ----------
        n_dim, deg, current_page : int, int, int
            Postition on spectral sequence and current page.

        Returns
        -------
        betas : coordinates on first pages
        [lift_references, lift_coordinates]: local coordinates lifted by
            vertical differential.

        """
        # number of simplices to parallelize over in position (n_dim-1, deg)
        if n_dim == 1:
            n_spx_number = self.nerve[0]
        else:
            n_spx_number = len(self.nerve[n_dim-1])

        # store space for preimages
        lift_references = []
        lift_coordinates = []
        # coordinates in first page
        Betas_1_page = np.zeros((
            len(R), self.page_dim_matrix[1, deg, n_dim-1]
            ))
        if len(R)>0:
            for nerve_spx_index in range(n_spx_number):
                lift_references.append([])
                lift_coordinates.append([])

            partial_cech_diff_and_lift_local = partial(
                self.cech_diff_and_lift_local, R, chains[0], chains[1],
                n_dim - 1, deg, lift_references, lift_coordinates, Betas_1_page)
            for idx in range(n_spx_number):
                partial_cech_diff_and_lift_local(idx)
            # workers_pool = Pool()
            # workers_pool.map(partial_cech_diff_and_lift, range(n_spx_number))
            # workers_pool.close()

        return Betas_1_page, [lift_references, lift_coordinates]
    # end cech_diff_and_lift

    #######################################################################
    # Cech chain plus lift of preimage
    def cech_diff_and_lift_local(
            self, R, reference_preimage, local_preimage, n_dim, deg,
            lift_references, lift_coordinates, Betas_1_page,
            nerve_spx_index
            ):
        """ Takes some chains in position (n_dim+1, deg) and computes Cech diff
        followed by a lift by vertical differential. This is done locally at
        cover information in (n_dim, deg).

        parameters
        ----------
        R: vector of radii

        Assume that reference_preimages are indexed from 0 to len(R)-1


        """
        # if nerve_spx_index==0, then prev=0
        prev = self.cycle_dimensions[n_dim][deg][nerve_spx_index-1]
        next = self.cycle_dimensions[n_dim][deg][nerve_spx_index]
        # if trivial cover skip
        if prev == next:
            return
        coboundary = self.nerve_differentials[n_dim + 1][nerve_spx_index]
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
        # if there are no images to compute, return
        if len(generators) == 0:
            return
        # size of local complex
        if deg == 0:
            cpx_size = self.subcomplexes[n_dim][nerve_spx_index][0]
        else:
            cpx_size = len(self.subcomplexes[n_dim][nerve_spx_index][deg])

        local_chains = np.zeros((cpx_size, len(generators)))
        # IMAGE OF CECH DIFFERENTIAL #############################
        for coface_index, nerve_coeff in zip(cofaces, coefficients):
            # check that there are some local coordinates
            if len(reference_preimage[coface_index]) > 0:
                # generate boundary matrix
                cech_local = self.local_cech_matrix(
                    n_dim+1, deg, coface_index, nerve_spx_index,
                    nerve_coeff)
                active_generators = np.where(np.in1d(
                    generators, reference_preimage[coface_index])
                    )[0]
                    # image of cech complex
                local_chains[:, active_generators] += np.matmul(
                    cech_local, local_preimage[coface_index].T
                    )
            # end if
        # end for
        local_chains %= self.p
        # local lift to first page
        gammas, betas = self.first_page_local_lift(
            n_dim, deg, nerve_spx_index, local_chains, R[generators])
        # compute vertical preimage and store
        # look for indices of nonzero columns
        preimages = np.matmul(self.PreIm[0][n_dim][nerve_spx_index][deg+1],
                              gammas).T
        nonzero_idx = np.where(gammas.any(axis=0))[0]
        if np.any(nonzero_idx):
            lift_coordinates[nerve_spx_index] = preimages[nonzero_idx]
            lift_references[nerve_spx_index] = generators[nonzero_idx]
        # store first page coefficients
        betas_aux = np.zeros((len(R), next - prev))
        betas_aux[generators] = np.transpose(betas)
        Betas_1_page[:, prev:next] = betas_aux
    # end cech_diff_and_lift_local

    ############################################################################
    # self.first_page_local_lift

    def first_page_local_lift(
            self, n_dim, deg, nerve_spx_index, local_chains, lift_radii):
        """ Lift to first page on a given open cover.

        loca_chains with generators indexing columns.
        """
        # create Im_Hom Matrix
        Im_Hom = np.append(
            self.Im[0][n_dim][nerve_spx_index][deg].coordinates,
            self.Hom[0][n_dim][nerve_spx_index][deg].coordinates, axis=1)
        start_index = np.size(Im_Hom,1)
        # Gaussian elimination of  M = (Im | Hom | local_chains)
        M = np.append(Im_Hom, local_chains, axis = 1)
        # R_M : vector of birth radii of columns in M
        R_M = self.Im[0][n_dim][nerve_spx_index][deg].barcode[:,0]
        R_M = np.concatenate([
            R_M, self.Hom[0][n_dim][nerve_spx_index][deg].barcode[:,0]],
            axis=None
            )
        R_M = np.concatenate([R_M, lift_radii], axis=None)
        _, T = gauss_col_rad(M, R_M, start_index, self.p)
        # look at reductions on generators
        T = T[:, start_index:]
        gammas = T[0:self.Im[0][n_dim][nerve_spx_index][deg].dim]
        betas = T[self.Im[0][n_dim][nerve_spx_index][deg].dim : start_index]
        # return preimage coordinates and beta coordinates
        return gammas, betas
    # end cech_diff_and_lift_local


    ############################################################################
    # local boundary matrix
    def local_cech_matrix(self, n_dim, deg, nerve_spx_index,
                          nerve_face_index, nerve_coeff):

        """
        Returns matrix of Cech differential in (n_dim, deg) restricted
        on component (nerve_face_index, nerve_spx_index).
        nerve_coeff stores the sign of the nerve differential in position
        (nerve_face_index, nerve_spx_index).
        Returns a matrix of size:
         (subcpx[n_dim-1][nerve_face_index][deg],
          subcpx[n_dim][nerve_spx_index][deg])
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
    # end local_cech_matrix

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
    # compute_two_page_representatives

    def compute_two_page_representatives(self, n_dim, deg):
        """ Computes total complex representatives for second page classes in
        position (n_dim, deg).

        All info is written in self.Hom_reps
        """
        if self.Hom[1][n_dim][deg].dim == 0:
            return
        # obtain chain complex reps of 2 page classes on entry (n_dim, deg)
        coordinates_hom = self.Hom[1][n_dim][deg].coordinates
        references = []
        local_coordinates = []
        prev = 0
        # end else

        # localized_format for first page info
        for nerv_idx, next in enumerate(self.cycle_dimensions[
                n_dim][deg][:-1]):
            # if there are some local cycles
            if prev < next:
                references.append(range(self.Hom[1][n_dim][deg].dim))
                local_coordinates.append(np.matmul(self.Hom[0][n_dim][
                    nerv_idx][deg].coordinates, coordinates_hom[prev:next]
                    ).T)
            else:
                references.append(np.array([]))
                local_coordinates.append(np.array([]))
            # end if else
            prev = next
        # end for
        # chains at position (n_dim, deg) in localized format
        chains = [references, local_coordinates]
        self.Hom_reps[1][n_dim][deg] = [chains]
        # add extra entries when possible
        if n_dim > 0:
            # birth values classes
            R = self.Hom[1][n_dim][deg].barcode[:,0]
            # cech_diff_and_lift
            betas, lift = self.cech_diff_and_lift(n_dim, deg, chains, R)
            if np.any(betas):
                raise(ValueError)
            # vertical lift written on total_complex_reps
            self.Hom_reps[1][n_dim][deg].append(lift)
        # end if
    # end def

    ###########################################################################
    # compute_higher_representatives

    def compute_higher_representatives(self, n_dim, deg, current_page):
        """ Computes total complex representatives for current_page classes in
        position (n_dim, deg).

        All info is written in self.Hom_reps
        """
        # handle trivial cases
        if self.Hom[current_page][n_dim][deg].dim == 0:
            return
        # (n_dim, deg) --> (Sn_dim, Sdeg)
        Sn_dim = n_dim - current_page
        Sdeg = deg + current_page - 1
        # if differential is trivial, total reps do not change
        if Sn_dim < 0:
            self.Hom_reps[current_page][n_dim][deg] = self.Hom_reps[
                current_page - 1][n_dim][deg]
            return
        # lift image to page
        hom_barcode = self.Hom[current_page][n_dim][deg].barcode
        hom_sums = (self.Hom[current_page][n_dim][deg].coordinates).T
        total_complex_reps = []
        hom_total_reps = self.Hom_reps[current_page - 1][n_dim][deg]
        # compute
        for chains in hom_total_reps:
            total_complex_reps.append(local_sums(chains, hom_sums))
        # end for

        for target_page in range(current_page, 1, -1):
            Betas, _ = self.cech_diff_and_lift(
                Sn_dim + 1, Sdeg, total_complex_reps[-1], hom_barcode[:,0])
            # go up to target_page and modify total_complex_reps
            Betas, Gammas = self.lift_to_page(
                Sn_dim, Sdeg, target_page, Betas, hom_barcode)
            # modify reps
            # from im_coord obtain total cpx chains
            if np.any(Gammas):
                Tn_dim = Sn_dim + current_page
                Tdeg = Sdeg - current_page + 1
                preimage_reps = []
                for chains in self.Hom_reps[target_page][Tn_dim][Tdeg]:
                    preimage_reps.append(local_sums(chains, Gammas))
                # add preimage reps to current tot_complex reps
                for idx, chains in enumerate(total_complex_reps):
                    total_complex_reps[idx] += preimage_reps[idx]
                # end for
            # end if
        # end for
        Betas, lift = self.cech_diff_and_lift(
            Sn_dim + 1, Sdeg, total_complex_reps[-1], hom_barcode[:,0])
        # check that there are no problems
        if np.any(Betas):
            raise RuntimeError
        # add vertical lift to reps
        total_complex_reps.append(lift)
    # end compute_total_representatives

    ############################################################################
    # extension

    def extension(self, start_n_dim, start_deg):
        """ Take information from spectral sequence class, and calculate
        extension coefficients for a given position (start_deg, start_n_dim).

        """
        print("start_n_dim:{}, start_deg:{}".format(start_n_dim, start_deg))
        death_radii = self.Hom[self.no_pages-1][start_n_dim][
            start_deg].barcode[:,1]
        no_gens = len(death_radii)
        Hom_reps = self.Hom_reps[self.no_pages-1][start_n_dim][start_deg]
        # bars for extension problem of infty page classes
        barcode_extension = np.ones((len(death_radii), 2))
        barcode_extension[:, 0] = death_radii
        barcode_extension[:, 1] *= self.max_r
        # go through all diagonal
        Sdeg = start_deg
        Sn_dim = start_n_dim
        for chains in Hom_reps:
            print("Sdeg:{}, Sn_dim:{}".format(Sdeg, Sn_dim))
            for target_page in range(self.no_pages, 1, -1):
                print("target_page:{}".format(target_page))
                # get coefficients on first page
                Betas = np.zeros((
                    no_gens, self.page_dim_matrix[1, Sdeg, Sn_dim]))
                print("Betas:({},{})".format(
                    np.size(Betas, 0), np.size(Betas, 1)
                ))
                prev = 0
                print("covers:{}".format(self.cycle_dimensions[Sn_dim][Sdeg]))
                for nerve_spx_index, next in enumerate(
                        self.cycle_dimensions[Sn_dim][Sdeg][:-1]):
                    if (prev < next) and (len(chains[0][nerve_spx_index]) > 0):
                        _, Betas_first_page = self.first_page_local_lift(
                            Sn_dim, Sdeg, nerve_spx_index,
                            chains[1][nerve_spx_index].T, death_radii)
                        Betas_aux = np.zeros((no_gens, next-prev))
                        Betas_aux[chains[0][nerve_spx_index]] = Betas_first_page.T
                        Betas[:, prev:next] = Betas_aux
                    # end if
                    prev = next
                # go up to target_page and modify total_complex_reps
                print("lift to {} page".format(target_page))
                print("Betas:({},{})".format(
                    np.size(Betas, 0), np.size(Betas, 1)
                ))
                Betas, Gammas = self.lift_to_page(
                    Sn_dim, Sdeg, target_page, Betas, barcode_extension)
                # check that we have vanished Betas
                if np.any(Betas):
                    raise RuntimeError
                # modify reps
                if np.any(Gammas):
                    print("Gammas")
                    print("({},{})",format(np.size(Gammas,0),np.size(Gammas,1)))
                    # compute coefficients of Gammas in 1st page
                    Im_coefficients = np.matmul(
                        self.Im[target_page-1][Sn_dim][Sdeg].coordinates, Gammas.T)
                    print("Im_coefficients")
                    print(np.size(Im_coefficients,0))
                    print(np.size(Im_coefficients,1))
                    for page in range(target_page - 1, 1, -1):
                        Im_coefficients = np.matmul(
                            self.Hom[target_page-1][Sn_dim][Sdeg].coordinates,
                            Im_coefficients)
                    # coordinates of images on 1st page
                    Im_coefficients = Im_coefficients.T
                    # compute local chains of image and add to chains
                    prev = 0
                    for nerve_spx_index, next in enumerate(
                            self.cycle_dimensions[n_dim][deg][:-1]):
                        chains[-1][nerve_spx_index] += np.matmul(
                            self.Hom[1][n_dim][nerve_spx_index][deg],
                            Im_coefficients[:, prev:next]
                        )
                        prev = next
                    # end for
                # end if
            # end for
            Sdeg += 1
            Sn_dim -= 1
        # end for
    # end def

# end spectral_sequence class ##################################################


################################################################################
# local_sums
#

def local_sums(chains, sums):
    """Sum chains as indicated by "sums"

    Each sum is given by rows in sums. These store the coefficients that
    the chain entries need to be added.

    chains are given in localized form
    """
    no_sums = np.size(sums, axis=0)
    new_ref = []
    new_coord = []
    for local_idx, ref in enumerate(chains[0]):
        if len(ref) > 0:
            new_ref.append(range(no_sums))
            new_coord.append(np.matmul(chains[1][local_idx].T,
                (sums.T)[ref]).T)
        else:
            new_ref.append([])
            new_coord.append([])
    # end for
    return [new_ref, new_coord]
