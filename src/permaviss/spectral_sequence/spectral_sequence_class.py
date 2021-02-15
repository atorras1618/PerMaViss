#
#    spectral_sequence_class.py
#

import numpy as np

from multiprocessing import Pool
from functools import partial

from ..simplicial_complexes.differentials import complex_differentials
from ..gauss_mod_p.gauss_mod_p import gauss_col_rad, gauss_barcodes

from ..persistence_algebra.barcode_bases import barcode_basis

from .local_chains_class import local_chains


class spectral_sequence(object):
    """Space and methods for Mayer-Vietoris spectral sequences

    Parameters
    ----------
    nerve : :obj:`list(Numpy Array)`
        Simplicial complex storing the nerve of the covering. This is stored as
        a list, where the ith entry contains a :obj:`Numpy Array` storing
        all the ith simplices; a simplex for each row.
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
    nerve_differentials : :obj:`list(Numpy Array)`
        Differentials of Nerve. Used for computing Cech Complex.
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
    zero_diff : :obj:`list(list(list(Numpy Array)))`
        List storing the vertical differential matrices on the 0 page of the
        spectral sequence. For  integers `n_dim`, `k` and `dim` the variable
        `zero_diff[n_dim][k][dim]` stores the `dim` differential of
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
        self.no_columns = len(nerve) - 1
        self.max_r = max_r
        self.p = p
        local_chains.p = p
        # add nerve_point_cloud to spectral_sequence info
        self.nerve_point_cloud = nerve_point_cloud
        # add points IN to support Cech Differential
        self.points_IN = points_IN
        # add nerve and compute nerve differentials
        self.nerve = nerve
        # count number of simplices in nerve
        self.nerve_spx_number = []
        self.nerve_spx_number.append(self.nerve[0])
        for nerve_simplices in nerve[1:self.no_columns]:
            self.nerve_spx_number.append(np.size(nerve_simplices, 0))
        # end for
        self.nerve_spx_number
        self.nerve_differentials = complex_differentials(nerve, p)
        # list containing barcode bases for Hom, Im and PreIm
        # Hom and Im go through all pages, whereas
        # PreIm is only contained in the 0 page
        self.Hom = [[]]
        self.Im = [[]]
        self.PreIm = [[]]
        self.subcomplexes = []
        self.zero_diff = []
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
            self.zero_diff.append([])
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
            :meth:`..local_persistent_homology`
        n_dim : int
            Column of `0`-page whose data has been computed.

        """
        self.subcomplexes[n_dim] = [it[0] for it in output]
        self.zero_diff[n_dim] = [it[1] for it in output]
        self.Hom[0][n_dim] = [it[2] for it in output]
        self.Im[0][n_dim] = [it[3] for it in output]
        self.PreIm[0][n_dim] = [it[4] for it in output]

        # check that the level of intersection is not empty.
        if len(self.Hom[0][n_dim]) > 0:
            for deg in range(self.no_rows):
                no_cycles = 0
                # cumulative dimensions
                self.cycle_dimensions[n_dim].append(
                    np.zeros(self.nerve_spx_number[n_dim]+1).astype(int))
                for k in range(self.nerve_spx_number[n_dim]):
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
                    for k in range(self.nerve_spx_number[n_dim]):
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
    # self.first_differential(self, n_dim, deg):

    def first_differential(self, n_dim, deg):
        """ Compute differential on first page (n_dim, deg) --> (n_dim-1, deg)

        Parameters
        ----------
        n_dim, deg : int, int
            Differential domain position on first page.

        Returns
        -------
        Betas : np.array
            Coefficients of image of first page differentials. The image of
            each class from (n_dim, deg) is given as a row.

        """
        # handle trivial cases
        if self.page_dim_matrix[1, deg, n_dim] == 0:
            return np.array([])
        # generate chains for sending to cech_diff_and_lift
        domain_chains = local_chains(self.nerve_spx_number[n_dim])
        # compute array of initial radii
        R = np.zeros(self.page_dim_matrix[1, deg, n_dim])
        # birth radii and localized_coordinates for classes
        prev = 0
        for nerve_spx_index, next in enumerate(
                self.cycle_dimensions[n_dim][deg][:-1]):
            if prev < next:
                domain_chains.add_entry(
                    nerve_spx_index, np.array(range(prev, next)),
                    (self.Hom[0][n_dim][nerve_spx_index][deg].coordinates).T)
                R[prev:next] = self.Hom[0][n_dim][nerve_spx_index][
                    deg].barcode[:, 0]
                prev = next
        # end for
        # call cech_diff_and_lift
        Betas, _ = self.cech_diff_and_lift(n_dim, deg, domain_chains, R)
        return Betas

    ###########################################################################
    # self.high_differential(self, n_dim, deg, current_page):

    def high_differential(self, n_dim, deg, current_page):
        """ Compute differential on `current-page`
        (n_dim, deg) --> (n_dim - current_page, deg + current_page - 1).

        Parameters
        ----------
        n_dim, deg : int, int
            Differential domain position.

        Returns
        -------
        Betas : np.array
            Coefficients of image of current_page differentials. The image of
            each class from (n_dim, deg) is given as a row.

        """
        # handle trivial case
        if self.Hom[current_page-1][n_dim][deg].dim == 0:
            return np.array([])
        # take last total complex entry of Hom reps
        chains = self.Hom_reps[current_page-1][n_dim][deg][current_page-1]
        Hom_barcode = self.Hom[current_page-1][n_dim][deg].barcode
        # codomain position
        Sn_dim = n_dim - current_page
        Sdeg = deg + current_page - 1
        # differential (n_dim, deg) --> (Sn_dim, Sdeg)
        Betas, _ = self.cech_diff_and_lift(Sn_dim + 1, Sdeg, chains,
                                           Hom_barcode[:, 0])
        Betas, _ = self.lift_to_page(Sn_dim, Sdeg, current_page, Betas,
                                     Hom_barcode)
        return Betas
    # end high_differential

    ###########################################################################
    # self.lift_to_page(self, n_dim, deg, page, chains):

    def lift_to_page(self, n_dim, deg, target_page, Betas, Beta_barcode):
        """ Lifts chains in position (n_dim, deg) from page 1 to target_page

        Returns Betas and image coordinates.

        Parameters
        ----------
        n_dim, deg : int, int
            Differential domain position.
        target_page : int
            Lift classes up to this page.
        Betas : np.array
            Coordinates of classes on first page.
        Betas_barcode : np.array
            Barcodes of classes to be lifted.

        Returns
        -------
        Betas.T : np.array
            Coefficients of image of current_page differentials. The image of
            each class from (n_dim, deg) is given as a row.
        Gammas.T : np.array
            Coefficients of added differentials of (current_page - 1) page.
            This is such that the sum of differentials using Gammas,
            plus adding classes using target_betas leads to the original Betas.

        """
        Betas = Betas.T
        # lift up to target_page
        for k in range(1, target_page):
            Im = self.Im[k][n_dim][deg]
            Hom = self.Hom[k][n_dim][deg]
            if Hom.dim > 0:
                if Im.dim > 0:
                    Im_Hom = np.append(Im.coordinates, Hom.coordinates, axis=1)
                    barcode_col = np.append(Im.barcode, Hom.barcode, axis=0)
                else:
                    Im_Hom = Hom.coordinates
                    barcode_col = Hom.barcode
            else:
                Betas = np.array([])
                Gammas = np.array([])
                break

            start_index = Im.dim + Hom.dim
            # add radii coordinates and radii
            barcode_col = np.append(barcode_col, Beta_barcode, axis=0)
            A = np.append(Im_Hom, Betas, axis=1)
            # barcodes of rows
            barcode_row = self.Hom[k][n_dim][deg].prev_basis.barcode
            # gaussian reduction on matrix between persistence modules
            # order here barcode_row
            rows_basis = barcode_basis(barcode_row)
            order = rows_basis.sort(send_order=True)
            # order row barcodes as well
            ordered_barcode_row = barcode_row[order]
            A = A[order]
            coefficients = gauss_barcodes(
                A, ordered_barcode_row, barcode_col, start_index, self.p)
            # next page coefficients
            Gammas = coefficients[:Im.dim]
            Betas = coefficients[Im.dim:]
        # end for
        return Betas.T, Gammas.T
    # end lift_to_page

    ###########################################################################
    # self.cech_diff_and_lift

    def cech_diff_and_lift(self, n_dim, deg, start_chains, R):
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
        chains : :obj:`list(list(Numpy Array))`

        Returns
        -------
        betas : coordinates on first pages
        [lift_references, lift_coordinates]: local coordinates lifted by
            vertical differential.

        """
        # store space for coordinates in first page
        Betas_1_page = np.zeros((
            len(R), self.page_dim_matrix[1, deg, n_dim-1]
            ))
        # store space for preimages
        lift_chains = local_chains(self.nerve_spx_number[n_dim-1])
        if len(R) > 0:
            partial_cech_diff_and_lift_local = partial(
                self.cech_diff_and_lift_local, R, start_chains, n_dim - 1, deg)
            # map reduce local cech differential and lifts
            workers_pool = Pool()
            output = workers_pool.map(
                partial_cech_diff_and_lift_local,
                range(self.nerve_spx_number[n_dim-1]))
            workers_pool.close()
            workers_pool.join()
            # output = []
            # for j in range(self.nerve_spx_number[n_dim-1]):
            #     output.append(partial_cech_diff_and_lift_local(j))
            prev = 0
            # store results
            for nerve_spx_index, next in enumerate(
                    self.cycle_dimensions[n_dim-1][deg][:-1]):
                if output[nerve_spx_index] is not None:
                    Betas_1_page[:, prev:next] = output[nerve_spx_index][0]
                    lift_chains.add_entry(
                        nerve_spx_index, output[nerve_spx_index][1],
                        output[nerve_spx_index][2])
                prev = next
            # end for
        return Betas_1_page, lift_chains
    # end cech_diff_and_lift

    ###########################################################################
    # self.cech_diff

    def cech_diff(self, n_dim, deg, start_chains):
        """ Given chains in (n_dim + 1, deg), compute Cech differential.

        Parameters
        ----------
        n_dim, deg: int, int
            Codomain position in spectral sequence.
        chains : :class:`local_chains` object
            Chains on (n_dim+1, deg) that are stored as references in chains[0]
            and local coordinates as rows in chains[1].

        Returns
        -------
        image_chains : :obj:`Local Coordinates`
            Image coordinates of Cech differential.

        """
        image_chains = local_chains(self.nerve_spx_number[n_dim])
        # CECH DIFFERENTIAL
        for nerve_spx_index in range(self.nerve_spx_number[n_dim]):
            loc_im_ref, loc_im_coord = self.cech_diff_local(
                start_chains, n_dim, deg, nerve_spx_index)
            image_chains.add_entry(nerve_spx_index, loc_im_ref, loc_im_coord)
        # end for
        return image_chains

    #######################################################################
    # Cech chain plus lift of preimage

    def cech_diff_and_lift_local(
            self, R, start_chains, n_dim, deg, nerve_spx_index):
        """ Takes some chains in position (n_dim+1, deg) and computes Cech diff
        followed by a lift by vertical differential. This is done locally at
        cover information in (n_dim, deg).

        This method is meant to be run in parallel.

        Parameters
        ----------
        R : :obj:`list`
            Vector of radii
        start_chains : :class:`local_chains` object
            Chains in position (n_dim + 1, deg)
        n_dim, deg, nerve_spx_index : int, int, int
            Position in spectral sequence and local index.

        Returns
        -------
        betas_1_page : :obj:`Numpy Array`
            Coefficients of lift to 1st page on position (n_dim, deg)
        local_lift_references : :obj:`list`
            List of local references of lift.
        local_lift_coordinates : :obj:`Numpy Array`
            Local coordinates of lift.

        """
        # if nerve_spx_index==0, then prev=0
        prev = self.cycle_dimensions[n_dim][deg][nerve_spx_index-1]
        next = self.cycle_dimensions[n_dim][deg][nerve_spx_index]
        # if trivial cover skip
        if prev == next:
            return
        # CECH DIFFERENTIAL
        generators, local_chains = self.cech_diff_local(
            start_chains, n_dim, deg, nerve_spx_index)
        # if there are no images to compute, return
        if len(generators) == 0:
            return
        # LOCAL LIFT TO FIRST PAGE
        gammas, betas = self.first_page_local_lift(
            n_dim, deg, local_chains, R[generators], nerve_spx_index)
        # store first page coefficients
        betas_1_page = np.zeros((len(R), next - prev))
        betas_1_page[generators] = np.transpose(betas)
        # compute vertical preimage and store
        preimages = np.matmul(self.PreIm[0][n_dim][nerve_spx_index][deg+1],
                              gammas).T
        # look for indices of nonzero columns
        nonzero_idx = np.where(gammas.any(axis=0))[0]
        if len(nonzero_idx) > 0:
            local_lift_ref = generators[nonzero_idx]
            # correct sign
            local_lift_coord = -preimages[nonzero_idx] % self.p
        else:
            local_lift_ref, local_lift_coord = [], []
        # end if else
        return betas_1_page, local_lift_ref, local_lift_coord
    # end cech_diff_and_lift_local

    ###########################################################################
    # self.cech_diff_local

    def cech_diff_local(
            self, start_chains, n_dim, deg, nerve_spx_index):
        """ Local Cech differential, starting from chains in (n_dim + 1, deg).

        Parameters
        ----------
        start_chains : :class:`local_chains` object
            Chains to compute Cech differential from.
        n_dim, deg, nerve_spx_index : int, int, int
            Position in spectral sequence and local index.

        Returns
        -------
        local_image_ref : :obj:`list`
            List of local references of image.
        local_image_coord.T : :obj:`Numpy Array`
            Local coordinates of image. Expressions correspond to rows while
            local simplices correspond to columns.
        """
        coboundary = self.nerve_differentials[n_dim + 1][nerve_spx_index]
        # cofaces and coefficients on cech differential
        cofaces = np.nonzero(coboundary)[0]
        coefficients = coboundary[cofaces]
        # indices of generators that are nontrivial by cech diff
        generators = start_chains.ref[cofaces[0]]
        for coface_index in cofaces[1:]:
            generators = np.append(generators, start_chains.ref[
                coface_index]).astype(int)
        # end for
        local_image_ref = np.unique(generators)
        # if there are no images to compute, return
        if len(local_image_ref) == 0:
            return [], []
        # size of local complex
        if deg == 0:
            cpx_size = self.subcomplexes[n_dim][nerve_spx_index][0]
        else:
            cpx_size = len(self.subcomplexes[n_dim][nerve_spx_index][deg])

        local_image_coord = np.zeros((cpx_size, len(local_image_ref)))
        # IMAGE OF CECH DIFFERENTIAL #############################
        for coface_index, nerve_coeff in zip(cofaces, coefficients):
            # check that there are some local coordinates
            if len(start_chains.ref[coface_index]) > 0:
                # generate boundary matrix
                cech_local = self.local_cech_matrix(
                    n_dim+1, deg, coface_index, nerve_spx_index,
                    nerve_coeff)
                active_generators = np.where(np.in1d(
                    local_image_ref, start_chains.ref[coface_index])
                    )[0]
                # image of cech complex
                local_image_coord[:, active_generators] += np.matmul(
                    cech_local, start_chains.coord[coface_index].T
                    )
            # end if
        # end for
        local_image_coord %= self.p
        return local_image_ref, local_image_coord.T

    ###########################################################################
    # local boundary matrix
    def local_cech_matrix(self, n_dim, deg, nerve_spx_index,
                          nerve_face_index, nerve_coeff):

        """Returns matrix of Cech differential in (n_dim, deg) restricted
        on component (nerve_face_index, nerve_spx_index).

        Parameters
        ----------
        n_dim, deg: int, int
            Position in spectral sequence.
        nerve_spx_index, nerve_face_index : int, int
            Local indices in domain and codomain respectively.
        nerve_coeff : int
            Coefficient in nerve differential determined by the pair
            nerve_spx_index and nerve_face_index.

        Returns
        -------
        boundary : :obj:`Numpy Array`
            Matrix of size (subcpx[n_dim-1][nerve_face_index][deg],
            subcpx[n_dim][nerve_spx_index][deg]) that represents the local
            cech differential.

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
    # self.first_page_local_lift

    def first_page_local_lift(
            self, n_dim, deg, local_coord, lift_radii, nerve_spx_index):
        """ Lift to first page on a given open cover.

        Parameters
        ----------
        n_dim, deg : int, int
            Position on spectral sequence.
        local_coord : :obj:`Numpy Array`
            Local coordinates to be lifted to first page and vertical
            differential. Rows are expressions while columns correspond
            to local simplices.
        lift_radi : :obj:`list`
            Values at which we want to lift `start_chains` by the
            vertical differential.
        nerve_spx_inex : int
            Local index. This function is meant to be parallelized over this.

        Returns
        -------
        gammas : :obj:`Numpy Array`
            2D Matrix expressing coefficients of lift. Each expression
            corresponds to a column, while image generators correspond to rows.
        betas : :obj:`Numpy Array`
            2D Matrix expressing coefficients in terms of homology classes on
            page 1. Expressions correspond to columns, while homology classes
            correspond to rows.

        """
        # take care of case when local_coord is a `local_chains` object
        if isinstance(local_coord, local_chains):
            if len(local_coord.ref[nerve_spx_index]) > 0:
                lift_radii = lift_radii[local_coord.ref[nerve_spx_index]]
                local_coord = local_coord.coord[nerve_spx_index]
            else:
                return [], []

        # return if nothing to lift
        if len(lift_radii) == 0:
            return [], []
        # R_M : vector of birth radii of columns in M
        # distinguish from trivial case where images are zero
        if self.Im[0][n_dim][nerve_spx_index][deg].dim > 0 and self.Hom[0][
                n_dim][nerve_spx_index][deg].dim > 0:
            Im_Hom = np.append(
                self.Im[0][n_dim][nerve_spx_index][deg].coordinates,
                self.Hom[0][n_dim][nerve_spx_index][deg].coordinates, axis=1)
            R_M = self.Im[0][n_dim][nerve_spx_index][deg].barcode[:, 0]
            R_M = np.concatenate([
                R_M, self.Hom[0][n_dim][nerve_spx_index][deg].barcode[:, 0]],
                axis=None)
        elif self.Hom[0][n_dim][nerve_spx_index][deg].dim > 0:
            Im_Hom = self.Hom[0][n_dim][nerve_spx_index][deg].coordinates
            R_M = self.Hom[0][n_dim][nerve_spx_index][deg].barcode[:, 0]
        elif self.Im[0][n_dim][nerve_spx_index][deg].dim > 0:
            Im_Hom = self.Im[0][n_dim][nerve_spx_index][deg].coordinates
            R_M = self.Im[0][n_dim][nerve_spx_index][deg].barcode[:, 0]
        else:
            return [], []

        R_M = np.concatenate([R_M, lift_radii], axis=None)
        start_index = np.size(Im_Hom, 1)
        # Gaussian elimination of  M = (Im | Hom | start_chains (local))
        M = np.append(Im_Hom, local_coord.T, axis=1)
        _, T = gauss_col_rad(M, R_M, start_index, self.p)
        # look at reductions on generators and correct sign
        T = -T[:, start_index:] % self.p
        gammas = T[0:self.Im[0][n_dim][nerve_spx_index][deg].dim]
        betas = T[self.Im[0][n_dim][nerve_spx_index][deg].dim:start_index]
        # return preimage coordinates and beta coordinates
        return gammas, betas
    # end first_page_local_lift

    ###########################################################################
    # self.first_page_lift

    def first_page_lift(self, n_dim, deg, start_chains, R):
        """Given some chains in position (n_dim, deg), lift to first page
        accross several covers.

        Parameters
        ----------
        n_dim, deg, current_page : int, int, int
            Postition on spectral sequence and current page.
        start_chains : :class:`local_chains` object
            Chains in position (n_dim, deg) that we lift to first page.
        R : :obj:`list`
            Values at which we lift `start_chains`

        Returns
        -------
        Betas_1_page : :obj:`Numpy Array`
            Coordinates on first page. Rows correspond to expressions and
            columns to homology classes.
        lift_chains : :class:`local_coord` object
            Chains after lifting vertically by horizontal differential.

        """
        # store space for preimages
        lift_chains = local_chains(self.nerve_spx_number[n_dim])
        # coordinates in first page
        Betas_1_page = np.zeros((
            len(R), self.page_dim_matrix[1, deg, n_dim]))
        # return if trivial
        if len(R) == 0:
            return Betas_1_page, lift_chains
        # compute vertical lifts in parallel
        partial_first_page_local_lift = partial(
            self.first_page_local_lift, n_dim, deg, start_chains, R)
        workers_pool = Pool()
        output = workers_pool.map(
            partial_first_page_local_lift, range(self.nerve_spx_number[n_dim]))
        workers_pool.close()
        workers_pool.join()
        # proceed to store the result
        prev = 0
        for nerve_spx_index, next in enumerate(
                self.cycle_dimensions[n_dim][deg][:-1]):
            gammas, betas = output[nerve_spx_index]
            # save betas
            if len(betas) > 0:
                Betas_aux = np.zeros((len(R), next-prev))
                Betas_aux[start_chains.ref[nerve_spx_index]] = betas.T
                Betas_1_page[:, prev:next] = Betas_aux
            # save lifts
            if len(gammas) > 0:
                lift_chains.add_entry(
                    nerve_spx_index, start_chains.ref[nerve_spx_index],
                    (np.matmul(self.PreIm[
                        0][n_dim][nerve_spx_index][deg+1], gammas)).T)
            # end if
            prev = next
        # end for
        return Betas_1_page, lift_chains

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

        Resulting representatives are written in self.Hom_reps[1][n_dim][deg]

        Parameters
        ----------
        n_dim, deg : int, int
            These specify the position on the spectral sequence where we want
            to compute and store the second page representatives.
        """
        # return if nothing to compute
        if self.Hom[1][n_dim][deg].dim == 0:
            return
        coordinates_hom = self.Hom[1][n_dim][deg].coordinates
        rep_chains = [local_chains(self.nerve_spx_number[n_dim])]
        prev = 0
        # localized_format for first page info
        for nerv_idx, next in enumerate(self.cycle_dimensions[
                n_dim][deg][:-1]):
            if prev < next:
                rep_chains[0].add_entry(
                    nerv_idx, range(self.Hom[1][n_dim][deg].dim),
                    np.matmul(self.Hom[0][n_dim][
                        nerv_idx][deg].coordinates, coordinates_hom[prev:next]
                        ).T)
            # end if
            prev = next
        # end for
        # add extra entries when possible
        if n_dim > 0:
            # birth values classes
            R = self.Hom[1][n_dim][deg].barcode[:, 0]
            # cech_diff_and_lift
            betas, lift = self.cech_diff_and_lift(n_dim, deg, rep_chains[0], R)
            if np.any(betas):
                raise(ValueError)
            # vertical lift written on total_complex_reps
            rep_chains.append(lift)
        # end if
        # store on allocated space for second page representatives
        self.Hom_reps[1][n_dim][deg] = rep_chains
    # end def

    ###########################################################################
    # compute_higher_representatives

    def compute_higher_representatives(self, n_dim, deg, current_page):
        """ Computes total complex representatives for current_page classes in
        position (n_dim, deg).

        Resulting representatives written in
        `self.Hom_reps[current_page][n_dim][deg]`

        Parameters
        ----------
        n_dim, deg, current_page : int, int, int
            Position on the spectral sequence and current page.
        """
        # handle trivial cases
        if self.Hom[current_page][n_dim][deg].dim == 0:
            self.Hom_reps[current_page][n_dim][deg] = []
            return
        # lift image to page
        hom_barcode = self.Hom[current_page][n_dim][deg].barcode
        hom_sums = (self.Hom[current_page][n_dim][deg].coordinates).T
        # create total complex reps up to last entry by using coefficients and
        # total complex representatives on previous page
        total_complex_reps = []
        # compute
        for chains in self.Hom_reps[current_page - 1][n_dim][deg]:
            total_complex_reps.append(local_chains.sums(chains, hom_sums))
        # end for
        # (n_dim, deg) --> (Sn_dim, Sdeg)
        Sn_dim = n_dim - current_page
        Sdeg = deg + current_page - 1
        # if differential is trivial, no need to compute cech differential
        if Sn_dim < 0:
            self.Hom_reps[current_page][n_dim][deg] = total_complex_reps
            return
        for target_page in range(current_page, 1, -1):
            Betas, _ = self.cech_diff_and_lift(
                Sn_dim + 1, Sdeg, total_complex_reps[-1], hom_barcode[:, 0])
            # go up to target_page and modify total_complex_reps

            Betas, Gammas = self.lift_to_page(
                Sn_dim, Sdeg, target_page, Betas, hom_barcode)
            # modify reps
            # from im_coord obtain total cpx chains
            Tn_dim = Sn_dim + target_page - 1
            Tdeg = Sdeg - target_page + 2
            preimage_reps = []
            # obtain preimage_coefficients from expressions in -Gammas
            preimage_coefficients = np.matmul(
                self.PreIm[target_page - 1][Tn_dim][Tdeg], -Gammas.T
            )
            # case of target_page is special due to local coordinates
            if np.any(Gammas) and target_page == 2:
                prev = 0
                for spx_idx, next in enumerate(
                        self.cycle_dimensions[Sn_dim + 1][Sdeg]):
                    if prev < next:
                        local_preimage = (np.matmul(
                            self.Hom[0][Sn_dim + 1][spx_idx][Sdeg].coordinates,
                            preimage_coefficients[prev:next])).T
                        # if non empby, add
                        if len(total_complex_reps[-1].ref[spx_idx]) > 0:
                            total_complex_reps[-1].coord[
                                spx_idx] += local_preimage
                        else:
                            total_complex_reps[-1].add_entry(
                                spx_idx, range(np.size(local_preimage, 0)),
                                local_preimage)
                    prev = next
                # end for
            elif np.any(Gammas):
                for chains in self.Hom_reps[target_page][Tn_dim][Tdeg]:
                    preimage_reps.append(local_chains.sums(
                        chains, preimage_coefficients))
                # add preimage reps to current tot_complex reps
                for idx, chains in enumerate(preimage_reps):
                    total_complex_reps[
                        current_page - target_page + 1 + idx] += chains
                # end for
            # end elif
        # end for
        Betas, lift = self.cech_diff_and_lift(
            Sn_dim + 1, Sdeg, total_complex_reps[-1], hom_barcode[:, 0])
        # check that there are no problems
        if np.any(Betas):
            raise RuntimeError
        # add vertical lift to reps
        total_complex_reps.append(lift)
        # store modified total complex reps
        self.Hom_reps[current_page][n_dim][deg] = total_complex_reps
    # end compute_total_representatives

    ###########################################################################
    # extension

    def extension(self, start_n_dim, start_deg):
        """ Take information from spectral sequence class, and calculate
        extension coefficients for a given position (start_deg, start_n_dim).

        """
        death_radii = self.Hom[self.no_pages-1][start_n_dim][
            start_deg].barcode[:, 1]
        Hom_reps = local_chains.copy_seq(
            self.Hom_reps[self.no_pages - 1][start_n_dim][start_deg])
        # bars for extension problem of infty page classes
        barcode_extension = np.ones((len(death_radii), 2))
        barcode_extension[:, 0] = death_radii
        barcode_extension[:, 1] *= self.max_r
        # if death_radii is equal to max_r, no need to compute ext coefficients
        ext_bool = death_radii < self.max_r
        # initialize extension matrices as zero matrices
        Sdeg = start_deg
        Sn_dim = start_n_dim
        for chains in Hom_reps:
            self.extensions[start_n_dim][start_deg].append(np.zeros((
                self.Hom[self.no_pages-1][Sn_dim][Sdeg].dim,
                len(death_radii))))
            Sdeg += 1
            Sn_dim -= 1
        # end for
        # if there are no extension coefficients to compute, return
        if not np.any(ext_bool):
            return
        # zero out representatives not in ext_bool
        for chains in Hom_reps:
            for k, local_ref in enumerate(chains.ref):
                if len(local_ref) > 0 and np.any(
                        np.invert(ext_bool[local_ref])):
                    chains.coord[k][np.invert(ext_bool[local_ref])] *= 0
                # end if
            # end for
        # end for
        # COMPUTE EXTENSION COEFFICIENTS
        # go through all diagonal
        Sdeg = start_deg
        Sn_dim = start_n_dim
        for idx, chains in enumerate(Hom_reps):
            print("lift to infty page")
            # lift to infinity page and substract betas
            Betas, _ = self.first_page_lift(Sn_dim, Sdeg, chains,
                                            death_radii)
            # go up to target_page
            Betas, _ = self.lift_to_page(Sn_dim, Sdeg, self.no_pages, Betas,
                                         barcode_extension)
            # STORE EXTENSION COEFFICIENTS
            self.extensions[start_n_dim][start_deg][idx] = Betas.T
            # MODIFY TOTAL COMPLEX REPS using BETAS
            print("modify using betas")
            if np.any(Betas):
                for ext_deg, Schains in enumerate(
                        self.Hom_reps[self.no_pages - 1][Sn_dim][Sdeg]):
                    # compute chains using betas and substract to reps
                    # problem with local sums!!!
                    local_chains_beta = local_chains.sums(Schains, -Betas)
                    for k, local_coord in enumerate(
                            Hom_reps[ext_deg + idx].coord):
                        local_ref = Hom_reps[ext_deg + idx].ref[k]
                        if (len(local_ref)) > 0 and (
                                len(local_chains_beta.ref[k]) > 0):
                            if not np.array_equal(
                                    local_ref, local_chains_beta.ref[k]):
                                raise ValueError
                            Hom_reps[ext_deg + idx].coord[k] = (
                                local_coord + local_chains_beta.coord[k]
                                ) % self.p
                        elif len(local_chains_beta.ref[k]) > 0:
                            Hom_reps[ext_deg + idx].ref[
                                k] = local_chains_beta.ref[k]
                            Hom_reps[ext_deg + idx].coord[
                                k] = local_chains_beta.coord[k]
                        # end elif
                    # end for
                # end for
            # end if
            # reduce up to 1st page using gammas
            print("reduce using gammas")
            for target_page in range(self.no_pages, 1, -1):
                # get coefficients on first page
                Betas, _ = self.first_page_lift(Sn_dim, Sdeg, chains,
                                                death_radii)
                # go up to target_page
                Betas, Gammas = self.lift_to_page(
                    Sn_dim, Sdeg, target_page, Betas, barcode_extension)
                # if lift target_page is nonzero, raise an error
                if np.any(Betas):
                    raise(RuntimeError)
                # MODIFY TOTAL COMPLEX REPS using GAMMAS
                if np.any(Gammas):
                    # compute coefficients of Gammas in 1st page
                    image_classes = np.matmul(
                        self.Im[target_page-1][Sn_dim][Sdeg].coordinates,
                        -Gammas.T)
                    # look for bars that might be already zero
                    if target_page == 2:
                        # obtain coefficients for gammas
                        image_chains = local_chains(
                            self.nerve_spx_number[Sn_dim])
                        prev = 0
                        for nerv_idx, next in enumerate(self.cycle_dimensions[
                                Sn_dim][Sdeg]):
                            if prev < next:
                                image_chains.add_entry(
                                    nerv_idx, range(np.size(Gammas, 0)),
                                    np.matmul(
                                        image_classes[prev:next].T,
                                        self.Hom[0][Sn_dim][nerv_idx][
                                            Sdeg].coordinates.T)
                                )
                            prev = next
                        # end for
                    else:
                        image_chains = local_chains.sums(
                            self.Hom_reps[target_page-2][Sn_dim][Sdeg][0],
                            image_classes.T
                        )
                    # end else
                    chains += image_chains
                # end if
            # end for
            # lift to first page
            print("lift and cech diff")
            Betas, lift_coord = self.first_page_lift(Sn_dim, Sdeg, chains,
                                                     death_radii)

            # correct sign of lift_coord and trivialise references
            lift_coord.minus()
            # end for
            # if lift to first page is nonzero, raise an error
            if np.any(Betas):
                raise(RuntimeError)

            if Sn_dim > 0:
                # compute Cech differential of lift_coord
                # and add to current reps
                image_chains = self.cech_diff(Sn_dim - 1, Sdeg + 1, lift_coord)
                Hom_reps[idx+1] = Hom_reps[idx + 1] + image_chains
            # advance reduction position
            Sdeg += 1
            Sn_dim -= 1
        # end for
    # end def

# end spectral_sequence class #################################################
