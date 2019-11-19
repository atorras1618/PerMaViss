"""
    spectral_sequence_class.py
    
    This module will implement barcode basis systems for spectral sequences.  
    These will be used in order to suppport the spectral sequence logic. 
    In particular, dimensions for each position will be stored for each page
    of the spectral sequence. 
    This will include a method for translating a vector to zero page and
    backwards to current page. 

"""
import numpy as np

from ..persistence_algebra.persistence_algebra import barcode_basis
from ..simplicial_complexes.differentials import complex_differentials
from ..gauss_mod_p.functions import solve_mod_p, solve_matrix_mod_p, multiply_mod_p

class spectral_sequence(object):
    """
        # __init__
        # zig_zag
        # load_to_zero_page 
        # cech_differential 
        # lift_to_page
        # lift_preimage
        # extension
    """
    def __init__(self, nerve, nerve_point_cloud, points_IN, max_dim, max_r, no_pages, p):
        """
        This saves enough space for the spectral sequence.
        Perhaps add subcomplexes and differentials into here. 
        Add a vector thay would translate local coordinates to global
        and the other way round.
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
        self.differentials = []
        self.cycle_dimensions = []
        # vectors that translate local indices to global
        self.local_coord = []
        self.tot_complex_reps = []
        for n_dim in range(len(nerve)):
            self.Hom[0].append([])
            self.Im[0].append([])
            self.PreIm[0].append([])
            self.subcomplexes.append([])
            self.differentials.append([])
            self.cycle_dimensions.append([])
            self.tot_complex_reps.append([])
            for deg in range(self.no_rows):
                self.tot_complex_reps[n_dim].append([])
        
        
        # make lists to store information in higher pages
        for k in range(1, no_pages):
            self.Hom.append([])
            self.Im.append([])
            self.PreIm.append([])
            for n_dim in range(self.no_columns):
                self.Hom[k].append([])
                self.Im[k].append([])
                self.PreIm[k].append([])
                for deg in range(self.no_rows):
                    self.Hom[k][n_dim].append([])
                    self.Im[k][n_dim].append([])
                    self.PreIm[k][n_dim].append([])
        
        # save space for dimension matrices for all pages
        # the order of variables is for easier printing of the spectral sequence
        self.page_dim_matrix = np.zeros((no_pages+1, max_dim, self.no_columns)).astype(int)
        # Save space for extension matrices
        self.extension_matrices = []
        for deg in range(self.no_rows):
            self.extension_matrices.append([])
            for n_dim in range(self.no_columns):
                self.extension_matrices[deg].append([])
        
        # define persistent homology and order of diagonal basis
        self.persistent_homology = []
        self.order_diagonal_basis = []
        
            
    ###############################################################################
    # add content to first page
        
    def add_output_first(self, output, n_dim):
        self.subcomplexes[n_dim] = [it[0] for it in output]
        self.differentials[n_dim] = [it[1] for it in output]
        self.Hom[0][n_dim] = [it[2] for it in output]
        self.Im[0][n_dim] = [it[3] for it in output]
        self.PreIm[0][n_dim] = [it[4] for it in output]

        # Number of simplices in nerve[n_dim]
        if n_dim > 0:
            n_spx_number = np.size(self.nerve[n_dim],0)
        else:
            n_spx_number = self.nerve[0]

        # add this to check that the level of intersection is not empty.
        # otherwise we might run out of index range
        if len(self.Hom[0][n_dim]) > 0:
            for deg in range(self.no_rows):
                no_cycles = 0
                # cumulative dimensions
                self.cycle_dimensions[n_dim].append(np.zeros(n_spx_number+1).astype(int))
                for k in range(n_spx_number):
                    # Generate page dim matrix and local_coordinates info
                    cycles_in_cover = self.Hom[0][n_dim][k][deg].dim
                    no_cycles += cycles_in_cover
                    self.cycle_dimensions[n_dim][deg][k] = no_cycles
                # end for
                self.page_dim_matrix[1, deg, n_dim] = no_cycles
            # end for
        # end if
        
    ###############################################################################
    # double complex check

    def double_complex_check(self):
        print("checking vertical diff")
        # check that complex is a double complex
        for n_dim in range(self.no_columns):
            # check for vertical differential
            if n_dim == 0:
                rk_nerve = self.nerve[n_dim]
            else:
                rk_nerve = len(self.nerve[n_dim])
            for nerv_spx in range(rk_nerve):
                for deg in range(2, self.no_rows):
                    zero_coordinates = []
                    coord_matrix = np.identity(len(self.subcomplexes[n_dim][nerv_spx][deg]))
                    for spx_idx in range(len(self.subcomplexes[n_dim][nerv_spx][deg])):
                        zero_coordinates.append({nerv_spx : coord_matrix[spx_idx]})
                    # end for
                    im_vert = self.vert_image(zero_coordinates, n_dim, deg)
                    im_vert = self.vert_image(im_vert, n_dim, deg-1)
                    for i, coord in enumerate(im_vert):
                        for n in iter(coord):
                            if np.any(coord[n]):
                                print("vertical not differentials")
                                raise RuntimeError
                            # end if
                        # end for
                    # end for
                # end for
            # end for
        # end for
        print("checking Cech diff")
        # check for Cech differentials
        for deg in range(self.no_rows):
            for n_dim in range(2, self.no_columns):
                for nerv_spx in range(len(self.nerve[n_dim])):
                    zero_coordinates = []
                    if deg == 0:
                        no_simplexes = self.subcomplexes[n_dim][nerv_spx][deg]
                    else:
                        no_simplexes = len(self.subcomplexes[n_dim][nerv_spx][deg])
                    coord_matrix = np.identity(no_simplexes)
                    for spx_idx in range(no_simplexes):
                        zero_coordinates.append({nerv_spx : coord_matrix[spx_idx]})
                    # end for
                    cech_diff = self.cech_differential(zero_coordinates, n_dim, deg)
                    cech_diff = self.cech_differential(cech_diff, n_dim-1, deg)
                    for i, coord in enumerate(cech_diff):
                        for n in iter(coord):
                            if np.any(coord[n]):
                                print("Cech differentials not differentials")
                                raise RuntimeError
                            # end if
                        # end for
                    # end for
                # end for
            # end for
        # end for

        print("checking anticommutativity")
        # check for anticommutativity
        for n_dim in range(1, self.no_columns):
            for nerv_spx in range(len(self.nerve[n_dim])):
                for deg in range(1, self.no_rows): 
                    zero_coordinates = []
                    coord_matrix = np.identity(len(self.subcomplexes[n_dim][nerv_spx][deg]))
                    for spx_idx in range(len(self.subcomplexes[n_dim][nerv_spx][deg])):
                        zero_coordinates.append({nerv_spx : coord_matrix[spx_idx]})
                    # end for
                    im_left = self.cech_differential(zero_coordinates, n_dim, deg)
                    im_left = self.vert_image(im_left, n_dim - 1, deg)
                    im_right = self.vert_image(zero_coordinates, n_dim, deg)
                    im_right = self.cech_differential(im_right, n_dim, deg-1)
                    for i, coord in enumerate(im_left):
                        result = _add_dictionaries([1,1], [coord, im_right[i]], self.p)
                        for n in iter(result):
                            if np.any(result[n]):
                                print("n_dim:{}, deg:{}".format(n_dim, deg))
                                print("anticommutativity fails")
                                raise RuntimeError
                            # end if
                        # end for
                    # end for
                # end for
            # end for
        # end for

    ###############################################################################
    # add higher page contents

    def add_higher_output(self, Hom, Im, PreIm, start_n_dim, start_deg, current_page):
        n_dim = start_n_dim
        deg = start_deg
        for i, h in enumerate(Hom):
            self.PreIm[current_page][n_dim][deg] = PreIm[i]
            self.Hom[current_page][n_dim][deg] = h
            self.page_dim_matrix[current_page+1, deg, n_dim] = h.dim
            self.Im[current_page][n_dim][deg] = Im[i]
            deg = start_deg - current_page + 1
            n_dim += current_page
         


    ###############################################################################
    # zig_zag 
    
    def zig_zag(self, n_dim, deg, current_page, lift_sum=True, initial_sum = [], store_reps=False):
        """
        This will take an expression on the nth page of the spectral sequence.
        It then computes its image under the nth differential and returns it.
        INPUT:
            -n_dim, deg: position on the page.
            -current_page: n, number of page where we are.
            -lift_sum (default True): whether we return a lifted sum on current_page or
                        we return an expresion on the 0 page instead. 
            -initial_sum (default identity(see below)): expression for which we want to compute zig-zag
                        Coordinates come as columns. 
            -store_reps (default False): whether we want to store total complex representatives. 
        OUTPUT:
            target_coordinates: image of start_coordinates under the current_page differential.
        """
        if len(initial_sum)==0:
            initial_sum = np.identity(self.page_dim_matrix[current_page, deg, n_dim])
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
        zero_coordinates, R = self.load_to_zero_page(initial_sum, n_dim, deg, current_page)
        if store_reps:
            self.tot_complex_reps[start_n_dim][start_deg].append(zero_coordinates)
            if n_dim == 0:
                return
            # end if
        # end if
                
        # Compute zigzag for the zero_page expression
        zero_coordinates = self.cech_differential(zero_coordinates, n_dim, deg)
        
        n_dim -= 1
        for k in range(current_page - 1):
            zero_coordinates = self.lift_preimage(zero_coordinates, R,  n_dim, deg)
            deg += 1
            if store_reps:
                self.tot_complex_reps[start_n_dim][start_deg].append(zero_coordinates)
                if n_dim == 0:
                    return

            zero_coordinates = self.cech_differential(zero_coordinates, n_dim, deg) 
            n_dim -= 1
    
        # Lift back to current_page 
        if lift_sum:
            target_coordinates = self.lift_to_page(zero_coordinates, R,  n_dim, deg, current_page) 
        else:
            target_coordinates = zero_coordinates

        return target_coordinates
   

    ###############################################################################
    # vert_image

    def vert_image(self, initial_sum, n_dim, deg):
        """
        Given an expression on zero page, computes the image under the vertical differentials.
        """
        result = []
        for i, rep in enumerate(initial_sum):
            result.append({})
            for nerv_spx in iter(rep):
                image = multiply_mod_p(self.differentials[n_dim][nerv_spx][deg], rep[nerv_spx], self.p)
                if np.any(image):
                    result[i][nerv_spx] = image
                # end if
            # end for
        # end for
        return result
    
    ###############################################################################
    # load_to_zero_page 
    
    def load_to_zero_page(self, initial_sum, n_dim, deg, current_page):
        """
        Given an expression on the nth page of the spectral sequence, we pick a
        representative on the 0th page and return it.
        INPUT:
            -initial_sum: expression to load to zero page. Coordinates for each element are on the columns.
            -n_dim, deg: position and number in page
            -current_page: page where we start
            -Hom: barcode basis for homology
        OUTPUT:
            -target_coordinates: List of coordinates in zero page. 
            -R: inital radius of each bar.
        """
        # Load sum to 1st page
        aux_sum = initial_sum
        for k in range(current_page-1, 0, -1):
            aux_sum = multiply_mod_p(self.Hom[k][n_dim][deg].coordinates, aux_sum, self.p)
        # end for

        # Load sum from 1 to 0 page
	# Transform coordinates to dictionary data types
        target_coordinates = []
        for i, coord in enumerate(aux_sum.T):
            target_coordinates.append({}) 
            for nerve_spx in range(len(self.cycle_dimensions[n_dim][deg])-1):
                local_coordinates = coord[self.cycle_dimensions[n_dim][deg][nerve_spx-1]: 
                                          self.cycle_dimensions[n_dim][deg][nerve_spx]]

                # If the local coordinates are nontrivial, add them to dictionary
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
                for nerve_spx in range(len(self.cycle_dimensions[n_dim][deg])-1):
                    local_coordinates = coord[self.cycle_dimensions[n_dim][deg][nerve_spx-1]: 
                                              self.cycle_dimensions[n_dim][deg][nerve_spx]]
                    # Take the maximum of birth radius over all nontrivial local coordinates
                    if len(local_coordinates) > 0 and np.any(local_coordinates):
                        R[i] = max(R[i], self.Hom[0][n_dim][nerve_spx][deg].birth_radius(local_coordinates))
                    # end if
                # end for
            # end for
        else:
            for i, coord in enumerate(initial_sum.T):
                R[i] = self.Hom[current_page -1][n_dim][deg].birth_radius(coord)        
            # end for
        # end else

        return target_coordinates, R
    
    
    ###############################################################################
    # cech_differential 
    
    def cech_differential(self, start_coordinates, n_dim, deg):
        """
        Performs the Cech Differential on start_coordinates
        INPUT:
            - n_dim, deg : current position in double complex.
            - start_coordinates: list of coordiantes on zero page
        OUTPUT:
            - target_coordinates: list containing the image of Cech Differential
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
                nerv_boundary_indx = np.nonzero(self.nerve_differentials[n_dim][:,nerve_spx_index])[0]
                nerv_boundary_coeff = self.nerve_differentials[n_dim][:,nerve_spx_index][nerv_boundary_indx]
                # go through all faces for each simplex in nerve
                for nerve_face_index, nerve_coeff in zip(nerv_boundary_indx, nerv_boundary_coeff):
                    if deg == 0:
                        # inclusions for points
                        # Preallocate space
                        if nerve_face_index not in cech_image:
                            cech_image[nerve_face_index] = np.zeros(self.subcomplexes[n_dim-1
                                                                        ][nerve_face_index][0])
                        # end if
                        for point_idx, point_coeff in enumerate(coordinates[nerve_spx_index]):
                            if point_coeff != 0:
                                face_point_idx = np.argmax(self.points_IN[n_dim-1][
                                                                nerve_face_index] == points_IN[point_idx])
                                cech_image[nerve_face_index][face_point_idx] += nerve_coeff * point_coeff * deg_sign
                                cech_image[nerve_face_index][face_point_idx] %= self.p
                            # end if
                        # end for
                    else:
                        # inclusions for edges, 2-simplices and higher simplices
                        # Preallocate space
                        if nerve_face_index not in cech_image:
                            cech_image[nerve_face_index] = np.zeros(len(self.subcomplexes[n_dim-1
                                                                        ][nerve_face_index][deg]))
                        # end if
                        # Iterate over nontrivial local simplices in domain
                        spx_indices = np.nonzero(coordinates[nerve_spx_index])[0]
                        spx_coefficients = coordinates[nerve_spx_index][spx_indices]
                        for spx_index, spx_coeff in zip(spx_indices, spx_coefficients):
                            # Obtain IN for vertices of simplex
                            vertices_spx = points_IN[self.subcomplexes[n_dim][nerve_spx_index][deg][spx_index]]
                            # Iterate over simplices in range to see which one has vertices_spx as vertices.
                            for im_indx, im_spx in enumerate(self.subcomplexes[n_dim-1][nerve_face_index][deg]):
                                vertices_face = self.points_IN[n_dim-1][nerve_face_index][im_spx.astype(int)] 
                                # When the vertices coincide, break the loop
                                if len(np.intersect1d(vertices_spx, vertices_face)) == deg + 1: 
                                    cech_image[nerve_face_index][im_indx] += spx_coeff * nerve_coeff * deg_sign
                                    cech_image[nerve_face_index][im_indx] %= self.p
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
    
    
    ###############################################################################
    # lift_to_page
    
    def lift_to_page(self, start_coordinates, R,  n_dim, deg, target_page):
        """
        Lifts some zero page element to a target page. 
        INPUT:
            - start_coordinates: list of dictionary pairs on zero page
            - R: list of radius at which we want to lift
            - n_dim, deg: position on page 
            - target_page: where we want to lift this sum
        OUTPUT:
            - target_coordinates: lifted start_coordinates to target page
                                Target coordinates are in rows. 
        """
        # Lift from 0 to 1 page
        # Transform from dictionary to np.array
        target_coordinates = np.zeros((len(start_coordinates), self.page_dim_matrix[1, deg, n_dim]))
        for i, coord in enumerate(start_coordinates):
            for nerve_spx_index in iter(coord):
                # solve (Im|Hom) localy
                # This is with respect to the active coordinates at R[i]
                Hom_dim = self.Hom[0][n_dim][nerve_spx_index][deg].dim
                if Hom_dim > 0:
                    Hom = self.Hom[0][n_dim][nerve_spx_index][deg].active_domain(R[i])
                Im_dim = self.Im[0][n_dim][nerve_spx_index][deg].dim 
                if Hom_dim > 0 and len(Hom) > 0:
                    if Im_dim > 0:
                        Im = self.Im[0][n_dim][nerve_spx_index][deg].active_domain(R[i])
                        Im_dim = np.size(Im,1)
                        Im_Hom = np.append(Im, Hom, axis=1)
                            
                    else:
                        Im_Hom = Hom
                    # end else

                    active_coordinates = solve_mod_p(Im_Hom, coord[nerve_spx_index], self.p)[Im_dim:]
                    # Save resulting coordinates as 1 page coordinates
                    local_lifted_coordinates = np.zeros(self.cycle_dimensions[n_dim][deg][nerve_spx_index] -
                                                        self.cycle_dimensions[n_dim][deg][nerve_spx_index-1])
                    local_active = self.Hom[0][n_dim][nerve_spx_index][deg].active(R[i])
                    local_lifted_coordinates[local_active] = active_coordinates
                    target_coordinates[i, self.cycle_dimensions[n_dim][deg][nerve_spx_index-1]: 
                                       self.cycle_dimensions[n_dim][deg][nerve_spx_index]] = local_lifted_coordinates            
                # end if
            # end for
        # end for
   
        # Lift from 1 to target page 
        for k in range(1, target_page):
            prev_target_coordinates = target_coordinates
            target_coordinates = np.zeros((len(start_coordinates), self.page_dim_matrix[k+1, deg, n_dim]))
            for i, coord in enumerate(prev_target_coordinates):
                # solve (Im|Hom) localy
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
                    if np.size(Im_Hom,1) > 0:
                        prev_active_coordinates = prev_target_coordinates[i][self.Hom[k][n_dim][deg].prev_basis.active(R[i])]
                        if np.any(prev_active_coordinates):
                            active_coordinates = solve_mod_p(Im_Hom, prev_active_coordinates, self.p)[Im_dim:]
                            lifted_coordinates[self.Hom[k][n_dim][deg].active(R[i])] = active_coordinates
                        # end if
                    # end if
                    target_coordinates[i] = lifted_coordinates
                # end if
            # end for 
        # end for
                   
        return target_coordinates
    
    ###############################################################################
    # lift_preimage
    
    def lift_preimage(self,  start_coordinates, R, n_dim, deg, sign=-1):
        """
        Given a zero page element, we lift it using the vertical differentials. 
        Additionally, we multiply the lift by -1 mod self.p
        In this method we assume that this can be done. 
        INPUT:
            - start_coordinates: list of dictionaries on zero page
            - R: list of birth radius of start_coordinates 
            - n_dim, deg: position on spectral sequence
        OUTPUT:
            - lifted_coordinates: lifted sum on position (n_dim, deg+1)
        """
   
        lifted_coordinates = []
        for i, coord in enumerate(start_coordinates):
            lifted_coordinates.append({})
            for nerve_spx_index in iter(coord):
                # Only lift nontrivial expressions
                if np.any(coord[nerve_spx_index]):
                    Im = self.Im[0][n_dim][nerve_spx_index][deg].active_domain(R[i])
                    solution = solve_mod_p(Im, coord[nerve_spx_index], self.p)
                    local_lift = multiply_mod_p(self.PreIm[0][n_dim][nerve_spx_index][deg+1][:, 
                                                self.Im[0][n_dim][nerve_spx_index][deg].active(R[i])],
                                                solution, self.p)
                    # lifted coordinates in terms of all coordinates 
                    lifted_coordinates[i][nerve_spx_index] = sign * local_lift % self.p
                # end if
            # end for
        # end for
    
        return lifted_coordinates

    ###############################################################################
    # extension
        
    def extension(self, start_n_dim, start_deg):
        """
        Generate extension block matrices for given n_dim and deg
        INPUT:
            -start_n_dim, start_deg: position on spectral sequence where we want
                        to compute the extensions of the contained barcodes. 
        OUTPUT:
            -extensions: list containing matrix blocks of extensions. 
        """
        extensions = []
        death_R = self.Hom[self.no_pages-1][start_n_dim][start_deg].barcode[:,1]
        dim_domain = self.Hom[self.no_pages-1][start_n_dim][start_deg].dim
        not_infty = death_R < self.max_r
        death_R = death_R[not_infty]
        not_infty = np.sort(np.nonzero(not_infty)[0])
        dim_ext = len(not_infty)
        # Take only the representatives that are not infinity
        total_representatives = self.tot_complex_reps[start_n_dim][start_deg]
        representatives = []
        for i, zero_coord in enumerate(total_representatives):
            representatives.append([_copy_dictionary(zero_coord[i]) for i in not_infty])

        # Start in specified entry, and go backwards
        ext_deg = 0
        deg = start_deg
        for n_dim in range(start_n_dim, -1, -1):
            # find and store extension coefficients
            extension = np.zeros((self.page_dim_matrix[self.no_pages-1][deg][n_dim], dim_domain))
            if len(not_infty)>0:
                extension_coefficients = self.lift_to_page(representatives[ext_deg], death_R, 
                                                            n_dim, deg, self.no_pages-1)
                extension[:, not_infty] = extension_coefficients.T
                # change total chain representatives accordingly
                current_representatives = self.tot_complex_reps[n_dim][deg]
                for i in range(dim_ext):
                    # substract sum of coefficients along all total complex, starting at n_dim, deg
                    for k, rep in enumerate(current_representatives):
                        substractor = _add_dictionaries(extension_coefficients[i], rep, self.p) 
                        representatives[k+ext_deg][i] = _add_dictionaries([1,-1], 
                                                            [representatives[k+ext_deg][i], substractor], 
                                                            self.p)
                    # end for
                # end for 
                # substract images to representatives, so that we can lift the vertical differentials 
                if n_dim > 0:
                    for page in range(self.no_pages-1, 0, -1):
                        image_coefficients = self.image_coordinates(representatives[ext_deg], death_R, 
                                                                    n_dim, deg, page)
                        # If image in position is nontrivial
                        if len(image_coefficients)>0 and np.any(image_coefficients):
                            preimages_sums = multiply_mod_p(image_coefficients, 
                                                    self.PreIm[page][n_dim+page][deg-page+1].T, self.p)
                            # substract images to zero page sums
                            if np.any(preimages_sums):
                                images = self.zig_zag(n_dim-page, deg+page-1, page, lift_sum=False, 
                                                      initial_sum = preimages_sums.T) 
                                for i in range(dim_ext):
                                    representatives[ext_deg][i] = _add_dictionaries([1,-1], 
                                                                    [representatives[ext_deg][i], images[i]], 
                                                                    self.p)
                                # end for
                            # end if
                        # end if
                    # end for 
                    # lift through vertical differential the last nontrivial entry in total complex diagonal
                    preimages_sums = self.lift_preimage(representatives[ext_deg], death_R, n_dim, deg)
                    representatives[ext_deg] = {}
                    # compute the cech differential for preimage_sums
                    cech_images = self.cech_differential(preimages_sums, n_dim, deg+1)
                    # substract cech differential of preimage
                    for i in range(dim_ext):
                        representatives[ext_deg + 1][i] = _add_dictionaries([1,1], 
                                            [representatives[ext_deg + 1][i], cech_images[i]], self.p)
                    # end for
                # end if
            # end if
            extensions.append(extension)
            deg += 1
            ext_deg += 1
        # end for
        return extensions
            
            
    ###############################################################################
    # image_coordinates

    def image_coordinates(self, start_coordinates, R,  n_dim, deg, target_page):
        """
        Lifts 0 page expression to target_page. 
        Then, assuming that this lies on the image of the spectral sequence
        differential, it computes the coordinates on the image basis. 
        INPUT:
            - start_coordinates: coordinates on zero page for several barcodes
            - R: initial radius of barcodes
            - n_dim, deg: position in spectral sequence
            - target_page: page of spectral sequence at which we want to solve the image equation.
        OUTPUT:
            - image_coordinates: start_coordinates classes in target_page written in terms of image of differential.
        """ 
        # Lift start_coordinates from 0 to target_page
        lifted_sum = self.lift_to_page(start_coordinates, R,  n_dim, deg, target_page)
        # If image is trivial, return
        Im_dim = self.Im[target_page][n_dim][deg].dim
        if Im_dim == 0:
            return []

        # Compute each coordinate in terms of images
        image_coordinates = np.zeros((len(start_coordinates), Im_dim))
        for i, rad in enumerate(R):
            Im = self.Im[target_page][n_dim][deg].active_coordinates(rad)
            active_lifted_sum = lifted_sum[i][self.Im[target_page][n_dim][deg].prev_basis.active(rad)]
            if len(active_lifted_sum)>0:
                image_coordinates[i,self.Im[target_page][n_dim][deg].active(rad)] = solve_mod_p(
                                                                Im, active_lifted_sum, self.p)
       
        return image_coordinates
    # end def
# end spectral_sequence class

###############################################################################
# _add_dictionaries
# 
# This mainly supports extension. Does what its name says.
# That is, entries with the same key are added. 

def _add_dictionaries(coefficients, representatives, p):
    rep_sum = {}
    for i, rep in enumerate(representatives):
        for spx_idx in iter(rep): 
            if spx_idx not in rep_sum:
                rep_sum[spx_idx] = (coefficients[i] * rep[spx_idx]) % p
            else:
                rep_sum[spx_idx] = (rep_sum[spx_idx] + coefficients[i] * rep[spx_idx]) % p
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
# _copy_dictionary
# 

def _copy_dictionary(original):
    copy = {}
    for spx_idx in iter(original):
        copy[spx_idx] = np.copy(original[spx_idx])
    
    return copy
