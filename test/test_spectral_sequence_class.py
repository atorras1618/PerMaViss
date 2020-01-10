
# # test for double complex well defined
    # # check for Cech differential to be an actual differential
    # for deg in range(max_dim):
    #     for start_n in range(2, nerve_dim):
    #         for nerve_spx in range(len(nerve[start_n])): 
    #             
    #     # end for
    # # end for
    # MV_ss.double_complex_check()

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

