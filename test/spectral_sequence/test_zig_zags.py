


# TEST cycles (inside add_output_first)
########################################################################
# check that we added cycles in dim 1
if n_dim > 0:
    dim_local = len(self.nerve[n_dim])
else:
    dim_local = self.nerve[n_dim]
for nerve_spx_index in range(dim_local):
    if self.Hom[0][n_dim][nerve_spx_index][1].dim > 0:
        trivial_image = np.matmul(
            self.zero_diff[n_dim][nerve_spx_index][1],
            self.Hom[0][n_dim][nerve_spx_index][1].coordinates
        ) % self.p
        if np.any(trivial_image):
            print("n_dim:{}, nerve_spx_index:{}".format(n_dim, nerve_spx_index))
            raise(ValueError)
    # end if
# end for
########################################################################

# TEST commutativity and cycles (inside high_differential)
########################################################################
# check that chains[-1] is zero through  first vertical then horizontal
if n_dim == 2 and deg == 0:
    # compute vertical differential
    chains_new = [[],[]]
    for idx, ref in enumerate(chains[0]):
        chains_new[0].append(ref)
        if len(ref) > 0:
            chains_new[1].append(np.matmul(
                self.zero_diff[1][idx][1],
                chains[1][idx].T
            ).T)
        else:
            chains_new[1].append([])
    # check that representative is correct
    # compute horizontal differential of first entry
    horiz_image = self.cech_diff(1,0, self.Hom_reps[current_page-1][n_dim][deg][0])
    for idx, local_ch in enumerate(horiz_image[1]):
        if np.any(local_ch % self.p):
            if np.any((local_ch[chains[0][idx]] + chains_new[1][idx]) % self.p):
                print((local_ch[chains[0][idx]] + chains_new[1][idx]) % self.p)
                raise(RuntimeError)
    #compute horizontal differential
    trivial_image = self.cech_diff(0, 0, chains_new)
    for triv in trivial_image[1]:
        if len(triv) > 0:
            if np.any(triv % self.p):
                print(triv % self.p)
                raise(RuntimeError)
    image_chains = self.cech_diff(0,1,chains)
    trivial_im = [[],[]]
    for idx, ref in enumerate(image_chains[0]):
        trivial_im[0].append(ref)
        if len(ref) > 0:
            trivial_im[1].append(np.matmul(
                self.zero_diff[0][idx][1],
                image_chains[1][idx].T
            ).T)
            if np.any(trivial_im[1][-1] % self.p):
                print(trivial_im[1][-1] % self.p)
                raise(RuntimeError)
        else:
            chains_new[1].append([])

########################################################################

# TEST lift to page: whether it lifts hom coordinates (inside high_differential)
########################################################################


########################################################################


# TEST commumativity of second page reps (in compute_two_page_representatives)
########################################################################
if n_dim > 0:
    cech_diff_im = self.cech_diff(n_dim-1, deg, chains)
    betas, _ = self.first_page_lift(n_dim-1, deg, cech_diff_im, R)
    if np.any(betas):
        print(betas)
        raise(RuntimeError)
    for idx, ref in enumerate(lift[0]):
        if len(ref) > 0:
            vert_im = np.matmul(
                self.zero_diff[n_dim-1][idx][deg+1],
                lift[1][idx].T
            ).T
            cech_local_nontrivial = cech_diff_im[1][idx][
                cech_diff_im[1][idx].any(axis=1)]
            if np.any((vert_im + cech_local_nontrivial)%self.p):
                print((vert_im + cech_local_nontrivial)%self.p)
                raise(RuntimeError)
        else:
            if len(cech_diff_im[0][idx]) > 0 and np.any(cech_diff_im[1][idx]):
                print("here")
                print("local_lift")
                print(lift[1][idx])
                print("local_cech_im")
                print(cech_diff_im[1][idx])
                raise(ValueError)
########################################################################

# TEST that image of Cech differential is a cycle (in cech_diff_and_lift_local)
########################################################################
if deg > 0:
    trivial_image = np.matmul(
        self.zero_diff[n_dim][nerve_spx_index][deg],
        local_chains) % self.p
    if np.any(trivial_image):
        print("Image of Cech diff not a cycle")
        raise(RuntimeError)
########################################################################

# TEST radius nonzero entries in betas_aux (in cech_diff_and_lift_local)
########################################################################
for g in generators:
    nonzero_coeff_bars = self.Hom[0][n_dim][nerve_spx_index][
        deg].barcode[np.nonzero(betas_aux[g])[0]]
    if len(nonzero_coeff_bars) > 0:
        if R[g] < np.max(nonzero_coeff_bars[:,0]):
            raise(ValueError)
########################################################################
