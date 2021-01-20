
    # check that Hom are indeed cycles
#    for idx, hom in enumerate(Hom):
#        if hom.dim > 0 and idx > 0:
#            trivial_image = np.matmul(local_differentials[idx], hom.coordinates)
#            if np.any(trivial_image % p):
#                print(trivial_image % p)
#                raise(RuntimeError)
#
