import permaviss
import numpy as np

from permaviss.sample_point_clouds.examples import random_cube, grid
from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss
from permaviss.gauss_mod_p.gauss_mod_p import gauss_col_rad

if __name__=='__main__':
    # Here check the correctness of gauss_col_rad
    A = np.array([[0,0,1,1,1],
                  [1,1,0,1,1],
                  [0,0,1,0,1]])
    R = np.array([4,1,2,3,4])
    p = 5
    Res, T = gauss_col_rad(A, R, 3, p)
    Exp_Res = np.array([[0,0,1,1,0],
                        [1,1,0,0,0],
                        [0,0,1,0,0]])
    Exp_T = np.array([[1.,0., 0., 0., 4.],
                      [0.,1., 0., 4., 0.],
                      [0.,0., 1., 0., 4.],
                      [0.,0., 0., 1., 0.],
                      [0.,0., 0., 0., 1.]])
    if (not np.array_equal(Exp_Res,Res)) or (not np.array_equal(Exp_T,T)):
        raise RuntimeError
    # Here we check how local coordinate data structures work
    X = grid(5, 5) #+ np.random.rand(25,2)*0.03
    S = create_MV_ss(X, 1.7, 3, 2, 2, 5)
    current_page = 1
    n_dim = 1
    deg = 1

    print("no_classes:{}".format(S.page_dim_matrix[current_page, deg, n_dim]))
    initial_sum = np.identity(S.page_dim_matrix[current_page, deg, n_dim])
    reference, local_coordinates = S.localize_coordinates(
        initial_sum, n_dim, deg)
    R = np.zeros(S.page_dim_matrix[current_page, deg, n_dim])
    prev=0
    for nerv_spx_index, _ in enumerate(S.nerve[n_dim]):
        next= S.cycle_dimensions[n_dim][deg][nerv_spx_index]
        R[prev:next] = S.Hom[0][n_dim][nerv_spx_index][deg].barcode[:,0]
        prev=next
    # end for
    n_dim = 0
    S.cech_diff_and_lift(R, reference, local_coordinates, n_dim, deg)
