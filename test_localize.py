import permaviss
import numpy as np

from permaviss.sample_point_clouds.examples import random_cube
from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss

if __name__=='__main__':
    X = random_cube(30, 2)
    S = create_MV_ss(X, 0.3, 3, 2, 0.31, 5)
    current_page = 1
    n_dim = 0
    deg = 0
    initial_sum = np.identity(S.page_dim_matrix[current_page, deg, n_dim])
    S.localize_coordinates(initial_sum, n_dim, deg)
