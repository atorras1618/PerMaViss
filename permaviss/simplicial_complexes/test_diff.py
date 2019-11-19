"""
    test_rip.py
"""

import numpy as np
from vietoris_rips import vietoris_rips as viRip
from differentials import *
import scipy.spatial.distance as dist # this module computes distance matrices and compressed distance matrices

X = np.array([[1,2],[0,0],[1,0],[1,1],[0,1]])
print("X:")
print(X)
Dist = dist.squareform(dist.pdist(X))
print("Dist:")
print(Dist)
C = viRip(Dist, 3, 3)
D = complex_differentials(C, 5)
print("Complex")
for i, c in enumerate(C):
    print("dim: {}".format(i))
    print(c)
print("Differential")
for i, d in enumerate(D):
    print("dim: {}".format(i))
    print(d)
