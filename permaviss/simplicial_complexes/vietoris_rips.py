"""
    vietoris_rips.py
"""

import numpy as np
import scipy.spatial.distance as dist # this module computes distance matrices and compressed distance matrices

def _lower_neighbours(G, u):
    """
    INPUT:
        -G: graph of neighbours
        -u: vertex of the graph
    OUTPUT:
        -l: list of lower neighbours of u
    """
    l = []  # list of lowest neighbours to be computed.
    for e in G:
        if e[1] == u:
            l.append(e[0])

    l.sort()
    return l


def vietoris_rips(Dist, max_r, max_dim):
    """
    This computes the Vietoris-Rips complex with simplexes of dimension less
    or equal to max_dim, and with maximum radius specified by max_r
    INPUT:
        - Dist : numpy.array distance matrix.
        - max_r: maximum radius of filtration.
        - max_dim: maximum dimension of computed Rips complex.
    OUTPUT:
        - C: simplicial complex numpy.array. Each dimension stores the simplexes
        given by their respective list of vertices. Additionally, at the start
        of each simplex list, there is attached the minimal radius of inclusion
        of the corresponding simplex.
        - R: list with radius of simplices
    """
    if max_dim < 1:  # at least returns always the neighbourhood graph
        max_dim = 1

    C = []
    R = []
    for i in range(max_dim + 1):
        C.append([])
        R.append([])

    # The zero component contains the number of vertices of C
    C[0] = len(Dist)
    R[0] = np.zeros(len(Dist))

    # Start with C[1]
    for i in range(C[0]):
        for j in range(i):
            if Dist[i][j] < max_r:
                C[1].append([j, i])
                R[1].append(Dist[i][j])

   
    # Sort edges according to their birth radius 
    sort_R = np.argsort(R[1])
    R[1], C[1] = np.array(R[1]), np.array(C[1])
    R[1] = R[1][sort_R]
    C[1] = C[1][sort_R]

    # Build VR-complex inductivelly
    for d in range(1, max_dim):
        # Assume that C[d] is already defined, then we compute C[d+1]
        for k, s in enumerate(C[d]):
            """Find neighbouring vertices of s in C, such that they are smaller than
            all the vertices from s. """
            low_n = _lower_neighbours(C[1], s[0])
            for v in s[1:]:
                low_n = [n for n in low_n if
                               n in _lower_neighbours(C[1], v)]

            for n in low_n:
                simplex_rad = R[d][k]
                for v in s:
                    if Dist[v][n] > simplex_rad:
                        simplex_rad = Dist[v][n]

                C[d + 1].append(np.insert(s, 0, n))
                R[d + 1].append(simplex_rad)

        # Sort simplices according to their birth radius 
        sort_R = np.argsort(R[d + 1])
        R[d + 1], C[d + 1] = np.array(R[d + 1]), np.array(C[d + 1])
        R[d + 1] = R[d + 1][sort_R]
        C[d + 1] = C[d + 1][sort_R]
  
    # store complexes as integers 
    for c in C[1:]:
        c = c.astype(int)
 
    return C, R

X = np.array([
    [0,0],[1,0],[0,1],[1,1]])
Dist = dist.squareform(dist.pdist(X))
C , R= vietoris_rips(Dist, 4,4)
