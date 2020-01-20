"""
    vietoris_rips.py
"""

import numpy as np


def _lower_neighbours(G, u):
    """ Given a graph `G` and a vertex `u` in `G`, we return a list with the
    vertices in `G` that are lower than `u` and are connected to `u` by an
    edge in `G`.

    Parameters
    ----------
    G : :obj:`Numpy Array(no. of edges, 2)`
        Matrix storing the edges of the graph.
    u : int
        Vertex of the graph.

    Returns
    -------
    lower_neighbours : :obj:`list`
        List of lower neighbours of `u` in `G`.
    """
    lower_neighbours = []  # list of lowest neighbours to be computed.
    for e in G:
        if e[1] == u:
            lower_neighbours.append(e[0])

    lower_neighbours.sort()
    return lower_neighbours


def vietoris_rips(Dist, max_r, max_dim):
    """This computes the Vietoris-Rips complex with simplexes of dimension less
    or equal to max_dim, and with maximum radius specified by max_r

    Parameters
    ----------
    Dist : :obj:`Numpy Array(no. of points, no. of points)`
        Distance matrix of points.
    max_r : float
        Maximum radius of filtration.
    max_dim : int
        Maximum dimension of computed Rips complex.

    Returns
    -------
    C : :obj:`list(Numpy Array)`
        Vietoris Rips complex generated for the given parameters. List where
        the first entry stores the number of vertices, and all other entries
        contain a :obj:`Numpy Array` with the list of simplices in `C`.
    R : :obj:`list(Numpy Array)`
        List with radius of birth for the simplices in `C`. The `i` entry
        contains a 1D :obj:`Numpy Array` containing each of the birth radii
        for each `i` simplex in `C`.

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

    # Build VR-complex inductively
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
