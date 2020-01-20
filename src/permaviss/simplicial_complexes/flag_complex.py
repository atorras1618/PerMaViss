###############################################################################
# flag complexes for graphs:
# This is used e.g. for computing the nerve of a cover.

import numpy as np

###############################################################################


def _lower_neighbours(G, u):
    """Given a graph `G` and a vertex `u` in `G`, we return a list with the
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
    lower_neighbours = []
    for e in G:
        if max(e) == u:
            lower_neighbours.append(min(e))

    return np.unique(lower_neighbours)

###############################################################################
# Main function


def flag_complex(G, no_vertices, max_dim):
    """Compute the flag complex of a graph `G` up to a maximum dimension `max_dim`.

    Parameters
    ----------
    G : :obj:`Numpy Array(no. of edges, 2)`
        Matrix storing the edges of the graph.
    no_vertices : int
        Number of vertices in graph `G`
    max_dim : int
        Maximum dimension

    Returns
    -------
    fl_cpx : :obj:`list(Numpy Array)`
        Flag complex of `G`. The `0` entry stores the number of vertices. For
        a higher entry `i`, `fl_cpx[i]` stores a :obj:`Numpy Array` matrix
        with the `i` simplices from `fl_cpx`.

    """
    if max_dim < 0:
        print("Cannot compute a complex of dimension: {}".format(max_dim))
        raise ValueError

    fl_cpx = []
    for i in range(max_dim + 1):
        fl_cpx.append([])

    fl_cpx[0] = no_vertices

    if max_dim == 0:
        return fl_cpx

    fl_cpx[1] = np.copy(G)

    # Build flag complex inductively
    for d in range(1, max_dim):
        for simplex in fl_cpx[d]:
            N = _lower_neighbours(fl_cpx[1], simplex[0])
            for v in simplex[1:]:
                N = np.intersect1d(N, _lower_neighbours(fl_cpx[1], v))

            # find simplices containing simplex and add them
            if np.size(N) > 0:
                simplices = np.ones((np.size(N), 1 + np.size(simplex)))
                simplices = np.multiply(np.append([1], simplex), simplices)
                simplices[:, 0] = N
                if isinstance(fl_cpx[d+1], list):
                    fl_cpx[d + 1] = simplices
                else:
                    fl_cpx[d+1] = np.append(fl_cpx[d + 1], simplices, 0)

    return fl_cpx
