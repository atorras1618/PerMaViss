###############################################################################
# flag complexes for graphs:
# This is used e.g. for computing the nerve of a cover.
# Notice: the returned simplicial complex has an arbitrary order on the simplices.

import numpy as np

###############################################################################
# 

def _lower_neighbours(graph, v):
    """
        Returns lower edges to vertex in a graph.
        INPUT:
            - graph: list of pairs storing edges.
            - v: index of vertex.
        OUTPUT:
            - lower_neighbours: list storing lower neighbours to v in graph.
    """
    lower_neighbours = [] 
    for e in graph:
        if max(e) == v:
            lower_neighbours.append(min(e))

    return np.unique(lower_neighbours)

###############################################################################
# Main function

def flag_complex(graph, no_vertices, max_dim):
    """
        Given a graph, compute its flag complex up to a maximum dimension.
        INPUT:
            - graph: list of edges given as pairs. 
            - no_vertices: int 
            - max_dim: int
        OUTPUT:
            - fl_cpx: flag complex as a list of arrays.
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

    fl_cpx[1] = np.copy(graph)

    # Build flag commplex inductivelly
    for d in range(1,max_dim):
        for simplex in fl_cpx[d]:
            N = _lower_neighbours(fl_cpx[1], simplex[0])
            for v in simplex[1:]:
                N =  np.intersect1d(N, _lower_neighbours(fl_cpx[1], v))

            # find simplices containing simplex and add them
            if np.size(N) > 0:
                simplices = np.ones((np.size(N), 1 + np.size(simplex)))
                simplices = np.multiply(np.append([1],simplex), simplices)
                simplices[:,0] = N
                if type(fl_cpx[d+1]) != type([]):
                    fl_cpx[d+1] = np.append(fl_cpx[d + 1], simplices, 0)
                else:
                    fl_cpx[d + 1] = simplices


    return fl_cpx

