import math
import numpy as np

from ..simplicial_complexes import flag_complex as flCpx

###############################################################################
# Jump to next hypercube in cover

def _next_hypercube(pos,div):
    """
    How to advance to the next hypercube taking into acount
    the divisions list div.
    """
    for i in range(len(pos)):
        if pos[-i-1] < div[-i-1]-1:
            pos[-i-1] += 1
            for k in range(i):
                pos[-k-1]=0
            break

###############################################################################
# Generate cubical cover of a point cloud

def generate_cover(max_div, overlap, point_cloud):
    """
    Receives a point cloud point_cloud in R^n and returns it divided into cubes
    and their respective intersections.
    INPUT:
        -max_div: number of divisions on the maximum side of point_cloud
        -overlap: overlap between hypercubes.
        -point_cloud: numpy.array listing points

    OUTPUT:
        -divided_point_cloud: list of divided points. The n^th entry contains
            the point cloud contained in the n^th hypercube.
        -points_IN: list containing Identification Numbers for the points in 
                    the covers. These will be their index in point_cloud.
        -nerve: the nerve of the hypercube cover.
    """
    #get the dimension of our dataset
    dim = np.size(point_cloud, 1)
    #Find the 'corners' of the dataset
    min_corner, max_corner = _corners_hypercube(point_cloud)
    #Find the largest coordinate difference between min_corner and max_corner
    max_length = np.amax(np.absolute(max_corner - min_corner))

    side = max_length / max_div  # side of division hypercubes
    number_hypercubes = 1 
    div = []  # list containing no of divisions per dimension
    pos = []  # list containing current position
    for d in range(dim):
        div.append(math.ceil((max_corner[d] - min_corner[d]) / side))
        number_hypercubes *= div[d]  # compute total number of hypercubes
        pos.append(0)  # initialize position list to zero

    divided_point_cloud = []
    points_IN = []
    for d in range(number_hypercubes):
        divided_point_cloud.append([])
        points_IN.append([])

    """
    Now we assign points to each division hypercube.
    Also we compute the underlying graph of the nerve.
    """
    neighbour_graph = []
    # dist: distance from the center of an hypercube to a face
    dist = overlap / 2. + side / 2.

    for i in range(number_hypercubes):
        #Find two corners and center point of small hypercube
        lcor = np.copy(min_corner)      # lower corner
        ucor = np.copy(min_corner)      # upper corner
        center = np.copy(min_corner)    # center point
        for d in range(dim):
            lcor[d] += side * pos[d]
            ucor[d] += side * pos[d]
            center[d] += side * pos[d] + side / 2.
            lcor[d] -= dist - side / 2.
            ucor[d] += dist + side / 2.

        """
        Include points in divided_point_cloud[i] if they lie on this hypercube
        """
        for idx_pt, pt in enumerate(point_cloud):
            in_hypercube = True  # assume the point lies on the hypercube
            """Now we check whether x lies really on the hypercube"""
            for k in range(dim):
                if lcor[k] > pt[k]:
                    in_hypercube = False
                    break
                if ucor[k] < pt[k]:
                    in_hypercube = False
                    break
            """If x is in hypercube we include it"""
            if in_hypercube:
                divided_point_cloud[i].append(pt)
                points_IN[i].append(idx_pt)

        divided_point_cloud[i] = np.array(divided_point_cloud[i])
        points_IN[i] = np.array(points_IN[i])
        # advance position of hypercube
        _next_hypercube(pos, div)  

    # Compute nerve of the cover
    nerve = _nerve_hypercube_cover(div)

    # Compute the point cloud and IN for each simplex in Nerve
    # Return this information so that the function spectral_sequence has less work to do
    # This can be parallelized
    nerve_point_cloud = [divided_point_cloud]
    nerve_points_IN = [points_IN]
    for k, k_simplices_nerve in enumerate(nerve[1:]):
        nerve_point_cloud.append([])
        nerve_points_IN.append([])
        for simplex in k_simplices_nerve:
            points_IN_intersection = _intersection_covers(points_IN, simplex)
            points_coord_intersection = point_cloud[points_IN_intersection]
            nerve_points_IN[k+1].append(np.array(points_IN_intersection))
            nerve_point_cloud[k+1].append(np.array(points_coord_intersection))
    
    return nerve_point_cloud, nerve_points_IN, nerve



###############################################################################
# Background functions supporting generate_cover

def _corners_hypercube(point_cloud):
    """
    This takes a point cloud and returns the minimum and maximum corners
    of the hypercube containing them.
    INPUT:
        - point_cloud: np.array
    OUNTPUT:
        - min_corner, max_corner : np.arrays
    """
    dim = np.size(point_cloud, 1)
    min_corner = np.amin(point_cloud, axis=0)
    max_corner = np.amax(point_cloud, axis=0)

    return min_corner, max_corner

assert np.array_equal(np.array([[0,-1,-0.5],[2,1,1]]), np.array(
        _corners_hypercube(np.array([[0,0,0],[0,-1,1],[1,1,0], [2,1,-0.5]]))))
assert np.array_equal(np.array([[0,-10],[1,1]]), np.array(
            _corners_hypercube(np.array([[0,0],[1,1], [1,-10], [0.5,0.5]]))))

###############################################################################
# Generate the nerve of an hypercube covering

def _nerve_hypercube_cover(div):
    """
    Given an array of divisions of an hypercube cover, this generates the nerve.
    INPUT:
        -div: numpy.array detailing the amount of divisions per dimension.
    OUTPUT:
        -nerve: list with simplices on each dimension listed as numpy.array  
    """
    dim = np.size(div)
    # compute number of hypercubes
    number_hypercubes = 1 
    for d in range(dim):
        number_hypercubes *= div[d]  

    nerve_graph = []
    pos = np.zeros(dim)
    for current_index in range(number_hypercubes):
        neighbour_step = -1 * np.ones(dim) # step on current position to neighbour
        # Add all the edges to the neighbour graph
        while np.any(neighbour_step < np.zeros(dim)):
            step = 1  # this step tracks how much we need to move the index for each coordinate
            # Initialize and compute the index of the neighbour
            neighbour_index = current_index  # initialize the lower index
            for i in range(dim):
                # if the hypercube is not a neighbour pass
                if (
                        ((pos[i] == 0) & (neighbour_step[i] == -1)) or
                        ((pos[i] == div[i] - 1) & (neighbour_step[i] == 1))):
                    neighbour_index = -1
                    break

                neighbour_index += neighbour_step[-i - 1] * step  # update lower index
                step *= div[-i - 1]  # increase coordinate step

            if (neighbour_index >= 0) & (sorted([neighbour_index, current_index]) not in nerve_graph):
                """
                    Include edge (index_neighbour,index_current) to the nerve graph
                    Only do this whenever this makes sense and the edge is not in the nerve already
                """
                nerve_graph.append(sorted([neighbour_index, current_index]))

            # advance to next neigbhour step
            for i in range(dim):
                if neighbour_step[-i-1] < 1:
                    neighbour_step[-i-1] += 1
                    for k in range(i):
                        neighbour_step[-k-1]=-1
                    break
        
        _next_hypercube(pos, div) 
 
    return flCpx.flag_complex(nerve_graph, number_hypercubes, 2**dim)    

###############################################################################
# Compute the common points on the interesection determined by simplex

def _intersection_covers(points_IN,  simplex):
    """
    Computes the points in the intersection determined by simplex
    INPUT:
        -points_IN: list of cover elements containing Identification Numbers for points.
        -simplex: a simplex in the nerve of the cover
    OUTPUT:
        -points_IN_intersection: list of points IN on intersection
    """
    points_IN_intersection = points_IN[int(simplex[0])]

    for k in range(1,len(simplex)):
        points_IN_intersection = [
                idx_pt for idx_pt in points_IN_intersection
                if idx_pt in points_IN[int(simplex[k])]]

    return points_IN_intersection

###############################################################################
# Determine whether a point is part of a cover element

# def _membership_points(point_cloud, min_corner, max_corner, div, overlap, side):
#     """
#     This takes a point cloud, together with the enclosing hypercube and
#     the number of divisions we want to perform. Taking into account an
#     overlap, we return an array detailing to which hypercubes each point
#     belongs to.
#     INPUT:
#         - point_cloud: np.array[number points, dim],
#         - min_corner, max_corner: np.arrays[dim], corners of hypercube.
#         - div: np.array[dim], number of divisions per dimension.
#     OUTPUT:
#         - list_simplices: np.array[num_points, *]
#     """
#     assert overlap > side, f"Overlap is too big."
# 
#     dim = point_cloud.shape[1]
#     regions = []
#     for d in range(dim):
#         regions.append(np.linspace(min_corner[d], max_corner[d],
#                        num=div[d] + 1)[1:])
# 
#     regions = np.asarray(regions)
#     list_simplices = []
#     for pt in point_cloud:
#         simplex = np.array([0])
#         for d in range(dim-1, -1, -1):
#             pos = np.argmax(pt[d] < regions[d] + (overlap / 2))
#             step = np.prod(div[d+1:])
#             simplex = simplex + step * pos
#             if (pos > 0) and (pt[d] < regions[d][pos-1] + (overlap / 2)):
#                 simplex = np.concatenate((simplex - step, simplex), axis=0)
#             if (pos < div[d]-1) and (pt[d] > regions[d][pos] - (overlap / 2)):
#                 simplex = np.concatenate((simplex, simplex + step), axis=0)
# 
#         list_simplices.append(simplex)
# 
#     return np.asarray(list_simplices)
# 
#assert np.array_equal(np.array([[0], [0,2], [2], [1], [0,1,2,3]]),
#                      _membership_points(np.array([[0,0], [1,0], [2,0], [0,2],
#                                        [1,1]]), [0,0], [2,2], [2,2], 0.5, 1))
