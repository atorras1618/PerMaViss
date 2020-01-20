import math
import numpy as np

from ..simplicial_complexes import flag_complex as flCpx

###############################################################################
# Jump to next hypercube in cover


def next_hypercube(pos, div):
    """Jumps to next hypercube in cubical cover

    Parameters
    ----------
    pos : :obj:`list`
        List of integer values specifying the position of the current
        hypercube. This is edited to the next hypercube.

    div : :obj:`list`
        List of integer values specifying how many hypercubes divide each
        dimension.

    Example
    -------
        >>> pos = [1,1,1]
        >>> div = [3,3,2]
        >>> next_hypercube(pos, div)
        >>> pos
        [1, 2, 0]

    """
    for i in range(len(pos)):
        if pos[-i-1] < div[-i-1]-1:
            pos[-i-1] += 1
            for k in range(i):
                pos[-k-1] = 0
            break

###############################################################################
# Generate cubical cover of a point cloud


def generate_cover(max_div, overlap, point_cloud):
    """Divides a point cloud into a cubical cover.

    Receives a point cloud point_cloud in R^n and returns it divided into cubes
    and their respective intersections. It also generates the nerve of the
    covering.

    Parameters
    ----------
    max_div : int
        Number of divisions on the maximum side of point_cloud
    overlap : float
        Overlap between adjacent hypercubes.
    point_cloud : :obj:`Numpy Array`
        Each row contains the coordinates of a point

    Returns
    -------
    divided_point_cloud : :obj:`list(list(Numpy Array 2D))`
        Point cloud coordinates indexed by nerve. The `i` entry contains the
        point cloud coordinates indexed by the `i` simplices of the nerve.
        That is, the first entry contains the coordinates contained in
        hypercubes. The second entry the coordinates of points in double
        intersections of hypercubes. And so on.
    points_IN : :obj:`list(list(Numpy Array 1D))`
        Identification Numbers (IN) of points in regions indexed by nerve.
        That is, this is the same as `divided_point_cloud` but storing IN
        instead of coordinates.
    nerve : :obj:`list(Numpy Array)`
        The nerve of the hypercube cover.

    Example
    -------
        >>> point_cloud = circle(5, 1)
        >>> max_div = 2
        >>> overlap = 0.5
        >>> divided_point_cloud, points_IN, nerve = generate_cover(max_div,
        ... overlap, point_cloud)
        >>> divided_point_cloud[0]
        [array([[-0.80901699, -0.58778525],
               [ 0.30901699, -0.95105652]]), array([[ 0.30901699,  0.95105652],
               [-0.80901699,  0.58778525]]), array([[ 1.        ,  0.        ],
               [ 0.30901699, -0.95105652]]), array([[ 1.        ,  0.        ],
               [ 0.30901699,  0.95105652]])]
        >>> divided_point_cloud[1]
        [array([], shape=(0, 2), dtype=float64), array([[ 0.30901699,
        -0.95105652]]), array([], shape=(0, 2), dtype=float64), array([],
        shape=(0, 2), dtype=float64), array([[ 0.30901699,  0.95105652]]),
        array([[ 1.,  0.]])]
        >>> points_IN[0]
        [array([3, 4]), array([1, 2]), array([0, 4]), array([0, 1])]
        >>> points_IN[1]
        [array([], dtype=float64), array([4]), array([], dtype=float64),
        array([], dtype=float64), array([1]), array([0])]
        >>> nerve[0]
        4
        >>> nerve[1]
        array([[ 0.,  1.],
               [ 0.,  2.],
               [ 1.,  2.],
               [ 0.,  3.],
               [ 1.,  3.],
               [ 2.,  3.]])
        >>> nerve[2]
        array([[ 0.,  1.,  2.],
               [ 0.,  1.,  3.],
               [ 0.,  2.,  3.],
               [ 1.,  2.,  3.]])
        >>> nerve[3]
        array([[ 0.,  1.,  2.,  3.]])


    """
    # Get the dimension of our dataset
    dim = np.size(point_cloud, 1)
    # Find the 'corners' of the dataset
    min_corner, max_corner = corners_hypercube(point_cloud)
    # Find the largest coordinate difference between min_corner and max_corner
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
    # dist: distance from the center of an hypercube to a face
    dist = overlap / 2. + side / 2.

    for i in range(number_hypercubes):
        # Find two corners and center point of small hypercube
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
        next_hypercube(pos, div)

    # Compute nerve of the cover
    nerve = nerve_hypercube_cover(div)

    # Compute the point cloud and IN for each simplex in Nerve
    # Return this information so that the function spectral_sequence
    # has less work to do. This can be parallelized
    nerve_point_cloud = [divided_point_cloud]
    nerve_points_IN = [points_IN]
    for k, k_simplices_nerve in enumerate(nerve[1:]):
        nerve_point_cloud.append([])
        nerve_points_IN.append([])
        for simplex in k_simplices_nerve:
            points_IN_intersection = intersection_covers(points_IN, simplex)
            points_coord_intersection = point_cloud[points_IN_intersection]
            nerve_points_IN[k+1].append(np.array(points_IN_intersection))
            nerve_point_cloud[k+1].append(np.array(points_coord_intersection))

    return nerve_point_cloud, nerve_points_IN, nerve


###############################################################################
# Background functions supporting generate_cover


def corners_hypercube(point_cloud):
    """Returns maximum and minimum corners of hypercube containing point_cloud.

    Parameters
    ----------
    point_cloud : :obj:`Numpy Array`
        Coordinates of point data. Each row corresponds to a point.

    Returns
    -------
    min_corner, max_corner : :obj:`Numpy Array`, :obj:`Numpy Array`
        Minimum and maximum corners of containing hypercube.

    Example
    -------
        >>> from permaviss.sample_point_clouds.examples import random_cube
        >>> point_cloud = random_cube(5,2)
        >>> point_cloud
        array([[-0.30392908, -0.40559307],
               [ 0.4736051 , -0.28257937],
               [-0.41760472, -0.30445089],
               [-0.02406966,  0.001455  ],
               [-0.28425041, -0.11212227]])
        >>> min_corner, max_corner = corners_hypercube(point_cloud)
        >>> min_corner
        array([-0.41760472, -0.40559307])
        >>> max_corner
        array([ 0.4736051,  0.001455 ])

    """
    min_corner = np.amin(point_cloud, axis=0)
    max_corner = np.amax(point_cloud, axis=0)

    return min_corner, max_corner

###############################################################################
# Generate the nerve of an hypercube covering


def nerve_hypercube_cover(div):
    """Generates the nerve of an hypercube covering.

    Given an array of divisions of an hypercube cover, this returns the nerve.

    Parameters
    ----------
    div : :obj:`Numpy Array`
        1D array of divisions per dimension.

    Returns
    -------
    nerve : :obj:`list(Numpy Array)`
        Nerve associated to hypercube covering.

    Example
    -------
    Nerve for three dimensional covering. There are 6 hypercubes and the
    divisions per dimension are given as 1 x 3 x 2.

        >>> div = [1,3,2]
        >>> nerve = nerve_hypercube_cover(div)
        >>> nerve[0]
        6
        >>> nerve[1]
        array([[ 0.,  1.],
               [ 0.,  2.],
               [ 1.,  2.],
               [ 0.,  3.],
               [ 1.,  3.],
               [ 2.,  3.],
               [ 2.,  4.],
               [ 3.,  4.],
               [ 2.,  5.],
               [ 3.,  5.],
               [ 4.,  5.]])
        >>> nerve[2]
        array([[ 0.,  1.,  2.],
               [ 0.,  1.,  3.],
               [ 0.,  2.,  3.],
               [ 1.,  2.,  3.],
               [ 2.,  3.,  4.],
               [ 2.,  3.,  5.],
               [ 2.,  4.,  5.],
               [ 3.,  4.,  5.]])
        >>> nerve[3]
        array([[ 0.,  1.,  2.,  3.],
               [ 2.,  3.,  4.,  5.]])
        >>> nerve[4]
        []

    """
    dim = np.size(div)
    # compute number of hypercubes
    number_hypercubes = 1
    for d in range(dim):
        number_hypercubes *= div[d]

    nerve_graph = []
    pos = np.zeros(dim)
    for current_index in range(number_hypercubes):
        neighbour_step = -1 * np.ones(dim)  # current step to neighbour
        # Add all the edges to the neighbour graph
        while np.any(neighbour_step < np.zeros(dim)):
            step = 1  # how much we need to move the index for each coordinate
            # Initialize and compute the index of the neighbour
            neighbour_index = current_index  # initialize the lower index
            for i in range(dim):
                # if the hypercube is not a neighbour pass
                if (
                        ((pos[i] == 0) & (neighbour_step[i] == -1)) or
                        ((pos[i] == div[i] - 1) & (neighbour_step[i] == 1))):
                    neighbour_index = -1
                    break

                neighbour_index += neighbour_step[-i - 1] * step
                step *= div[-i - 1]  # increase coordinate step

            if (neighbour_index >= 0) & (sorted(
                    [neighbour_index, current_index]) not in nerve_graph):
                """Include edge (index_neighbour,index_current) to the nerve
                graph. Only do this whenever this makes sense and the edge is
                not in the nerve already.
                """
                nerve_graph.append(sorted([neighbour_index, current_index]))

            # advance to next neighbour step
            for i in range(dim):
                if neighbour_step[-i-1] < 1:
                    neighbour_step[-i-1] += 1
                    for k in range(i):
                        neighbour_step[-k-1] = -1
                    break

        next_hypercube(pos, div)

    return flCpx.flag_complex(nerve_graph, number_hypercubes, 2**dim)

###############################################################################
# Compute the common points on the intersection determined by simplex


def intersection_covers(points_IN,  simplex):
    """Computes the points in the intersection specified by a nerve simplex.

    Parameters
    ----------
    points_IN : :obj:`list(list(Numpy Array 1D))`
        Identification Numbers (IN) of points in covering hypercubes.
    simplex : :obj:`Numpy Array`
        Simplex in nerve which specifies intersection between hypercubes.

    Returns
    -------
    points_IN_intersection : :obj:`list(int)`
        IN of points in the intersection specified by simplex.

    Example
    -------
        >>> import numpy as np
        >>> points_IN = [np.array([0,3,5]),np.array([0,1]), np.array([1,5])]
        >>> simplex = np.array([0,1])
        >>> intersection_covers(points_IN, simplex)
        [0]

    """
    points_IN_intersection = points_IN[int(simplex[0])]

    for k in range(1, len(simplex)):
        points_IN_intersection = [
                idx_pt for idx_pt in points_IN_intersection
                if idx_pt in points_IN[int(simplex[k])]]

    return points_IN_intersection
