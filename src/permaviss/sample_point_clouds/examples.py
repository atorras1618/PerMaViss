import numpy as np

import scipy.spatial.distance as dist


def take_sample(point_cloud, no_samples):
    """ Take a subsample from samples using a minmax algorithm.

    We start from a random point. Then choose the point further appart. Next,
    we take the point that is further appart from the taken points. Continuing
    we take all the samples from `point_cloud`.

    Parameters
    ----------
    point_cloud : :obj:`Numpy Array`
        List of points to take samples from.
    np_samples : int
        Number of samples that we want to take. It has to be smaller than
        the dimension of `point_cloud`.

    Returns
    -------
    :obj:`Numpy Array`
        Matrix storing the coordinates of the sampled points.

    """
    no_points = len(point_cloud)
    no_samples = min(no_points, no_samples)
    Dist = dist.squareform(dist.pdist(point_cloud))
    k = int(np.random.rand() * no_points) - 1
    selection = [k]
    point_selection = [point_cloud[k]]
    for i in range(no_samples):
        not_selected = np.setdiff1d(range(no_points), selection)
        rest_Dist = np.min(Dist[not_selected][:, selection], axis=1)
        k = not_selected[np.argmax(rest_Dist)]
        selection.append(k)
        point_selection.append(point_cloud[k])

    return np.array(point_selection)


def circle(no_points, radius):
    """ Take random points from a circle on the plane. This circle has
    centre `(0, 0)`.

    Parameters
    ----------
    no_points : int
        Number of points wanted.
    radius : float
        Radius of the circle.

    Returns
    -------
    :obj:`Numpy Array`
        Coordinates of sampled points from the circle.

    """
    point_cloud = []
    step = 2 * np.pi / (no_points)
    for n in range(no_points):
        point_cloud.append([radius * np.cos(n * step),
                            radius * np.sin(n * step)])

    return np.asarray(point_cloud)


def random_circle(no_points, radius, epsilon, center=[0, 0]):
    """ Take random points around a circle on the plane.

    Parameters
    ----------
    no_points : int
        Number of points wanted.
    radius : float
        Radius of the circle.
    epsilon : float
        Noise that we want to apply to each sampled point.
    centre : `list(float, float)`
        Two entries specifying the position of the centre.

    Returns
    -------
    :obj:`Numpy Array`
        Coordinates of sampled points from around the circle.

    """
    point_cloud = []
    for n in range(no_points):
        random_angle = np.random.rand() * 2 * np.pi
        random_rad = radius * (1 + np.random.rand() * epsilon)
        point_cloud.append([center[0] + random_rad * np.cos(random_angle),
                            center[1] + random_rad * np.sin(random_angle)])

    return np.asarray(point_cloud)


def ball(no_points, radius, dim):
    """ Take random points from a ball of a given dimension. This ball has
    centre `(0, 0, 0)`.

    Parameters
    ----------
    no_points : int
        Number of points wanted.
    radius : float
        Radius of the ball.
    dim : int
        Dimension of the ball.

    Returns
    -------
    :obj:`Numpy Array`
        Coordinates of sampled points from the ball.

    """
    point_cloud = []
    for n in range(no_points):
        valid_point = False
        while valid_point is False:
            point = []
            for d in range(dim):
                point.append((np.random.rand() - 0.5) * radius)

            v = np.array(point)
            if np.linalg.norm(v) < radius:
                valid_point = True
                point_cloud.append(point)

    return np.asarray(point_cloud)


def random_sphere(no_points, radius, dim):
    """ Take random points from a sphere of a given dimension. This sphere has
    centre `(0, 0, 0)`.

    Parameters
    ----------
    no_points : int
        Number of points wanted.
    radius : float
        Radius of the sphere.
    dim : int
        Dimension of the sphere.

    Returns
    -------
    :obj:`Numpy Array`
        Coordinates of sampled points from the sphere.

    """
    point_cloud = []
    for n in range(no_points):
        point = []
        for d in range(dim):
            point.append((np.random.rand() - 0.5) * radius)

        v = np.array(point)
        norm = np.linalg.norm(v)
        for d in range(dim):
            point[d] = point[d] / norm

        point_cloud.append(point)

    return np.asarray(point_cloud)


def random_cube(no_points, dim):
    """ Take random points from a unit cube around the origin. This cube can be
    of various dimensions.

    Parameters
    ----------
    no_points : int
        Number of points wanted.
    dim : int
        Dimension of the cube.

    Returns
    -------
    :obj:`Numpy Array`
        Coordinates of sampled points from the cube.

    """
    point_cloud = []
    for n in range(no_points):
        point = []
        for d in range(dim):
            point.append(np.random.rand() - 0.5)

        point_cloud.append(point)

    return np.asarray(point_cloud)


def grid(hdiv, vdiv):
    """ Take the nodes from a hdiv x vdiv grid on 2D plane.

    Parameters
    ----------
    hdiv : int
        Number of rows.
    vdiv : int
        Number of columns.

    Returns
    -------
    :obj:`Numpy Array`
        List of points.

    """
    hshift = hdiv / 2. - 0.5
    vshift = vdiv / 2. - 0.5
    point_cloud = []
    for i in range(hdiv):
        for j in range(vdiv):
            point_cloud.append([i - hshift, j - vshift])

    return np.asarray(point_cloud)


def grid_tridimensional(hdiv, vdiv, ddiv):
    """ Take the nodes from a hdiv x vdiv x ddiv grid on 3D space.

    Parameters
    ----------
    hdiv : int
        Number of rows.
    vdiv : int
        Number of columns.
    ddiv : int
        Number of flats.

    Returns
    -------
    :obj:`Numpy Array`
        List of points.

    """
    hshift = hdiv / 2. - 0.5
    vshift = vdiv / 2. - 0.5
    dshift = ddiv / 2. - 0.5

    point_cloud = []
    for i in range(hdiv):
        for j in range(vdiv):
            for k in range(ddiv):
                point_cloud.append([i - hshift, j - vshift, k - dshift])

    return np.asarray(point_cloud)


def torus3D(no_points, min_rad=1, max_rad=3):
    """ Take samples from a torus embedded in 3D space.

    Parameters
    ----------
    no_points : int
        Number of points to be taken.
    min_rad : float, default is `1`
        Radius of circle on section of the torus.
    max_rad : float, default is `3`
        Distance from the torus centre to the centre of the section.

    Returns
    -------
    :obj:`Numpy Array`
        List of points.

    """
    if min_rad > max_rad:
        raise ValueError
    point_cloud = []
    for i in range(no_points):
        theta = np.random.rand() * 2 * np.pi
        alpha = np.random.rand() * 2 * np.pi
        point_cloud.append([
            (max_rad + min_rad * np.cos(alpha)) * np.cos(theta),
            (max_rad + min_rad * np.cos(alpha)) * np.sin(theta),
            min_rad * np.sin(alpha)])
    return np.asarray(point_cloud)


def torus(div, min_rad, max_rad):
    """ Take samples from a torus in 4D space.

    Parameters
    ----------
    no_points : int
        Number of points to be taken.
    min_rad : float, default is `1`
        Radius of circle on section of the torus.
    max_rad : float, default is `3`
        Distance from the torus centre to the centre of the section.

    Returns
    -------
    :obj:`Numpy Array`
        List of points.

    """
    point_cloud = []
    step_small = 2 * np.pi / div
    step_big = step_small * min_rad / max_rad
    ang_max = 0

    while(ang_max < 2 * np.pi):

        ang_min = 0

        while(ang_min < 2 * np.pi):

            point_cloud.append([min_rad * np.cos(ang_min),
                                min_rad * np.sin(ang_min),
                                max_rad * np.cos(ang_max),
                                max_rad * np.sin(ang_max)])
            ang_min += step_small

        ang_max += step_big
    return np.asarray(point_cloud)
