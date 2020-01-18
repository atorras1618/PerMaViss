import numpy as np

from permaviss.sample_point_clouds.examples import grid

from permaviss.covers.cubical_cover import (
    next_hypercube, corners_hypercube, generate_cover, nerve_hypercube_cover)


def test_next_hypercube():
    pos = [1, 1]
    div = [3, 2]
    next_hypercube(pos, div)
    assert [2, 0] == pos


def test_corners_hypercube():
    assert np.array_equal(
            np.array([[0, -1, -0.5], [2, 1, 1]]),
            np.array(corners_hypercube(
                np.array([[0, 0, 0], [0, -1, 1], [1, 1, 0], [2, 1, -0.5]])
                    ))
        )
    assert np.array_equal(
            np.array([[0, -10], [1, 1]]),
            np.array(corners_hypercube(
                np.array([[0, 0], [1, 1], [1, -10], [0.5, 0.5]])
                    ))
       )


def test_nerve_hypercube_cover():
    div = [2, 3]
    # Expected nerve
    expected_nerve = [
        6,
        np.array([[0, 1], [1, 2], [0, 3], [1, 3], [0, 4],
                  [1, 4], [2, 4], [3, 4], [1, 5], [2, 5], [4, 5]]),
        np.array([[0, 1, 3], [0, 1, 4], [1, 2, 4], [0, 3, 4],
                  [1, 3, 4], [1, 2, 5], [1, 4, 5], [2, 4, 5]]),
        np.array([[0, 1, 3, 4], [1, 2, 4, 5]]),
        []
    ]
    nerve = nerve_hypercube_cover(div)
    assert nerve[0] == expected_nerve[0]
    for dim, simplices in enumerate(nerve[1:4]):
        assert np.array_equal(simplices, expected_nerve[dim+1])


def test_generate_cover():
    """We divide a 3x3 grid into four overlapping regions.
        We compute the different coordinates for each region
        and intersections. Also, associated ID numbers are
        given to different points. Finally, we also check
        whether the given simplicial complex is correct.
    """
    point_cloud = grid(3, 3)
    # Expected points
    # expected points on each cube
    points_cube_0 = np.array([[-1., -1.], [-1., 0.], [0., -1.], [0., 0.]])
    points_cube_1 = np.array([[-1., 0.], [-1., 1.], [0., 0.], [0., 1.]])
    points_cube_2 = np.array([[0., -1.], [0., 0.], [1., -1.], [1., 0.]])
    points_cube_3 = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    # expected points in double intersections
    points_inter_01 = np.array([[-1., 0.], [0., 0.]])
    points_inter_02 = np.array([[0., -1.], [0., 0.]])
    points_inter_03 = np.array([[0., 0.]])
    points_inter_12 = np.array([[0., 0.]])
    points_inter_13 = np.array([[0., 0.], [0., 1.]])
    points_inter_23 = np.array([[0., 0.], [1, 0.]])
    # expected points in triple intersections
    points_inter_012 = np.array([[0., 0.]])
    points_inter_013 = np.array([[0., 0.]])
    points_inter_023 = np.array([[0., 0.]])
    points_inter_123 = np.array([[0., 0.]])
    # expected points in fourth intersections
    points_inter_0123 = np.array([[0., 0.]])
    expected_nerve_point_cloud = [
        [points_cube_0, points_cube_1, points_cube_2, points_cube_3],
        [points_inter_01, points_inter_02, points_inter_03,
         points_inter_12, points_inter_13, points_inter_23],
        [points_inter_012, points_inter_013,
         points_inter_023, points_inter_123],
        [points_inter_0123]
    ]
    # Expected IN
    # expected IN on each cube
    IN_cube_0 = np.array([0, 1, 3, 4])
    IN_cube_1 = np.array([1, 2, 4, 5])
    IN_cube_2 = np.array([3, 4, 6, 7])
    IN_cube_3 = np.array([4, 5, 7, 8])
    # expected IN in double intersections
    IN_inter_01 = np.array([1, 4])
    IN_inter_02 = np.array([3, 4])
    IN_inter_03 = np.array([4])
    IN_inter_12 = np.array([4])
    IN_inter_13 = np.array([4, 5])
    IN_inter_23 = np.array([4, 7])
    # expected IN in triple intersections
    IN_inter_012 = np.array([4])
    IN_inter_013 = np.array([4])
    IN_inter_023 = np.array([4])
    IN_inter_123 = np.array([4])
    # expected IN in fourth intersections
    IN_inter_0123 = np.array([4])
    expected_nerve_points_IN = [
        [IN_cube_0, IN_cube_1, IN_cube_2, IN_cube_3],
        [IN_inter_01, IN_inter_02, IN_inter_03,
         IN_inter_12, IN_inter_13, IN_inter_23],
        [IN_inter_012, IN_inter_013,
         IN_inter_023, IN_inter_123],
        [IN_inter_0123]
    ]
    # Expected Nerve
    no_cubes = 4
    double_inter = np.array(
        [[0., 1.], [0., 2.], [1., 2.], [0., 3.], [1., 3.], [2., 3.]]
    )
    triple_inter = np.array(
        [[0., 1., 2.], [0., 1., 3.], [0., 2., 3.], [1., 2., 3.]]
    )
    fourth_inter = np.array(
        [[0., 1., 2., 3.]]
    )
    expected_nerve = [no_cubes, double_inter,
                      triple_inter, fourth_inter]

    # Check that the result is correct
    nerve_point_cloud, nerve_points_IN, nerve = generate_cover(
        2, 0.5, point_cloud)
    for dim, points_covers in enumerate(nerve_point_cloud):
        for idx, local_point_cloud in enumerate(points_covers):
            assert np.array_equal(
                local_point_cloud,
                expected_nerve_point_cloud[dim][idx])

    for dim, IN_covers in enumerate(nerve_points_IN):
        for idx, local_IN in enumerate(IN_covers):
            assert np.array_equal(
                local_IN,
                expected_nerve_points_IN[dim][idx])

    assert expected_nerve[0] == nerve[0]
    for dim, simplices in enumerate(nerve[1:4]):
        assert np.array_equal(simplices, expected_nerve[dim+1])
