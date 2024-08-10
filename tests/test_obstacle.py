import numpy as np
import pytest
import obstacle


def test_convexity():
    with pytest.raises(ValueError):
        # direction changing, not self-intersecting
        obst = obstacle.ConvexObstacle(
            np.array([[0, 0], [1, 0], [0.5, 0.5], [1, 1], [0, 1]])
        )

    with pytest.raises(ValueError):
        # direction changing, self intersecting
        obst = obstacle.ConvexObstacle(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))

    with pytest.raises(ValueError):
        # direction maintaining, self intersecting
        obst = obstacle.ConvexObstacle(
            np.array([[0, 0], [1, 0], [0.5, 0.5], [0.5, -0.5], [1, 0]])
        )


def test_check_intersection():
    # For each polygon, test when the point is inside (true), on
    # vertex, and outside (all false). On edge has trouble with numeric
    # error

    obst = obstacle.ConvexObstacle(np.array([[0, 0], [0, 1]]))
    assert obst.check_intersection(np.array([-1, 0]))
    assert not obst.check_intersection(np.array([0, 0]))
    assert not obst.check_intersection(np.array([0, 0.5]))
    assert not obst.check_intersection(np.array([1, 0]))

    np.random.seed(0)
    random_points = np.random.rand(1000, 3, 2)
    centroids = np.mean(random_points, axis=1)
    diffs = random_points - centroids[:, None, :]
    angles = np.arctan2(diffs[:, :, 1], diffs[:, :, 0]) % (np.pi * 2)
    sort_idx = np.argsort(angles, axis=1)
    ccw_vertices = np.take_along_axis(random_points, sort_idx[:, :, None], axis=1)
    for points, centroid in zip(ccw_vertices, centroids):
        obst = obstacle.ConvexObstacle(points)
        assert obst.check_intersection(centroid)
        # assert not obst.check_intersection((points[0] + points[1]) / 2)
        assert not obst.check_intersection(points[0])
        assert not obst.check_intersection(points[0] + (points[0] - centroid))
