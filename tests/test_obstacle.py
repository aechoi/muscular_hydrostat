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
    polygons = [random_convex_polygons(sides=3, count=100)]
    polygons.append(random_convex_polygons(sides=4, count=100))
    polygons.append(random_convex_polygons(sides=10, count=100))
    polygons.append(random_convex_polygons(sides=100, count=100))
    for polygon_type in polygons:
        centroids = np.mean(polygon_type, axis=1)
        for polygon, centroid in zip(polygon_type, centroids):
            obst = obstacle.ConvexObstacle(polygon)
            assert obst.check_intersection(centroid)

            edge_point = (polygon[0] + polygon[1]) / 2
            assert obst.check_intersection(edge_point - 1e-12 * (edge_point - centroid))
            assert not obst.check_intersection(
                edge_point + 1e-12 * (edge_point - centroid)
            )
            assert not obst.check_intersection(polygon[0])
            assert not obst.check_intersection(polygon[0] + (polygon[0] - centroid))


def random_convex_polygons(sides, count=1):
    """Algorithm for efficiently generating a random convex polygon. Taken from
    https://cglab.ca/~sander/misc/ConvexGeneration/convex.html which references
    https://refubium.fu-berlin.de/bitstream/handle/fub188/17874/1994_01.pdf?sequence=1&isAllowed=y
    """
    rand_x = np.sort(np.random.rand(sides, count), axis=0)
    rand_y = np.sort(np.random.rand(sides, count), axis=0)

    extreme_x = [rand_x[0], rand_x[-1]]
    extreme_y = [rand_y[0], rand_y[-1]]

    selection_x = np.round(np.random.rand(len(rand_x[1:-1])))
    selection_y = np.round(np.random.rand(len(rand_y[1:-1])))

    sublist_x_1 = np.vstack(
        ([extreme_x[0]], rand_x[1:-1][selection_x == 0], [extreme_x[1]])
    )
    sublist_x_2 = np.vstack(
        ([extreme_x[0]], rand_x[1:-1][selection_x == 1], [extreme_x[1]])
    )
    sublist_y_1 = np.vstack(
        ([extreme_y[0]], rand_y[1:-1][selection_y == 0], [extreme_y[1]])
    )
    sublist_y_2 = np.vstack(
        ([extreme_y[0]], rand_y[1:-1][selection_y == 1], [extreme_y[1]])
    )

    x_vecs = np.vstack(
        (np.diff(sublist_x_1, axis=0), np.diff(sublist_x_2[::-1], axis=0))
    )
    y_vecs = np.vstack(
        (np.diff(sublist_y_1, axis=0), np.diff(sublist_y_2[::-1], axis=0))
    )
    np.random.shuffle(y_vecs)

    angles = np.arctan2(y_vecs, x_vecs)
    vecs = np.stack((x_vecs, y_vecs), axis=2)
    sort_idx = np.argsort(angles, axis=0)
    vecs = np.take_along_axis(vecs, sort_idx[:, :, None], axis=0)
    points = np.cumsum(vecs, axis=0)
    # points = np.vstack((np.zeros((1, count, 2)), np.cumsum(vecs, axis=0)))
    points = points.transpose(1, 0, 2)

    return points
