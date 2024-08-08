import cvxpy as cp
import numpy as np


class ConvexObstacle:
    """A convex obstacle is defined as an intersection of half spaces.

    The class provides functions for determining whether a particle is inside
    the obstacle, a function for determining the nearest boundary point. The
    obstacle is currently assumed to be 2 dimensional.

    The hyperplanes are defined by A(x-x0) <= 0 where A is a stack of the
    transposes of each hyperplane's normal vector and x0 is any point on the
    hyperplane.
    """

    def __init__(self, vertices):
        """
        Args:
            vertices: an nx2 np.ndarray where n is the number of vertices. The
                points must be in counter-clockwise order and form a convex set.
        """
        self.vertices = np.vstack([vertices, vertices[0]])
        self.normal_matrix = np.array(
            [
                [v1[1] - v2[1], v2[0] - v1[0]]
                for v1, v2 in zip(self.vertices[:-1], self.vertices[1:])
            ]
        )
        self.normal_matrix = (
            self.normal_matrix / np.linalg.norm(self.normal_matrix, axis=1)[:, None]
        )

        # Check convexity, normals should never increase by more than pi rad
        tan = np.arctan2(self.normal_matrix[:, 1], self.normal_matrix[:, 0])
        angle_diffs = (np.roll(tan, -1) - tan) % (2 * np.pi)

        if any(angle_diffs > np.pi):
            raise ValueError("Vertices must form a convex object.")

    def check_intersection(self, point):
        """Returns true if the point intersects the obstacle.

        Args:
            point: a length 2 np.array
        Return:
            Returns true if the point intersects the obstacle. Points on the
            edge are considered intersecting.
        """
        return np.all(
            np.diag(self.normal_matrix @ (point[:, None] - self.vertices[:-1].T)) > 0
        )
        # Might be a faster way to do it? Above is more readable though
        # return np.all(
        #     np.einsum("ij,ji->i", self.normal_matrix, point[:, None] - self.vertices.T)
        #     > 0
        # )

    def nearest_point(self, point):
        """Returns the nearest point on the surface of the obstacle to a point.
        Only works when the point is inside the polygon.

        Args:
            point: a length 2 np.array
        Return:
            Returns another length 2 np.array that has the x,y coordinate of
            the closest point."""
        # closest_point = cp.Variable(2)
        # objective = cp.Minimize(cp.sum_squares(closest_point - point))
        # constraints = [
        #     row @ (closest_point - vertex) == 0
        #     for vertex, row in zip(self.vertices[:-1], self.normal_matrix)
        # ]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # return closest_point.value
        distances = np.diag(
            self.normal_matrix @ (point[:, None] - self.vertices[:-1].T)
        )
        if np.any(distances < 0):
            raise ValueError("Point is outside the polygon.")
        nearest_idx = np.argmin(distances)
        return point - self.normal_matrix[nearest_idx] * distances[nearest_idx]
