import numpy as np


class ConvexObstacle2D:
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
        self.vertices = vertices
        if len(vertices) > 2:
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
        distances = np.diag(
            self.normal_matrix @ (point[:, None] - self.vertices[:-1].T)
        )
        if np.any(distances < 0):
            raise ValueError("Point is outside the polygon.")
            # closest_point = cp.Variable(2)
            # objective = cp.Minimize(cp.sum_squares(closest_point - point))
            # constraints = [
            #     row @ (closest_point - vertex) == 0
            #     for vertex, row in zip(self.vertices[:-1], self.normal_matrix)
            # ]
            # prob = cp.Problem(objective, constraints)
            # prob.solve()
            # return closest_point.value
        nearest_idx = np.argmin(distances)
        buffer_factor = 1.1  # ensures that the point is pushed out of the wall instead of asymptotically approaching edge.
        return (
            point
            - self.normal_matrix[nearest_idx] * distances[nearest_idx] * buffer_factor
        )


class ConvexObstacle3D:
    """A convex obstacle is defined as an intersection of half spaces.

    The class provides functions for determining whether a particle is inside
    the obstacle, a function for determining the nearest boundary point. The
    obstacle is assumed to be 3 dimensional.

    The hyperplanes are defined by A(x-x0) <= 0 where A is a stack of the
    transposes of each hyperplane's normal vector and x0 is any point on the
    hyperplane.

    An obstacle is made of vertices, edges, and faces. Each face has a centroid
    which is represented as a row of the centroids matrix.

    check_intersections() takes the inner product of the normal with the
    relative vector (point - centroids). If the answer is non-positive for
    every inner product, then the point intersects.

    nearest_point() finds the nearest point on the surface of the obstacle
    """

    def __init__(self, vertices, edges, faces):
        """
        Args:
            vertices: an nx3 np.ndarray where n is the number of vertices. The
                points must be in counter-clockwise order and form a convex set.
            edges: an ex2 np.ndarray where e is the number of edges. This is
                just for drawing
            faces: a list of index lists denoting faces in a clockwise direction
        """
        self.vertices = np.array(vertices)
        self.edges = np.array(edges)
        centroid = np.average(vertices, axis=0)

        self.normal_matrix = np.empty((len(faces), 3))
        self.face_centroids = np.empty_like(self.normal_matrix)
        for idx, face in enumerate(faces):
            self.face_centroids[idx] = np.average(self.vertices[face], axis=0)
            normalized_points = self.vertices[face] - self.face_centroids[idx]
            _, eigvecs = np.linalg.eigh(np.cov(normalized_points.T))
            normal = eigvecs[:, 0]

            # ensure that normals face outwards
            if (self.face_centroids[idx] - centroid) @ normal < 0:
                normal = -normal

            self.normal_matrix[idx] = normal

    def check_intersection(self, point):
        """Returns true if the point intersects the obstacle.

        Args:
            point: a length 2 np.array
        Return:
            Returns true if the point intersects the obstacle. Points on the
            edge are considered intersecting.
        """
        return np.all(np.diag(self.normal_matrix @ (point - self.face_centroids).T) < 0)

    def nearest_point(self, point):
        """Returns the nearest point on the surface of the obstacle to a point.
        Only works when the point is inside the polygon.

        Args:
            point: a length 2 np.array
        Return:
            Returns another length 2 np.array that has the x,y coordinate of
            the closest point."""
        distances = -np.diag(self.normal_matrix @ (point - self.face_centroids).T)
        if np.any(distances < 0):
            raise ValueError("Point is outside the polygon.")
        nearest_idx = np.argmin(distances)

        buffer_factor = 0.01  # ensures that the point is pushed out of the wall instead of asymptotically approaching edge.
        return point + self.normal_matrix[nearest_idx] * (
            distances[nearest_idx] + buffer_factor
        )

    def calc_many_intersections(self, points):
        centered_points = points[:, :, None] - self.face_centroids.T[None, :, :]
        # i points, j dimensions, k faces
        distances = np.einsum("ijk,kj->ik", centered_points, self.normal_matrix)
        # if all distances negative or 0, then intersecting
        return np.all(distances <= 0, axis=1)
