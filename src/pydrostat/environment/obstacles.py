"""This module holds the abstract class and concrete instances of obstacles.

An obstacle is a special type of constraint which conditionally applies a constraint to
vertices that intersect the interior of the obstacle. 

"""

from abc import ABC, abstractmethod

import numpy as np

from ..constraint.constraint_interface import IConstraint
from ..structure.structure_interface import IStructure


class IObstacle(ABC, IConstraint):
    @abstractmethod
    def check_intersection(self, points: np.ndarray):
        """Check if a set of points intersect with the obstacle volume.

        Args:
            points: an nxd array of points to check for intersections where n is the
                number of points and d is the dimension

        Returns:
            a boolean array where True indices are intersecting the obstacle

        Raises:
            NotImplementedError if not implemneted in concrete instance.
        """
        raise NotImplementedError

    def initialize_constraint(structure: IStructure) -> None:
        """Nothing to initialize for obstacles"""
        pass

    @abstractmethod
    def calculate_constraints(
        structure: IStructure,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Check if any points intersect the obstacle, if so, constrain
        motion to be along the obstacle. Otherwise, no constraint."""
        pass


class ConvexPolytope(IObstacle):
    """A convex polytope obstacle is defined as an intersection of half spaces.

    The hyperplanes are defined by A(x-x0) <= 0 where A is a stack of the
    transposes of each hyperplane's normal vector and x0 is any point on the
    hyperplane.
    """

    def __init__(self, vertices: np.ndarray, facets: list[list[int]]):
        """
        Args:
            - vertices
        """
        self.vertices = np.array(vertices)
        self.dim = vertices.shape[-1]
        self.facets = np.array(facets)
        self.normal_matrix, self.facet_centroids = self._calculate_facet_data()

    def _calculate_facet_data(self):
        centroid = np.average(self.vertices, axis=0)

        normal_matrix = np.empty((len(self.facets), self.dim))  # fxd
        facet_centroids = np.empty_like(normal_matrix) # fxd

        for idx, facet in enumerate(self.facets):
            facet_centroids[idx] = np.average(self.vertices[facet], axis=0)
            normalized_points = self.vertices[facet] - facet_centroids[idx]
            _, eigvecs = np.linalg.eigh(np.cov(normalized_points.T))
            normal = eigvecs[:, 0]

            # ensure that normals face outwards
            if (facet_centroids[idx] - centroid) @ normal < 0:
                normal = -normal

            normal_matrix[idx] = normal

        return normal_matrix, facet_centroids

    def check_intersection(self, points: np.ndarray):
        """Check if any points intersect with the obstacle
        """
        relative_facet_vectors = (
            points[None, :, :] - self.facet_centroids[:, None, :]
        )  # f faces, n vertices,  d dimensions
        distances = np.einsum("ijk,ik->ij", relative_facet_vectors, self.normal_matrix) 
        # distances is normal vector normed with relative vector
        # for each face, take the relative vector to the centroid, and dot with normal
        # take the ith normal and norm with all n vertices for the ith face
        return np.all(distances <= 0, axis=0)
    
    def calculate_constraints(self, structure: IStructure):
        intersecting_points = structure.positions[self.check_intersection(structure.positions)]
        if not np.all
