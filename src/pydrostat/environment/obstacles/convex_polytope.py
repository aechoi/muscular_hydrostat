from itertools import product

import numpy as np
import OpenGL.GL as gl

from ..obstacle_interface import IObstacle
from ...structure.structure_interface import IStructure


class ConvexPolytope(IObstacle):
    """A convex polytope obstacle is defined as an intersection of half spaces.

    The hyperplanes are defined by A(x-x0) <= 0 where A is a stack of the
    transposes of each hyperplane's normal vector and x0 is any point on the
    hyperplane.
    """

    def __init__(self, vertices: np.ndarray, facets: list[list[int]]):
        self.vertices = np.array(vertices)
        self.dim = vertices.shape[-1]
        self.facets = np.array(facets)
        self.normal_matrix, self.facet_centroids = self._calculate_facet_data()

    def _calculate_facet_data(self):
        centroid = np.average(self.vertices, axis=0)

        normal_matrix = np.empty((len(self.facets), self.dim))  # fxd
        facet_centroids = np.empty_like(normal_matrix)  # fxd

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
        """Check if any points intersect with the obstacle"""
        distances = self._calc_distance_to_faces(points)
        return np.all(distances <= 0, axis=1)

    def _calc_distance_to_faces(self, points: np.ndarray) -> np.ndarray:
        """Negative distance if inside the obstacle"""
        relative_facet_vectors = (
            points[:, :, None] - self.facet_centroids.T[None, :, :]
        )  # n vertices,  d dimensions, f faces
        distances = np.diagonal(
            self.normal_matrix @ relative_facet_vectors,
            axis1=-2,
            axis2=-1,
        )  # NxF
        return distances

    def calculate_constraints(self, structure: IStructure):
        distances = self._calc_distance_to_faces(structure.positions)
        intersecting_idxs = np.all(distances <= 0, axis=1)
        intersecting_points = structure.positions[intersecting_idxs]
        if not intersecting_points:
            return [], [], []

        intersected_faces = np.argmin(distances[intersecting_idxs], axis=1)
        constraints = distances[intersecting_idxs, intersected_faces]
        num_constraints = len(constraints)
        jacobians = np.zeros((num_constraints) + structure.positions.shape)
        djacobian_dts = np.zeros_like(jacobians)

        intersected_normals = self.normal_matrix[intersected_faces]
        jacobians[np.arange(num_constraints), intersecting_idxs, :] = (
            intersected_normals
        )

        return constraints, jacobians, djacobian_dts

    def draw(self):
        vertex_difs = self.vertices[:, None, :] - self.vertices[None, :, :]
        edge_mask = np.sum(vertex_difs != 0, axis=-1) == 1
        edge_indices = np.where(np.triu(edge_mask, 1) > 0)

        for edge in zip(edge_indices[0], edge_indices[1]):
            for vertex in edge:
                gl.glVertex3fv(self.vertices[vertex])


def build_rectangular_obstacle(min_coord: list[float], max_coord: list[float]):
    if len(min_coord) != len(max_coord):
        raise ValueError("Min and max coords must have same dimension.")
    dim = len(min_coord)
    min_coord = np.array(min_coord)
    max_coord = np.array(max_coord)
    if np.any(max_coord <= min_coord):
        raise ValueError(
            "All max_coord elements must be greater than the corresponding min_coord element."
        )

    combinations = list(product([0, 1], repeat=dim))
    vertices = min_coord + combinations * (max_coord - min_coord)

    facets = []
    for dim in range(dim):
        fixed_values = [min_coord[dim], max_coord[dim]]

        for value in fixed_values:
            facet = [i for i, vertex in enumerate(vertices) if vertex[dim] == value]
            facets.append(facet)

    return ConvexPolytope(vertices, facets)
