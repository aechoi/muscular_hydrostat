"""This module holds concrete implementations of the Cell classes inherited from abstract structure.

   Instances of this class construct the 3D hydrostat muscular arm.
"""

from dataclasses import dataclass

import numpy as np

@dataclass
class Cell3D:
    """A dataclass which holds shape info for 3D arms"""

    vertices: list[float]  # index of vertices
    edges: list[list[int]]  # each tuple indexes 2 points
    faces: list[
        list[int]
    ]  # indices of points, tuples may be ragged, must be ordered counter-clockwise from outside

    fixed_indices: list[int] = None
    masses: list[float] = None
    vertex_damping: list[float] = None
    edge_damping: list[float] = None

    def __post_init__(self):
        if self.fixed_indices is None:
            self.fixed_indices = []
        if self.masses is None:
            self.masses = np.ones(len(self.vertices)) / len(self.vertices)
        if self.vertex_damping is None:
            self.vertex_damping = np.ones(len(self.vertices)) / len(self.vertices)
        if self.edge_damping is None:
            self.edge_damping = np.ones(len(self.edges))

        self.triangles = self.triangulate_faces()

    def triangulate_faces(self):
        """Decompose each face into triangles for the purposes of volume calculation.
        Each face is assumed to be arranged counter clockwise from the outside.

        Returns:
            a tx3 np.ndarray of vertex indices for all t triangles
        """
        triangles = []
        for face in self.faces:
            if self.vertices[0] in face:
                continue
            for v1, v2 in zip(face[1:-1], face[2:]):
                triangles.append([face[0], v1, v2])
        return np.array(triangles)