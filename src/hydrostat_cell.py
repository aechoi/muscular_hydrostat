from __future__ import annotations

import cvxpy as cp
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HydrostatCell:
    # Still have force on legs, but geometry is
    # instantaneously updated to enforce constant volume
    """
    2
    |\
    | \
    0--1
    """
    vertices: np.ndarray = field(
        default_factory=lambda: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    )
    edges: list[tuple[int, int]] = field(
        default_factory=lambda: [(0, 1), (1, 2), (0, 2)]
    )

    masses = np.array([1, 1, 1])  # mass of vertices [kg]
    dampers = np.array([1, 1, 1])  # damping rate of edges [N-s/m]
    springs = np.array([1, 1, 1])  # spring rate of edges [N/m]

    # the base of an arm has a specific boundary condition
    base: bool = True

    def __post_init__(self):
        self.dof_matrix = np.ones_like(self.vertices)
        if self.base:
            self.dof_matrix[0] = [0, 0]
            self.dof_matrix[1] = [1, 0]

        edge_array = np.array(self.edges)
        self.resting_edge_length = np.linalg.norm(
            self.vertices[edge_array[:, 0]] - self.vertices[edge_array[:, 1]],
            axis=1,
        )

        # actuation force of edges in newtons
        # positive is extension, negative is contraction
        self.motors = np.array([0, 0, 0])

    def move_vertex(
        self, index: int, abs_coord: np.ndarray = None, rel_coord: np.ndarray = None
    ) -> np.ndarray:
        """"""
        if (abs_coord is None) == (rel_coord is None):
            raise ValueError("Need a single coordinate as input")

        if abs_coord is not None:
            rel_coord = abs_coord - self.vertices[index]

        self.vertices[index] += self.dof_matrix[index] * rel_coord

        return self.vertices

    def apply_forces(
        self, forces: np.ndarray, dof_matrix: np.ndarray = None
    ) -> np.ndarray:
        """Given forces on each node and the degrees of freedom of a node,
        calculate the node accelerations.



        Args
            forces: an Nx2 array of (x,y) component forces. N is the number of nodes
            dof_matrix: an Nx2 array of (x,y) degrees of freedom where 1 represents
                free and 0 represents constrained

        Returns
            Return an Nx2 array of (x,y) acceleration components."""
        if dof_matrix is None:
            dof_matrix = self.dof_matrix
