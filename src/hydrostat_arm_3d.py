from dataclasses import dataclass
from typing import Sequence
import numpy as np

from data_logger import DataLogger


@dataclass
class HydrostatCell3D:
    """Defines cell relations"""

    vertices: list[float]  # index of vertices
    edges: list[tuple]  # each tuple indexes 2 points
    faces: list[
        tuple
    ]  # indices of points, tuples may be ragged, must be ordered counter-clockwise from outside

    fixed_indices: list[int] = None

    masses: list[float] = None
    vertex_damping: list[float] = None
    edge_damping: list[float] = None

    def __post_init__(self):
        self.triangles = self.triangulate_faces()

    def triangulate_faces(self):
        """Decompose each face into triangles for the purposes of volume
        calculation."""
        triangles = []
        for face in self.faces:
            for v1, v2 in zip(face[1:-1], face[2:]):
                triangles.append([face[0], v1, v2])
        return triangles

    def cell_volume(self, points):
        """Calculate the volume of the polyhedron"""
        volume = 0
        apex_coordinate = points[self.vertices[0]]
        for triangle in self.triangles:
            if 0 in triangle:
                continue
            relative_vectors = points[triangle] - apex_coordinate
            volume += np.linalg.det(relative_vectors)
        return volume / 6

    def calc_next_states(self, dt):
        pass


@dataclass
class HydrostatArm3D:
    points: np.ndarray  # nxd array of point coordinates
    cells: list[HydrostatCell3D]
    odor_func: callable = None
    obstacles: list[object] = None

    constraint_spring: float = 500
    constraint_damper: float = 10

    def __post_init__(self):
        # Arm geometry
        self.velocities = np.zeros_like(self.points)  # nxd array of point velocities

        # numpy quirk, ravel creates a view rather than a copy, so changing one
        # of the below changes the original points/velocities arrays and vice
        # versa. Essentially a built in automatic setter!
        self.position_vector = np.ravel(self.points)
        self.velocity_vector = np.ravel(self.velocities)

        self.edges = None  # list of edge tuples

        # Arm parameters/variables
        self.inv_mass_mat = None  # square matrix with shape of number of vertices
        self.damping_mat = None  # square matrix with shape of number of vertices
        self.external_forces = np.zeros_like(
            self.points
        )  # force vector for each vertex
        self.muscles = np.zeros(
            len(self.edges)
        )  # scalar force of contraction for each edge

        # Environment parameters
        self.obstacles = []
        self.odors = []

        # Simulation variables
        self.timestamp = 0

    def add_obstacle(self, obstacle):
        """Add a ConvexObstacle that could collide with the arm."""
        self.obstacles.append(obstacle)

    def constraints(self) -> np.ndarray:
        constraints = None
        return constraints

    def jacobian(self) -> np.ndarray:
        jacobian = None
        return jacobian

    def jacobian_derivative(self) -> np.ndarray:
        djacobian = None
        return djacobian

    def set_external_forces(
        self, vertex_indices: list[int], force: Sequence[float]
    ) -> np.ndarray:
        """Modify external force array to have new force. Return the current
        external force configuration."""
        self.external_forces[vertex_indices] = force
        return self.external_forces

    def set_muscle_actuations(
        self, edge_indices: list[int], force: float
    ) -> np.ndarray:
        """Modify muscle array to have new force. Return the current muscle
        configuration."""
        self.muscles[edge_indices] = force
        return self.muscles

    def active_edge_forces(self) -> np.ndarray:
        """Given the current state of muscle actuations, return the forces
        on each vertex."""
        internal_forces = np.zeros_like(self.points)
        return internal_forces

    def passive_edge_forces(self) -> np.ndarray:
        """Forces from passive elements"""
        edge_forces = np.zeros_like(self.points)
        return edge_forces

    def calc_next_states(self, dt):
        self.timestamp += dt

        jac = self.jacobian()
        djac = self.jacobian_derivative()
        active_edge_forces = self.active_edge_forces()
        passive_edge_forces = self.passive_edge_forces()

        lagrange_mult = np.linalg.inv(jac @ self.inv_mass_mat @ jac.T) @ (
            (jac @ self.inv_mass_mat @ self.damping_mat - djac) @ self.velocity_vector
            - jac
            @ self.inv_mass_mat
            @ (self.external_forces + active_edge_forces - passive_edge_forces)
            - self.constraint_spring * self.constraints()
            - self.constraint_damper * jac @ self.velocity_vector
        )

        reactions = jac.T @ lagrange_mult

        accel = self.inv_mass_mat @ (
            self.external_forces
            + active_edge_forces
            + reactions
            - self.damping_mat @ self.velocity_vector
            - passive_edge_forces
        )

        self.velocity_vector = self.velocity_vector + accel * dt
        self.position_vector = self.position_vector + self.velocity_vector * dt
