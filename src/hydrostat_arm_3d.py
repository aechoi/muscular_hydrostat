from dataclasses import dataclass
from typing import Sequence
import numpy as np
import sys

from data_logger import DataLogger


@dataclass
class HydrostatCell3D:
    """Defines cell relations"""

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
            self.masses = np.ones(len(self.vertices))
        if self.vertex_damping is None:
            self.vertex_damping = np.ones(len(self.vertices))
        if self.edge_damping is None:
            self.edge_damping = np.ones(len(self.edges))

        self.triangles = self.triangulate_faces()

    def triangulate_faces(self):
        """Decompose each face into triangles for the purposes of volume
        calculation.

        TODO: omit triangles with index 0 to avoid filtering out later.
        I don't think 0 index triangles are ever used.
        """
        triangles = []
        for face in self.faces:
            for v1, v2 in zip(face[1:-1], face[2:]):
                triangles.append([face[0], v1, v2])
        return triangles

    def volume(self, positions):
        """Calculate the volume of the polyhedron

        TODO Get rid of all the /6 factors in volume and jacobian calculations.
        It doesn't have to literally be volume"""
        volume = 0
        apex_position = positions[self.vertices[0]]
        for triangle in self.triangles:
            if 0 in triangle:
                continue
            relative_positions = positions[triangle] - apex_position
            if np.linalg.cond(relative_positions) >= 1 / sys.float_info.epsilon:
                continue
            volume += np.linalg.det(relative_positions)
        return volume / 6

    def volume_jacobian(self, positions):
        """Calculate the vector shaped volume jacobian for the cell.

        TODO: Merge with volume calc to reduce redundant calculations"""
        jacobian = np.zeros(positions.size)
        apex_position = positions[self.vertices[0]]
        for triangle in self.triangles:
            if 0 in triangle:
                continue
            relative_positions = positions[triangle] - apex_position
            if np.linalg.cond(relative_positions) >= 1 / sys.float_info.epsilon:
                continue
            cofactors = (
                np.linalg.det(relative_positions) * np.linalg.inv(relative_positions).T
            )
            jacobian[self.get_vec_indices([0])] -= np.sum(cofactors, axis=0)
            jacobian[self.get_vec_indices(triangle)] += cofactors.flatten()
        return jacobian / 6

    def volume_jacobian_derivative(self, positions, velocities):
        djacobian = np.zeros(positions.size)
        apex_position = positions[self.vertices[0]]
        apex_velocity = velocities[self.vertices[0]]
        for triangle in self.triangles:
            if 0 in triangle:
                continue
            relative_positions = positions[triangle] - apex_position
            relative_velocities = positions[triangle] - apex_velocity
            if np.linalg.cond(relative_positions) >= 1 / sys.float_info.epsilon:
                continue

            cofactors = (
                np.linalg.det(relative_positions) * np.linalg.inv(relative_positions).T
            )

            dcofactors = np.trace(cofactors.T @ relative_velocities) * np.linalg.inv(
                relative_positions
            ) - cofactors.T @ relative_velocities @ np.linalg.inv(relative_positions)
            dcofactors = dcofactors.T
            djacobian[self.get_vec_indices([0])] -= np.sum(dcofactors, axis=0)
            djacobian[self.get_vec_indices(triangle)] += dcofactors.flatten()

        return djacobian / 6

    def face_normal(self, positions, face_idx):
        """Calculate the best fit normal plane and centroid for a face. The
        sign of the normal is not guaranteed."""
        centroid = np.average(positions[self.faces[face_idx]], axis=0)
        _, _, Vh = np.linalg.svd(positions[self.faces[face_idx]] - centroid)
        return Vh[-1], centroid

    def get_vec_indices(self, vertex_indices: list):
        """Return the list of indices that correspond to the vertex indices."""
        if type(vertex_indices) is int:
            vertex_indices = [vertex_indices]
        vertex_indices = np.array(vertex_indices)
        return (vertex_indices[:, None] * 3 + np.arange(3)[None, :]).flatten()

    def calc_next_states(self, dt):
        pass


@dataclass
class HydrostatArm3D:
    positions: np.ndarray  # nxd array of point coordinates
    cells: list[HydrostatCell3D]
    odor_func: callable = None
    obstacles: list[object] = None

    constraint_spring: float = 500
    constraint_damper: float = 10

    def __post_init__(self):
        ## Arm geometry
        self.positions = self.positions.astype(float)
        self.positions_init = self.positions.copy()
        self.velocities = np.zeros_like(self.positions)  # nxd array of point velocities

        # numpy quirk, ravel creates a view rather than a copy, so changing one
        # of the below changes the original points/velocities arrays and vice
        # versa. Essentially a built in automatic setter!
        self.position_vector = np.ravel(self.positions)
        self.velocity_vector = np.ravel(self.velocities)

        self.edges = []
        for cell in self.cells:
            for edge in cell.edges:
                # TODO check if sorting necessary. Don't remember why...
                if sorted(edge) not in self.edges:
                    self.edges.append(edge)

        ## Arm parameters/variables
        self.inv_mass_mat = np.eye(
            self.positions.size
        )  # square matrix with shape of number of vertices
        self.damping_mat = np.eye(
            self.positions.size
        )  # square matrix with shape of number of vertices
        for cell in self.cells:
            for vertex, mass, damper in zip(
                cell.vertices, cell.masses, cell.vertex_damping
            ):
                indices = cell.get_vec_indices(vertex)
                self.inv_mass_mat[indices, indices] = mass
                self.damping_mat[indices, indices] = damper
        # TODO get edge damping rates from cell info and check for contradiction
        self.edge_damping = [1] * len(self.edges)

        self.external_forces = np.zeros_like(
            self.positions
        )  # force vector for each vertex
        self.muscles = np.zeros(
            len(self.edges)
        )  # scalar force of contraction for each edge

        ## Environment parameters
        self.obstacles = []
        self.odors = []

        ## Simulation variables
        self.timestamp = 0

    def add_obstacle(self, obstacle):
        """Add a ConvexObstacle that could collide with the arm."""
        self.obstacles.append(obstacle)

    def constraints(self) -> np.ndarray:
        constraints = []
        for cell in self.cells:
            # Add boundary constraints
            for idx in cell.fixed_indices:
                constraints.append(self.positions[idx][0] - self.positions_init[idx][0])
                constraints.append(self.positions[idx][1] - self.positions_init[idx][1])
                constraints.append(self.positions[idx][2] - self.positions_init[idx][2])

            # Add volume constraints
            if cell.volume(self.positions_init) != 0:
                constraints.append(
                    cell.volume(self.positions) - cell.volume(self.positions_init)
                )

            # Add face constraints
            for idx, face in enumerate(cell.faces):
                if len(face) <= 3:
                    continue
                normal, centroid = cell.face_normal(self.positions, idx)
                errors = (self.positions[face] - centroid) @ normal
                # print("normal", normal)
                # print("centroid", centroid)
                # print("errors", errors)
                constraints.extend([*errors])

            # Add self intersection constraints
            # TODO

        # Add obstacle collision constraints
        # TODO design 3d obstacle calculations
        # for obstacle in self.obstacles:
        #     for vertex in cell.vertices:
        #         point = self.points[vertex]
        #         if obstacle.check_intersection(point):
        #             nearest_point = obstacle.nearest_point(point)
        #             constraints.append(point[0] - nearest_point[0])
        #             constraints.append(point[1] - nearest_point[1])
        #             constraints.append(point[2] - nearest_point[2])

        return np.array(constraints)

    def jacobian(self) -> np.ndarray:
        # TODO don't recalculate constraints with every jacobian. Honestly,
        # it might make sense to merge all of these into a single function
        # because of repeated calculations.
        jacobian = np.zeros((len(self.constraints()), len(self.position_vector)))
        constraint_idx = 0

        for cell in self.cells:
            # Add boundary constraints
            for idx in cell.fixed_indices:
                jacobian[constraint_idx * 3 : constraint_idx * 3 + 3, idx] = 1
                constraint_idx += 3

            # Add volume constraints
            if cell.volume(self.positions_init) != 0:
                jacobian[constraint_idx] = cell.volume_jacobian(self.positions)
                constraint_idx += 1

            # Add face constraints
            for idx, face in enumerate(cell.faces):
                if len(face) <= 3:
                    continue
                normal, _ = cell.face_normal(self.positions, idx)
                for vertex in face:
                    jacobian[constraint_idx, cell.get_vec_indices(vertex)] = normal * (
                        1 - 1 / len(face)
                    )
                    constraint_idx += 1

            # Add self intersection constraints
            # TODO

        # Add obstacle collision constraints
        return jacobian

    def jacobian_derivative(self) -> np.ndarray:
        djacobian = np.zeros((len(self.constraints()), len(self.position_vector)))
        constraint_idx = 0

        for cell in self.cells:
            # Add boundary constraints
            constraint_idx += 3 * len(cell.fixed_indices)

            # Add volume constraints
            if cell.volume(self.positions_init) != 0:
                djacobian[constraint_idx] = cell.volume_jacobian_derivative(
                    self.positions, self.velocities
                )
                constraint_idx += 1

            # Add face constraints
            for face in cell.faces:
                if len(face) <= 3:
                    continue
                for vertex in face:
                    constraint_idx += 1

            # Add self intersection constraints TODO

        # Add obstacle collision constrainst TODO
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
        on each vertex. TODO"""
        internal_forces = np.zeros_like(self.positions)
        return internal_forces

    def passive_edge_forces(self) -> np.ndarray:
        """Forces from passive elements"""
        edge_forces = np.zeros_like(self.positions)
        for edge, damping_rate in zip(self.edges, self.edge_damping):
            edge_vector = self.positions[edge[1]] - self.positions[edge[0]]
            edge_unit_vector = edge_vector / np.linalg.norm(edge_vector)
            relative_velocity = self.velocities[edge[1]] - self.velocities[edge[0]]
            edge_velocity = (
                np.dot(edge_unit_vector, relative_velocity) * edge_unit_vector
            )  # extension positive, contraction negative
            edge_damp_force = damping_rate * edge_velocity
            edge_forces[edge[0]] -= edge_damp_force
            edge_forces[edge[1]] += edge_damp_force

        return edge_forces

    def calc_next_states(self, dt):
        self.timestamp += dt

        jac = self.jacobian()
        djac = self.jacobian_derivative()
        active_edge_forces = self.active_edge_forces()
        passive_edge_forces = self.passive_edge_forces()

        # lagrange_mult = np.linalg.inv(jac @ self.inv_mass_mat @ jac.T) @ (
        #     (jac @ self.inv_mass_mat @ self.damping_mat - djac) @ self.velocity_vector
        #     - jac
        #     @ self.inv_mass_mat
        #     @ (
        #         self.external_forces + active_edge_forces - passive_edge_forces
        #     ).flatten()
        #     - self.constraint_spring * self.constraints()
        #     - self.constraint_damper * jac @ self.velocity_vector
        # )
        front_inverse = np.linalg.inv(jac @ self.inv_mass_mat @ jac.T)
        velocity_term = (
            jac @ self.inv_mass_mat @ self.damping_mat - djac
        ) @ self.velocity_vector
        force_term = (
            jac
            @ self.inv_mass_mat
            @ (
                self.external_forces + active_edge_forces - passive_edge_forces
            ).flatten()
        )
        constraint_control = (
            self.constraint_spring * self.constraints()
            + self.constraint_damper * jac @ self.velocity_vector
        )
        lagrange_mult = front_inverse @ (
            velocity_term - force_term - constraint_control
        )

        reactions = jac.T @ lagrange_mult

        accel = self.inv_mass_mat @ (
            (self.external_forces + active_edge_forces).flatten()
            + reactions
            - self.damping_mat @ self.velocity_vector
            - passive_edge_forces.flatten()
        )

        self.velocity_vector += accel * dt
        self.position_vector += self.velocity_vector * dt

        return self.position_vector, self.velocity_vector, accel