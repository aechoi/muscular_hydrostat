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

        """
        triangles = []
        for face in self.faces:
            if self.vertices[0] in face:
                continue
            for v1, v2 in zip(face[1:-1], face[2:]):
                triangles.append([face[0], v1, v2])
        return triangles

    def volume_constraints(self, positions, velocities):
        volume = 0
        jacobian = np.zeros(positions.size)
        djacdt = np.zeros(positions.size)

        apex_position = positions[self.vertices[0]]
        apex_velocity = velocities[self.vertices[0]]

        for triangle in self.triangles:
            relative_positions = positions[triangle] - apex_position
            relative_velocities = velocities[triangle] - apex_velocity

            if np.linalg.cond(relative_positions) >= 1 / sys.float_info.epsilon:
                # Only run calculations on non-degenerate tetrahedrons
                continue

            volume += np.linalg.det(relative_positions)

            cofactors = (
                np.linalg.det(relative_positions) * np.linalg.inv(relative_positions).T
            )
            jacobian[self.get_vec_indices([0])] -= np.sum(cofactors, axis=0)
            jacobian[self.get_vec_indices(triangle)] += cofactors.flatten()

            dcofactors = np.trace(cofactors.T @ relative_velocities) * np.linalg.inv(
                relative_positions
            ) - cofactors.T @ relative_velocities @ np.linalg.inv(relative_positions)
            dcofactors = dcofactors.T
            djacdt[self.get_vec_indices([0])] -= np.sum(dcofactors, axis=0)
            djacdt[self.get_vec_indices(triangle)] += dcofactors.flatten()

        return volume, jacobian, djacdt

    def face_constraints(self, positions, velocities, face_idx):
        """Calculate the constraint vector, Jacobian, and Jacobian time
        derivative for maintaining face planarity constraints."""
        points = positions[self.faces[face_idx]]
        dpointsdt = velocities[self.faces[face_idx]]
        N = len(points)
        D = len(points[0])

        centroid = np.average(points, axis=0)
        dcentroiddt = np.average(dpointsdt, axis=0)
        centered_points = points - centroid
        centered_velocities = dpointsdt - dcentroiddt

        cov, dcdp, dcdt, ddcdpdt = self.calc_covariance_variables(
            centered_points, centered_velocities
        )
        normal, dndp, dndt, ddndpdt = self.calc_normal_variables(
            cov, dcdp, dcdt, ddcdpdt
        )

        indices = self.get_vec_indices(self.faces[face_idx])
        constraints = centered_points @ normal

        dPdP = np.zeros((N, N, D, D))
        dPdP[np.arange(N), np.arange(N), :, :] = np.eye(D)
        dcentereddP = dPdP - np.eye(D)[None, None, :, :] / N
        jacobian = np.zeros((N, positions.size))
        jacobian[:, indices] = (
            dcentereddP @ normal + np.einsum("ij,klj->ikl", centered_points, dndp)
        ).reshape(N, -1)

        djacdt = np.zeros((N, positions.size))
        djacdt[:, indices] = (
            dcentereddP @ dndt
            + np.einsum("ij,klj->ikl", centered_velocities, dndp)
            + np.einsum("ij,klj->ikl", centered_points, ddndpdt)
        ).reshape(N, -1)

        return constraints, jacobian, djacdt

    def calc_covariance_variables(self, centered_points, centered_velocities):
        N = len(centered_points)

        cov = np.cov(centered_points.T)

        # dCdP
        units = np.expand_dims(np.eye(3), (0, 2))
        points_tensor = np.expand_dims(centered_points, (1, 3))
        dcdp_single = points_tensor @ units
        dcdp = 1 / (N - 1) * (dcdp_single.transpose(0, 1, 3, 2) + dcdp_single)

        # dCdt
        dcdt_single = np.einsum("ij,ik->jk", centered_velocities, centered_points)
        dcdt = 1 / (N - 1) * (dcdt_single + dcdt_single.T)

        # ddCdPdt
        velocity_tensor = np.expand_dims(centered_velocities, (1, 3))
        ddcdpdt_single = velocity_tensor @ units
        ddcdpdt = 1 / (N - 1) * (ddcdpdt_single.transpose(0, 1, 3, 2) + ddcdpdt_single)

        return cov, dcdp, dcdt, ddcdpdt

    def calc_normal_variables(self, cov, dcdp, dcdt, ddcdpdt):
        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigvec = eigvecs[:, np.argmin(eigvals)]

        normal = min_eigvec

        # dNdP
        eigval_dif = eigvals[:, None] - eigvals
        eigval_dif = np.divide(1, eigval_dif, out=eigval_dif, where=eigval_dif != 0)
        dndp = np.einsum(
            "ijkl,lk,mk->ijm",
            eigval_dif[0][:, None] * eigvecs.T @ dcdp,
            min_eigvec.reshape(-1, 1),
            eigvecs,
        )

        dvdts = (-eigval_dif * (eigvecs.T @ dcdt @ eigvecs))[:, :, None] * eigvecs.T[
            :, None, :
        ]
        dvdts = np.sum(dvdts, axis=0).T
        dndt = dvdts[:, 0]

        dldts = np.einsum("ij,ji->i", eigvecs.T @ dcdt, eigvecs)

        deigval_difdt = dldts[:, None] - dldts
        deigval_difdt = -np.divide(
            deigval_difdt,
            (eigvals[:, None] - eigvals) ** 2,
            out=deigval_difdt,
            where=deigval_difdt != 0,
        )

        ddndpdt = (
            np.einsum(
                "ijkl,lk,mk->ijm",
                deigval_difdt[0][:, None] * eigvecs.T @ dcdp,
                min_eigvec.reshape(-1, 1),
                eigvecs,
            )
            + np.einsum(
                "ijkl,lk,mk->ijm",
                eigval_dif[0][:, None] * dvdts.T @ dcdp,
                min_eigvec.reshape(-1, 1),
                eigvecs,
            )
            + np.einsum(
                "ijkl,lk,mk->ijm",
                eigval_dif[0][:, None] * eigvecs.T @ ddcdpdt,
                min_eigvec.reshape(-1, 1),
                eigvecs,
            )
            + np.einsum(
                "ijkl,lk,mk->ijm",
                eigval_dif[0][:, None] * eigvecs.T @ dcdp,
                dvdts[:, 0].reshape(-1, 1),
                eigvecs,
            )
            + np.einsum(
                "ijkl,lk,mk->ijm",
                eigval_dif[0][:, None] * eigvecs.T @ dcdp,
                min_eigvec.reshape(-1, 1),
                dvdts,
            )
        )

        return normal, dndp, dndt, ddndpdt

    # def self_intersection_constraint(self, positions, velocities):
    #     """return the constraints, jacobian, and jacobian time derivative for
    #     cell points intersecting the cell edges"""
    #     points = positions[self.vertices]
    #     print(points)
    #     # face vertices are ordered counter clockwise from outside
    #     constraints = []
    #     normals = np.zeros((len(self.faces), 3))
    #     centroids = np.zeros_like(normals)
    #     for idx, face in enumerate(self.faces):
    #         face_points = positions[face]
    #         centroids[idx] = np.average(face_points, axis=0)
    #         normals[idx] = np.cross(*(face_points - centroids[idx])[:2])
    #     intersect = np.einsum(
    #         "ijk,jk->ij", points[:, None, :] - centroids[None, :, :], normals
    #     )
    #     np.where(intersect > 0)[0]

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
                edge = sorted(edge)
                if edge not in self.edges:
                    self.edges.append(edge)

        self.num_fixed = 0
        self.num_face_vertices = 0
        self.init_volumes = []
        for cell in self.cells:
            self.num_fixed += len(cell.fixed_indices)
            for face in cell.faces:
                self.num_face_vertices += len(face)
            init_volume, _, _ = cell.volume_constraints(
                self.positions_init, self.velocities
            )
            self.init_volumes.append(init_volume)

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

    def add_odor(self, odor):
        self.odors.append(odor)

    def smell(self, coordinate):
        scent_strength = 0
        for odor in self.odors:
            scent_strength += odor(coordinate)
        return scent_strength

    def calc_constraints(self):
        # Check collisions first, then construct matrices
        num_collisions = 0
        constraints = np.zeros(
            num_collisions
            + self.num_fixed * 3
            + len(self.cells)
            + self.num_face_vertices
        )
        jacobians = np.zeros((len(constraints), len(self.position_vector)))
        djacdts = np.zeros((len(constraints), len(self.position_vector)))

        constraint_idx = 0
        for cell_idx, cell in enumerate(self.cells):
            # Add boundary constraints
            for idx in cell.fixed_indices:
                constraints[constraint_idx] = (
                    self.positions[idx][0] - self.positions_init[idx][0]
                )
                constraints[constraint_idx + 1] = (
                    self.positions[idx][1] - self.positions_init[idx][1]
                )
                constraints[constraint_idx + 2] = (
                    self.positions[idx][2] - self.positions_init[idx][2]
                )

                jacobians[constraint_idx, idx * 3] = 1
                jacobians[constraint_idx + 1, idx * 3 + 1] = 1
                jacobians[constraint_idx + 2, idx * 3 + 2] = 1
                constraint_idx += 3

            # Add volume constraints
            volume, jacobian, djacdt = cell.volume_constraints(
                self.positions, self.velocities
            )
            if volume != 0:
                constraints[constraint_idx] = volume - self.init_volumes[cell_idx]
                jacobians[constraint_idx] = jacobian
                djacdts[constraint_idx] = djacdt
                constraint_idx += 1

            # Add face constraints
            for idx, face in enumerate(cell.faces):
                if len(face) <= 3:
                    continue
                face_constraints, face_jacobians, face_djacdts = cell.face_constraints(
                    self.positions, self.velocities, idx
                )
                constraints[constraint_idx : constraint_idx + len(face)] = (
                    face_constraints
                )
                jacobians[constraint_idx : constraint_idx + len(face)] = face_jacobians
                djacdts[constraint_idx : constraint_idx + len(face)] = face_djacdts
                constraint_idx += len(face)

            # TODO Self Intersection
            # TODO Obstacle Collisions
        return constraints, jacobians, djacdts

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
        configuration. Muscle actuation is always positive and relates to a
        contraction of the muscle."""
        if force < 0:
            raise ValueError("Muscle activation must be positive")
        self.muscles[edge_indices] = force
        return self.muscles

    def control_muscles(self):
        """Set muscle actuations based using controller"""
        self.muscles = self.muscles * 0
        if not self.odors:
            return

        tip_cell = self.cells[-1]
        # find gradient using vertex scents
        points = self.positions[tip_cell.vertices]
        scents = self.smell(points)
        gradient = (
            np.linalg.pinv(np.column_stack((points, np.ones(len(points))))) @ scents
        )[:-1]
        gradient = gradient / np.linalg.norm(gradient)

        forward_backward_gradient = gradient[-1]
        print(forward_backward_gradient)
        for cell in self.cells:
            if forward_backward_gradient > 0:
                edge_index = [
                    self.edges.index(sorted(edge)) for edge in cell.edges[-4:]
                ]
                self.set_muscle_actuations(edge_index, forward_backward_gradient)
            else:
                edge_index = [
                    self.edges.index(sorted(edge)) for edge in cell.edges[4:-4]
                ]
                self.set_muscle_actuations(edge_index, -forward_backward_gradient)

    def active_edge_forces(self) -> np.ndarray:
        """Given the current state of muscle actuations, return the forces
        on each vertex."""
        edge_forces = np.zeros_like(self.positions)
        for edge, muscle_force in zip(self.edges, self.muscles):
            edge_vector = self.positions[edge[1]] - self.positions[edge[0]]
            edge_vector = edge_vector / np.linalg.norm(edge_vector)
            edge_forces[edge[1]] -= edge_vector * muscle_force
            edge_forces[edge[0]] += edge_vector * muscle_force
        return edge_forces

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

        constraint, jac, djac = self.calc_constraints()

        self.control_muscles()
        active_edge_forces = self.active_edge_forces()
        passive_edge_forces = self.passive_edge_forces()

        # TODO figure out how to rigorously size the regularization term
        front_inverse = np.linalg.inv(
            jac @ self.inv_mass_mat @ jac.T + np.eye(len(constraint)) * 1e-6
        )
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
            self.constraint_spring * constraint
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
