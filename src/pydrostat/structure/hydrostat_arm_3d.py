# TODO: use cupy for the large matrix manipulations (front inverse and probably
# face matrix calculations). Look into using @njit from numba for necessary
# for loops in constraint calculation

from dataclasses import dataclass
from typing import Sequence
import time

# import logging

import numpy as np

# import cupy as cp

# from numba import jit

# from data_logger import DataLogger

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename="log.log",
#     filemode="w",
#     format="%(asctime)s %(levelname)s: %(message)s",
#     level=logging.DEBUG,
# )


def get_vec_indices(vertex_indices: list):
    """Return the list of indices that correspond to the vertex indices."""
    if type(vertex_indices) is int:
        vertex_indices = [vertex_indices]
    vertex_indices = np.array(vertex_indices)
    return (vertex_indices[:, None] * 3 + np.arange(3)[None, :]).flatten()


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
        return np.array(triangles)

    def volume_constraints(self, positions, velocities):
        jacobian = np.zeros(positions.size)
        djacdt = np.zeros(positions.size)

        apex_position = positions[self.vertices[0]]
        apex_velocity = velocities[self.vertices[0]]

        relative_positions = positions[self.triangles] - apex_position  # Tx3xD
        relative_velocities = velocities[self.triangles] - apex_velocity
        volumes = np.linalg.det(relative_positions)  # T
        volume = np.sum(volumes)

        position_inverse = np.linalg.inv(relative_positions)  # Tx3xD

        cofactors = volumes[:, None, None] * position_inverse.swapaxes(-1, -2)  # TxDx3
        adjugate = cofactors.swapaxes(-1, -2)
        jacobian[get_vec_indices([0])] = -np.sum(cofactors, axis=(0, 1))
        vec_indices = get_vec_indices(self.triangles.flatten())
        np.add.at(jacobian, vec_indices, cofactors.flatten())

        dcofactors = (
            np.trace(adjugate @ relative_velocities, axis1=1, axis2=2)[:, None, None]
            * position_inverse
            - adjugate @ relative_velocities @ position_inverse
        )
        djacdt[get_vec_indices([0])] = -np.sum(dcofactors, axis=(0, 1))
        np.add.at(djacdt, vec_indices, dcofactors.flatten())

        return volume, jacobian, djacdt


def calc_face_constraints(positions, velocities, face_indices):
    """Calculate the constraint vector, Jacobian, and Jacobian time
    derivative for maintaining face planarity constraints."""
    # Faces are looped over because faces can be ragged. Can we vectorize anyways?
    # face_indices, f_i, can have different length for different i

    # logger.debug("start face constraint calculations")
    # last_time = time.perf_counter()

    points = positions[face_indices]
    dpointsdt = velocities[face_indices]
    N, D = points.shape

    centroid = np.average(points, axis=0)
    dcentroiddt = np.average(dpointsdt, axis=0)
    centered_points = points - centroid
    centered_velocities = dpointsdt - dcentroiddt

    # logger.debug(f"[{time.perf_counter() - last_time}] calc setup")
    # last_time = time.perf_counter()

    cov, dcdp, dcdt, ddcdpdt = calc_covariance_variables(
        centered_points, centered_velocities
    )
    # logger.debug(f"[{time.perf_counter() - last_time}] covar calcs")
    # last_time = time.perf_counter()

    normal, dndp, dndt, ddndpdt = calc_normal_variables(cov, dcdp, dcdt, ddcdpdt)
    # logger.debug(f"[{time.perf_counter() - last_time}] normal calcs")
    # last_time = time.perf_counter()

    indices = get_vec_indices(face_indices)
    constraints = centered_points @ normal  # NxD @ D = N

    dPdP = np.zeros((N, N, D, D))
    dPdP[np.arange(N), np.arange(N), :, :] = np.eye(D)
    dcentereddP = dPdP - np.eye(D)[None, None, :, :] / N  # NxNxDxD
    jacobian = np.zeros((N, positions.size))
    jacobian[:, indices] = (
        dcentereddP @ normal  # NxNxDxD @ D = NxNxD
        + (centered_points @ dndp).swapaxes(0, 1)  # (NxD @ NxDxD)swap = NxNxD
    ).reshape(N, -1)

    djacdt = np.zeros((N, positions.size))
    djacdt[:, indices] = (
        dcentereddP @ dndt
        + (centered_velocities @ dndp).swapaxes(0, 1)
        + (centered_points @ ddndpdt).swapaxes(0, 1)
    ).reshape(N, -1)
    # logger.debug(f"[{time.perf_counter() - last_time}] assembly calcs")

    return constraints, jacobian, djacdt


def calc_covariance_variables(centered_points, centered_velocities):
    dof = len(centered_points) - 1

    cov = np.cov(centered_points.T)

    units = np.eye(3)[None, :, None, :]

    # dCdP
    dcdp_single = centered_points[:, None, :, None] @ units
    dcdp = (dcdp_single.swapaxes(-1, -2) + dcdp_single) / dof

    # dCdt
    dcdt_single = centered_velocities.T @ centered_points
    dcdt = (dcdt_single + dcdt_single.T) / dof

    # ddCdPdt
    ddcdpdt_single = centered_velocities[:, None, :, None] @ units
    ddcdpdt = (ddcdpdt_single.swapaxes(-1, -2) + ddcdpdt_single) / dof

    return cov, dcdp, dcdt, ddcdpdt


def calc_normal_variables(cov, dcdp, dcdt, ddcdpdt):
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecsT = eigvecs.T
    min_eigvec = eigvecs[:, np.argmin(eigvals)]

    normal = min_eigvec

    # dNdP axes 1 and 2 swapped because not used elsewhere otherwise
    eigval_dif = eigvals[:, None] - eigvals
    eigval_dif[:] = np.divide(1, eigval_dif, out=eigval_dif, where=eigval_dif != 0)
    min_eigval_dif = eigval_dif[0][:, None]
    dndp = (min_eigval_dif * eigvecsT @ dcdp @ min_eigvec @ eigvecsT).swapaxes(-2, -1)

    dvdts = np.sum(
        (-eigval_dif * (eigvecsT @ dcdt @ eigvecs))[:, :, None] * eigvecsT[:, None, :],
        axis=0,
    ).T
    dndt = dvdts[:, 0]

    dldts = np.diagonal(eigvecsT @ dcdt @ eigvecs)

    deigval_difdt = dldts[:, None] - dldts
    deigval_difdt[:] = -np.divide(
        deigval_difdt,
        (eigvals[:, None] - eigvals) ** 2,
        out=deigval_difdt,
        where=deigval_difdt != 0,
    )

    # ugly and ragged to reduce computation
    ddndpdt = (
        (
            (
                (deigval_difdt[0][:, None] * eigvecsT + min_eigval_dif * dvdts.T) @ dcdp
                + min_eigval_dif * eigvecsT @ ddcdpdt
            )
            @ min_eigvec
            + min_eigval_dif * eigvecsT @ dcdp @ dvdts[:, 0]
        )
        @ eigvecsT
        + min_eigval_dif * eigvecsT @ dcdp @ min_eigvec @ dvdts.T
    ).swapaxes(-2, -1)

    return normal, dndp, dndt, ddndpdt


@dataclass
class HydrostatArm3D:
    positions: np.ndarray  # nxd array of point coordinates
    cells: list[HydrostatCell3D]
    odor_func: callable = None
    obstacles: list[object] = None

    constraint_spring: float = 500
    constraint_damper: float = 10

    environment = None

    def __post_init__(self):
        ## Arm geometry
        self.positions = self.positions.astype(float)
        self.positions_init = self.positions.copy()
        self.velocities = np.zeros_like(self.positions)  # nxd array of point velocities

        # ravel creates a view rather than a copy, so changing one
        # of the below changes the original points/velocities arrays and vice
        # versa. Essentially a built in automatic setter!
        self.position_vector = np.ravel(self.positions)
        self.velocity_vector = np.ravel(self.velocities)

        self.num_fixed = 0
        self.num_face_vertices = 0
        self.init_volumes = []
        self.edges = []
        self.faces = []
        for cell in self.cells:
            self.num_fixed += len(cell.fixed_indices)
            init_volume, _, _ = cell.volume_constraints(
                self.positions_init, self.velocities
            )
            self.init_volumes.append(init_volume)

            for edge in cell.edges:
                edge = sorted(edge)
                if edge not in self.edges:
                    self.edges.append(edge)

            for face in cell.faces:
                face = sorted(face)
                if face not in self.faces:
                    self.faces.append(face)
                    self.num_face_vertices += len(face)
        self.edges = np.array(self.edges)

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
                indices = get_vec_indices(vertex)
                self.inv_mass_mat[indices, indices] = 1 / mass
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
        if self.environment is not None:
            self.obstacles = self.environment.obstacles

        ## Simulation variables
        self.timestamp = 0

    def set_environment(self, environment):
        self.environment = environment
        self.obstacles = environment.obstacles

    def smell(self, coordinate):
        return self.environment.sample_scent(coordinate)

    def calc_constraints(self):
        ## Calculate variable length constraints first.
        # start_time = time.perf_counter()
        # logger.debug("calc_constraints start")

        # last_time = time.perf_counter()
        # Obstacles
        collated_collision_mask = np.empty(0, dtype=bool)
        collated_nearest_points = np.empty((0, 3))
        for obstacle in self.obstacles:
            collision_mask = obstacle.check_intersection(self.positions)
            nearest_points = obstacle.nearest_point(self.positions[collision_mask])
            collated_collision_mask = np.hstack(
                (collated_collision_mask, collision_mask)
            )
            collated_nearest_points = np.vstack(
                (collated_nearest_points, nearest_points)
            )
        num_collisions = np.sum(collated_collision_mask)

        # logger.debug(
        #     f"[{time.perf_counter() - last_time}] calc collisions and nearest points"
        # )
        # last_time = time.perf_counter()

        # Edge length. No edge less than minimum edge length
        min_edge_length = 0.4
        edge_points = self.positions[self.edges]
        edge_lengths = np.linalg.norm(edge_points[:, 0] - edge_points[:, 1], axis=1)
        edge_mask = (edge_lengths < min_edge_length).astype(bool)
        num_short = np.sum(edge_mask)

        # logger.debug(
        #     f"[{time.perf_counter() - last_time}] calc minimum edge constraints"
        # )
        # last_time = time.perf_counter()

        ## Instantiate constraint matrices
        constraints = np.zeros(
            num_collisions * 3
            + num_short
            + self.num_fixed * 3
            + len(self.cells)
            + self.num_face_vertices
        )
        jacobians = np.zeros((len(constraints), len(self.position_vector)))
        djacdts = np.zeros((len(constraints), len(self.position_vector)))

        # logger.debug(f"[{time.perf_counter() - last_time}] instantiate matrices")
        # last_time = time.perf_counter()

        constraint_idx = 0
        if num_collisions > 0:
            constraints[: num_collisions * 3] = (
                self.positions[collated_collision_mask] - collated_nearest_points
            ).flatten()
            pos_indices = get_vec_indices(
                np.where(collated_collision_mask)[0] % len(self.positions)
            )
            jacobians[np.arange(num_collisions * 3), pos_indices] = 1
            constraint_idx += num_collisions * 3

        # logger.debug(f"[{time.perf_counter() - last_time}] add collision constraints")
        # last_time = time.perf_counter()

        if num_short > 0:
            edge_lengths = edge_lengths[edge_mask]
            constraints[constraint_idx : constraint_idx + np.sum(edge_mask)] = (
                edge_lengths - min_edge_length - 0.01
            )
            short_points = edge_points[edge_mask]
            short_dif = short_points[:, 0] - short_points[:, 1]
            norm_difs = (short_dif) / edge_lengths[:, None]

            pos_indices = get_vec_indices(self.edges[edge_mask].reshape(-1)).reshape(
                -1, 2, 3
            )
            constraint_indices = constraint_idx + np.arange(num_short).reshape(-1, 1)

            jacobians[constraint_indices, pos_indices[:, 0]] = norm_difs
            jacobians[constraint_indices, pos_indices[:, 1]] = -norm_difs

            short_velocities = self.velocities[self.edges[edge_mask]]
            vel_dif = short_velocities[:, 0] - short_velocities[:, 1]
            dunit_edge = (
                vel_dif / edge_lengths[:, None]
                - short_dif
                * (np.diag(short_dif @ vel_dif.T) / edge_lengths**3)[:, None]
            )

            djacdts[constraint_indices, pos_indices[:, 0]] = dunit_edge
            djacdts[constraint_indices, pos_indices[:, 1]] = -dunit_edge
            constraint_idx += num_short

        # logger.debug(f"[{time.perf_counter() - last_time}] add short edge constraints")
        # last_time = time.perf_counter()

        for cell_idx, cell in enumerate(self.cells):
            # Add boundary constraints
            for idx in cell.fixed_indices:
                # TODO precompile a list of fixed indices, fix them all at once
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

            # logger.debug(
            #     f"[{time.perf_counter() - last_time}] add cell {cell_idx} boundary constraints"
            # )
            # last_time = time.perf_counter()

            # Add volume constraints
            volume, jacobian, djacdt = cell.volume_constraints(
                self.positions, self.velocities
            )
            if volume != 0:
                constraints[constraint_idx] = volume - self.init_volumes[cell_idx]
                jacobians[constraint_idx] = jacobian
                djacdts[constraint_idx] = djacdt
                constraint_idx += 1

            # logger.debug(
            #     f"[{time.perf_counter() - last_time}] add cell {cell_idx} volume constraints"
            # )
            # last_time = time.perf_counter()

        # Add face constraints
        for face in self.faces:
            if len(face) <= 3:
                continue
            face_constraints, face_jacobians, face_djacdts = calc_face_constraints(
                self.positions, self.velocities, face
            )
            constraints[constraint_idx : constraint_idx + len(face)] = face_constraints
            jacobians[constraint_idx : constraint_idx + len(face)] = face_jacobians
            djacdts[constraint_idx : constraint_idx + len(face)] = face_djacdts
            constraint_idx += len(face)

        # logger.debug(
        #     f"[{time.perf_counter() - last_time}] add cell {cell_idx} face constraints"
        # )
        # last_time = time.perf_counter()

        # TODO Self Intersection
        # logger.debug(f"[{time.perf_counter() - start_time}] total constraint time")
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
        if self.environment is None:
            return

        forward_backward_gradient = 0
        strength_scales = np.logspace(-1, 2, len(self.cells), base=4.5)

        for idx, cell in enumerate(self.cells[::-1]):
            strength_scale = strength_scales[idx]

            points = self.positions[cell.vertices]
            scents = self.smell(points)

            gradient = (
                np.linalg.pinv(np.column_stack((points, np.ones(len(points))))) @ scents
            )[:-1]
            gradient = gradient / np.linalg.norm(gradient)

            top_face = cell.faces[-1]
            normal = np.cross(
                np.diff(self.positions[top_face[0:2]], axis=0),
                np.diff(self.positions[top_face[1:3]], axis=0),
            ).flatten()
            normal = normal / np.linalg.norm(normal)

            forward_backward_gradient = np.dot(gradient, normal)

            if forward_backward_gradient > 0:
                edge_index = np.array(
                    [
                        # self.edges.index(sorted(edge)) for edge in cell.edges[-4:]
                        np.where(np.all(self.edges == sorted(edge), axis=1))[0]
                        for edge in cell.edges[-4:]
                    ]
                ).flatten()
                self.muscles[edge_index] = forward_backward_gradient / 1.5
                # if idx == 0:
                #     self.muscles[edge_index] = forward_backward_gradient / 2
            else:
                edge_index = np.array(
                    [
                        # self.edges.index(sorted(edge)) for edge in cell.edges[4:-8]
                        np.where(np.all(self.edges == sorted(edge), axis=1))[0]
                        for edge in cell.edges[4:-8]
                    ]
                ).flatten()
                self.muscles[edge_index] = -forward_backward_gradient

            desired_motion = gradient - normal
            # desired_motion = desired_motion / np.linalg.norm(desired_motion)
            top_centroid = np.average(self.positions[top_face], axis=0)
            rel_vertices = self.positions[top_face] - top_centroid
            activations = rel_vertices @ desired_motion
            # edge_index = [self.edges.index(sorted(edge)) for edge in cell.edges[4:-8]]
            edge_index = np.array(
                [
                    np.where(np.all(self.edges == sorted(edge), axis=1))[0]
                    for edge in cell.edges[4:-8]
                ]
            ).flatten()
            self.muscles[edge_index] += activations * strength_scale * 4
            # edge_index = [self.edges.index(sorted(edge)) for edge in cell.edges[8:-4]]
            edge_index = np.array(
                [
                    np.where(np.all(self.edges == sorted(edge), axis=1))[0]
                    for edge in cell.edges[8:-4]
                ]
            ).flatten()
            self.muscles[edge_index] += activations * strength_scale * 2
            self.muscles = np.clip(self.muscles, 0, None)

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

        # start_time = time.perf_counter()
        # logger.debug(f"dynamics start")

        # last_time = time.perf_counter()
        constraint, jac, djac = self.calc_constraints()
        # logger.debug(f"[{time.perf_counter() - last_time}] constraints calculated")
        # last_time = time.perf_counter()

        self.control_muscles()
        # logger.debug(f"[{time.perf_counter() - last_time}] muscle control")
        # last_time = time.perf_counter()
        active_edge_forces = self.active_edge_forces()
        # logger.debug(f"[{time.perf_counter() - last_time}] set active forces")
        # last_time = time.perf_counter()
        passive_edge_forces = self.passive_edge_forces()
        # logger.debug(f"[{time.perf_counter() - last_time}] set passive forces")
        # last_time = time.perf_counter()

        # TODO figure out how to rigorously size the regularization term
        front_inverse = np.linalg.inv(
            jac @ self.inv_mass_mat @ jac.T + np.eye(len(constraint)) * 1e-6
        )
        # logger.debug(f"[{time.perf_counter() - last_time}] front inverse calculated")
        # last_time = time.perf_counter()

        velocity_term = (
            jac @ self.inv_mass_mat @ self.damping_mat - djac
        ) @ self.velocity_vector
        # logger.debug(f"[{time.perf_counter() - last_time}] velocity component")
        # last_time = time.perf_counter()

        force_term = (
            jac
            @ self.inv_mass_mat
            @ (
                self.external_forces + active_edge_forces - passive_edge_forces
            ).flatten()
        )
        # logger.debug(f"[{time.perf_counter() - last_time}] force component")
        # last_time = time.perf_counter()

        constraint_control = (
            self.constraint_spring * constraint
            + self.constraint_damper * jac @ self.velocity_vector
        )
        # logger.debug(f"[{time.perf_counter() - last_time}] constraint control")
        # last_time = time.perf_counter()

        lagrange_mult = front_inverse @ (
            velocity_term - force_term - constraint_control
        )
        reactions = jac.T @ lagrange_mult
        # logger.debug(f"[{time.perf_counter() - last_time}] reactions calculated")
        # last_time = time.perf_counter()

        accel = self.inv_mass_mat @ (
            (self.external_forces + active_edge_forces).flatten()
            + reactions
            - self.damping_mat @ self.velocity_vector
            - passive_edge_forces.flatten()
        )

        self.velocity_vector += accel * dt
        self.position_vector += self.velocity_vector * dt

        # logger.debug(f"[{time.perf_counter() - last_time}] integration calculated")
        # last_time = time.perf_counter()

        # logger.debug(f"[{time.perf_counter() - start_time}] total dynamics time")

        return self.position_vector, self.velocity_vector, accel
