from dataclasses import dataclass, field
import numpy as np

from data_logger import DataLogger


@dataclass
class HydrostatArm:
    """Simulate a hydrostat arm using constrained dynamics"""

    vertices: np.ndarray = field(
        default_factory=lambda: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    )
    cells: np.ndarray = field(default_factory=lambda: np.array([[0, 1, 2]], dtype=int))

    dof_dict: dict = field(default_factory=lambda: {})

    masses = None  # mass of vertices [kg]
    dampers = None  # damping rate of substrate [N-s/m]
    edge_damping_rate = 1  # damping rate of edges

    dim: int = 2

    odor_func: callable = None
    obstacles = []

    def __post_init__(self):
        # Convert Mass and Damping parameters into matrix forms
        if self.masses is None:
            self.masses = np.ones(len(self.vertices)) * 1
        if self.dampers is None:
            self.dampers = np.ones(len(self.vertices)) * 0.2
        if len(self.vertices) != len(self.masses) or len(self.vertices) != len(
            self.dampers
        ):
            raise ValueError("Vertices, masses, and dampers must have the same length")
        stateful_masses = np.column_stack((self.masses, self.masses)).ravel()
        self.inv_mass_mat = np.linalg.inv(np.diag(stateful_masses))
        stateful_dampers = np.column_stack((self.dampers, self.dampers)).ravel()
        self.damping_mat = np.diag(stateful_dampers)

        # Convert cell coordinates into one dimensional state vectors
        self.stateful_cells = np.empty(
            (len(self.cells), len(self.cells[0]) * self.dim), dtype=int
        )
        for idx in range(self.dim):
            self.stateful_cells[:, idx :: self.dim] = self.cells * self.dim + idx

        self.pos_init = np.ravel(self.vertices)
        self.vertices_init = self.vertices.copy()
        self.vel_init = np.zeros_like(self.pos_init)
        self.stateful_pos = self.pos_init.copy()  # stateful position
        self.stateful_vel = self.vel_init.copy()  # stateful velocity
        self.velocities = np.zeros_like(self.vertices)

        # list of tuples for the purposes of drawing the network. In order
        # for coloring to work, they must be sorted.
        # the cell_edge_map is needed because we need to know which muscles to
        # actuate in the closed loop control which is calculated per cell. That
        # way we don't need to search a bunch of times to find out which index
        # the edge is on.
        # Instead, we could use cell objects
        self.cell_edge_map = []
        self.edges = []
        for cell in self.cells:
            for i, _ in enumerate(cell):
                vertex_indices = (cell[i], cell[(i + 1) % len(cell)])
                edge = (min(vertex_indices), max(vertex_indices))
                if edge not in self.edges:
                    self.edges.append(edge)
        self.edges = sorted(self.edges)

        for cell in self.cells:
            cell_edges = []
            for i, _ in enumerate(cell):
                vertex_indices = (cell[i], cell[(i + 1) % len(cell)])
                edge = (min(vertex_indices), max(vertex_indices))
                cell_edges.append(self.edges.index(edge))
            self.cell_edge_map.append(cell_edges)

        self.muscles = np.zeros(len(self.edges))

        self.external_forces = np.zeros_like(self.stateful_pos)
        self.internal_forces = np.zeros_like(self.stateful_pos)

        # fix first two vertices if no dof specified
        if not self.dof_dict:
            self.dof_dict[0] = [0, 0]
            self.dof_dict[1] = [0, 0]
        self.boundary_indices = []
        for vertex, dofs in self.dof_dict.items():
            for i, dof in enumerate(dofs):
                if dof == 0:
                    self.boundary_indices.append(2 * vertex + i)

        self.timestamp = 0.0

        self.logger = DataLogger(self.edges)
        self.logger.log(
            self.timestamp,
            self.stateful_pos,
            self.stateful_vel,
            np.zeros_like(self.stateful_pos),
            self.external_forces,
            self.internal_forces,
        )

        self.errors = [[0] * len(self.cells)]

    def cell_volume(self, vertices):
        """Calculate the signed volume of an nxd array"""
        rolled_vertices = np.roll(vertices, -1, axis=0)
        return 0.5 * np.sum(-np.diff(vertices * rolled_vertices[:, ::-1], axis=1))

    def constraints(self):
        constraints_array = []
        for idx in self.boundary_indices:
            constraints_array.append(self.stateful_pos[idx] - self.pos_init[idx])

        for cell in self.cells:
            constraints_array.append(
                self.cell_volume(self.vertices[cell])
                - self.cell_volume(self.vertices_init[cell])
            )

        for obstacle in self.obstacles:
            for vertex in self.vertices:
                if obstacle.check_intersection(vertex):
                    nearest_point = obstacle.nearest_point(vertex)
                    constraints_array.append(vertex[0] - nearest_point[0])
                    constraints_array.append(vertex[1] - nearest_point[1])
        return np.array(constraints_array)

    def jacobian(self):
        jacobian_mat = np.zeros((len(self.constraints()), len(self.stateful_pos)))
        current_constraint = 0
        for idx in self.boundary_indices:
            jacobian_mat[current_constraint, idx] = 1
            current_constraint += 1

        for cell, stateful_cell in zip(self.cells, self.stateful_cells):
            diff_array = 0.5 * (
                np.roll(self.vertices[cell], 1, axis=0)
                - np.roll(self.vertices[cell], -1, axis=0)
            )
            jacobian_entry = np.empty((diff_array.size,))
            jacobian_entry[0::2] = -diff_array[:, 1]
            jacobian_entry[1::2] = diff_array[:, 0]
            jacobian_mat[current_constraint, stateful_cell] = jacobian_entry
            current_constraint += 1

        for obstacle in self.obstacles:
            for v_idx, vertex in enumerate(self.vertices):
                if obstacle.check_intersection(vertex):
                    jacobian_mat[current_constraint, 2 * v_idx] = 1
                    jacobian_mat[current_constraint + 1, 2 * v_idx + 1] = 1
                    current_constraint += 2

        return jacobian_mat

    def jacobian_derivative(self):
        jacobian_derivative_mat = np.zeros(
            (len(self.constraints()), len(self.stateful_pos))
        )
        current_constraint = len(self.boundary_indices)

        for cell, stateful_cell in zip(self.cells, self.stateful_cells):
            diff_array = 0.5 * (
                np.roll(self.velocities[cell], 1, axis=0)
                - np.roll(self.velocities[cell], -1, axis=0)
            )
            jacobian_entry = np.empty((diff_array.size,))
            jacobian_entry[0::2] = -diff_array[:, 1]
            jacobian_entry[1::2] = diff_array[:, 0]
            jacobian_derivative_mat[current_constraint, stateful_cell] = jacobian_entry
            current_constraint += 1

        return jacobian_derivative_mat

    def add_obstacle(self, obstacle):
        """Add an obstacle that the arm may interact with"""
        self.obstacles.append(obstacle)

    def apply_external_force(self, vertex_idx, force=(0, 0)):
        """Set the external force for a particular vertex"""
        self.external_forces = np.zeros_like(self.stateful_pos)
        self.external_forces[vertex_idx * 2] = force[0]
        self.external_forces[vertex_idx * 2 + 1] = force[1]

    def actuate_muscle(self, edge_idx, force: float = 0):
        """Set the internal force for a particular edge"""
        self.muscles[edge_idx] = force

    def calc_internal_forces(self):
        """Calculate the internal forces based on muscle activations"""
        self.internal_forces = np.zeros_like(self.stateful_pos)
        for idx, edge in enumerate(self.edges):
            # each edge has the index of the vertex affected
            # the x component is edge[0]*2, y component is edge[0]*2+1
            # muscles stores the magnitude of force, the direction is along the edge

            self.internal_forces[edge[0] * 2] += self.muscles[idx] * (
                self.stateful_pos[edge[1] * 2] - self.stateful_pos[edge[0] * 2]
            )
            self.internal_forces[edge[0] * 2 + 1] += self.muscles[idx] * (
                self.stateful_pos[edge[1] * 2 + 1] - self.stateful_pos[edge[0] * 2 + 1]
            )

            self.internal_forces[edge[1] * 2] += self.muscles[idx] * (
                self.stateful_pos[edge[0] * 2] - self.stateful_pos[edge[1] * 2]
            )
            self.internal_forces[edge[1] * 2 + 1] += self.muscles[idx] * (
                self.stateful_pos[edge[0] * 2 + 1] - self.stateful_pos[edge[1] * 2 + 1]
            )

    def control_muscles(self):
        """By probing the odor, determine how to actuate muscles"""
        if self.odor_func is None:
            return
        self.muscles = self.muscles * 0
        # for cell_idx, cell in enumerate(self.cells):
        #     cell_vertices = self.vertices[cell]
        #     scents = self.odor_func(*cell_vertices.T)
        # gradient = (
        #     np.linalg.inv(np.column_stack((cell_vertices, np.ones(3)))) @ scents
        # )[:-1]
        # gradient = gradient / np.linalg.norm(gradient)
        # print(gradient)

        # # wedge product of unit edge and unit grad is activation
        # for edge_idx, (v1, v2) in enumerate(
        #     zip(cell_vertices, np.roll(cell_vertices, -1, axis=0))
        # ):
        #     vector = v2 - v1
        #     vector = vector / np.linalg.norm(vector)

        #     ortho = np.abs(vector[0] * gradient[1] - vector[1] * gradient[0]) * 10
        #     print(v1, v2, vector, ortho)
        #     if self.muscles[self.cell_edge_map[cell_idx][edge_idx]] == 0:
        #         self.muscles[self.cell_edge_map[cell_idx][edge_idx]] = ortho
        #     else:
        #         self.muscles[self.cell_edge_map[cell_idx][edge_idx]] = (
        #             self.muscles[self.cell_edge_map[cell_idx][edge_idx]] + ortho
        #         ) / 2
        tip_left = self.cells[-2]
        tip_right = self.cells[-1]
        vertices_left = self.vertices[tip_left]
        vertices_right = self.vertices[tip_right]
        scents_left = self.odor_func(*vertices_left.T)
        scents_right = self.odor_func(*vertices_right.T)

        forward_backward_gradient = (
            (-(scents_left[0] + scents_right[0]) + (scents_left[-1] + scents_right[-1]))
            / (scents_left[0] + scents_right[0] + scents_left[-1] + scents_right[-1])
            * 1
        )

        for cell_idx, (cell_left, cell_right) in enumerate(
            zip(self.cells[::2], self.cells[1::2])
        ):
            vertices_left = self.vertices[cell_left]
            vertices_right = self.vertices[cell_right]

            scents_left = self.odor_func(*vertices_left.T)
            scents_right = self.odor_func(*vertices_right.T)

            left_right_gradient = (
                (scents_left[2] - scents_right[2])
                / (scents_left[2] + scents_right[2])
                * 1
            )
            self.muscles[self.cell_edge_map[2 * cell_idx][2]] = left_right_gradient
            self.muscles[self.cell_edge_map[2 * cell_idx + 1][2]] = -left_right_gradient
            self.muscles[self.cell_edge_map[2 * cell_idx + 1][0]] = (
                -left_right_gradient * 2
            )

            self.muscles[self.cell_edge_map[2 * cell_idx][0]] = (
                forward_backward_gradient
            )
            self.muscles[self.cell_edge_map[2 * cell_idx + 1][1]] = (
                forward_backward_gradient
            )
        self.muscles *= 20
        self.muscles = np.clip(self.muscles, 0, 50)

    def calc_edge_damping(self):
        """Return a length n array of forces caused by linear damping in the
        edges."""
        edge_forces = np.zeros_like(self.vertices, dtype=float)
        vertex_velocities = self.stateful_vel.reshape(-1, 2)
        for edge in self.edges:
            edge_vector = self.vertices[edge[0]] - self.vertices[edge[1]]
            edge_unit_vector = edge_vector / np.linalg.norm(edge_vector)
            relative_velocity = vertex_velocities[edge[0]] - vertex_velocities[edge[1]]
            edge_velocity = (
                np.dot(edge_unit_vector, relative_velocity) * edge_unit_vector
            )
            edge_damp_force = self.edge_damping_rate * edge_velocity

            edge_forces[edge[0]] += edge_damp_force
            edge_forces[edge[1]] += -edge_damp_force

        return edge_forces.reshape(-1)

    def calc_next_states(self, dt):
        """Calculate the next state using the particular system parameters"""
        self.timestamp += dt
        self.control_muscles()
        self.calc_internal_forces()
        edge_forces = self.calc_edge_damping()

        # jac = self.jacobian(self.stateful_pos)
        # djac = self.jacobian_derivative(self.stateful_pos, self.stateful_vel)
        jac = self.jacobian()
        djac = self.jacobian_derivative()
        ks = 500
        kd = 10

        lagrange_mult = np.linalg.inv(jac @ self.inv_mass_mat @ jac.T) @ (
            (jac @ self.inv_mass_mat @ self.damping_mat - djac) @ self.stateful_vel
            - jac
            @ self.inv_mass_mat
            @ (self.external_forces + self.internal_forces - edge_forces)
            - ks * self.constraints()
            - kd * jac @ self.stateful_vel
        )

        reactions = jac.T @ lagrange_mult

        accel = self.inv_mass_mat @ (
            self.external_forces
            + self.internal_forces
            + reactions
            - self.damping_mat @ self.stateful_vel
            - edge_forces
        )

        self.stateful_vel = self.stateful_vel + accel * dt
        self.stateful_pos = self.stateful_pos + self.stateful_vel * dt

        self.logger.log(
            self.timestamp,
            self.stateful_pos,
            self.stateful_vel,
            accel,
            self.external_forces,
            self.internal_forces,
        )

        self.errors.append(
            [
                self.cell_volume(self.vertices[cell])
                - self.cell_volume(self.vertices_init[cell])
                for cell in self.cells
            ]
        )

        return self.stateful_pos, self.stateful_vel, accel

    def save(self, filename: str = None):
        """Save positions, velocities, accelerations, external forces, and internal forces."""
        self.logger.save(filename)

    @property
    def stateful_pos(self):
        return self._stateful_pos

    @stateful_pos.setter
    def stateful_pos(self, value):
        self._stateful_pos = value
        self.vertices = self._stateful_pos.reshape(-1, 2)

    @property
    def stateful_vel(self):
        return self._stateful_vel

    @stateful_vel.setter
    def stateful_vel(self, value):
        self._stateful_vel = value
        self.velocities = self._stateful_vel.reshape(-1, 2)
