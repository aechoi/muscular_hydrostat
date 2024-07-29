from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class HydrostatArm:
    """Simulate a hydrostat arm using constrained dynamics"""

    vertices: np.ndarray = field(
        default_factory=lambda: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    )
    cells: np.ndarray = field(default_factory=lambda: np.array([[0, 1, 2]], dtype=int))

    masses = None  # mass of vertices [kg]
    dampers = None  # damping rate of edges [N-s/m]

    dof_dict = {}

    dim: int = 2

    def __post_init__(self):
        if self.masses is None:
            self.masses = np.ones(len(self.vertices))
        if self.dampers is None:
            self.dampers = np.ones(len(self.vertices))
        if len(self.vertices) != len(self.masses) or len(self.vertices) != len(
            self.dampers
        ):
            raise ValueError("Vertices, masses, and dampers must have the same length")

        self.stateful_cells = np.empty(
            (len(self.cells), len(self.cells[0]) * self.dim), dtype=int
        )
        for idx in range(self.dim):
            self.stateful_cells[:, idx :: self.dim] = self.cells * self.dim + idx

        if not self.dof_dict:
            # Add in simply supported if no dof specified
            self.dof_dict[0] = [0, 0]
            self.dof_dict[1] = [1, 0]

        self.pos_init = np.ravel(self.vertices)
        self.vel_init = np.zeros_like(self.pos_init)
        self.pos = self.pos_init.copy()
        self.vel = self.vel_init.copy()

        stateful_masses = np.column_stack((self.masses, self.masses)).ravel()
        self.inv_mass_mat = np.linalg.inv(np.diag(stateful_masses))
        stateful_dampers = np.column_stack((self.dampers, self.dampers)).ravel()
        self.damping_mat = np.diag(stateful_dampers)

        # list of tuples for the purposes of drawing the network
        self.edges = []
        for cell in self.cells:
            for i, _ in enumerate(cell):
                edge = (cell[i], cell[(i + 1) % 3])
                if edge not in self.edges:
                    self.edges.append(edge)
        self.muscles = np.zeros(len(self.edges))

        self.external_forces = np.zeros_like(self.pos)
        self.internal_forces = np.zeros_like(self.pos)

        self.constraints, self.jacobian, self.jacobian_derivative = (
            self.construct_matrices()
        )

    def cell_volume(self, q):
        """Calculate the volume of a list of vectors"""
        return 0.5 * np.abs(
            (q[2] - q[0]) * (q[5] - q[1]) - (q[3] - q[1]) * (q[4] - q[0])
        )

    def construct_matrices(self):
        """Construct the constraint array.

        The elements of the constraint array change depending on boundary
        conditions and the number of cells. Instead of recalculating which
        need to be included every time, this method precalculates it once and
        then returns a constraint_array function which takes the position
        states as input."""

        boundary_constraints = []
        for vertex, dofs in self.dof_dict.items():
            for i, dof in enumerate(dofs):
                if dof == 0:
                    boundary_constraints.append(2 * vertex + i)

        def constraint_array(pos_states):
            constraint_array = []
            for idx in boundary_constraints:
                constraint_array.append(pos_states[idx])
            for cell in self.stateful_cells:
                constraint_array.append(
                    self.cell_volume(pos_states[cell])
                    - self.cell_volume(self.pos_init[cell])
                )
            return np.array(constraint_array)

        def jacobian(pos_states):
            jacobian_mat = np.zeros(
                (len(constraint_array(pos_states)), len(pos_states))
            )
            current_constraint = 0
            for idx in boundary_constraints:
                jacobian_mat[current_constraint, idx] = 1
                current_constraint += 1

            for cell in self.stateful_cells:
                jacobian_mat[current_constraint, cell[0]] = 0.5 * (
                    pos_states[cell[3]] - pos_states[cell[5]]
                )
                jacobian_mat[current_constraint, cell[1]] = 0.5 * (
                    pos_states[cell[4]] - pos_states[cell[2]]
                )
                jacobian_mat[current_constraint, cell[2]] = 0.5 * (
                    pos_states[cell[5]] - pos_states[cell[1]]
                )
                jacobian_mat[current_constraint, cell[3]] = 0.5 * (
                    pos_states[cell[0]] - pos_states[cell[4]]
                )
                jacobian_mat[current_constraint, cell[4]] = 0.5 * (
                    pos_states[cell[1]] - pos_states[cell[3]]
                )
                jacobian_mat[current_constraint, cell[5]] = 0.5 * (
                    pos_states[cell[2]] - pos_states[cell[0]]
                )
                current_constraint += 1
            return jacobian_mat

        def jacobian_derivative(pos_states, vel_states):
            jacobian_derivative_mat = np.zeros(
                (len(constraint_array(pos_states)), len(pos_states))
            )
            current_constraint = len(boundary_constraints)

            for cell in self.stateful_cells:
                jacobian_derivative_mat[current_constraint, cell[0]] = 0.5 * (
                    vel_states[cell[3]] - vel_states[cell[5]]
                )
                jacobian_derivative_mat[current_constraint, cell[1]] = 0.5 * (
                    vel_states[cell[4]] - vel_states[cell[2]]
                )
                jacobian_derivative_mat[current_constraint, cell[2]] = 0.5 * (
                    vel_states[cell[5]] - vel_states[cell[1]]
                )
                jacobian_derivative_mat[current_constraint, cell[3]] = 0.5 * (
                    vel_states[cell[0]] - vel_states[cell[4]]
                )
                jacobian_derivative_mat[current_constraint, cell[4]] = 0.5 * (
                    vel_states[cell[1]] - vel_states[cell[3]]
                )
                jacobian_derivative_mat[current_constraint, cell[5]] = 0.5 * (
                    vel_states[cell[2]] - vel_states[cell[0]]
                )
                current_constraint += 1
            return jacobian_derivative_mat

        return constraint_array, jacobian, jacobian_derivative

    def apply_external_force(self, vertex_idx, force=(0, 0)):
        """Set the external force for a particular vertex"""
        self.external_forces = np.zeros_like(self.pos)
        self.external_forces[vertex_idx * 2] = force[0]
        self.external_forces[vertex_idx * 2 + 1] = force[1]

    def actuate_muscle(self, edge_idx, force: float = 0):
        """Set the internal force for a particular edge"""
        self.muscles[edge_idx] = force

    def calc_internal_forces(self):
        """Calculate the internal forces based on muscle activations"""
        self.internal_forces = np.zeros_like(self.pos)
        for idx, edge in enumerate(self.edges):
            # each edge has the index of the vertex affected
            # the x component is edge[0]*2, y component is edge[0]*2+1
            # muscles stores the magnitude of force, the direction is along the edge

            self.internal_forces[edge[0] * 2] += self.muscles[idx] * (
                self.pos[edge[1] * 2] - self.pos[edge[0] * 2]
            )
            self.internal_forces[edge[0] * 2 + 1] += self.muscles[idx] * (
                self.pos[edge[1] * 2 + 1] - self.pos[edge[0] * 2 + 1]
            )

            self.internal_forces[edge[1] * 2] += self.muscles[idx] * (
                self.pos[edge[0] * 2] - self.pos[edge[1] * 2]
            )
            self.internal_forces[edge[1] * 2 + 1] += self.muscles[idx] * (
                self.pos[edge[0] * 2 + 1] - self.pos[edge[1] * 2 + 1]
            )

    def calc_next_states(self, dt):
        """Calculate the next state using the particular system parameters"""
        self.calc_internal_forces()

        jac = self.jacobian(self.pos)
        djac = self.jacobian_derivative(self.pos, self.vel)

        lagrange_mult = np.linalg.inv(jac @ self.inv_mass_mat @ jac.T) @ (
            (jac @ self.inv_mass_mat @ self.damping_mat - djac) @ self.vel
            - jac @ self.inv_mass_mat @ (self.external_forces + self.internal_forces)
        )  # - ks * C(q) - kd * J @ dq)

        reactions = jac.T @ lagrange_mult

        accel = self.inv_mass_mat @ (
            self.external_forces
            + self.internal_forces
            + reactions
            - self.damping_mat @ self.vel
        )

        self.pos = self.pos + self.vel * dt
        self.vertices = self.pos.reshape(-1, 2)
        self.vel = self.vel + accel * dt

        return self.pos, self.vel, accel
