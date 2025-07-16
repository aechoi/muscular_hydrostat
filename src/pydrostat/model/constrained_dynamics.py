"""An abstract class for constrained particle dynamics.

Note, control systems typically use a state vector while constrained particle dynamics
in multiple dimensions are more naturally expressed as position and velocity matrices.
This class interfaces between the two so that state vector arguments are converted to
position and velocity matrices internally.

This module only deals with 3D dynamics.

State is not tracked here, only the dynamics of the system are represented."""

from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from pydrostat.model.dynamics import DynamicModel

if TYPE_CHECKING:
    from ..constraint.constraint_interface import IConstraint


class ConstrainedDynamics(DynamicModel):
    """An abstract class for constrained particle dynamics.

    We let a constrained particle dynamics system be a collection of verticies with mass
    and constraints. Forces can act on these vertices and induce accelerations. These
    accelerations are integrated over time to get velocities and positions. The state
    of this system is a length 6n vector of positions and velocities where `n` is the
    number of vertices.

    Properties:
        inv_masses: a length n array of the reciprocals of vertex masses
        constraints: a list of constraints to apply to the system
        external_forces: a length nxd array of external forces acting on the vertices
        constraint_damping_rate: the rate of damping for the constraints
        constraint_spring_rate: the rate of spring force for the constraints
    """

    def __init__(
        self,
        num_controls: int,
        masses: jnp.ndarray,
        constraints: list[IConstraint] = None,
        constraint_damping_rate=50,
        constraint_spring_rate=50,
    ):
        """Generate the state vector and initialize the constraints

        Args:
            num_particles: the number of particles in the system
            masses: a length n array of the masses of each vertex
            constraints: a list of intrinsic constraints to apply to the system (ie
                typically will not include environmental constraints like obstacles)
            constraint_damping_rate: the rate of damping for the constraints
            constraint_spring_rate: the rate of spring force for the constraints"""
        num_particles = len(masses)
        num_states = num_particles * 6
        super().__init__(num_states, num_controls)

        self.inv_masses = jnp.repeat(1 / masses, 3)

        self.constraints = constraints
        if self.constraints is None:
            self.constraints = []

        self.external_forces = jnp.zeros((num_particles, 3))

        self.constraint_damping_rate = constraint_damping_rate
        self.constraint_spring_rate = constraint_spring_rate

    def set_environment(self, environment, state, current_obstacles):
        """TODO: The problem is that now the constraints may not be initialized, but
        maybe that's okay since the initial condition is not stored here."""
        for obstacle in current_obstacles:
            self.remove_constraint(obstacle)

        current_obstacles = environment.obstacles
        for obstacle in current_obstacles:
            self.add_constraint(obstacle, state)
        self.initialize_constraints(state)
        return current_obstacles

    def initialize_constraints(self, initial_state: jnp.ndarray):
        for constraint in self.constraints:
            constraint.initialize_constraint(self, initial_state)

    def continuous_dynamics(self, state, control, t):
        """Returns the current state derivative"""
        actuation_forces = self._calc_actuation_forces(state, control)
        explicit_forces = self._calc_explicit_forces(
            state, actuation_forces
        )  # Anything that's not a constraint force, ie spring rates, viscous damping
        reaction_forces = self._calc_reaction_forces(state, explicit_forces)
        pos, vel = self.state2posvel(state)
        dstate = jnp.hstack(
            (
                vel.ravel(),
                self.inv_masses * (reaction_forces.ravel() + explicit_forces.ravel()),
            )
        )
        return dstate

    def add_constraint(self, constraint: IConstraint, state):
        """Add a constraint"""
        constraint.initialize_constraint(self, state)
        self.constraints.append(constraint)

    def remove_constraint(self, constraint: IConstraint):
        """Remove a constraint"""
        if constraint in self.constraints:
            self.constraints.remove(constraint)

    def apply_external_forces(self, vertices: jnp.ndarray, forces: jnp.ndarray):
        """Set the force acting on a particular vertex

        Args:
            vertices: a length l array of vertex indices to apply forces to
            forces: an lxd array of forces"""
        self.external_forces = self.external_forces.at[vertices].set(forces)

    @abstractmethod
    def _calc_actuation_forces(self, state, control) -> jnp.ndarray:
        """Given a control input, determine the actuation force vectors.

        Args:
            state: the current state of the system
            control: a vector of control inputs

        Returns:
            An nxd jnp.ndarray of forces acting on each vertex

        Raises:
            NotImplemnetedError: if function is not implemented in subclass
        """
        raise NotImplementedError

    def _calc_explicit_forces(self, state, actuation_forces) -> jnp.ndarray:
        """Calculate and sum all forces that are not calculated via constrained
        dynamics.

        Typically, forces are notated as PassiveForces=ActiveForces like in the case
        of m*ddx + b*dx + k*x = F. In this case, our external forces are any actuations
        or external forces not caused by constrainst. This is why there is a negative
        sign.

        Args:
            state: the current state of the system
            actuation_forces: an nxd jnp.ndarray where n is the number of vertices and
                d is the dimension of the space

        Returns:
            An nxd array of total explicit force vectors on the vertices.

        Raises:
            NotImplementedError: if function is not implemented in subclass
        """
        passive_forces = self._calc_passive_forces(state)

        explicit_forces = self.external_forces + actuation_forces - sum(passive_forces)
        return explicit_forces

    @abstractmethod
    def _calc_passive_forces(self, state):
        """Calculate forces that are not caused by constraints or actuation."""
        raise NotImplementedError

    # def _calc_reaction_forces(self, state, explicit_forces):
    #     """Calculate the reaction forces from the constraints.

    #     Args:
    #         state: the current state of the system
    #         explicit_forces: an nxd jnp.ndarray of vertex forces not caused by constraints

    #     Returns:
    #         An nxd array of reaction forces on the vertices."""
    #     _, vel = self.state2posvel(state)

    #     def no_constraints(_):
    #         return jnp.zeros_like(explicit_forces)

    #     def with_constraints(_):
    #         constraints, jacobian, djacobian_dt = self._calculate_constraints(state)

    #         front_matrix = jacobian @ (self.inv_masses[:, None] * jacobian)
    #         dependent_array = -(
    #             djacobian_dt @ vel
    #             + jacobian @ (self.inv_masses * explicit_forces)
    #             + self.constraint_damping_rate * jacobian @ vel
    #             + self.constraint_spring_rate * constraints
    #         )
    #         lagrange_multipliers = jnp.linalg.solve(front_matrix, dependent_array)
    #         return jacobian.T @ lagrange_multipliers

    #     return jax.lax.cond(
    #         len(self.constraints) == 0,
    #         no_constraints,
    #         with_constraints,
    #         operand=None,
    #     )

    def _calc_reaction_forces(self, state, explicit_forces):
        if not self.constraints:
            return jnp.zeros_like(explicit_forces)

        pos, vel = self.state2posvel(state)
        constraints, jacobian, djacobian_dt = self._calculate_constraints(state)
        front_matrix = jacobian @ (self.inv_masses[:, None] * jacobian.T)
        front_matrix = (
            front_matrix + jnp.eye(front_matrix.shape[0]) * 1e-6
        )  # Regularization
        dependent_array = -(
            djacobian_dt @ vel.ravel()
            + jacobian @ (self.inv_masses * explicit_forces.ravel())
            + self.constraint_damping_rate * jacobian @ vel.ravel()
            + self.constraint_spring_rate * constraints
        )
        lagrange_multipliers = jnp.linalg.solve(front_matrix, dependent_array)
        return jacobian.T @ lagrange_multipliers

    def _calculate_constraints(
        self, state
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """TODO replace with autograd calculation of constraints"""
        constraints = []
        jacobians = []
        djacobian_dts = []
        for constraint in self.constraints:
            constraint, jacobian, djacobian_dt = constraint.calculate_constraints(
                self, state
            )

            # sometimes a constraint doesn't apply and returns no constraints
            if len(constraint) == 0:
                continue

            jacobian = jnp.reshape(jacobian, (len(constraint), -1))
            djacobian_dt = jnp.reshape(djacobian_dt, (len(constraint), -1))

            constraints.extend(constraint)
            jacobians.append(jacobian)
            djacobian_dts.append(djacobian_dt)
        constraints = jnp.array(constraints)
        jacobians = jnp.vstack(jacobians)
        djacobian_dts = jnp.vstack(djacobian_dts)

        return constraints, jacobians, djacobian_dts

    def state2posvel(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Convert a state vector to position and velocity matrices.

        Args:
            state: a 6n length jnp.ndarray where n is the number of vertices
        Returns:
            A tuple of position and velocity matrices, each nx3."""
        pos, vel = state._split(2)
        return pos.reshape(-1, 3), vel.reshape(-1, 3)
