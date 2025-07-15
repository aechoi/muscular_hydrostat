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
        state: an 6n length array of vertex positions and velocities
        inv_masses: a length n array of the reciprocals of vertex masses
    """

    def __init__(
        self,
        num_particles: int,
        masses: jnp.ndarray,
        constraints: list[IConstraint] = None,
        constraint_damping_rate=50,
        constraint_spring_rate=50,
    ):
        """Generate the state vector and initialize the constraints"""
        super().__init__()

        self.inv_masses = 1 / masses

        self.constraints = constraints
        if self.constraints is None:
            self.constraints = []

        self.external_forces = jnp.zeros(num_particles, 3)

        self.constraint_damping_rate = constraint_damping_rate
        self.constraint_spring_rate = constraint_spring_rate

        for constraint in self.constraints:
            constraint.initialize_constraint(self)

    def continuous_dynamics(self, state, control, t):
        """Returns the current state derivative"""
        actuation_forces = self._calc_actuation_forces(state, control)
        explicit_forces = self._calc_explicit_forces(
            state, actuation_forces
        )  # Anything that's not a constraint force, ie spring rates, viscous damping
        reaction_forces = self._calc_reaction_forces(state, explicit_forces)
        pos, vel = self.state2posvel(state)

        return jnp.vstack(
            (
                vel,
                self.inv_masses * (reaction_forces + explicit_forces),
            )
        )

    def add_constraint(self, constraint: IConstraint):
        """Add a constraint"""
        constraint.initialize_constraint(self)
        self.constraints.append(constraint)

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

    def _calc_reaction_forces(self, state, explicit_forces):
        """Calculate the reaction forces from the constraints.

        Args:
            state: the current state of the system
            explicit_forces: an nxd jnp.ndarray of vertex forces not caused by constraints

        Returns:
            An nxd array of reaction forces on the vertices."""
        _, vel = self.state2posvel(state)

        def no_constraints(_):
            return jnp.zeros_like(explicit_forces)

        def with_constraints(_):
            constraints, jacobian, djacobian_dt = self._calculate_constraints(state)

            front_matrix = jacobian @ (self.inv_masses[:, None] * jacobian)
            dependent_array = -(
                djacobian_dt @ vel
                + jacobian @ (self.inv_masses * explicit_forces)
                + self.constraint_damping_rate * jacobian @ vel
                + self.constraint_spring_rate * constraints
            )
            lagrange_multipliers = jnp.linalg.solve(front_matrix, dependent_array)
            return jacobian.T @ lagrange_multipliers

        return jax.lax.cond(
            len(self.constraints) == 0,
            no_constraints,
            with_constraints,
            operand=None,
        )

    def _calculate_constraints(
        self, state
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """TODO replace with autograd calculation of constraints"""
        constraints = []
        jacobians = []
        djacobian_dts = []
        for constraint in self.constraints:
            constraint, jacobian, djacobian_dt = constraint.calculate_constraints(state)

            # sometimes a constraint doesn't apply and returns no constraints
            if len(constraint) == 0:
                continue

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
