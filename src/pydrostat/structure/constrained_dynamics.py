"""An abstract class for simulated structures.

Typical usage example:
    class SomeStructure(IStructure):
        ...

TODO:
- How to generalize sense data?
- How do controllers know what to actuate?
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp


if TYPE_CHECKING:
    from ..control.controller_interface import IController
    from ..constraint.constraint_interface import IConstraint
    from ..sensing.sensor_interface import ISensor
    from ..environment.environment import Environment


class ConstrainedDynamics(ABC):
    """An abstract class for constrained particle dynamics.

    A structure is some combination of 0-d vertices, 1-d edges, 2-d faces, and 3-d
    cells. These structures are 2d, where some set of edges are actuators. The
    dynamics of a structure are defined using the framework of constrained dynamics.

    Attributes:
        positions: an nxd jnp.ndarray where n is the number of vertices. Each row
            describes the position of each vertex.
        velocities: an nxd jnp.ndarray where n is the number of vertices. Each row
            describes the velocity of each vertex.
        controller: a IController object
        constraints: a list of IConstraint objects
        sensor_readings: TODO

        edges: an ex2 jnp.ndarray of ints where e is the number of edges. Each element of
            the array is an index of a particular vertex. The two elements describe the
            end points of the edge and are sorted by index.
        faces: a, potentially ragged, length f sequence of jnp.ndarrays. Each element of
            the ith array is an index of a vertex that is a member of the ith face. The
            vertices in each array must be ordered counter-clockwise from the outside.
            If all faces have the same number of vertices, this may be a fxa jnp.ndarray
            where a is the number of vertices on each face.
        cells: a, potentially ragged, length c sequence of jnp.ndarrays. Each element of
            the ith array is an index of a vertex that is a member of the ith cell. The
            vertices are sorted by index. If all cells have the same number of vertices,
            this may be a cxb jnp.ndarray where b is the number of vertices on each cell.

    Have one class be the calculation class, the other be the object class.
    The calculation class operates directly on states and controls. The structure object
    should just hold states and controls, and pass to calculation class, b/c need to operate
        on general states and controls for some of the controllers.

    The integrator requires a state state dynamics function and a timestep.
    Some integrators need a dynamics function that can return dynamics for any state/control

    The controller takes the u-grad of the stage cost and dynamics, the x-grad of stage cost and dynamics
    For finite horizon, the controller needs to be able to calculate all states from integration. So makes
    a copy of the structure, integrates forward, saving the states at every point. At every point, also
    calc the u-grad and x-grad. This allows calculation of controller lagrange multipliers.

    In the TO controller:
        - need structure, initial conditions, dynamics function for grad
        - generate copy of structure
        - integrate forward in time while calculating gradients and states
        - calculate lagrangians backwards in time
        - repeat multiple times until convergence
        - this would typically be done offline and only calculates policy for
            single initial state.

    In online CL controller:
        - take state and sensor data
        - create current control based on state and sensor data
        - return control

    For TO MPC:
        - Run TO on finite horizon, return first control

    For random MPC:
        - Simulate many random controls
        - return the best

    Assume zero order hold control
    """

    def __init__(
        self,
        initial_positions: jnp.ndarray,
        initial_velocities: jnp.ndarray,
        masses: jnp.ndarray,
        constraints: list[IConstraint] = None,
        constraint_damping_rate=1,
        constraint_spring_rate=1,
    ):
        self.positions = initial_positions
        self.velocities = initial_velocities
        self.states = jnp.stack(
            (jnp.ravel(initial_positions), jnp.ravel(initial_velocities))
        )
        self.n = len(self.states)

        self.inv_masses = 1 / masses

        self.constraints = constraints
        if self.constraints is None:
            self.constraints = []

        self.external_forces = jnp.zeros_like(self.positions)

        self.constraint_damping_rate = constraint_damping_rate
        self.constraint_spring_rate = constraint_spring_rate

        for constraint in self.constraints:
            constraint.initialize_constraint(self)

    def add_constraint(self, constraint):
        """Add a constraint"""
        self.constraints.append(constraint)

    def apply_external_forces(self, vertices: jnp.ndarray, forces: jnp.ndarray):
        """Set the force acting on a particular vertex

        Args:
            vertices: a length l array of vertex indices to apply forces to
            forces: an lxd array of forces"""
        self.external_forces[vertices] = forces

    def integrate(self, state, control, dt: float) -> None:
        """Calculate the next state based on the current state and controls"""
        k1 = self._dynamics(state, control)
        k2 = self._dynamics(state + dt / 2 * k1, control)
        k3 = self._dynamics(state + dt / 2 * k2, control)
        k4 = self._dynamics(state + dt * k3, control)

        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _dynamics(self, state, control) -> jnp.ndarray:
        """Calculate the acceleration of every vertex by calculating constraint forces."""
        actuation_forces = self._calc_actuation_forces(state, control)
        explicit_forces = self._calc_explicit_forces(
            state, actuation_forces
        )  # Anything that's not a constraint force
        reaction_forces = self._calc_reaction_forces(state, explicit_forces)

        return jnp.vstack(
            (
                state[int(self.n / 2) :],
                self.inv_masses * (reaction_forces + explicit_forces),
            )
        )

    @abstractmethod
    def _calc_actuation_forces(self, state, control_input):
        """Given a control input, actuate the muscles accordingly.

        Args:
            control_input: an jnp.ndarray of the same shape as the actuators

        Returns:
            An nxd jnp.ndarray of forces acting on each vertex

        Raises:
            NotImplemnetedError: if function is not implemented in subclass
        """
        raise NotImplementedError

    @abstractmethod
    def _calc_explicit_forces(self, state, actuation_forces) -> jnp.ndarray:
        """Calculate and sum all forces that are not calculated via constrained
        dynamics.

        Args:
            actuation_forces: an nxd jnp.ndarray where n is the number of vertices and
                d is the dimension of the space

        Returns:
            An nxd array of total explicit forces on the vertices.

        Raises:
            NotImplementedError: if function is not implemented in subclass
        """
        raise NotImplementedError

    def _calc_reaction_forces(self, state, explicit_forces):
        if not self.constraints:
            return jnp.zeros_like(explicit_forces)

        constraints, jacobian, djacobian_dt = self._calculate_constraints(state)

        front_matrix = jacobian @ (self.inv_masses[:, None] * jacobian)
        dependent_array = -(
            djacobian_dt @ state[int(self.n / 2) :]
            + jacobian @ (self.inv_masses * explicit_forces)
            + self.constraint_damping_rate * jacobian @ state[int(self.n / 2) :]
            + self.constraint_spring_rate * constraints
        )
        lagrange_multipliers = jnp.linalg.solve(front_matrix, dependent_array)
        reaction_forces = jacobian.T @ lagrange_multipliers

        return reaction_forces

    def _calculate_constraints(
        self, state
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        constraints = []
        jacobians = []
        djacobian_dts = []
        for constraint in self.constraints:
            constraint, jacobian, djacobian_dt = constraint.calculate_constraints(state)

            # sometimes a constraint doesn't apply and returns no constraints
            if len(constraint) > 0:
                constraints.extend(constraint)
                jacobians.append(jacobian)
                djacobian_dts.append(djacobian_dt)
        constraints = jnp.array(constraints)
        jacobians = jnp.vstack(jacobians)
        djacobian_dts = jnp.vstack(djacobian_dts)

        return constraints, jacobians, djacobian_dts
