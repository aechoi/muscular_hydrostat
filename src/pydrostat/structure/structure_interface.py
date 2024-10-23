"""An interface for simulated structures.

Typical usage example:
    class SomeStructure(IStructure):
        ...
"""

from abc import ABC, abstractmethod

import numpy as np

from constraint_interface import IConstraint
from .control.controller_interface import IController


class IStructure(ABC):
    """An abstract class used as an interface for concrete structure objects.

    A structure is some combination of 0-d vertices, 1-d edges, 2-d faces, and 3-d
    cells. A structure is not required to have all of those components (eg 2D structures
    will not have 3-d cells) except for vertices. The dynamics of a structure are
    defined using the framework of constrained dynamics.

    Attributes:
        positions: an nxd np.ndarray where n is the number of vertices. Each row
            describes the position of each vertex.
        velocities: an nxd np.ndarray where n is the number of vertices. Each row
            describes the velocity of each vertex.
        controller: a IController object
        constraints: a list of IConstraint objects
        sensor_readings: TODO

        # edges: an ex2 np.ndarray of ints where e is the number of edges. Each element of
        #     the array is an index of a particular vertex. The two elements describe the
        #     end points of the edge and are sorted by index.
        # faces: a, potentially ragged, length f sequence of np.ndarrays. Each element of
        #     the ith array is an index of a vertex that is a member of the ith face. The
        #     vertices in each array must be ordered counter-clockwise from the outside.
        #     If all faces have the same number of vertices, this may be a fxa np.ndarray
        #     where a is the number of vertices on each face.
        # cells: a, potentially ragged, length c sequence of np.ndarrays. Each element of
        #     the ith array is an index of a vertex that is a member of the ith cell. The
        #     vertices are sorted by index. If all cells have the same number of vertices,
        #     this may be a cxb np.ndarray where b is the number of vertices on each cell.
    """

    def __init__(
        self,
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
        masses: np.ndarray,
        damping_rates: np.ndarray,
        controller: IController,
        constraints: list[IConstraint] = [],
        constraint_damping_rate=10,
        constraint_spring_rate=500,
    ):
        self.positions = initial_positions
        self.velocities = initial_velocities
        self.inv_masses = 1 / masses
        self.damping_rates = damping_rates

        self.controller = controller
        self.constraints = constraints
        self.sensor_readings = None  # TODO: Is there a general way to do sensor readings? Maybe need interface for sense object

        self.external_forces = np.zeros_like(self.positions)

        self.constraint_damping_rate = constraint_damping_rate
        self.constraint_spring_rate = constraint_spring_rate

    def _calculate_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        constraints = []
        jacobians = []
        djacobian_dts = []
        for constraint in self.constraints:
            constraint, jacobian, djacobian_dt = constraint.calculate_constraints(self)
            constraints.append(constraint)
            jacobians.append(jacobian)
            djacobian_dts.append(djacobian_dt)
        constraints = np.vstack(constraints)
        jacobians = np.vstack(jacobians)
        djacobian_dts = np.vstack(djacobian_dts)

        return constraints, jacobians, djacobian_dts

    def _sense(self):
        """Take sensor measurements and store in self.sensor_readings"""
        pass

    def apply_external_forces(self, vertices: np.ndarray, forces: np.ndarray):
        """Set the force acting on a particular vertex

        Args:
            vertices: a length l array of vertex indices to apply forces to
            forces: an lxd array of forces"""
        self.external_forces[vertices] = forces

    def _calc_explicit_forces(self, control_inputs) -> np.ndarray:
        """Calculate and sum all forces that are not calculated via constrained dynamics"""
        # active_edge_forces = self._calc_active_edge_forces(control_inputs)
        # passive_edge_forces = self._calc_passive_edge_forces()

        # explicit_forces = (
        #     self.external_forces
        #     + active_edge_forces
        #     - passive_edge_forces
        #     - self.damping_rates[:, None] * self.velocities
        # )
        # return explicit_forces
        pass

    def _calculate_acceleration(self) -> np.ndarray:
        """Calculate the acceleration of every vertex by calculating constraint forces."""
        # TODO how does controller know what is actuateable? Should this be explicitly edges?
        control_inputs = self.controller.calc_inputs(self)

        explicit_forces = self._calc_explicit_forces(control_inputs)

        constraints, jacobians, djacobian_dts = self._calculate_constraints()
        front_inverse = np.linalg.inv(
            np.tensordot(jacobians, self.inv_masses[None, :, None] * jacobians, 2)
        )
        lagrange_multipliers = -front_inverse @ (
            np.tensordot(djacobian_dts, self.velocities, 2)
            + np.tensordot(jacobians, self.inv_masses[:, None] * explicit_forces, 2)
            + self.constraint_damping_rate * np.tensordot(jacobians, self.velocities, 2)
            + self.constraint_spring_rate * constraints
        )
        reaction_forces = np.tensordot(lagrange_multipliers, jacobians, 1)

        accelerations = self.inv_masses[:, None] * (reaction_forces + explicit_forces)
        return accelerations

    def iterate(self, dt: float) -> None:
        """Update the position and velocity of the structure vertices.

        TODO: Currently this uses euler integration. We may get more stable results by
        using a more sophisticated integration scheme (RK2, RK4, etc). Seems like a
        good use of a strategy design pattern.
        """
        acceleration = self._calculate_acceleration(dt)
        self.positions = self.positions + self.velocities * dt
        self.velocities = self.velocities + acceleration * dt
