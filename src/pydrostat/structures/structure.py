"""An interface for simulated structures.

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

import numpy as np
from .cell import Cell3D

if TYPE_CHECKING:
    from ..control.controller_interface import IController
    from ..constraint.constraint_interface import IConstraint
    from ..sensing.sensor_interface import ISensor
    from ...pydrostat.environment.environment import Environment


class AStructure(ABC):
    """An abstract class for a concrete structure objects.

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
        cells: Cell3D,
        controller: IController = None,
        environment: Environment = None,
        constraints: list[IConstraint] = [],
        sensors: list[ISensor] = [],
        constraint_damping_rate=1,
        constraint_spring_rate=1,
    ):
        self.positions = initial_positions
        self.velocities = initial_velocities
        self.cells = cells
        self.controller = controller
        self.environment = environment
        self.constraints = constraints
        self.sensors = sensors
        self.constraint_damping_rate = constraint_damping_rate
        self.constraint_spring_rate = constraint_spring_rate

        self.masses = np.zeros(len(initial_positions))
        self.damping_rates = np.zeros(len(initial_positions))
        self.inv_masses = 1 / self.masses

        for obstacle in self.environment.obstacles:
            self.constraints.append(obstacle)

        self.external_forces = np.zeros_like(self.positions)

        for constraint in self.constraints:
            constraint.initialize_constraint(self)


    def _calculate_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        constraints = []
        jacobians = []
        djacobian_dts = []
        for constraint in self.constraints:
            constraint, jacobian, djacobian_dt = constraint.calculate_constraints(self)
            if len(constraint) > 0:
                constraints.extend(constraint)
                jacobians.append(jacobian)
                djacobian_dts.append(djacobian_dt)
        constraints = np.array(constraints)
        jacobians = np.vstack(jacobians)
        djacobian_dts = np.vstack(djacobian_dts)

        return constraints, jacobians, djacobian_dts

    def _calc_reaction_forces(self, explicit_forces):
        if not self.constraints:
            return np.zeros_like(explicit_forces)

        constraints, jacobians, djacobian_dts = self._calculate_constraints()

        front_matrix = (
            np.tensordot(
                jacobians,
                np.moveaxis(self.inv_masses[None, :, None] * jacobians, 0, -1),
                2,
            )
            + np.eye(len(constraints)) * 1e-6
        )
        dependent_array = -(
            np.tensordot(djacobian_dts, self.velocities, 2)
            + np.tensordot(jacobians, self.inv_masses[:, None] * explicit_forces, 2)
            + self.constraint_damping_rate * np.tensordot(jacobians, self.velocities, 2)
            + self.constraint_spring_rate * constraints
        )
        lagrange_multipliers = np.linalg.solve(front_matrix, dependent_array)
        reaction_forces = np.tensordot(lagrange_multipliers, jacobians, 1)
        # print("Reaction Forces \n", reaction_forces)

        return reaction_forces

    def _sense(self) -> dict[str : np.ndarray]:
        """Take sensor measurements for all sensors and return a dictionary of data.

        Returns:
            A dictionary of sensor data where each key is the sensor type"""
        sensor_data = {}
        for sensor in self.sensors:
            sensor_data[sensor.sensor_type] = sensor.sense(self, self.environment)
        return sensor_data

    @abstractmethod
    def _actuate(self, control_input):
        """Given a control input, actuate the muscles accordingly.

        Args:
            control_input: an np.ndarray of the same shape as the actuators

        Returns:
            An nxd np.ndarray of forces acting on each vertex

        Raises:
            NotImplemnetedError: if function is not implemented in subclass
        """
        raise NotImplementedError

    def apply_external_forces(self, vertices: np.ndarray, forces: np.ndarray):
        """Set the force acting on a particular vertex

        Args:
            vertices: a length l array of vertex indices to apply forces to
            forces: an lxd array of forces"""
        self.external_forces[vertices] = forces

    @abstractmethod
    def _calc_explicit_forces(self, actuation_forces) -> np.ndarray:
        """Calculate and sum all forces that are not calculated via constrained
        dynamics.

        Args:
            actuation_forces: an nxd np.ndarray where n is the number of vertices and
                d is the dimension of the space

        Returns:
            An nxd array of total explicit forces on the vertices.

        Raises:
            NotImplementedError: if function is not implemented in subclass
        """
        raise NotImplementedError

    def _calculate_acceleration(self) -> np.ndarray:
        """Calculate the acceleration of every vertex by calculating constraint forces."""
        # TODO how does controller know what is actuateable? Should this be explicitly edges?
        self.control_inputs = np.zeros(len(self.edges))
        if self.controller is not None:
            self.control_inputs = self.controller.calc_inputs(self, self._sense())

        # TODO may be worth formatting as some sort of a(x) + b(u) instead.
        actuation_forces = self._actuate(self.control_inputs)
        explicit_forces = self._calc_explicit_forces(actuation_forces)
        reaction_forces = self._calc_reaction_forces(explicit_forces)

        accelerations = self.inv_masses[:, None] * (reaction_forces + explicit_forces)
        return accelerations

    def iterate(self, dt: float) -> None:
        """Update the position and velocity of the structure vertices.

        TODO: Currently this uses euler integration. We may get more stable results by
        using a more sophisticated integration scheme (RK2, RK4, etc). Seems like a
        good use of a strategy design pattern.
        """
        acceleration = self._calculate_acceleration()
        self.positions = self.positions + self.velocities * dt
        self.velocities = self.velocities + acceleration * dt
