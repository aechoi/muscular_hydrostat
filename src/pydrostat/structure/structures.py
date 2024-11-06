"""This module holds concrete implementations of the structure interface. These
include but are not limited to 
    - cuboid arms
    - single cells
    - iso-cylinders

Typical use case:
    # In a simulator
    structure = structures.arm_3d(...)
    while simulating:
        structure.iterate(dt)
        # code to display structure
"""

from dataclasses import dataclass

import numpy as np

from .structure_interface import IStructure


@dataclass
class Cell3D:
    """A dataclass which holds shape info for 3D arms"""

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
        """Decompose each face into triangles for the purposes of volume calculation.
        Each face is assumed to be arranged counter clockwise from the outside.

        Returns:
            a tx3 np.ndarray of vertex indices for all t triangles
        """
        triangles = []
        for face in self.faces:
            if self.vertices[0] in face:
                continue
            for v1, v2 in zip(face[1:-1], face[2:]):
                triangles.append([face[0], v1, v2])
        return np.array(triangles)


class Arm3D(IStructure):
    def __init__(
        self,
        initial_positions,
        initial_velocities,
        cells: Cell3D,
        controller=None,
        environment=None,
        constraints=[],
        sensors=[],
        constraint_damping_rate=10,
        constraint_spring_rate=500,
    ):
        # collect edges and faces from cells
        self.cells = cells
        self.edges = []
        self.faces = []
        self.edge_damping = []

        for cell in self.cells:
            for e, edge in enumerate(cell.edges):
                edge = sorted(edge)
                if edge not in self.edges:
                    self.edges.append(edge)
                    self.edge_damping.append(cell.edge_damping[e])

            for face in cell.faces:
                face = sorted(face)
                if face not in self.faces:
                    self.faces.append(face)
        self.edges = np.array(self.edges)

        # REMOVE AFTER COMPATIBLE
        self.muscles = np.zeros(len(self.edges))

        masses = np.zeros(len(initial_positions))
        damping_rates = np.zeros(len(initial_positions))
        for cell in self.cells:
            for v, vertex in enumerate(cell.vertices):
                masses[vertex] = cell.masses[v]
                damping_rates[vertex] = cell.vertex_damping[v]

        self.control_inputs = np.zeros(len(self.edges))

        super().__init__(
            initial_positions,
            initial_velocities,
            masses,
            damping_rates,
            controller,
            environment,
            constraints,
            sensors,
            constraint_damping_rate,
            constraint_spring_rate,
        )

    def _actuate(self, control_input):
        edge_forces = np.zeros_like(self.positions)
        for edge, muscle_force in zip(self.edges, control_input):
            edge_vector = self.positions[edge[1]] - self.positions[edge[0]]
            edge_vector = edge_vector / np.linalg.norm(edge_vector)
            edge_forces[edge[1]] -= edge_vector * muscle_force
            edge_forces[edge[0]] += edge_vector * muscle_force
        return edge_forces

    def _calc_explicit_forces(self, actuation_forces):
        passive_edge_forces = self._calc_passive_edge_forces()

        explicit_forces = (
            self.external_forces
            + actuation_forces
            - passive_edge_forces
            - self.damping_rates[:, None] * self.velocities
        )
        # print("Explicit forces: \n", explicit_forces)
        # print("External forces: \n", self.external_forces)
        # print("Actuation forces: \n", actuation_forces)
        # print("Passive Edge forces: \n", passive_edge_forces)
        # print(
        #     "Vertex Damping forces: \n", self.damping_rates[:, None] * self.velocities
        # )
        return explicit_forces

    def _calc_passive_edge_forces(self):
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


class CubicArmBuilder:
    """A builder for 3D cubic arms

    create cell structure
    choose controller
    set environment
    add multiple constraints
    add multiple sensors


    """

    def __init__(
        self,
        height: int,
        width: float = 1,
        base_centroid: np.ndarray = np.array([0, 0, 0]),
    ):
        self.controller = None
        self.environment = None
        self.constraints = []
        self.sensors = []

        self.cells = []
        base_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        default_centroid = np.mean(base_points, axis=0)
        cube_vertices = np.arange(8)
        cube_edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [0, 6],
                [1, 7],
                [2, 4],
                [3, 5],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
            ]
        )
        cube_faces = np.array(
            [
                [0, 3, 2, 1],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
                [4, 5, 6, 7],
            ]
        )

        self.positions = base_points.copy()

        for level in range(height):
            new_points = base_points + np.array([0, 0, level + 1])
            self.positions = np.vstack((self.positions, new_points))

            index_offset = 4 * level
            self.cells.append(
                Cell3D(
                    cube_vertices + index_offset,
                    cube_edges + index_offset,
                    cube_faces + index_offset,
                )
            )

        self.velocities = np.zeros_like(self.positions)
        self.positions = self.positions * width - default_centroid + base_centroid

    def add_controller(self, controller):
        self.controller = controller

    def add_environment(self, environment):
        self.environment = environment

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def construct_arm(self):
        return Arm3D(
            self.positions,
            self.velocities,
            self.cells,
            self.controller,
            self.environment,
            self.constraints,
            self.sensors,
        )
