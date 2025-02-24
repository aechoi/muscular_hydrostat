"""This module holds concrete implementations of the Arm classes.

Typical use case:
    # In a simulator
    structure = structures.arm_3d(...)
    while simulating:
        structure.iterate(dt)
        # code to display structure
"""

import numpy as np

from .structure import AStructure
from .cell import Cell3D


class Arm3D(AStructure):
    def __init__(
        self,
        initial_positions,
        initial_velocities,
        cells: Cell3D,
        controller=None,
        environment=None,
        constraints=[],
        sensors=[],
        constraint_damping_rate=50,
        constraint_spring_rate=50,
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

        # # REMOVE AFTER COMPATIBLE
        # self.muscles = np.zeros(len(self.edges))

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