import numpy as np

from .constraint_interface import IConstraint
from ..structure.structure_interface import AStructure


class ClipLength(IConstraint):
    def __init__(self, min_length: float = 0.0, max_length: float = np.inf):
        self.min_length = min_length
        self.max_length = max_length
        self.limits = np.array([self.min_length, self.max_length])

    def initialize_constraint(self, structure: AStructure):
        pass

    def calculate_constraints(self, structure: AStructure):
        edge_points = structure.positions[structure.edges]  # shape ex2xd
        edge_lengths = np.linalg.norm(edge_points[:, 0] - edge_points[:, 1], axis=1)
        edge_mask = np.logical_or(
            edge_lengths > self.max_length, edge_lengths < self.min_length
        )
        num_constrained = np.sum(edge_mask)

        if num_constrained == 0:
            return [], [], []

        length_diffs = edge_lengths[edge_mask][:, None] - self.limits[None, :]
        min_or_max = np.argmin(np.abs(length_diffs), axis=1)
        constraints = length_diffs[np.arange(num_constrained), min_or_max]
        jacobians = np.zeros((num_constrained,) + structure.positions.shape)
        djac_dts = np.zeros((num_constrained,) + structure.positions.shape)

        constrained_points = edge_points[edge_mask]  # Shape sx2xd where s is num_long

        relative_constrained_edge = (
            constrained_points[:, 0] - constrained_points[:, 1]
        )  # shape sxd
        unit_relative_edge = (
            relative_constrained_edge / edge_lengths[edge_mask, None]
        )  # shape sxd

        constrained_edges = structure.edges[edge_mask]  # shape sx2
        jacobians[np.arange(num_constrained), constrained_edges[:, 0]] = (
            unit_relative_edge
        )
        jacobians[np.arange(num_constrained), constrained_edges[:, 1]] = (
            -unit_relative_edge
        )

        constrained_velocities = structure.velocities[structure.edges[edge_mask]]
        vel_dif = (
            constrained_velocities[:, 0] - constrained_velocities[:, 1]
        )  # shape sxd
        # a'b - b'a / b^2
        # edge_length = (rel_vec^T rel_vec)^0.5
        # A = rel_vec^T rel_vec
        # del/dt = del/dA . dA/dt = 0.5(A)^-0.5 * dA/dt
        # dA/dt = dA/drel_vec . drel_vec/dt
        # dedge_length = 0.5 (rel_vec^T rel_vec)^-0.5 * sum(rel_vec .* drel_vec)
        dunit_relative_edge = (
            vel_dif * edge_lengths[edge_mask, None]
            - relative_constrained_edge
            * np.sum(relative_constrained_edge * vel_dif, axis=1)[:, None]
        ) / edge_lengths[edge_mask, None] ** 2

        djac_dts[np.arange(num_constrained), constrained_edges[:, 0]] = (
            dunit_relative_edge
        )
        djac_dts[np.arange(num_constrained), constrained_edges[:, 1]] = (
            -dunit_relative_edge
        )

        return constraints, jacobians, djac_dts
