import numpy as np

from .constraint_interface import IConstraint
from ..structure.structure_interface import AStructure


class FixedVertex(IConstraint):
    def __init__(self, fixed_vertices):
        self.fixed_vertices = np.array(fixed_vertices)

    def initialize_constraint(self, structure: AStructure):
        self.initial_positions = structure.positions[self.fixed_vertices]

    def calculate_constraints(self, structure: AStructure):
        # num_constraints = len(self.fixed_vertices)
        # relative_vecs = (
        #     structure.positions[self.fixed_vertices] - self.initial_positions
        # )
        # constraints = 0.5 * (relative_vecs * relative_vecs).sum(axis=1)

        # jacobians = np.zeros((num_constraints,) + structure.positions.shape)
        # djac_dts = np.zeros((num_constraints,) + structure.positions.shape)

        # jacobians[np.arange(num_constraints), self.fixed_vertices, :] = relative_vecs
        # djac_dts[np.arange(num_constraints), self.fixed_vertices, :] = (
        #     structure.velocities[self.fixed_vertices]
        # )

        dim = structure.positions.shape[-1]
        num_constraints = len(self.fixed_vertices) * dim
        relative_vecs = (
            structure.positions[self.fixed_vertices] - self.initial_positions
        )
        constraints = relative_vecs.flatten()

        dim_idx = np.arange(dim)
        dim_IDX, fixed_IDX = np.meshgrid(dim_idx, self.fixed_vertices)

        jacobians = np.zeros((num_constraints,) + structure.positions.shape)
        djac_dts = np.zeros((num_constraints,) + structure.positions.shape)

        jacobians[
            np.arange(num_constraints), fixed_IDX.flatten(), dim_IDX.flatten()
        ] = 1

        return constraints, jacobians, djac_dts
