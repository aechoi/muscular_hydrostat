import numpy as np

from .constraint_interface import IConstraint
from pydrostat.model.structure import Arm3D


class FixedVertex(IConstraint):
    def __init__(self, fixed_vertices):
        self.fixed_vertices = np.array(fixed_vertices)

    def initialize_constraint(self, structure: Arm3D, initial_state):
        pos, _ = structure.state2posvel(initial_state)
        self.initial_positions = pos[self.fixed_vertices]

    def calculate_constraints(self, structure: Arm3D, state):
        pos, _ = structure.state2posvel(state)
        dim = pos.shape[-1]
        num_constraints = len(self.fixed_vertices) * dim
        relative_vecs = pos[self.fixed_vertices] - self.initial_positions
        constraints = relative_vecs.flatten()

        dim_idx = np.arange(dim)
        dim_IDX, fixed_IDX = np.meshgrid(dim_idx, self.fixed_vertices)

        jacobians = np.zeros((num_constraints,) + pos.shape)
        djac_dts = np.zeros((num_constraints,) + pos.shape)

        jacobians[
            np.arange(num_constraints), fixed_IDX.flatten(), dim_IDX.flatten()
        ] = 1

        return constraints, jacobians, djac_dts
