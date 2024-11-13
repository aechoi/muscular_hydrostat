import numpy as np

from .constraint_interface import IConstraint
from ..structure.structure_interface import IStructure


class ConstantVolume(IConstraint):
    """Constant volume constraint for a 3D cell. Assume that"""

    def __init__(self):
        self.initial_volumes = None

    def initialize_constraint(self, structure: IStructure):
        self.initial_volumes = []
        for cell in structure.cells:
            apex_position = structure.positions[cell.vertices[0]]

            relative_positions = (
                structure.positions[cell.triangles] - apex_position
            )  # Tx3xD
            volumes = np.linalg.det(relative_positions)  # T
            volume = np.sum(volumes)
            self.initial_volumes.append(volume)

        self.initial_volumes = np.array(self.initial_volumes)

    def calculate_constraints(self, structure: IStructure):
        if self.initial_volumes is None:
            raise ValueError(
                "Initial volumes have not been calculated. Call initialize_constraint()"
            )
        pos = structure.positions
        vel = structure.velocities

        constraints = np.zeros(len(structure.cells))
        jacobians = np.zeros((len(structure.cells),) + pos.shape)
        djac_dts = np.zeros((len(structure.cells),) + pos.shape)

        for c_idx, cell in enumerate(structure.cells):
            # TODO vectorize across all cells assuming same cell repeated
            jacobian = np.zeros_like(pos)
            djac_dt = np.zeros_like(pos)

            apex_position = pos[cell.vertices[0]]
            apex_velocity = vel[cell.vertices[0]]

            relative_positions = pos[cell.triangles] - apex_position  # Tx3xD
            relative_velocities = vel[cell.triangles] - apex_velocity
            volumes = np.linalg.det(relative_positions)  # T
            volume = np.sum(volumes)

            position_inverse = np.linalg.inv(relative_positions)  # Tx3xD

            cofactors = volumes[:, None, None] * position_inverse.swapaxes(
                -1, -2
            )  # TxDx3
            adjugate = cofactors.swapaxes(-1, -2)
            jacobian[0] = -np.sum(cofactors, axis=(0, 1))
            # jacobian is nx3
            # each triangle has 3 indices 1 to n, total of t*3 vertices
            # add the cofactor
            # vec_indices = get_vec_indices(self.triangles.flatten())
            # for each index in the trinagles, add the ith cofactor to the jacobian at that index
            np.add.at(jacobian, cell.triangles.flatten(), cofactors.reshape(-1, 3))

            dcofactors = (
                np.trace(adjugate @ relative_velocities, axis1=1, axis2=2)[
                    :, None, None
                ]
                * position_inverse
                - adjugate @ relative_velocities @ position_inverse
            )
            djac_dt[0] = -np.sum(dcofactors, axis=(0, 1))
            np.add.at(djac_dt, cell.triangles.flatten(), dcofactors.reshape(-1, 3))

            constraints[c_idx] = volume - self.initial_volumes[c_idx]
            jacobians[c_idx] = jacobian
            djac_dts[c_idx] = djac_dt

            # jacobian = np.zeros(structure.positions.size)
            # djacdt = np.zeros(structure.positions.size)

            # apex_position = structure.positions[cell.vertices[0]]
            # apex_velocity = structure.velocities[cell.vertices[0]]

            # relative_positions = (
            #     structure.positions[cell.triangles] - apex_position
            # )  # Tx3xD
            # relative_velocities = structure.velocities[cell.triangles] - apex_velocity
            # volumes = np.linalg.det(relative_positions)  # T
            # volume = np.sum(volumes)

            # position_inverse = np.linalg.inv(relative_positions)  # Tx3xD

            # cofactors = volumes[:, None, None] * position_inverse.swapaxes(
            #     -1, -2
            # )  # TxDx3
            # adjugate = cofactors.swapaxes(-1, -2)
            # jacobian[get_vec_indices([0])] = -np.sum(cofactors, axis=(0, 1))
            # vec_indices = get_vec_indices(cell.triangles.flatten())
            # np.add.at(jacobian, vec_indices, cofactors.flatten())

            # dcofactors = (
            #     np.trace(adjugate @ relative_velocities, axis1=1, axis2=2)[
            #         :, None, None
            #     ]
            #     * position_inverse
            #     - adjugate @ relative_velocities @ position_inverse
            # )
            # djacdt[get_vec_indices([0])] = -np.sum(dcofactors, axis=(0, 1))
            # np.add.at(djacdt, vec_indices, dcofactors.flatten())

            # constraints[c_idx] = volume - self.initial_volumes[c_idx]
            # jacobians[c_idx] = jacobian.reshape(-1, 3)
            # djac_dts[c_idx] = djacdt.reshape(-1, 3)

        return constraints, jacobians, djac_dts


# def get_vec_indices(vertex_indices: list):
#     """Return the list of indices that correspond to the vertex indices."""
#     if type(vertex_indices) is int:
#         vertex_indices = [vertex_indices]
#     vertex_indices = np.array(vertex_indices)
#     return (vertex_indices[:, None] * 3 + np.arange(3)[None, :]).flatten()
