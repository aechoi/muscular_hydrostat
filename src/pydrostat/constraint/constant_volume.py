import numpy as np

from .constraint_interface import IConstraint
from ..structure.structure import AStructure


class ConstantVolume(IConstraint):
    """Constant volume constraint for a 3D cell. Assume that the cells may have
    different structures from each other."""

    def __init__(self):
        self.initial_volumes = None

    def initialize_constraint(self, structure: AStructure):
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

    def calculate_constraints(self, structure: AStructure):
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

        return constraints, jacobians, djac_dts


class ConstantVolumeCommon(IConstraint):
    """Constant volume constraint for a 3D cell. Assume that the cells have a common
    structure."""

    def __init__(self):
        self.initial_volumes = None

    def initialize_constraint(self, structure: AStructure):
        self.vertices = np.array([cell.vertices for cell in structure.cells])  # CxV
        self.triangles = np.array([cell.triangles for cell in structure.cells])  # CxTxS

        apex_positions = structure.positions[self.vertices[:, 0]]  # CxD
        relative_positions = (
            structure.positions[self.triangles] - apex_positions[:, None, None, :]
        )  # CxTx3xD - CxD
        tet_volumes = np.linalg.det(relative_positions)  # CxT
        self.initial_volumes = np.sum(tet_volumes, axis=1)  # C

    def calculate_constraints(self, structure: AStructure):
        if self.initial_volumes is None:
            raise ValueError(
                "Initial volumes have not been calculated. Call initialize_constraint()"
            )
        pos = structure.positions
        vel = structure.velocities

        constraints = np.zeros(len(structure.cells))
        jacobians = np.zeros((len(structure.cells),) + pos.shape)
        djac_dts = np.zeros((len(structure.cells),) + pos.shape)

        apex_positions = pos[self.vertices[:, 0]]
        apex_velocities = vel[self.vertices[:, 0]]

        relative_positions = pos[self.triangles] - apex_positions[:, None, None, :]
        relative_velocities = (
            vel[self.triangles] - apex_velocities[:, None, None, :]
        )  # CxTxSxD

        tet_volumes = np.linalg.det(relative_positions)
        volumes = np.sum(tet_volumes, axis=1)
        constraints = volumes - self.initial_volumes

        position_inverse = np.linalg.inv(relative_positions)  # CxTxDxS
        cofactors = tet_volumes[:, :, None, None] * position_inverse.swapaxes(
            -1, -2
        )  # CxTxSxD
        adjugates = cofactors.swapaxes(-1, -2)  # CxTxDxS

        jacobians[:, 0] = -np.sum(cofactors, axis=(1, 2))  # CxD

        dcofactors = (
            np.sum(cofactors * relative_velocities, axis=(-1, -2))[:, :, None, None]
            * position_inverse
            - adjugates @ relative_velocities @ position_inverse
        )
        djac_dts[:, 0] = -np.sum(dcofactors, axis=(1, 2))  # CxD

        # TODO: try to vectorize this? Trickier than it seems
        for cell_idx, _ in enumerate(structure.cells):
            np.add.at(
                jacobians[cell_idx], self.triangles[cell_idx], cofactors[cell_idx]
            )
            np.add.at(
                djac_dts[cell_idx], self.triangles[cell_idx], dcofactors[cell_idx]
            )

        return constraints, jacobians, djac_dts
