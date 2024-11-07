import numpy as np

from .constraint_interface import IConstraint
from ..structure.structure_interface import IStructure


class RaggedPlanarFaces(IConstraint):
    """Does not assume each face has the same number of vertices"""

    def initialize_constraint(self, structure: IStructure):
        pass

    def calculate_constraints(self, structure: IStructure):
        constraints = []
        jacobians = []
        djac_dts = []
        for face in structure.faces:

            points = structure.positions[face]
            dpointsdt = structure.velocities[face]
            F, D = points.shape

            centroid = np.average(points, axis=0)
            dcentroiddt = np.average(dpointsdt, axis=0)
            centered_points = points - centroid
            centered_velocities = dpointsdt - dcentroiddt

            cov, dcdp, dcdt, ddcdpdt = self._calc_covariance_variables(
                centered_points, centered_velocities
            )

            normal, dndp, dndt, ddndpdt = self._calc_normal_variables(
                cov, dcdp, dcdt, ddcdpdt
            )

            constraint = centered_points @ normal  # FxD @ D = F

            dPdP = np.zeros((F, F, D, D))
            dPdP[np.arange(F), np.arange(F), :, :] = np.eye(D)
            dcentereddP = dPdP - np.eye(D)[None, None, :, :] / F  # FxFxDxD
            jacobian = np.zeros((F,) + structure.positions.shape)
            jacobian[:, face, :] = dcentereddP @ normal + (  # FxFxDxD @ D = FxFxD
                centered_points @ dndp
            ).swapaxes(
                0, 1
            )  # (FxD @ FxDxD)swap = FxFxD

            djacdt = np.zeros((F,) + structure.positions.shape)
            djacdt[:, face, :] = (
                dcentereddP @ dndt
                + (centered_velocities @ dndp).swapaxes(0, 1)
                + (centered_points @ ddndpdt).swapaxes(0, 1)
            )

            constraints.extend(constraint)
            jacobians.extend(jacobian)
            djac_dts.extend(djacdt)
        constraints = np.array(constraints)
        jacobians = np.array(jacobians)
        djac_dts = np.array(djac_dts)
        return constraints, jacobians, djac_dts

    def _calc_covariance_variables(self, centered_points, centered_velocities):
        dof = len(centered_points) - 1

        cov = np.cov(centered_points.T)

        units = np.eye(3)[None, :, None, :]

        # dCdP
        dcdp_single = centered_points[:, None, :, None] @ units
        dcdp = (dcdp_single.swapaxes(-1, -2) + dcdp_single) / dof

        # dCdt
        dcdt_single = centered_velocities.T @ centered_points
        dcdt = (dcdt_single + dcdt_single.T) / dof

        # ddCdPdt
        ddcdpdt_single = centered_velocities[:, None, :, None] @ units
        ddcdpdt = (ddcdpdt_single.swapaxes(-1, -2) + ddcdpdt_single) / dof

        return cov, dcdp, dcdt, ddcdpdt

    def _calc_normal_variables(self, cov, dcdp, dcdt, ddcdpdt):
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecsT = eigvecs.T
        min_eigvec = eigvecs[:, np.argmin(eigvals)]

        normal = min_eigvec

        # dNdP axes 1 and 2 swapped because not used elsewhere otherwise
        eigval_dif = eigvals[:, None] - eigvals
        eigval_dif[:] = np.divide(1, eigval_dif, out=eigval_dif, where=eigval_dif != 0)
        min_eigval_dif = eigval_dif[0][:, None]
        dndp = (min_eigval_dif * eigvecsT @ dcdp @ min_eigvec @ eigvecsT).swapaxes(
            -2, -1
        )

        dvdts = np.sum(
            (-eigval_dif * (eigvecsT @ dcdt @ eigvecs))[:, :, None]
            * eigvecsT[:, None, :],
            axis=0,
        ).T
        dndt = dvdts[:, 0]

        dldts = np.diagonal(eigvecsT @ dcdt @ eigvecs)

        deigval_difdt = dldts[:, None] - dldts
        deigval_difdt[:] = -np.divide(
            deigval_difdt,
            (eigvals[:, None] - eigvals) ** 2,
            out=deigval_difdt,
            where=deigval_difdt != 0,
        )

        # ugly and ragged to reduce computation
        ddndpdt = (
            (
                (
                    (deigval_difdt[0][:, None] * eigvecsT + min_eigval_dif * dvdts.T)
                    @ dcdp
                    + min_eigval_dif * eigvecsT @ ddcdpdt
                )
                @ min_eigvec
                + min_eigval_dif * eigvecsT @ dcdp @ dvdts[:, 0]
            )
            @ eigvecsT
            + min_eigval_dif * eigvecsT @ dcdp @ min_eigvec @ dvdts.T
        ).swapaxes(-2, -1)

        return normal, dndp, dndt, ddndpdt


class EqualPlanarFaces(IConstraint):
    """Assume that each face has the same number of vertices for vectorization"""

    def initialize_constraint(structure):
        pass

    def calculate_constraints(structure):
        pass
