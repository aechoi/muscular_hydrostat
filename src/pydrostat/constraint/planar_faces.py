import numpy as np

from .constraint_interface import IConstraint
from ..structure.structure_interface import AStructure


class PlanarFacesRagged(IConstraint):
    """Does not assume each face has the same number of vertices"""

    def initialize_constraint(self, structure: AStructure):
        pass

    def calculate_constraints(self, structure: AStructure):
        constraints = []
        jacobians = []
        djac_dts = []
        for face in structure.faces:

            points = structure.positions[face]  # VxD
            dpointsdt = structure.velocities[face]  # VxD
            F, D = points.shape

            centroid = np.average(points, axis=0)
            dcentroiddt = np.average(dpointsdt, axis=0)
            centered_points = points - centroid
            centered_velocities = dpointsdt - dcentroiddt

            cov, dcdp, dcdt, ddcdpdt = self._calc_covariance_variables(
                centered_points, centered_velocities
            )  # DxD, VxDxDxD, DxD, VxDxDxD

            normal, dndp, dndt, ddndpdt = self._calc_normal_variables(
                cov, dcdp, dcdt, ddcdpdt
            )

            constraint = centered_points @ normal  # VxD @ D = V

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
        # centered_points: VxD
        # units: DxD
        # Vx-xDx- @ -xDx-xD -> VxDxDxD
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
        eigvals, eigvecs = np.linalg.eigh(cov)  # D, DxD
        eigvecsT = eigvecs.T
        min_eigvec = eigvecs[:, np.argmin(eigvals)]  # D

        normal = min_eigvec

        # dNdP axes 1 and 2 swapped because not used elsewhere otherwise
        eigval_dif = eigvals[:, None] - eigvals  # DxD
        eigval_dif[:] = np.divide(1, eigval_dif, out=eigval_dif, where=eigval_dif != 0)
        min_eigval_dif = eigval_dif[0][:, None]
        dndp = (min_eigval_dif * eigvecsT @ dcdp @ min_eigvec @ eigvecsT).swapaxes(
            -2, -1
        )  # VxDxD

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


class PlanarFacesCommon(IConstraint):
    """Assume that each face has the same number of vertices for vectorization"""

    def initialize_constraint(self, structure):
        num_vertices = len(structure.faces[0])
        for face in structure.faces:
            if len(face) != num_vertices:
                raise ValueError(
                    "Faces have different numbers of vertices. Use PlanarFacesRagged"
                )
        self.face_indices = np.array(structure.faces)

    def calculate_constraints(self, structure: AStructure):
        N, D = structure.positions.shape
        F, V = self.face_indices.shape

        constraints = np.zeros((F, V))
        jacobians = np.zeros((F, V, N, D))
        djac_dts = np.zeros((F, V, N, D))

        points = structure.positions[self.face_indices]  # FxVxD
        dpoints_dt = structure.velocities[self.face_indices]  # FxVxD

        centroids = np.average(points, axis=1)  # FxD
        dcentroid_dts = np.average(dpoints_dt, axis=1)  # FxD
        relative_positions = points - centroids[:, None, :]  # FxVxD
        relative_velocities = dpoints_dt - dcentroid_dts[:, None, :]  # FxVxD

        VV, FF = np.meshgrid(np.arange(V), np.arange(F))
        dFdP = np.zeros((F, V, N, D, D))
        dFdP[FF, VV, self.face_indices, :, :] = np.eye(D)

        dcentroiddP = np.zeros((F, N, D, D))
        dcentroiddP[FF, self.face_indices, :, :] = np.eye(D) / V

        drelative_dP = (dFdP - dcentroiddP[:, None, :, :, :]).swapaxes(
            2, 3
        )  # FxVxDxNxD

        cov, dcdp, dcdt, ddcdpdt = self._calc_covariance_variables(
            relative_positions, relative_velocities, drelative_dP
        )  # FxDxD, FxDxDxNxD, FxDxD, FxDxDxNxD

        normal, dndp, dndt, ddndpdt = self._calc_normal_variables(
            cov, dcdp, dcdt, ddcdpdt
        )  # FxD, FxDxNxD, FxD, FxDxNxD

        constraints = relative_positions @ normal[:, :, None]
        # FxVxD @ FxDx1 -> FxV

        jacobians = np.sum(
            drelative_dP * normal[:, None, :, None, None], axis=2
        ) + np.sum(
            relative_positions[:, :, :, None, None] * dndp[:, None, :], axis=2
        )  # FxVxNxD

        djac_dts = (
            np.sum(drelative_dP * dndt[:, None, :, None, None], axis=2)
            + np.sum(
                relative_velocities[:, :, :, None, None] * dndp[:, None, :, :, :],
                axis=2,
            )
            + np.sum(
                relative_positions[:, :, :, None, None] * ddndpdt[:, None, :, :, :],
                axis=2,
            )
        )  # FxVxNxD

        constraints = constraints.reshape(-1)
        jacobians = jacobians.reshape(-1, N, D)
        djac_dts = djac_dts.reshape(-1, N, D)

        return constraints, jacobians, djac_dts

    def _calc_covariance_variables(
        self, relative_positions, relative_velocities, drelative_dP
    ):
        """

        Args:
            relative_positions: FxVxD
            relative_velocities: FxVxD
            drelative_dP: FxVxDxNxD"""
        dof = relative_positions.shape[1] - 1

        cov = relative_positions.swapaxes(-1, -2) @ relative_positions / dof

        # dCdP
        dcdp_single = np.sum(
            relative_positions[:, :, :, None, None, None]
            * drelative_dP[:, :, None, :, :, :],
            axis=(1),
        )  # FxDxDxNxD
        dcdp = (dcdp_single + dcdp_single.swapaxes(1, 2)) / dof

        # dCdt
        dcdt_single = relative_velocities.swapaxes(-1, -2) @ relative_positions
        dcdt = (dcdt_single + dcdt_single.swapaxes(-1, -2)) / dof

        # ddCdPdt
        ddcdpdt_single = np.sum(
            relative_velocities[:, :, :, None, None, None]
            * drelative_dP[:, :, None, :, :, :],
            axis=(1),
        )
        ddcdpdt = (ddcdpdt_single + ddcdpdt_single.swapaxes(1, 2)) / dof

        return cov, dcdp, dcdt, ddcdpdt

    def _calc_normal_variables(self, cov, dcdp, dcdt, ddcdpdt):
        """
        Args:
            cov: FxDxD
            dcdp: FxDxDxNxD
            dcdt: FxDxD
            ddcdpdt: FxDxDxNxD
        """
        eigvals, eigvecs = np.linalg.eigh(cov)  # FxD, FxDxD
        # eigvals are guaranteed ascending
        eigvecsT = eigvecs.swapaxes(-1, -2)  # FxDxD
        min_eigvecs = eigvecsT[:, 0]  # FxD

        normals = min_eigvecs

        # dNdP axes 1 and 2 swapped because not used elsewhere otherwise
        eigval_dif = eigvals[:, :, None] - eigvals[:, None, :]  # FxDxD
        eigval_dif[:] = np.divide(1, eigval_dif, out=eigval_dif, where=eigval_dif != 0)
        min_eigval_difs = eigval_dif[:, 0][:, :, None]  # FxDxD
        # In general, I've found this is faster than using einsum
        dndp = np.sum(
            (min_eigval_difs * eigvecsT)[:, None, :, :, None, None, None]
            * dcdp[:, None, None, :, :, :, :]
            * normals[:, None, None, None, :, None, None]
            * eigvecs[:, :, :, None, None, None, None],
            axis=(2, 3, 4),
        )  # FxDxNxD

        dvdts = np.sum(
            (-eigval_dif * (eigvecsT @ dcdt @ eigvecs))[:, :, :, None]
            * eigvecsT[:, :, None, :],
            axis=1,
        ).swapaxes(
            -1, -2
        )  # FxDxD
        dndt = dvdts[:, :, 0]  # FxD

        dldts = np.diagonal(eigvecsT @ dcdt @ eigvecs, axis1=-2, axis2=-1)  # FxD

        deigval_difdt = dldts[:, :, None] - dldts[:, None, :]
        deigval_difdt[:] = -np.divide(
            deigval_difdt,
            (eigvals[:, :, None] - eigvals[:, None, :]) ** 2,
            out=deigval_difdt,
            where=deigval_difdt != 0,
        )  # FxDxD

        ddndpdt = (
            self._bespoke_contraction(
                deigval_difdt[:, 0][:, :, None], eigvecsT, dcdp, normals, eigvecs
            )
            + self._bespoke_contraction(
                min_eigval_difs, dvdts.swapaxes(-1, -2), dcdp, normals, eigvecs
            )
            + self._bespoke_contraction(
                min_eigval_difs, eigvecsT, ddcdpdt, normals, eigvecs
            )
            + self._bespoke_contraction(min_eigval_difs, eigvecsT, dcdp, dndt, eigvecs)
            + self._bespoke_contraction(min_eigval_difs, eigvecsT, dcdp, normals, dvdts)
        )

        return normals, dndp, dndt, ddndpdt

    def _bespoke_contraction(self, eigval_difs, eigvecsT, dcdp, normals, eigvecs):
        contraction = np.sum(
            (eigval_difs * eigvecsT)[:, None, :, :, None, None, None]
            * dcdp[:, None, None, :, :, :, :]
            * normals[:, None, None, None, :, None, None]
            * eigvecs[:, :, :, None, None, None, None],
            axis=(2, 3, 4),
        )  # FxDxNxD
        return contraction
