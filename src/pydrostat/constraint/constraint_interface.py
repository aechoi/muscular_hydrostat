"""An interface for constraints used in constrained dynamics calculations.

This interface describes the structure for a constraint class used in calculating
matrices required for calculating lagrange multipliers. This includes a constraint
vector entry, the Jacobians, and the time derivative of the Jacobians.

Typical usage example:
    class SomeConstraint(IConstraint):
        ...
"""

from abc import ABC, abstractmethod

import numpy as np

from ..structure.structure_interface import AStructure


class IConstraint(ABC):
    """An abstract class used as an interface for constraint objects.

    Each concrete instance of this interface can have any number of functions which
    aid in calculating the constraints. Typically, there will not be separate functions
    for the constraints, jacobian, and jacobian time derivative because the calculation
    of one usually involves the calculation of intermediate variables which are shared
    between the calculation of others.
    """

    def initialize_constraint(structure: AStructure) -> None:
        """Initialize the constraint by calculating whatever data is needed

        Certain constraints, such as constant volume, require a calculation to be done
        before running the simulation (for instance, calculating the initial volume).

        Args:
            structure: A concrete instance of the structure interface.

        Raises:
            NotImplementedError if concrete class does not implement this method."""
        raise NotImplementedError

    @abstractmethod
    def calculate_constraints(
        structure: AStructure,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the constraint vector, the Jacobian, and the time derivative of the
        Jacobian.

        Given a particular structure object (arm, cell, etc), calculate a particular
        type of constraint (eg constant volume, planar faces, min/max edge length) and
        the associated Jacobian and Jacobian time derivative.

        Args:
            structure: A concrete instance of the structure interface.

        Returns:
            A tuple of three np.ndarrays. The first is the constraint array. This is a
            1-dimensional length m array where m is the number of generated constraints.

            The second is the Jacobian which is a 3-dimensional array of shape mxnxd
            where n is the number of vertices in the structure.

            The third is the time derivative of the Jacobian which is of the same shape
            as the Jacobian

        Raises:
            NotImplementedError if concrete class does not implement this method
        """
        raise NotImplementedError
