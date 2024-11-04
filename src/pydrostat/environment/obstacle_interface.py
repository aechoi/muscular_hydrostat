"""This module holds the abstract class and concrete instances of obstacles.

An obstacle is a special type of constraint which conditionally applies a constraint to
vertices that intersect the interior of the obstacle. 

"""

from abc import abstractmethod

import numpy as np

from ..constraint.constraint_interface import IConstraint
from ..structure.structure_interface import IStructure


class IObstacle(IConstraint):
    @abstractmethod
    def check_intersection(self, points: np.ndarray):
        """Check if a set of points intersect with the obstacle volume.

        Args:
            points: an nxd array of points to check for intersections where n is the
                number of points and d is the dimension

        Returns:
            a boolean array where True indices are intersecting the obstacle

        Raises:
            NotImplementedError if not implemneted in concrete instance.
        """
        raise NotImplementedError

    def initialize_constraint(structure: IStructure) -> None:
        """Nothing to initialize for obstacles"""
        pass

    @abstractmethod
    def calculate_constraints(
        structure: IStructure,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Check if any points intersect the obstacle, if so, constrain
        motion to be along the obstacle. Otherwise, no constraint."""
        pass
