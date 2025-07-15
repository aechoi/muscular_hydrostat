""""""

from abc import ABC, abstractmethod

from pydrostat.model.structure import Arm3D


class IPolicy(ABC):

    @abstractmethod
    def __call__(self, structure: Arm3D, t: float) -> list[float]:
        """Calculates and returns a vector of control inputs for the structure
        to implement. Must be the same shape as the structure actuators.

        Args:
            structure: a structure object which is being controlled

        Returns:
            an np.ndarray of control inputs which have the same shape as the actuators
            of the structure.

        Raises:
            NotImplemnetedError: if not implemneted by concrete class"""
        raise NotImplementedError
