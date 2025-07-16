"""TODO: I think typically observations would not be passed in, but instead
state estimation would be used to get the estimated state from observations.
For the hydrostat, this would mean that the state vector should really be all
of the positions, velocities, and smell concentrations."""

from abc import ABC, abstractmethod

from pydrostat.model.structure import Arm3D


class IPolicy(ABC):

    @abstractmethod
    def __call__(self, structure: Arm3D, states, observations, t: float) -> list[float]:
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
