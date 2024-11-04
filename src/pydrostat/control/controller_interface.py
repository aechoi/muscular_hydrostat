""""""

from abc import ABC, abstractmethod

from ..structure.structure_interface import IStructure


class IController(ABC):

    @abstractmethod
    def calc_inputs(structure: IStructure, sensor_data: dict):
        """Calculates and returns a vector of control inputs for the structure
        to implement. Must be the same shape as the structure actuators.

        Args:
            structure: a structure object which is being controlled
            sensor_data: a dictionary with sensor type strings as keys and data as the
                value.

        Returns:
            an np.ndarray of control inputs which have the same shape as the actuators
            of the structure.

        Raises:
            NotImplemnetedError: if not implemneted by concrete class"""
        raise NotImplementedError
