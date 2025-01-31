from __future__ import annotations
from abc import ABC, abstractmethod

from ..structures.structure import AStructure
from _old.environment import Environment


class ISensor(ABC):
    @property
    @abstractmethod
    def sensor_type(self) -> str:
        """Require that sensor objects have a sensor_type string which acts as a key
        to the controllers on what sort of sensor data is provided."""
        pass

    @abstractmethod
    def sense(structure: AStructure, environment: Environment):
        """Sense something about the environment using the position of sesnors on the
        structure.

        Args:
            structure: the structure on which the sensors are housed
            environment: the environment that the structure is in

        Returns:
            a SensorData object which has the type of data and the data itself

        Raises:
            NotImplemnetedError: if not implemneted by concrete class"""
        raise NotImplementedError
