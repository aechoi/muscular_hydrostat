from __future__ import annotations
from abc import ABC, abstractmethod

from .structure_interface import IStructure
from environment.environment import Environment


class ISensor(ABC):

    @abstractmethod
    def sense(structure: IStructure, environment: Environment):
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
