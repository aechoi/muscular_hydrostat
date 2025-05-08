from .sensor_interface import ISensor
from ..structures.structure import AStructure
from _old.environment import Environment


class VertexChemoceptors(ISensor):
    def __init__(self):
        self._sensor_type = "VertexChemoceptors"

    def sense(self, structure: AStructure, environment: Environment):
        return environment.sample_scent(structure.positions)

    @property
    def sensor_type(self):
        return self._sensor_type
