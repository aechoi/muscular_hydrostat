from .sensor_interface import ISensor
from pydrostat.model.structure import Arm3D
from _old.environment import Environment


class VertexChemoceptors(ISensor):
    def __init__(self):
        self._sensor_type = "VertexChemoceptors"

    def sense(self, structure: Arm3D, environment: Environment):
        return environment.sample_scent(structure.positions)

    @property
    def sensor_type(self):
        return self._sensor_type
