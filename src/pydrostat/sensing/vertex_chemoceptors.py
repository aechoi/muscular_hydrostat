from .sensor_interface import ISensor
from pydrostat.model.structure import Arm3D
from _old.environment import Environment


class VertexChemoceptors(ISensor):
    def __init__(self):
        self._sensor_type = "VertexChemoceptors"

    def sense(self, structure: Arm3D, state, environment: Environment):
        pos, _ = structure.state2posvel(state)
        return environment.sample_scent(pos)

    @property
    def sensor_type(self):
        return self._sensor_type
