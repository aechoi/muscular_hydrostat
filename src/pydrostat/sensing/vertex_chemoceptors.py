from sensor_interface import ISensor
from ..structure.structure_interface import IStructure
from _old.environment import Environment


class VertexChemoceptors(ISensor):
    def __init__(self):
        self.sensor_type = "VertexChemoceptors"

    def sense(structure: IStructure, environment: Environment):
        return environment.sample_scent(structure.positions)
