"""This module holds concrete implementations of the structure interface. These
include but are not limited to 
    - cuboid arms
    - single cells
    - iso-cylinders

Typical use case:
    # In a simulator
    structure = structures.arm_3d(...)
    while simulating:
        structure.iterate(dt)
        # code to display structure
"""

import numpy as np

from .structure import AStructure
from .cell import Cell3D
from .arm import Arm3D

class CubicArmBuilder:
    """A builder for 3D cubic arms

    create cell structure
    choose controller
    set environment
    add multiple constraints
    add multiple sensors

    """

    # class variables for base matrices of edges, faces, vertices, base_points
    cube_edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [0, 6],
                [1, 7],
                [2, 4],
                [3, 5],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
            ]
        )
    cube_faces = np.array(
            [
                [0, 3, 2, 1],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
                [4, 5, 6, 7],
            ]
        )
    cube_vertices = np.arange(8)
    base_points = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
        )
    
    def __init__(
        self,
        height: int,
        width: float = 1,
        base_centroid: np.ndarray = np.array([0, 0, 0]),
    ):
        self.controller = None
        self.environment = None
        self.constraints = []
        self.sensors = []
        self.cells = []
        self.positions = CubicArmBuilder.base_points.copy()
        self.velocities = np.zeros_like(self.positions)
        default_centroid = np.mean(CubicArmBuilder.base_points, axis=0)

        self.__initialize_positions(height, width, default_centroid, base_centroid)
        self.__initialize_cells(height)


    def __initialize_cells(self, _height):
        for level in range(_height):
            index_offset = 4 * level
            self.cells.append(
                Cell3D(
                    CubicArmBuilder.cube_vertices + index_offset,
                    CubicArmBuilder.cube_edges + index_offset,
                    CubicArmBuilder.cube_faces + index_offset
                    # masses=np.ones_like(cube_vertices) / len(cube_vertices),
                    # vertex_damping=np.ones_like(cube_vertices),
                )
            )


    def __initialize_positions(self, _height, _width, _default_centroid, _base_centroid):
        for level in range(_height):
            new_points = CubicArmBuilder.base_points + np.array([0, 0, level + 1])
            self.positions = np.vstack((self.positions, new_points))

        self.positions = self.positions * _width - _default_centroid + _base_centroid


    def set_controller(self, controller):
        self.controller = controller


    def set_environment(self, environment):
        self.environment = environment


    def add_constraint(self, constraint):
        self.constraints.append(constraint)


    def add_sensor(self, sensor):
        self.sensors.append(sensor)


    def construct_arm(self):
        return Arm3D(
            self.positions,
            self.velocities,
            self.cells,
            self.controller,
            self.environment,
            self.constraints,
            self.sensors,
        )
