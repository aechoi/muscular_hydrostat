"""A module for a simple hand-tuned gradient controller"""

from typing import TYPE_CHECKING

import numpy as np

from controller_interface import IController

if TYPE_CHECKING:
    from .structure.structure_interface import IStructure


class HandTunedGradient(IController):
    """A class for calculating edge actuations based on the estimated gradient of scent"""

    def calc_inputs(structure: IStructure, sensor_data: dict):
        control_inputs = np.zeros_like(structure.edges)
        if "VertexChemoceptors" not in sensor_data:
            return control_inputs

        sensor_data = sensor_data["VertexChemoceptors"]

        forward_backward_gradient = 0
        strength_scales = np.logspace(-1, 2, len(structure.cells), base=4.5)

        for idx, cell in enumerate(structure.cells[::-1]):
            strength_scale = strength_scales[idx]

            points = structure.positions[cell.vertices]
            scents = sensor_data[cell.vertices]

            gradient = (
                np.linalg.pinv(np.column_stack((points, np.ones(len(points))))) @ scents
            )[:-1]
            gradient = gradient / np.linalg.norm(gradient)

            top_face = cell.faces[-1]
            normal = np.cross(
                np.diff(structure.positions[top_face[0:2]], axis=0),
                np.diff(structure.positions[top_face[1:3]], axis=0),
            ).flatten()
            normal = normal / np.linalg.norm(normal)

            forward_backward_gradient = np.dot(gradient, normal)

            if forward_backward_gradient > 0:
                actuator_index = np.array(
                    [
                        np.where(np.all(structure.edges == sorted(edge), axis=1))[0]
                        for edge in cell.edges[-4:]
                    ]
                ).flatten()
                control_inputs[actuator_index] = forward_backward_gradient / 1.5
            else:
                actuator_index = np.array(
                    [
                        np.where(np.all(structure.edges == sorted(edge), axis=1))[0]
                        for edge in cell.edges[4:-8]
                    ]
                ).flatten()
                control_inputs[actuator_index] = -forward_backward_gradient

            desired_motion = gradient - normal
            # desired_motion = desired_motion / np.linalg.norm(desired_motion)
            top_centroid = np.average(structure.positions[top_face], axis=0)
            rel_vertices = structure.positions[top_face] - top_centroid
            activations = rel_vertices @ desired_motion
            actuator_index = np.array(
                [
                    np.where(np.all(structure.edges == sorted(edge), axis=1))[0]
                    for edge in cell.edges[4:-8]
                ]
            ).flatten()
            control_inputs[actuator_index] += activations * strength_scale * 4

            actuator_index = np.array(
                [
                    np.where(np.all(structure.edges == sorted(edge), axis=1))[0]
                    for edge in cell.edges[8:-4]
                ]
            ).flatten()
            control_inputs[actuator_index] += activations * strength_scale * 2
            control_inputs = np.clip(control_inputs, 0, None)

        return control_inputs
