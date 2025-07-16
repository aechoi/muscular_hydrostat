"""A module for a simple hand-tuned gradient controller"""

import jax
import jax.numpy as jnp

from .policy_interface import IPolicy

from pydrostat.model.structure import Arm3D


class HandTunedGradient(IPolicy):
    """A class for calculating edge actuations based on the estimated gradient of scent

    This policy calculates an internal gradient estimated from the sensor data at the
    vertices of each cell. It attempts to point its top face towards this gradient. If
    the end effector is behind the food, it will attempt to contract all lateral edges
    to lengthen. Otherwise it will try to contract longitudinal edges."""

    def __call__(self, structure: Arm3D, states, observations, t: float) -> jnp.ndarray:
        control_inputs = jnp.zeros(len(structure.edges), dtype=float)
        if "VertexChemoceptors" not in observations:
            return control_inputs

        sensor_data = observations["VertexChemoceptors"]
        pos, vel = structure.state2posvel(states)

        forward_backward_gradient = 0
        strength_scales = jnp.logspace(-1, 2, len(structure.cells), base=4.5)

        for idx, cell in enumerate(structure.cells[::-1]):
            strength_scale = strength_scales[idx]

            points = pos[cell.vertices]
            scents = sensor_data[cell.vertices]

            gradient = (
                jnp.linalg.pinv(jnp.column_stack((points, jnp.ones(len(points)))))
                @ scents
            )[:-1]
            gradient = gradient / jnp.linalg.norm(gradient)

            top_face = cell.faces[-1]
            normal = jnp.cross(
                jnp.diff(pos[top_face[0:2]], axis=0),
                jnp.diff(pos[top_face[1:3]], axis=0),
            ).flatten()
            normal = normal / jnp.linalg.norm(normal)

            forward_backward_gradient = jnp.dot(gradient, normal)

            if forward_backward_gradient > 0:
                actuator_index = jnp.array(
                    [
                        jnp.where(
                            jnp.all(
                                jnp.array(structure.edges) == jnp.sort(edge), axis=1
                            )
                        )[0]
                        for edge in cell.edges[-4:]
                    ]
                ).flatten()
                control_inputs = control_inputs.at[actuator_index].set(
                    forward_backward_gradient * 1
                )
            else:
                actuator_index = jnp.array(
                    [
                        jnp.where(
                            jnp.all(
                                jnp.array(structure.edges) == jnp.sort(edge), axis=1
                            )
                        )[0]
                        for edge in cell.edges[4:-8]
                    ]
                ).flatten()
                control_inputs = control_inputs.at[actuator_index].add(
                    -forward_backward_gradient
                )

            desired_motion = gradient - normal
            # desired_motion = desired_motion / jnp.linalg.norm(desired_motion)
            top_centroid = jnp.average(pos[top_face], axis=0)
            rel_vertices = pos[top_face] - top_centroid
            activations = rel_vertices @ desired_motion
            actuator_index = jnp.array(
                [
                    jnp.where(
                        jnp.all(jnp.array(structure.edges) == jnp.sort(edge), axis=1)
                    )[0]
                    for edge in cell.edges[4:-8]
                ]
            ).flatten()
            control_inputs = control_inputs.at[actuator_index].add(
                activations * strength_scale * 4
            )

            actuator_index = jnp.array(
                [
                    jnp.where(
                        jnp.all(jnp.array(structure.edges) == jnp.sort(edge), axis=1)
                    )[0]
                    for edge in cell.edges[8:-4]
                ]
            ).flatten()
            control_inputs = control_inputs.at[actuator_index].add(
                activations * strength_scale * 2
            )
            control_inputs = jnp.clip(control_inputs, 0, None)

        return control_inputs


class HandTunedGradient2(IPolicy):
    """A class for calculating edge actuations based on the estimated gradient of scent"""

    def policy(self, structure: Arm3D, states, observations, t):
        control_inputs = jnp.zeros(len(structure.edges), dtype=float)
        if "VertexChemoceptors" not in observations:
            return control_inputs

        sensor_data = observations["VertexChemoceptors"]

        pos, vel = structure.state2posvel(states)

        forward_backward_gradient = 0
        strength_scales = jnp.logspace(-1, 2, len(structure.cells), base=4.5) * 1

        for idx, cell in enumerate(structure.cells[::-1]):
            strength_scale = strength_scales[idx]

            points = pos[cell.vertices]
            scents = sensor_data[cell.vertices]

            gradient = (
                jnp.linalg.pinv(jnp.column_stack((points, jnp.ones(len(points)))))
                @ scents
            )[:-1]
            gradient = gradient / jnp.linalg.norm(gradient)

            top_face = cell.faces[-1]
            normal = jnp.cross(
                jnp.diff(pos[top_face[0:2]], axis=0),
                jnp.diff(pos[top_face[1:3]], axis=0),
            ).flatten()
            normal = normal / jnp.linalg.norm(normal)

            forward_backward_gradient = jnp.dot(gradient, normal)

            if forward_backward_gradient > 0:
                actuator_index = jnp.array(
                    [
                        jnp.where(jnp.all(structure.edges == sorted(edge), axis=1))[0]
                        for edge in cell.edges[-4:]
                    ]
                ).flatten()
                control_inputs[actuator_index] = forward_backward_gradient * 1
            else:
                actuator_index = jnp.array(
                    [
                        jnp.where(jnp.all(structure.edges == sorted(edge), axis=1))[0]
                        for edge in cell.edges[4:-8]
                    ]
                ).flatten()
                control_inputs[actuator_index] = -forward_backward_gradient

            desired_motion = gradient - normal
            # desired_motion = desired_motion / jnp.linalg.norm(desired_motion)
            top_centroid = jnp.average(pos[top_face], axis=0)
            rel_vertices = pos[top_face] - top_centroid
            activations = rel_vertices @ desired_motion
            actuator_index = jnp.array(
                [
                    jnp.where(jnp.all(structure.edges == sorted(edge), axis=1))[0]
                    for edge in cell.edges[4:-8]
                ]
            ).flatten()
            control_inputs[actuator_index] += activations * strength_scale * 4

            actuator_index = jnp.array(
                [
                    jnp.where(jnp.all(structure.edges == sorted(edge), axis=1))[0]
                    for edge in cell.edges[8:-4]
                ]
            ).flatten()
            control_inputs[actuator_index] += activations * strength_scale * 2
            control_inputs = jnp.clip(control_inputs, 0, None)

        return control_inputs
