"""This module defines a concrete instance of the ConstrainedDynamics class which models
a structure with vertices, edges, and faces. The explicit forces include vertex damping,
and edge damping."""

from dataclasses import dataclass

import OpenGL.GL as gl
import jax
import jax.numpy as jnp
from .constrained_dynamics import ConstrainedDynamics

from time import time


@dataclass
class Cell3D:
    """A dataclass which holds shape info for 3D arms"""

    vertices: list[float]  # index of vertices
    edges: list[list[int]]  # each tuple indexes 2 points
    faces: list[
        list[int]
    ]  # indices of points, tuples may be ragged, must be ordered counter-clockwise from outside

    masses: list[float] = None
    vertex_damping: list[float] = None
    edge_damping: list[float] = None

    def __post_init__(self):
        if self.masses is None:
            self.masses = jnp.ones(len(self.vertices)) / len(self.vertices)
        if self.vertex_damping is None:
            self.vertex_damping = jnp.ones(len(self.vertices)) / len(self.vertices)
        if self.edge_damping is None:
            self.edge_damping = jnp.ones(len(self.edges))

        self.triangles = self.triangulate_faces()

    def triangulate_faces(self):
        """Decompose each face into triangles for the purposes of volume calculation.
        Each face is assumed to be arranged counter clockwise from the outside.

        Returns:
            a tx3 jnp.ndarray of vertex indices for all t triangles
        """
        triangles = []
        for face in self.faces:
            if self.vertices[0] in face:
                continue
            for v1, v2 in zip(face[1:-1], face[2:]):
                triangles.append([face[0], v1, v2])
        return jnp.array(triangles)


class Arm3D(ConstrainedDynamics):
    """A concrete instance of ConstrainedDynamics that models a 3D muscular hydrostat
    with cells of constant volume.

    Properties:
        cells: a list of Cell3D objects that make up the arm
        edges: a list of edges, each edge is a tuple of vertex indices
        faces: a list of faces, each face is a tuple of vertex indices
        vertex_damping: a length n array of vertex damping rates
        edge_damping: a length m array of edge damping rates"""

    def __init__(
        self,
        cells: Cell3D,
        constraints=None,
    ):
        self.cells = cells
        vertices = []
        self.edges = []
        self.faces = []
        self.edge_damping = []

        for cell in self.cells:
            for vertex in cell.vertices:
                if vertex not in vertices:
                    vertices.append(vertex)

            for e, edge in enumerate(cell.edges):
                edge = sorted(edge)
                if edge not in self.edges:
                    self.edges.append(edge)
                    self.edge_damping.append(cell.edge_damping[e])

            for face in cell.faces:
                face = sorted(face)
                if face not in self.faces:
                    self.faces.append(face)

        num_particles = len(vertices)
        self.edges = jnp.array(self.edges)
        self.edge_damping = jnp.array(self.edge_damping)
        num_controls = len(self.edges)

        masses = jnp.zeros(num_particles)
        self.vertex_damping = jnp.zeros(num_particles)
        for cell in self.cells:
            for v, vertex in enumerate(cell.vertices):
                masses = masses.at[vertex].set(cell.masses[v])
                self.vertex_damping = self.vertex_damping.at[vertex].set(
                    cell.vertex_damping[v]
                )

        self.constraints = constraints if constraints is not None else []

        super().__init__(
            num_controls,
            masses,
            constraints,
        )

    def _calc_actuation_forces(self, state, control_input):
        pos, vel = self.state2posvel(state)
        edge_forces = jnp.zeros_like(pos)

        def actuation_force_fn(edge, muscle_force):
            i, j = edge
            edge_vector = pos[j] - pos[i]
            edge_vector = edge_vector / jnp.linalg.norm(edge_vector)
            edge_force = edge_vector * muscle_force
            return (i, edge_force), (j, -edge_force)

        (idx_i, force_i), (idx_j, force_j) = jax.vmap(
            actuation_force_fn, in_axes=[0, 0]
        )(self.edges, control_input)

        edge_forces = jnp.zeros_like(pos)
        edge_forces = edge_forces.at[idx_i].add(force_i)
        edge_forces = edge_forces.at[idx_j].add(force_j)
        return edge_forces

    def _calc_passive_forces(self, state):
        pos, vel = self.state2posvel(state)
        passive_forces = []
        passive_edge_forces = self._calc_passive_edge_forces(pos, vel)
        passive_forces.append(passive_edge_forces)
        passive_forces.append(self.vertex_damping[:, None] * vel)
        return passive_forces

    def _calc_passive_edge_forces(self, pos, vel):
        """Calculate the damping forces along edges using vmap."""

        def edge_force_fn(edge, damping_rate):
            i, j = edge
            edge_vector = pos[j] - pos[i]
            edge_unit_vector = edge_vector / jnp.linalg.norm(edge_vector)
            relative_velocity = vel[j] - vel[i]
            edge_velocity = (edge_unit_vector @ relative_velocity) * edge_unit_vector
            edge_damp_force = damping_rate * edge_velocity
            # Returns force contributions for both vertices
            return (i, -edge_damp_force), (j, edge_damp_force)

        # Vectorize over all edges
        (idx_i, force_i), (idx_j, force_j) = jax.vmap(edge_force_fn, in_axes=[0, 0])(
            self.edges, self.edge_damping
        )

        # Accumulate forces at each vertex
        edge_forces = jnp.zeros_like(pos)
        edge_forces = edge_forces.at[idx_i].add(force_i)
        edge_forces = edge_forces.at[idx_j].add(force_j)
        return edge_forces

    def draw(self, state, control, sensor_data, idx):
        color = jnp.array([0.0, 0.0, 0.0])
        pos, _ = self.state2posvel(state)

        # Draw edges
        gl.glBegin(gl.GL_LINES)
        for edge, input in zip(self.edges, control):
            activation = input / (1 + input)
            color = jnp.ones(3) * 1 - activation
            color = color.at[idx].set(1)
            gl.glColor3f(*color)
            for vertex in edge:
                gl.glVertex3f(*pos[vertex])
        gl.glEnd()

        # Draw vertices
        gl.glPointSize(10.0)
        gl.glBegin(gl.GL_POINTS)
        scents = sensor_data.get("scent", jnp.zeros(len(pos)))
        max_scent = max(scents)
        for vertex, scent in zip(pos, scents):
            color = jnp.ones(3) * 1 - scent / max_scent
            color = color.at[idx].set(1)
            gl.glColor3f(*color)
            gl.glVertex3f(*vertex)
        gl.glEnd()


class CubicArmBuilder:
    """A builder for 3D cubic arms

    Creates a stack of cubic cells. Edges are along the cube edges and across diagonals.
    """

    def __init__(
        self,
        height: int,
        width: float = 1,
        base_centroid: jnp.ndarray = jnp.array([0, 0, 0]),
    ):
        self.constraints = []
        self.sensors = []

        self.cells = []
        base_points = jnp.array(
            [[0, 0, 0], [width, 0, 0], [width, width, 0], [0, width, 0]], dtype=float
        )
        base_points = base_points - jnp.array([width / 2, width / 2, 0]) + base_centroid
        cube_vertices = jnp.arange(8)
        cube_edges = jnp.array(
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
        cube_faces = jnp.array(
            [
                [0, 3, 2, 1],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
                [4, 5, 6, 7],
            ]
        )

        self.positions = base_points.copy()

        for level in range(height):
            new_points = base_points + jnp.array([0, 0, level + 1])
            self.positions = jnp.vstack((self.positions, new_points))

            index_offset = 4 * level
            self.cells.append(
                Cell3D(
                    cube_vertices + index_offset,
                    cube_edges + index_offset,
                    cube_faces + index_offset,
                )
            )
        self.velocities = jnp.zeros_like(self.positions)
        self.states = jnp.hstack(
            (self.positions.flatten(), self.velocities.flatten())
        ).flatten()
