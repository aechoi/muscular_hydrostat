"""Set of premade polytopes for testing"""

import numpy as np


class Square:
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    vertices = [0, 1, 2, 3]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    faces = [[0, 1, 2, 3]]


class Tetrahedron:
    points = np.array(
        [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],  # 0  # 1  # 2  # 3
        dtype=float,
    )
    vertices = [0, 1, 2, 3]
    edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]


class SquarePyramid:
    points = np.array(
        [[0, 0, 0.2], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 1, 2]], dtype=float
    )
    vertices = [0, 1, 2, 3, 4]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
    faces = [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]


class TriangularPrism:
    points = np.array(
        [[0, 0, 0], [2, 0, 0], [1, 3**0.5, 0], [0, 0, 4], [2, 0, 4], [1, 3**0.5, 4]],
        dtype=float,
    )
    vertices = [0, 1, 2, 3, 4, 5]
    edges = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 4], [2, 5], [3, 4], [4, 5], [5, 3]]
    faces = [[0, 2, 1], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5], [3, 4, 5]]


class Cube:
    points = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ],
        dtype=float,
    )
    vertices = [0, 1, 2, 3, 4, 5, 6, 7]
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
    ]
    faces = [
        [0, 3, 2, 1],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [4, 5, 6, 7],
    ]


class CubeArm:
    def __init__(self, height=5):
        base_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        base_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        base_face = np.array([[0, 3, 2, 1]])
        upright_face_base = np.array(
            [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        )
        self.points = base_points.copy()
        self.edges = base_edges.copy()
        self.faces = base_face.copy()

        for level in range(height):
            new_points = base_points + np.array([0, 0, level + 1])
            self.points = np.vstack((self.points, new_points))

            upright_edges = np.column_stack(
                (np.arange(4) + 4 * (level), np.arange(4) + 4 * (level + 1))
            )
            self.edges = np.vstack(
                (self.edges, upright_edges, base_edges + 4 * (level + 1))
            )

            self.faces = np.vstack(
                (
                    self.faces,
                    upright_face_base + 4 * (level),
                    base_face + 4 * (level + 1),
                )
            )
        self.edges = self.edges.tolist()
        self.faces = self.faces.tolist()
        self.vertices = np.arange(len(self.points)).tolist()
