"""Module for anything related to the collection of items in the environment."""

import numpy as np
from scipy.ndimage import convolve, generate_binary_structure, laplace

from .food import Food
from .obstacle_interface import IObstacle


class Environment:
    """An environment has obstacles and food particles. The food particles emit an odor
    which diffuses through the environment.

    An enivornment can be any dimension."""

    def __init__(
        self,
        dim: int,
        obstacles: list[IObstacle] = [],
        foods: list[Food] = [],
        limits=None,
        spatial_resolution: float = 0.2,
        steady_state_error: float = 0.05,
        max_iterations: int = 1000,
    ):
        self.dim = dim
        self.obstacles = obstacles

        self.food_locations = [food.location for food in foods]
        self.food_magnitudes = [food.magnitude for food in foods]
        self.food_locations = np.array(self.food_locations)
        self.food_magnitudes = np.array(self.food_magnitudes)

        self.spatial_resolution = spatial_resolution
        # 0.99 is the diffusion coefficient which must be <1. Larger coefficient means
        # faster convergence, but >1 causes overshoot and eventual instability.
        self.dt = spatial_resolution**2 / (self.dim * 2) * 0.99
        self.steady_state_error = steady_state_error
        self.max_iterations = max_iterations

        self.limits = limits
        if limits is None:
            self.limits = [-5, 5] * (dim - 1)
            self.limits.extend([0, 10])
        self.limits = np.array(self.limits).reshape(self.dim, 2)

        coords = [np.arange(*limit, self.spatial_resolution) for limit in self.limits]
        grids = np.meshgrid(*coords)
        self.grid_points = np.stack(grids, axis=self.dim).reshape(
            -1, self.dim
        )  # list of all grid points

        self.concentration = np.zeros_like(grids[0])
        if len(self.food_locations) > 0:
            self.concentration = self._calc_diffusion()

    def _calc_diffusion(self):
        food_indices = self._coord_to_idx(self.food_locations)
        self.concentration[*food_indices.T] = self.food_magnitudes
        self.obstacle_mask, self.adjacency_mask = self._calc_obstacle_masks()

        converged = False
        loops = 0

        while not converged and (loops < self.max_iterations):
            loops += 1

            change = (
                self.dt
                * laplace(self.concentration, mode="constant")
                / self.spatial_resolution**2
            )
            change[*food_indices.T] = 0
            change += (
                self.dt
                / self.spatial_resolution**2
                * (self.concentration * self.adjacency_mask)
            )
            new_concentration = self.concentration + change
            new_concentration = new_concentration * (1 - self.obstacle_mask)
            if (
                np.max(
                    np.abs(change[new_concentration != 0])
                    / new_concentration[new_concentration != 0]
                )
                < self.steady_state_error
            ):
                converged = True

            if loops % 50 == 0:
                error = np.max(
                    np.abs(change[new_concentration != 0])
                    / new_concentration[new_concentration != 0]
                )
                print(f"Loop: {loops}, Error: {error:.3f}")
            self.concentration[:] = new_concentration

        if loops == self.max_iterations:
            raise ValueError(
                "Concentration failed to converge. Increase max iterations or increase acceptable steady state error."
            )
        else:
            print("Converged!")

    def add_obstacle(self, obstacle: IObstacle):
        self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle):
        self.obstacles.remove(obstacle)

    def sample_scent(self, coordinates):
        indices = self._coord_to_idx(coordinates)
        return self.concentration[*indices.T]

    def _coord_to_idx(self, coordinates: np.ndarray) -> np.ndarray[int]:
        """Given a coordinate in space, return the grid index"""
        return (
            (coordinates - self.limits[:, 0][None, :]) / self.spatial_resolution
        ).astype(int)

    def _calc_obstacle_masks(self):
        obstacle_mask = np.zeros_like(self.concentration)
        for obstacle in self.obstacles:
            obstacle_mask += obstacle.check_intersection(self.grid_points).reshape(
                self.concentration.shape
            )
        # obstacle_mask = obstacle_mask.transpose(
        #     1, 0, 2
        # )  # not sure why not reshaping properly, but this fixes it
        # TODO: see if commenting out ruins
        adjacency_kernel = generate_binary_structure(self.dim, 1)
        # adjacency_kernel = np.array(
        #     [
        #         [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        #         [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        #         [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        #     ]
        # )
        adjacency_mask = convolve(obstacle_mask, adjacency_kernel, mode="constant")
        return obstacle_mask, adjacency_mask
