"""Module for anything related to the collection of items in the environment."""

import numpy as np
from scipy.ndimage import convolve, generate_binary_structure, laplace

from .food import Food
from .obstacle_interface import IObstacle


class Environment:
    """An environment has obstacles and food particles. The food particles emit an odor
    which diffuses through the environment.

    An enivornment can be any dimension."""
    
    # Class Variables
    dim: int
    obstacles: list[IObstacle]
    spatial_resolution: float
    steady_state_error: float
    max_iterations: int
    foods: list[Food]
    dt: float
    limits=None
    food_locations: np.ndarray
    food_magnitudes: np.ndarray
    grid_points: np.ndarray
    concentration: np.ndarray

    # Methods
    def __init__(
        self,
        dim: int,
        obstacles: list[IObstacle] = [],
        spatial_resolution: float = 0.2,
        steady_state_error: float = 0.05,
        max_iterations: int = 1000,
        foods: list[Food] = [],
        limits = None
    ):
        self.dim = dim
        self.obstacles = obstacles
        self.spatial_resolution = spatial_resolution
        self.steady_state_error = steady_state_error
        self.max_iterations = max_iterations

        # 0.99 is the diffusion coefficient which must be <1. Larger coefficient means
        # faster convergence, but >1 causes overshoot and eventual instability.
        self.dt = spatial_resolution**2 / (self.dim * 2) * 0.99
        
        self.__initialize_limits(limits)

        self.food_locations = np.array([food.location for food in foods])
        self.food_magnitudes = np.array([food.magnitude for food in foods])

        grids = self.__calc_grid()
        self.__initialize_grid_points(grids)
        self.__initialize_concentration(grids)


    def __initialize_limits(self, limits):
        self.limits = limits
        if limits is None:
            self.limits = [-5, 5] * (self.dim - 1)
            self.limits.extend([-2, 10])
        self.limits = np.array(self.limits).reshape(self.dim, 2)


    def __calc_grid(self):
        coords = [np.arange(*limit, self.spatial_resolution) for limit in self.limits]
        grids = np.meshgrid(*coords)
        return grids


    def __initialize_grid_points(self, grids):
        # list of all grid points
        self.grid_points = np.stack(grids, axis=self.dim).reshape(-1, self.dim)  


    def __initialize_concentration(self, grids):
        self.concentration = np.zeros_like(grids[0])
        if len(self.food_locations) > 0:
            self.concentration = self._calc_diffusion()


    def _calc_diffusion(self):
        food_indices = self._coord_to_idx(self.food_locations)
        concentration = self.concentration.copy()

        concentration[tuple(food_indices.T)] = self.food_magnitudes
        obstacle_mask = self.__calc_obstacle_mask()

        converged = False
       
        loops = 0
        while not converged and (loops < self.max_iterations):
            loops += 1

            change = self.__calc_concentration_change(food_indices, concentration, obstacle_mask)
            new_concentration = self.__calc_new_concentration(concentration, obstacle_mask, change)
            error = self.__calc_error(change, new_concentration)
            
            if (error < self.steady_state_error):

                converged = True

            if loops % 50 == 0:
                print(f"Loop: {loops}, Error: {error:.3f}")

            concentration[:] = new_concentration

        if loops >= self.max_iterations:
            raise ValueError(
                "Concentration failed to converge. Increase max iterations or increase acceptable steady state error."
            )
        
        print("Converged")
        return concentration


    def __calc_error(self, change,concentration):
        return np.max(
                    np.abs(change[concentration != 0])
                    / concentration[concentration != 0]
                )

    
    def __calc_obstacle_mask(self):
        obstacle_mask = np.zeros_like(self.concentration)
        for obstacle in self.obstacles:
            obstacle_mask += obstacle.check_intersection(self.grid_points).reshape(
                self.concentration.shape
            )
        return obstacle_mask.transpose(1, 0, 2)  # not sure why not reshaping properly, but this fixes it


    def __calc_adjacency_mask(self, obstacle_mask):
         # TODO: see if commenting out ruins
        adjacency_kernel = generate_binary_structure(self.dim, 1)
        # adjacency_kernel = np.array(
        #     [
        #         [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        #         [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        #         [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        #     ]
        # )
        return convolve(obstacle_mask, adjacency_kernel, mode="constant")


    def __calc_new_concentration(self, concentration, obstacle_mask, change):
        return (concentration + change) * (1 - obstacle_mask)


    def __calc_concentration_change(self, food_indices, concentration, obstacle_mask):
        change = (
                self.dt
                * laplace(concentration, mode="constant")
                / self.spatial_resolution**2
            )
        change[*food_indices.T] = 0
        adjacency_mask = self.__calc_adjacency_mask(obstacle_mask)
        change += (
                self.dt
                / self.spatial_resolution**2
                * (concentration * adjacency_mask)
            )
        
        return change
    

    def add_obstacle(self, obstacle: IObstacle):
        self.obstacles.append(obstacle)


    def remove_obstacle(self, obstacle):
        self.obstacles.remove(obstacle)


    def sample_scent(self, coordinates):
        indices = self._coord_to_idx(coordinates)
        return self.concentration[tuple(indices.T)]


    def _coord_to_idx(self, coordinates: np.ndarray) -> np.ndarray[int]:
        """Given a coordinate in space, return the grid index"""
        return (
            (coordinates - self.limits[:, 0][None, :]) / self.spatial_resolution
        ).astype(int)

 
