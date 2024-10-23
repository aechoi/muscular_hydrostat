"""This module simulates the diffusion of scent through an environment with
obstacles. The final steady state distribution is then used by the hydrostat
arm."""

import numpy as np
import scipy.ndimage as nd

class Environment3D:
    """A 3D environment with obstacles and food"""

    def __init__(
        self,
        obstacles,
        food,
        lims=[[-5, 5], [-5, 5], [0, 10]],
        dx=0.1,
        ss_error=0.05,
    ):
        self.obstacles = obstacles
        self.food_locs = food[:, :3]
        self.food_strengths = food[:, 3]

        self.xlim = lims[0]
        self.ylim = lims[1]
        self.zlim = lims[2]
        self.dx = dx
        # diffusion coefficient doesn't matter since we are looking at steady
        # state, so assume 1
        self.dt = dx**2 / 6 * 0.99

        x = np.arange(*lims[0], self.dx)
        y = np.arange(*lims[1], self.dx)
        z = np.arange(*lims[2], self.dx)
        X, Y, Z = np.meshgrid(x, y, z)
        self.grid_points = np.stack((X, Y, Z), axis=3).reshape(-1, 3)

        self.concentration = np.zeros_like(X, dtype=float)
        food_indices = self.coord_to_idx(self.food_locs).T
        self.concentration[*food_indices] = self.food_strengths
        self.obstacle_mask, self.adjacency_mask = self.calc_obstacle_masks()

        converged = False
        max_loops = 1000
        loops = 0

        while not converged and (loops < max_loops):
            loops += 1

            change = (
                self.dt * nd.laplace(self.concentration, mode="constant") / self.dx**2
            )
            change[*food_indices] = 0
            change += self.dt / self.dx**2 * (self.concentration * self.adjacency_mask)
            new_concentration = self.concentration + change
            new_concentration = new_concentration * (1 - self.obstacle_mask)
            if (
                np.max(
                    np.abs(change[new_concentration != 0])
                    / new_concentration[new_concentration != 0]
                )
                < ss_error
            ):
                converged = True

            if loops % 50 == 0:
                error = np.max(
                    np.abs(change[new_concentration != 0])
                    / new_concentration[new_concentration != 0]
                )
                print(f"Loop: {loops}, Error: {error:.3f}")
            self.concentration[:] = new_concentration

        if loops == max_loops:
            print("Failed to converge")

    def coord_to_idx(self, coords):
        return (
            (coords - [self.xlim[0], self.ylim[0], self.zlim[0]]) // self.dx
        ).astype(int)

    def sample_scent(self, coordinate):
        indices = self.coord_to_idx(coordinate)
        return self.concentration[*indices.T]

    def calc_obstacle_masks(self):
        obstacle_mask = np.zeros_like(self.concentration)
        for obstacle in self.obstacles:
            obstacle_mask += obstacle.calc_many_intersections(self.grid_points).reshape(
                self.concentration.shape
            )
        obstacle_mask = np.clip(obstacle_mask, 0, 1).transpose(
            1, 0, 2
        )  # not sure why not reshaping properly
        adjacency_kernel = np.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ]
        )
        adjacency_mask = nd.convolve(obstacle_mask, adjacency_kernel, mode="constant")
        return obstacle_mask, adjacency_mask
