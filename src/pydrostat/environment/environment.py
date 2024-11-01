"""Module for anything related to the collection of items in the environment."""


class Environment:
    """An environment has obstacles and food particles. The food particles emit an odor
    which diffuses through the environment."""

    def __init__(
        self,
        obstacles: list[Obstacle] = [],
        foods: list[Food] = [],
        limits=None,
        spatial_resolution: float = 0.1,
        steady_state_error: float = 0.05,
    ):
        self.obstacles = obstacles
        self.foods = foods
        self.concentration = self._calc_diffusion()

    def _calc_diffusion(self):
        pass

    def add_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle):
        self.obstacles.remove(obstacle)

    def sample_scent(self, coordinates):
        indices = self.coord_to_idx(coordinates)
        return self.concentration[*indices.T]

    def coord_to_idx(self, coordinates):
        return (
            (coordinates - [self.xlim[0], self.ylim[0], self.zlim[0]]) // self.dx
        ).astype(int)
