from dataclasses import dataclass
import numpy as np


@dataclass
class Food:
    location: np.ndarray
    magnitude: float
